const $ = (id) => document.getElementById(id);

const LIVE_STATES = new Set(["starting", "running", "stopping"]);

let lastState = null;
let previewRfb = null;
let previewConnected = false;
let previewConnecting = false;
let previewCreds = null;
let previewConfig = null;
let previewConfigPromise = null;
let previewGeneration = 0;
let previewError = "";
let liveTimer = null;
let liveActive = false;
let liveFailures = 0;
let liveAvailable = false;
let liveBackoffUntil = 0;
const LIVE_PREVIEW_MS = 100;
const LIVE_BACKOFF_MS = 10000;
let pollInFlight = false;
let actionInFlight = false;
let clientError = "";
let lastServerError = null;

const MANUAL_SEND_MS = 20;
const MANUAL_KEY_CODES = {
  Digit1: "1", Digit2: "2", Digit3: "3", Digit4: "4", Digit5: "5",
  KeyQ: "q", KeyW: "w", KeyE: "e", KeyR: "r", KeyT: "t", KeyU: "u",
  KeyI: "i", KeyO: "o", KeyP: "p", KeyA: "a", KeyS: "s", KeyD: "d",
  KeyF: "f", KeyG: "g", KeyH: "h", KeyJ: "j", KeyK: "k", KeyL: "l",
  KeyZ: "z", KeyX: "x", KeyC: "c", KeyV: "v", KeyB: "b", KeyN: "n",
  KeyM: "m", Space: "space", Tab: "tab", Enter: "enter",
  ShiftLeft: "shift", ShiftRight: "shift",
  ControlLeft: "ctrl", ControlRight: "ctrl",
  ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
};

function manualClientId() {
  const bytes = new Uint8Array(12);
  crypto.getRandomValues(bytes);
  return `browser-${Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

const manualClient = manualClientId();
let manualLocked = false;
let manualTimer = null;
let manualInFlight = false;
let manualPendingDisarm = false;
let manualSeq = 0;
let manualLastError = "";
const manualKeys = new Set();
const manualButtons = new Set();
const manualMouse = { dx: 0, dy: 0, wheel: 0 };

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      message = (await response.json()).error || message;
    } catch {
      // keep the status text
    }
    throw new Error(message);
  }
  return response.json();
}

async function post(path, body) {
  return api(path, {
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

function showError(message) {
  $("error").textContent = message || "";
  $("error").style.display = message ? "block" : "none";
}

function renderError() {
  showError(clientError || lastServerError || "");
}

function setClientError(message) {
  clientError = message || "";
  renderError();
}

function applyServerError(serverError) {
  const next = serverError || null;
  if (next !== lastServerError) {
    lastServerError = next;
    if (next) clientError = "";
  }
  renderError();
}

function uptime(startedAt) {
  if (!startedAt) return "";
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000));
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function previewAuthVisible() {
  return $("vnc-auth").classList.contains("visible");
}

function setPreviewError(message) {
  previewError = message || "";
  $("vnc-error").textContent = previewError;
}

function setPreviewConnecting(connecting) {
  previewConnecting = connecting;
  const button = $("vnc-auth").querySelector("button[type=submit]");
  if (!button) return;
  button.disabled = connecting;
  button.textContent = connecting ? "Connecting…" : "Connect preview";
}

function showPreviewAuth() {
  const wasVisible = previewAuthVisible();
  $("vnc-auth").classList.add("visible");
  $("vnc-error").textContent = previewError || "";
  if (!wasVisible) $("vnc-pass").focus();
}

function hidePreviewAuth() {
  $("vnc-auth").classList.remove("visible");
  $("vnc-pass").value = "";
}

async function loadPreviewConfig() {
  if (previewConfig) return previewConfig;
  const generation = previewGeneration;
  if (!previewConfigPromise) {
    const promise = api("/api/preview/config");
    previewConfigPromise = promise;
    const clear = () => {
      if (previewConfigPromise === promise) previewConfigPromise = null;
    };
    promise.then(clear, clear);
  }
  const config = await previewConfigPromise;
  if (generation === previewGeneration) {
    previewConfig = config;
    $("vnc-user").value = config.username || "";
    if (config.assets_available === false) {
      setPreviewError("noVNC assets are unavailable");
    }
  }
  return config;
}

function stopLivePreview(failed = false) {
  liveActive = false;
  if (failed) liveBackoffUntil = Date.now() + LIVE_BACKOFF_MS;
  if (liveTimer) {
    clearTimeout(liveTimer);
    liveTimer = null;
  }
  liveAvailable = false;
  liveFailures = 0;
  const img = $("preview-live");
  if (img) {
    img.hidden = true;
    img.removeAttribute("src");
  }
}

function liveTick() {
  if (!liveActive) return;
  const probe = new Image();
  probe.onload = () => {
    if (!liveActive) return;
    liveAvailable = true;
    liveFailures = 0;
    const img = $("preview-live");
    img.src = probe.src;
    img.hidden = false;
    hidePreviewAuth();
    $("preview-wait").style.display = "none";
    liveTimer = setTimeout(liveTick, LIVE_PREVIEW_MS);
  };
  probe.onerror = () => {
    if (!liveActive) return;
    liveFailures += 1;
    if (liveFailures < 10) {
      liveTimer = setTimeout(liveTick, 300);
    } else {
      // The layer is not publishing: fall back to the VNC transport.
      stopLivePreview(true);
    }
  };
  probe.src = `/api/preview/live.bmp?t=${Date.now()}`;
}

function startLivePreview() {
  if (liveActive || Date.now() < liveBackoffUntil) return;
  liveActive = true;
  liveAvailable = false;
  liveFailures = 0;
  const img = $("preview-live");
  if (img) img.hidden = false;
  liveTick();
}

function closePreview() {
  stopLivePreview();
  previewGeneration += 1;
  const rfb = previewRfb;
  previewRfb = null;
  previewConnected = false;
  setPreviewConnecting(false);
  if (rfb) {
    try {
      rfb.disconnect();
    } catch {
      // already gone
    }
  }
  previewConfig = null;
  previewConfigPromise = null;
  hidePreviewAuth();
  $("vnc-pass").value = "";
  previewCreds = null;
  setPreviewError("");
}

async function connectPreview() {
  if (previewRfb || previewConnecting || !previewCreds) return;
  const generation = previewGeneration;
  setPreviewConnecting(true);
  try {
    const config = await loadPreviewConfig();
    if (generation !== previewGeneration || !previewCreds) return;
    if (!config || config.assets_available === false) {
      setPreviewError("noVNC assets are unavailable");
      previewCreds = null;
      showPreviewAuth();
      return;
    }
    const { default: RFB } = await import("/novnc/core/rfb.js");
    if (generation !== previewGeneration || !previewCreds) return;
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = config.ws_url || `${protocol}://${location.hostname}:${config.proxy_port}/`;
    const rfb = new RFB($("preview"), wsUrl, {
      credentials: previewCreds,
      wsProtocols: ["binary"],
    });
    if (generation !== previewGeneration || !previewCreds) {
      try {
        rfb.disconnect();
      } catch {
        // already gone
      }
      return;
    }
    previewRfb = rfb;
    previewConnected = false;
    rfb.viewOnly = true;
    rfb.scaleViewport = false;
    rfb.resizeSession = false;
    rfb.focusOnClick = false;

    rfb.addEventListener("connect", () => {
      if (previewRfb !== rfb) return;
      previewConnected = true;
      setPreviewConnecting(false);
      previewCreds = null;
      previewError = "";
      $("vnc-error").textContent = "";
      hidePreviewAuth();
      $("preview-wait").style.display = "none";
    });

    rfb.addEventListener("securityfailure", (event) => {
      if (previewRfb !== rfb) return;
      previewConnected = false;
      setPreviewConnecting(false);
      setPreviewError(event.detail?.reason || "authentication failed");
      showPreviewAuth();
    });

    rfb.addEventListener("credentialsrequired", () => {
      if (previewRfb !== rfb) return;
      previewConnected = false;
      setPreviewConnecting(false);
      setPreviewError("VNC credentials are required");
      showPreviewAuth();
      rfb.disconnect();
    });

    rfb.addEventListener("disconnect", (event) => {
      if (previewRfb !== rfb) return;
      previewGeneration += 1;
      previewRfb = null;
      previewConnected = false;
      setPreviewConnecting(false);
      previewCreds = null;
      if (!event.detail.clean) {
        if (!previewError) setPreviewError("VNC connection closed; check the password and that the stream is running");
        showPreviewAuth();
      } else if (previewError) {
        showPreviewAuth();
      } else {
        hidePreviewAuth();
      }
    });
  } catch (error) {
    if (generation !== previewGeneration) return;
    setPreviewConnecting(false);
    previewRfb = null;
    previewConnected = false;
    previewCreds = null;
    setPreviewError(error.message || "preview connection failed");
    showPreviewAuth();
  } finally {
    if (generation === previewGeneration && !previewRfb) setPreviewConnecting(false);
  }
}

function syncPreview(state) {
  if (!LIVE_STATES.has(state.state)) {
    closePreview();
    return;
  }
  startLivePreview();
  if (liveActive) {
    if (liveAvailable) hidePreviewAuth();
    return;
  }
  const preview = state.preview || {};
  const transportAlive = Boolean(preview.vnc_alive && preview.proxy_alive);
  if (transportAlive) {
    if (previewRfb && previewConnected) {
      hidePreviewAuth();
    } else if (previewCreds) {
      connectPreview();
    } else {
      if (!previewConfig) {
        loadPreviewConfig().catch((error) => {
          setPreviewError(error.message || "preview configuration unavailable");
        });
      }
      showPreviewAuth();
    }
  } else {
    hidePreviewAuth();
  }
}

function updatePreviewWait(state) {
  const wait = $("preview-wait");
  const stateText = {
    stopped: "session stopped",
    failed: "session failed",
    starting: "starting…",
  }[state.state];
  if (stateText) {
    wait.textContent = stateText;
    wait.style.display = "block";
    return;
  }
  if (liveActive) {
    if (liveAvailable) {
      wait.style.display = "none";
    } else {
      wait.textContent = "waiting for live capture…";
      wait.style.display = "block";
    }
    return;
  }
  const preview = state.preview || {};
  const transportAlive = Boolean(preview.vnc_alive && preview.proxy_alive);
  if (previewRfb && previewConnected) {
    wait.style.display = "none";
  } else if (!transportAlive) {
    wait.textContent = "waiting for VNC transport…";
    wait.style.display = "block";
  } else if (previewAuthVisible()) {
    wait.style.display = "none";
  } else {
    wait.textContent = "authenticate to view the preview";
    wait.style.display = "block";
  }
}

function applyControls(state) {
  const running = state.state === "running";
  $("start").disabled = actionInFlight || !["stopped", "failed"].includes(state.state);
  $("stop").disabled = actionInFlight || state.state === "stopped";
  document.querySelectorAll("#modes button").forEach((button) => {
    const active = button.dataset.mode === state.mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
    button.disabled = actionInFlight || !running;
  });
  $("kill-game").disabled = actionInFlight || !running;
  $("kill-steam").disabled = actionInFlight || !running;
  const audio = state.audio || {};
  const audioButton = $("mute-audio");
  audioButton.textContent = audio.muted ? "Unmute game audio" : "Mute game audio";
  audioButton.disabled = actionInFlight || !running || !audio.available;
  const body = state.body || null;
  const bodyRunning = Boolean(body && body.running);
  $("body-start").disabled = actionInFlight || !body || !body.enabled || bodyRunning;
  $("body-stop").disabled = actionInFlight || !body || !bodyRunning;
}

function clamp(value, limit) {
  return Math.max(-limit, Math.min(limit, value));
}

function clearManualMouse() {
  manualMouse.dx = 0;
  manualMouse.dy = 0;
  manualMouse.wheel = 0;
}

function clearManualState() {
  manualKeys.clear();
  manualButtons.clear();
  clearManualMouse();
}

function stopManualTimer() {
  if (manualTimer !== null) {
    clearInterval(manualTimer);
    manualTimer = null;
  }
}

function sendManualDisarm() {
  fetch("/api/manual/input", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ client: manualClient, seq: ++manualSeq, armed: false }),
  }).catch(() => {});
}

function releaseManual() {
  const wasActive = manualLocked || manualInFlight || manualPendingDisarm;
  manualLocked = false;
  stopManualTimer();
  clearManualState();
  if (wasActive) {
    if (manualInFlight) manualPendingDisarm = true;
    else sendManualDisarm();
  }
  if (document.pointerLockElement) document.exitPointerLock();
  updateManualUi(lastState);
}

async function manualTick() {
  if (!manualLocked || manualInFlight) return;
  manualInFlight = true;
  const payload = {
    client: manualClient,
    seq: ++manualSeq,
    armed: true,
    keys: [...manualKeys].sort(),
    buttons: [...manualButtons].sort(),
    dx: manualMouse.dx,
    dy: manualMouse.dy,
    wheel: manualMouse.wheel,
  };
  clearManualMouse();
  try {
    const response = await fetch("/api/manual/input", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      let message = `input rejected (${response.status})`;
      try {
        message = (await response.json()).error || message;
      } catch {
        // keep the status message
      }
      manualLastError = message;
      releaseManual();
    } else {
      manualLastError = "";
    }
  } catch {
    manualLastError = "input unavailable";
  } finally {
    manualInFlight = false;
    if (!manualLocked && manualPendingDisarm) {
      manualPendingDisarm = false;
      sendManualDisarm();
    }
  }
  updateManualUi(lastState);
}

function startManualTimer() {
  stopManualTimer();
  manualTick();
  manualTimer = setInterval(manualTick, MANUAL_SEND_MS);
}

function requestManualControl() {
  const state = lastState;
  if (!state || state.state !== "running" || state.mode !== "manual") return;
  if (!state.manual || state.manual.available === false) {
    manualLastError = state.manual?.last_error || "manual input is not ready yet";
    updateManualUi(state);
    return;
  }
  const preview = $("preview-box");
  const fallback = () => {
    try {
      const result = preview.requestPointerLock();
      if (result && typeof result.catch === "function") {
        result.catch(() => {
          manualLastError = "pointer lock was blocked";
          updateManualUi(lastState);
        });
      }
    } catch (error) {
      manualLastError = error.message;
      updateManualUi(lastState);
    }
  };
  try {
    const result = preview.requestPointerLock({ unadjustedMovement: true });
    if (result && typeof result.catch === "function") result.catch(fallback);
  } catch {
    fallback();
  }
}

function updateManualUi(state) {
  const overlay = $("manual-overlay");
  const manualRunning = Boolean(state && state.state === "running" && state.mode === "manual");
  overlay.classList.toggle("visible", manualRunning);
  document.body.classList.toggle("manual-mode", manualRunning);
  document.body.classList.toggle("manual-active", manualLocked);
  if (!manualRunning) return;

  const manual = state.manual || {};
  const available = manual.available !== false;
  const foreign = Boolean(manual.armed && manual.client && manual.client !== manualClient);
  let title = "Click preview to control";
  let detail = "Mouse look · WASD move · arrows hand · F punch · LMB fire · RMB grip · Tab hand · Esc releases";
  if (manualLocked) {
    if (manual.armed && manual.client === manualClient) {
      title = "Manual control active";
      detail = "Esc releases the mouse · click the preview to re-arm";
    } else {
      title = "Capturing input…";
      detail = "waiting for the manual client";
    }
  } else if (foreign) {
    title = "Another browser has control";
    detail = "Click the preview to take over";
  } else if (!available) {
    title = manual.last_error ? "Manual input unavailable" : "Waiting for the manual client…";
    detail = manual.last_error || "The session is still starting";
  } else if (manualLastError) {
    title = "Input problem";
    detail = manualLastError;
  }
  $("manual-title").textContent = title;
  $("manual-detail").textContent = detail;
}

function manualKey(code) {
  return MANUAL_KEY_CODES[code] || null;
}

function render(state) {
  lastState = state;

  $("state").textContent = state.state;
  $("state").className = `badge ${state.state}`;
  // Weston serves the full 1920x1080 framebuffer over VNC. The state's height
  // used to subtract a 32px capture panel that does not exist in this view,
  // letterboxing the game - so the clip is pinned to 16:9, not to the state.
  $("preview-clip").style.aspectRatio = "16 / 9";

  const showSession = state.state === "running" || state.state === "stopping";
  $("session-id").textContent = showSession ? state.session_id || "" : "";
  $("session-id").style.display = showSession ? "" : "none";
  $("uptime").textContent = showSession ? uptime(state.started_at) : "";
  $("uptime").style.display = showSession ? "" : "none";

  if (
    manualLocked
    && (state.state !== "running" || state.mode !== "manual")
  ) {
    releaseManual();
  }
  if (
    manualLocked
    && state.manual?.armed
    && state.manual.client
    && state.manual.client !== manualClient
  ) {
    releaseManual();
  }

  applyControls(state);
  syncPreview(state);
  updatePreviewWait(state);
  updateManualUi(state);
  applyServerError(state.last_error);

  renderHealth(state);
  bodySync();
}

function healthCard(label, value, valueClass, detail) {
  const card = document.createElement("div");
  card.className = "card";
  const strong = document.createElement("strong");
  strong.textContent = label;
  const span = document.createElement("span");
  span.className = valueClass || "";
  span.textContent = value;
  const small = document.createElement("small");
  small.textContent = detail || "";
  card.append(strong, span, small);
  return card;
}

function renderHealth(state) {
  const cards = Object.entries(state.processes).map(([name, process]) =>
    healthCard(name, process.alive ? "alive" : "down",
      process.alive ? "ok" : "bad", process.pid ?? "")
  );
  cards.push(healthCard("OpenXR", state.openxr_client || "none"));
  const preview = state.preview || {};
  const transportAlive = Boolean(preview.vnc_alive && preview.proxy_alive);
  cards.push(healthCard("Preview", transportAlive ? "live" : "waiting",
    transportAlive ? "ok" : "bad", `${preview.mode ?? ""} · ${preview.fps_target ?? 0} fps`));
  $("health").replaceChildren(...cards);
}

async function refresh() {
  if (pollInFlight) return;
  pollInFlight = true;
  try {
    render(await api("/api/state"));
  } catch (error) {
    setClientError(error.message);
  } finally {
    pollInFlight = false;
  }
}

async function runAction(action) {
  if (actionInFlight) return;
  actionInFlight = true;
  if (lastState) applyControls(lastState);
  try {
    await action();
    setClientError("");
    await refresh();
  } catch (error) {
    setClientError(error.message);
  } finally {
    actionInFlight = false;
    if (lastState) applyControls(lastState);
  }
}

$("vnc-auth").addEventListener("submit", (event) => {
  event.preventDefault();
  setPreviewError("");
  const username = $("vnc-user").value;
  const password = $("vnc-pass").value;
  $("vnc-pass").value = "";
  previewCreds = { username, password };
  connectPreview();
});

$("start").onclick = () => runAction(() => post("/api/session/start"));

$("stop").onclick = () => runAction(() => post("/api/session/stop"));

document.querySelectorAll("#modes button").forEach((button) => {
  button.onclick = () => runAction(() => post("/api/session/mode", { mode: button.dataset.mode }));
});

$("kill-game").onclick = () => {
  if (!confirm("Kill the game process?")) return;
  runAction(() => post("/api/session/kill", { target: "game" }));
};

$("kill-steam").onclick = () => {
  if (!confirm("Shut down the hidden Steam client?")) return;
  runAction(() => post("/api/session/kill", { target: "steam" }));
};

$("mute-audio").onclick = () => runAction(() => post("/api/audio/mute", { muted: !(lastState?.audio?.muted) }));

$("body-start").onclick = () => runAction(() => post("/api/body/start"));

$("body-stop").onclick = () => runAction(() => post("/api/body/stop"));

// Logs: every session log in one colour-coded stream. The chips filter sources;
// the console sticks to the bottom unless you scroll up to read history.
const LOG_SOURCES = ["session", "monado", "weston", "proxy", "steam", "pose"];
const LOG_POLL_MS = 2000;
const logHidden = new Set();
let logData = {};
let logSources = [];
let logChipKey = "";
let logSignature = "";

function buildLogChips(sources) {
  const box = $("log-filters");
  box.replaceChildren();
  for (const name of sources) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.dataset.source = name;
    chip.textContent = name;
    chip.setAttribute("aria-pressed", "true");
    chip.onclick = () => {
      if (logHidden.has(name)) logHidden.delete(name);
      else logHidden.add(name);
      const on = !logHidden.has(name);
      chip.classList.toggle("off", !on);
      chip.setAttribute("aria-pressed", String(on));
      renderLog(true);
    };
    box.appendChild(chip);
  }
}

function renderLog(stick) {
  const tail = $("log-tail");
  const frag = document.createDocumentFragment();
  for (const name of logSources) {
    if (logHidden.has(name)) continue;
    for (const line of logData[name] || []) {
      const row = document.createElement("div");
      row.className = `log-line src-${name}`;
      const tag = document.createElement("span");
      tag.className = "log-src";
      tag.textContent = name;
      const text = document.createElement("span");
      text.textContent = line || " ";
      row.append(tag, text);
      frag.appendChild(row);
    }
  }
  tail.replaceChildren(frag);
  if (stick) tail.scrollTop = tail.scrollHeight;
}

async function loadLog() {
  const tail = $("log-tail");
  const stick = tail.scrollTop + tail.clientHeight >= tail.scrollHeight - 8;
  try {
    const response = await fetch("/api/logs");
    if (!response.ok) throw new Error(`failed: ${response.status}`);
    const doc = await response.json();
    logData = doc.logs || {};
    logSources = Object.keys(logData).sort((a, b) => {
      const rank = (name) => (LOG_SOURCES.includes(name) ? LOG_SOURCES.indexOf(name) : 99);
      return rank(a) - rank(b);
    });
    const key = logSources.join(",");
    if (key !== logChipKey) {
      logChipKey = key;
      buildLogChips(logSources);
    }
    // Rebuilding ~1k rows every poll is wasteful; only redraw on new output.
    const signature = logSources
      .map((name) => `${name}:${(logData[name] || []).length}:${(logData[name] || []).at(-1) || ""}`)
      .join("|");
    if (signature === logSignature) return;
    logSignature = signature;
    renderLog(stick);
  } catch (error) {
    tail.textContent = error.message;
  }
}

loadLog();
setInterval(loadLog, LOG_POLL_MS);

$("preview-box").addEventListener("click", (event) => {
  if (event.target.closest("#vnc-auth")) return;
  if (!manualLocked) requestManualControl();
}, true);

document.addEventListener("pointerlockchange", () => {
  const locked = document.pointerLockElement === $("preview-box");
  if (locked) {
    manualLocked = true;
    manualPendingDisarm = false;
    manualLastError = "";
    clearManualState();
    startManualTimer();
  } else if (manualLocked) {
    releaseManual();
  }
  updateManualUi(lastState);
});

document.addEventListener("pointerlockerror", () => {
  manualLastError = "pointer lock was blocked";
  updateManualUi(lastState);
});

document.addEventListener("keydown", (event) => {
  if (!manualLocked) return;
  const key = manualKey(event.code);
  if (!key) return;
  event.preventDefault();
  manualKeys.add(key);
}, true);

document.addEventListener("keyup", (event) => {
  if (!manualLocked) return;
  const key = manualKey(event.code);
  if (!key) return;
  event.preventDefault();
  manualKeys.delete(key);
}, true);

document.addEventListener("mousemove", (event) => {
  if (!manualLocked) return;
  manualMouse.dx = clamp(manualMouse.dx + clamp(event.movementX, 500), 10000);
  manualMouse.dy = clamp(manualMouse.dy + clamp(event.movementY, 500), 10000);
});

document.addEventListener("mousedown", (event) => {
  if (!manualLocked || ![0, 1, 2].includes(event.button)) return;
  event.preventDefault();
  manualButtons.add(event.button);
});

document.addEventListener("mouseup", (event) => {
  if (!manualLocked) return;
  manualButtons.delete(event.button);
});

document.addEventListener("wheel", (event) => {
  if (!manualLocked) return;
  event.preventDefault();
  manualMouse.wheel = clamp(manualMouse.wheel - Math.sign(event.deltaY), 10);
}, { passive: false });

document.addEventListener("contextmenu", (event) => {
  if (manualLocked) event.preventDefault();
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") releaseManual();
});

window.addEventListener("blur", () => releaseManual());

refresh();
setInterval(refresh, 1000);

// Third-person body view. Separate from the session: it is its own MuJoCo
// process, so it can be live while the game session is stopped.
const BODY_REFRESH_MS = 120;
let bodyTimer = null;
let bodyFailures = 0;

function bodyTiles(info) {
  // One tile per configured angle, centred below the game cam. Tiles are
  // keyed by view name so a config change rebuilds them instead of stacking
  // duplicates.
  const box = $("body-tiles");
  const views = (info && info.views) || [];
  const wanted = new Set(views);
  for (const tile of Array.from(box.children)) {
    if (!wanted.has(tile.dataset.view)) tile.remove();
  }
  for (const name of views) {
    let tile = box.querySelector(`[data-view="${name}"]`);
    if (!tile) {
      tile = document.createElement("div");
      tile.className = "body-tile";
      tile.dataset.view = name;
      const img = document.createElement("img");
      img.alt = "";
      const caption = document.createElement("div");
      caption.className = "tile-caption";
      tile.append(img, caption);
      box.append(tile);
    }
    tile.querySelector(".tile-caption").textContent = name.replace(/_/g, " ");
  }
  return Array.from(box.children);
}

function bodyRefresh() {
  const info = (lastState && lastState.body) || null;
  const wait = $("body-wait");
  // "running" only means this dashboard spawned the viewer. A viewer started
  // another way still writes frames, and the server already reports that as
  // frames_unmanaged. Gating on running alone stopped the refresh loop and left
  // the tiles showing whatever they last loaded - live frames on disk, still
  // images on screen.
  const live = Boolean(info && info.frame && (info.running || info.frames_unmanaged));
  if (!live) {
    wait.hidden = false;
    if (!info) {
      wait.textContent = "body view off";
    } else if (info.error) {
      wait.textContent = info.error;
    } else if (info.running) {
      wait.textContent = "waiting for the first frame\u2026";
    } else {
      wait.textContent = "body view off";
    }
    return;
  }
  wait.hidden = true;
  for (const tile of bodyTiles(info)) {
    const img = tile.querySelector("img");
    if (img.dataset.busy) continue;
    img.dataset.busy = "1";
    const probe = new Image();
    const name = tile.dataset.view;
    probe.onload = () => {
      bodyFailures = 0;
      img.src = probe.src;
      delete img.dataset.busy;
    };
    probe.onerror = () => {
      delete img.dataset.busy;
      bodyFailures += 1;
      if (bodyFailures > 8) {
        wait.hidden = false;
        wait.textContent = "waiting for body view";
      }
    };
    probe.src = `/api/body.jpg?view=${encodeURIComponent(name)}&t=${Date.now()}`;
  }
}

function bodySync() {
  const info = (lastState && lastState.body) || null;
  // Same rule as bodyRefresh: a live frame counts even when this dashboard did
  // not spawn the viewer, otherwise the refresh timer never starts at all.
  const wanted = Boolean(info && info.frame && (info.running || info.frames_unmanaged));
  if (wanted && bodyTimer === null) {
    bodyTimer = setInterval(bodyRefresh, BODY_REFRESH_MS);
    bodyRefresh();
  } else if (!wanted && bodyTimer !== null) {
    clearInterval(bodyTimer);
    bodyTimer = null;
  }
}

// Manual verbs: the human is the BodyCommand writer. The dashboard only forwards
// a verb name; the viewer owns the mapping to a pose, so the two cannot drift.
async function bodyVerb(verb, button) {
  const status = $("verb-status");
  if (button) button.disabled = true;
  try {
    const reply = await post("/api/body/command", { verb });
    status.textContent = reply.ok
      ? `${verb} → seq ${reply.seq ?? "?"}, authority ${reply.authority ?? 0}`
      : `${verb} rejected: ${reply.error || "unknown error"}`;
  } catch (error) {
    status.textContent = `${verb} failed: ${error.message}`;
  } finally {
    if (button) button.disabled = false;
  }
}

document.addEventListener("click", (event) => {
  const verbButton = event.target.closest("button[data-verb]");
  if (verbButton) bodyVerb(verbButton.dataset.verb, verbButton);
});
