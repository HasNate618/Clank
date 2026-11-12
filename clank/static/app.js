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
const LIVE_PREVIEW_MS = 33;
const LIVE_BACKOFF_MS = 10000;

// Which picture the screen shows. "game" is the layer's colour frame; "model"
// is the same frame faded by depth, which is what the planner is handed. The
// fade is rebuilt here from `spike/wbc_game/fade.py` because that file needs
// numpy and this page needs to run at frame rate.
let viewMode = "game";
const VIEW_KEY = "clank-view";

//: The fade's constants, copied from fade.py: the reference depth is the fifth
//: percentile of valid depth (the working surface, when the object's own depth
//: is not known - and here it never is), the dead zone starts at 1.06x of it,
//: and everything past 1.45x is fully faded.
const FADE_PERCENTILE = 5;
const FADE_NEAR = 1.06;
const FADE_FAR = 1.45;
const FADE_BACKDROP = [18, 22, 26];
//: Missing samples: the layer writes 0 for a hole and has reported 11.8 m.
const FADE_SENTINEL = 12.0;

let frameTimes = [];
const FPS_WINDOW_MS = 1500;
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

// Theme: the inline script in index.html paints the first theme from
// localStorage or the OS preference; this owns every change after that. The
// button's glyph is constant; its label names the destination, like the mode
// toggle's label does.
const THEME_KEY = "clank-theme";

function currentTheme() {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function syncThemeButton() {
  const button = $("theme-toggle");
  const light = currentTheme() === "light";
  // The sun/moon swap is CSS, keyed on [data-theme]; only the label changes.
  const label = light ? "Switch to dark theme" : "Switch to light theme";
  button.setAttribute("aria-label", label);
  button.title = label;
}

// Stage layouts: one data-layout attribute on the deck, one CSS block per
// layout. Only "side" ships; a layout switcher later is a registry entry here
// plus a `[data-layout="..."]` block in style.css, no markup changes.
const LAYOUTS = {
  side: "preview centre, controls left, brain and logs right",
};

function setLayout(name) {
  if (!LAYOUTS[name]) return false;
  $("deck").dataset.layout = name;
  return true;
}
// Console hook until a switcher UI exists: clankLayout("side").
window.clankLayout = setLayout;

// Shape, not size: a 16:9 screen cannot fill a near-square window, so the thing
// that absorbs the leftover height changes. Tall puts the robot panel under the
// screen, where its tiles get the whole width; wide fills the height with the
// screen and moves the robot into the rail. Brain, logs and telemetry are
// vertical by nature and stay in the rail either way.
const WIDE = matchMedia("(min-aspect-ratio: 3/2)");

function syncAspect() {
  const deck = $("deck");
  const wide = WIDE.matches;
  deck.dataset.aspect = wide ? "wide" : "tall";
  const robot = $("body-box");
  const stage = document.querySelector(".stage");
  const rail = document.querySelector(".col-rail");
  if (!robot || !stage || !rail) return;
  if (wide) {
    if (robot.parentElement !== rail) rail.prepend(robot);
  } else if (robot.parentElement !== stage) {
    stage.append(robot);
  }
}

WIDE.addEventListener("change", syncAspect);
syncAspect();

// The overlay canvas is where future visual debugging (grounding boxes, glow
// patches) draws, in game-pixel coordinates: it matches the live frame when
// the depth layer publishes, else the VNC framebuffer's 1920x1080.
function syncPreviewOverlay() {
  const overlay = $("preview-overlay");
  const live = $("preview-live");
  const width = liveAvailable && live.width ? live.width : 1920;
  const height = liveAvailable && live.height ? live.height : 1080;
  if (overlay.width !== width) overlay.width = width;
  if (overlay.height !== height) overlay.height = height;
}

// ---------- grounding overlay ----------
// What the brain last claimed it could see. The board expires server-side, so nothing
// here has to remember to erase a box; the alpha fade only stops an old claim from
// looking as certain as a fresh one. Everything is drawn in the canvas's own pixels and
// scaled from the frame size the brain reported, so a 1705-wide still and a 960-wide
// live stream land in the same place.
const OVERLAY_POLL_MS = 700;
const OVERLAY_LIFE_S = 6;
let overlayTimer = null;

function cssColour(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function drawOverlay(items) {
  const overlay = $("preview-overlay");
  if (!overlay) return;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!items || !items.length) return;
  const go = cssColour("--go", "#3ecf8e");
  const halt = cssColour("--halt", "#ff6b6b");
  const fontPx = Math.max(12, Math.round(overlay.height / 34));
  ctx.font = `600 ${fontPx}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  ctx.lineWidth = Math.max(2, overlay.height / 280);
  const radius = Math.max(3, overlay.height / 110);
  for (const item of items) {
    const size = Array.isArray(item.size) && item.size.length === 2 ? item.size : [1920, 1080];
    const scaleX = overlay.width / (Number(size[0]) || 1920);
    const scaleY = overlay.height / (Number(size[1]) || 1080);
    const age = Number(item.age_s) || 0;
    const alpha = Math.max(0.25, 1 - age / OVERLAY_LIFE_S);
    const colour = item.present === false ? halt : go;
    const box = (item.box || []).map(Number);
    if (box.length !== 4) continue;
    const x = box[0] * overlay.width;
    const y = box[1] * overlay.height;
    const w = (box[2] - box[0]) * overlay.width;
    const h = (box[3] - box[1]) * overlay.height;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = colour;
    ctx.strokeRect(x, y, w, h);
    const pixel = (item.pixel || []).map(Number);
    if (pixel.length === 2) {
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(pixel[0] * scaleX, pixel[1] * scaleY, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    // The spread is the brain's own disagreement between reads, which is the number that
    // says whether to trust the box at all.
    const spread = Number(item.spread);
    const name = String(item.noun || "object");
    const short = name.length > 42 ? `${name.slice(0, 41)}…` : name;
    const label = spread > 0 ? `${short} · spread ${Math.round(spread)} px` : short;
    // Keep the label on the canvas: a box near an edge would otherwise run its caption
    // off the picture, which is exactly where a grounding claim tends to sit.
    const metrics = ctx.measureText(label);
    const labelX = Math.max(4, Math.min(x, overlay.width - metrics.width - 14));
    const textY = Math.min(overlay.height - 6, Math.max(fontPx + 6, y - 8));
    ctx.globalAlpha = Math.min(1, alpha + 0.1);
    ctx.fillStyle = "rgba(0, 0, 0, 0.72)";
    ctx.fillRect(labelX, textY - fontPx - 4, metrics.width + 12, fontPx + 10);
    ctx.fillStyle = colour;
    ctx.fillText(label, labelX + 6, textY + 2);
  }
  ctx.globalAlpha = 1;
}

async function overlayTick() {
  const overlay = $("preview-overlay");
  if (overlay) {
    try {
      const response = await fetch(`/api/overlay?t=${Date.now()}`, { cache: "no-store" });
      if (response.ok) {
        const body = await response.json();
        drawOverlay(body.items || []);
      }
    } catch {
      // The overlay is a convenience; a failed poll must not touch the page.
    }
  }
  overlayTimer = setTimeout(overlayTick, OVERLAY_POLL_MS);
}

function startOverlay() {
  if (overlayTimer === null) overlayTick();
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
  frameTimes = [];
  drawStatus(false);
  document.querySelector(".preview")?.removeAttribute("data-live");
  const canvas = $("preview-live");
  if (canvas) canvas.hidden = true;
}

// --- the screen's two views -------------------------------------------------
// One mask, built at depth resolution and sampled nearest onto the colour frame:
// `fade()` upsamples depth to the picture, and a mask has no detail to lose.
let fadeMask = null;
let fadeMaskFor = "";

function foregroundDepth(depth) {
  // The fifth percentile of valid depth, from every eighth sample. Sorting
  // 16k numbers is cheaper than 130k and lands in the same place.
  const samples = [];
  for (let i = 0; i < depth.length; i += 8) {
    const d = depth[i];
    if (d > 0 && d < FADE_SENTINEL) samples.push(d);
  }
  if (!samples.length) return 0;
  samples.sort((a, b) => a - b);
  return samples[Math.floor((FADE_PERCENTILE / 100) * (samples.length - 1))];
}

function fadeAlphas(depth) {
  const target = foregroundDepth(depth);
  // No usable depth means do not fade: a frame with no depth is still a frame.
  if (!target) return null;
  const near = FADE_NEAR * target;
  const far = FADE_FAR * target;
  const key = `${depth.length}:${target.toFixed(4)}`;
  if (fadeMaskFor === key && fadeMask) return fadeMask;
  const mask = new Uint8Array(depth.length);
  for (let i = 0; i < depth.length; i += 1) {
    const d = depth[i];
    let k = 1;
    if (d > 0 && d < FADE_SENTINEL) {
      if (d <= near) k = 0;
      else if (d < far) {
        const t = (d - near) / (far - near);
        k = t * t * (3 - 2 * t);
      }
    }
    mask[i] = Math.round(k * 255);
  }
  fadeMask = mask;
  fadeMaskFor = key;
  return mask;
}

function drawGame(canvas, colour, width, height) {
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  canvas.getContext("2d").putImageData(new ImageData(colour, width, height), 0, 0);
}

function drawModel(canvas, colour, colourWidth, colourHeight, depth, width, height) {
  const mask = fadeAlphas(depth);
  if (!mask) return false;
  const out = new Uint8ClampedArray(width * height * 4);
  const [br, bg, bb] = FADE_BACKDROP;
  // Nearest-neighbour colour sampling, cached per width: the layer's colour
  // buffer is an integer multiple of the depth buffer, so this is exact.
  if (!drawModel.xmap || drawModel.xmap.length !== width || drawModel.xfrom !== colourWidth) {
    const xmap = new Int32Array(width);
    for (let x = 0; x < width; x += 1) xmap[x] = Math.min(colourWidth - 1, Math.floor((x * colourWidth) / width));
    drawModel.xmap = xmap;
    drawModel.xfrom = colourWidth;
  }
  const xmap = drawModel.xmap;
  for (let y = 0; y < height; y += 1) {
    const sy = Math.min(colourHeight - 1, Math.floor((y * colourHeight) / height));
    const srcRow = sy * colourWidth;
    const dstRow = y * width;
    for (let x = 0; x < width; x += 1) {
      const s = (srcRow + xmap[x]) * 4;
      const d = dstRow + x;
      const k = mask[d] / 255;
      const o = d * 4;
      out[o] = colour[s] + (br - colour[s]) * k;
      out[o + 1] = colour[s + 1] + (bg - colour[s + 1]) * k;
      out[o + 2] = colour[s + 2] + (bb - colour[s + 2]) * k;
      out[o + 3] = 255;
    }
  }
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  canvas.getContext("2d").putImageData(new ImageData(out, width, height), 0, 0);
  return true;
}

function noteFrame() {
  const now = performance.now();
  frameTimes.push(now);
  while (frameTimes.length && now - frameTimes[0] > FPS_WINDOW_MS) frameTimes.shift();
}

function drawStatus(faded, wanted) {
  const badge = $("preview-status");
  if (!badge) return;
  if (frameTimes.length < 2) {
    badge.textContent = "";
    return;
  }
  const span = frameTimes[frameTimes.length - 1] - frameTimes[0];
  const fps = span > 0 ? Math.round(((frameTimes.length - 1) * 1000) / span) : 0;
  // Say when the fade is missing: the switch would otherwise show the plain
  // picture while claiming to show the model's.
  badge.textContent = faded
    ? `${fps} fps · depth`
    : wanted ? `${fps} fps · no depth` : `${fps} fps`;
}

function setViewMode(mode) {
  viewMode = mode === "model" ? "model" : "game";
  for (const button of document.querySelectorAll("[data-view]")) {
    const active = button.dataset.view === viewMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }
  try {
    localStorage.setItem(VIEW_KEY, viewMode);
  } catch {
    // Private mode: the choice still holds for this page load.
  }
}

async function liveTick() {
  if (!liveActive) return;
  const wantDepth = viewMode === "model";
  try {
    const stamp = Date.now();
    const [colourResponse, depthResponse] = await Promise.all([
      fetch(`/api/preview/live.rgba?t=${stamp}`, { cache: "no-store" }),
      wantDepth ? fetch(`/api/preview/live.depth?t=${stamp}`, { cache: "no-store" }) : null,
    ]);
    if (!colourResponse.ok) throw new Error("no live frame");
    const width = Number(colourResponse.headers.get("X-Frame-Width") || 0);
    const height = Number(colourResponse.headers.get("X-Frame-Height") || 0);
    const data = new Uint8ClampedArray(await colourResponse.arrayBuffer());
    let depth = null;
    let depthWidth = 0;
    let depthHeight = 0;
    if (depthResponse && depthResponse.ok) {
      depthWidth = Number(depthResponse.headers.get("X-Frame-Width") || 0);
      depthHeight = Number(depthResponse.headers.get("X-Frame-Height") || 0);
      const buffer = await depthResponse.arrayBuffer();
      if (depthWidth && depthHeight && buffer.byteLength === depthWidth * depthHeight * 4) {
        depth = new Float32Array(buffer);
      }
    }
    if (!liveActive) return;
    if (!width || !height || data.length !== width * height * 4) {
      throw new Error("bad live frame");
    }
    const canvas = $("preview-live");
    const faded = depth ? drawModel(canvas, data, width, height, depth, depthWidth, depthHeight) : false;
    if (!faded) drawGame(canvas, data, width, height);
    noteFrame();
    drawStatus(faded, wantDepth);
    canvas.hidden = false;
    liveAvailable = true;
    syncPreviewOverlay();
    liveFailures = 0;
    hidePreviewAuth();
    $("preview-wait").style.display = "none";
    document.querySelector(".preview")?.setAttribute("data-live", "true");
    liveTimer = setTimeout(liveTick, LIVE_PREVIEW_MS);
  } catch {
    if (!liveActive) return;
    liveFailures += 1;
    if (liveFailures < 10) {
      liveTimer = setTimeout(liveTick, 300);
    } else {
      // The layer is not publishing: fall back to the VNC transport.
      stopLivePreview(true);
    }
  }
}

function startLivePreview() {
  if (liveActive || Date.now() < liveBackoffUntil) return;
  liveActive = true;
  liveAvailable = false;
  liveFailures = 0;
  frameTimes = [];
  const canvas = $("preview-live");
  if (canvas) canvas.hidden = false;
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
  const text = $("preview-wait-text");
  // A spinner for the waits that resolve on their own; none for the ones that
  // need the operator (stopped, failed, log in), because a turning ring would
  // promise progress that is not coming.
  const show = (phrase, spinning) => {
    text.textContent = phrase;
    $("preview-spinner").hidden = !spinning;
    wait.style.display = "";
  };
  const hide = () => {
    $("preview-spinner").hidden = true;
    wait.style.display = "none";
  };
  const stateText = {
    stopped: "session stopped",
    failed: "session failed",
    starting: "starting…",
  }[state.state];
  if (stateText) {
    show(stateText, state.state === "starting");
    return;
  }
  if (liveActive) {
    if (liveAvailable) {
      hide();
    } else {
      show("waiting for live capture…", true);
    }
    return;
  }
  const preview = state.preview || {};
  const transportAlive = Boolean(preview.vnc_alive && preview.proxy_alive);
  if (previewRfb && previewConnected) {
    hide();
  } else if (!transportAlive) {
    show("waiting for VNC transport…", true);
  } else if (previewAuthVisible()) {
    hide();
  } else {
    show("authenticate to view the preview", false);
  }
}

function applyControls(state) {
  const running = state.state === "running";
  $("start").disabled = actionInFlight || !["stopped", "failed"].includes(state.state);
  $("stop").disabled = actionInFlight || state.state === "stopped";
  const run = state.run || null;
  const runActive = Boolean(run && !["idle", "failed"].includes(run.state));
  // `/api/state` already carries `model` from `brain.server_status()`. Grey the
  // button only when nothing is listening: a `loading` server is a wait the run's
  // own `checking` phase is built for, so disabling it would hide that path.
  const model = state.model || {};
  const modelDown = model.up === false;
  $("start-run").disabled = actionInFlight || !running || runActive || modelDown;
  $("start-run").title = modelDown
    ? `the model server is ${model.status || "down"}${model.model ? ` (${model.model})` : ""}`
    : "";
  $("stop-run").disabled = actionInFlight || !runActive;
  // What is loading, in words, while the ring turns: the game restart is 40 s of
  // otherwise-silent waiting, and "restarting the game…" is the whole answer.
  const runBusy = runActive || ["starting", "stopping"].includes(state.state);
  $("run-spinner").hidden = !runBusy;
  $("run-state").textContent = runActive
    ? (run.detail || (run.phase ? `run ${run.phase}` : ""))
    : (run && run.id ? `run ${run.id} · ${run.state}` : "");
  $("run-state").title = run && run.error ? run.error : "";
  $("brain-run").textContent = run && run.id ? run.id : "";
  // The diary is the run's; the recording of that run is one click away,
  // and it opens on this run rather than on the newest one.
  $("watch-run").href = run && run.id
    ? `/recordings?run=${encodeURIComponent(run.id)}`
    : "/recordings";
  const modeButton = $("mode-toggle");
  const manual = state.mode === "manual";
  // The label names where a press takes you, so it is the destination, not the
  // current mode. `manual` is one side of the switch; every other mode - the
  // `hold` fallback on a body-less checkout included - is the other.
  modeButton.textContent = manual ? "Switch to robot controller" : "Switch to manual override";
  modeButton.setAttribute("aria-pressed", manual ? "true" : "false");
  modeButton.classList.toggle("active", manual);
  modeButton.disabled = actionInFlight || !running;
  const audio = state.audio || {};
  const audioButton = $("mute-audio");
  // The button is an icon, not a label: its own glyph reports the state, so
  // only the accessibility name changes here. Writing textContent would eat
  // the SVGs.
  const audioLabel = audio.muted ? "Unmute game audio" : "Mute game audio";
  audioButton.setAttribute("aria-label", audioLabel);
  audioButton.title = audioLabel;
  audioButton.classList.toggle("muted", Boolean(audio.muted));
  audioButton.disabled = actionInFlight || !running || !audio.available;
  const body = state.body || null;
  const bodyRunning = Boolean(body && body.running);
  // The menu button needs a running session: without one there is no relay to ask.
  $("menu-press").disabled = actionInFlight || !running;
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
  syncPreviewOverlay();
  updateManualUi(state);
  // A refused run's reason (the model server is down, the game would not
  // relaunch) is the run's error, and it is the only instruction the operator
  // gets: it belongs on the error line, not in a title attribute.
  applyServerError(state.last_error || (state.run && state.run.error) || null);

  renderHealth(state);
  bodySync();
}

function healthCard(label, value, valueClass, detail, wide) {
  const card = document.createElement("div");
  card.className = wide ? "card card-wide" : "card";
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
  // The planner's server replaced the preview card here: the transport state is
  // already obvious from the picture being alive, while "is the model loaded"
  // was the one thing on this list nothing else could tell you.
  const model = state.model || {};
  const up = Boolean(model.up);
  const value = up ? "serving" : model.status === "loading" ? "loading" : "down";
  cards.push(healthCard("Model", model.model || value,
    model.status === "ok" ? "ok" : model.status === "loading" ? "warn" : "bad",
    model.model ? value : "", true));
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
$("start-run").onclick = () => runAction(() => post("/api/run/start"));
$("stop-run").onclick = () => runAction(() => post("/api/run/stop"));

$("mode-toggle").onclick = () => {
  const mode = lastState?.mode === "manual" ? "body" : "manual";
  runAction(() => post("/api/session/mode", { mode }));
};

$("mute-audio").onclick = () => runAction(() => post("/api/audio/mute", { muted: !(lastState?.audio?.muted) }));

// One press of the controller's menu button. It is a request to the relay file, so it works
// whichever pose mode is running - including `manual`, where the body is not driving anything.
$("menu-press").onclick = async () => {
  const button = $("menu-press");
  button.disabled = true;
  try {
    await post("/api/menu", { hand: "right" });
    button.classList.add("btn-fired");
    setTimeout(() => button.classList.remove("btn-fired"), 220);
  } catch (err) {
    button.title = `menu press failed: ${err.message}`;
  } finally {
    button.disabled = false;
  }
};

$("theme-toggle").onclick = () => {
  const next = currentTheme() === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch {
    // Private mode: the toggle still works for this page load.
  }
  syncThemeButton();
};
syncThemeButton();

for (const button of document.querySelectorAll("[data-view]")) {
  button.onclick = () => setViewMode(button.dataset.view);
}
try {
  setViewMode(localStorage.getItem(VIEW_KEY) || "game");
} catch {
  setViewMode("game");
}

$("body-start").onclick = () => runAction(() => post("/api/body/start"));

$("body-stop").onclick = () => runAction(() => post("/api/body/stop"));

// Brain: the diary from /api/brain/messages as a conversation - JobBot on the
// left, Clank on the right, actions as chips, outcomes as one-liners. It is a
// view of the diary, not a log console: sentences, timestamps only on hover.
const BRAIN_POLL_MS = 2000;
const BRAIN_KINDS = new Set(["heard", "think", "act", "result", "error"]);
let brainSignature = "";

function brainRow(row) {
  const kind = BRAIN_KINDS.has(row.kind) ? row.kind : "think";
  const bubble = document.createElement("div");
  bubble.className = `brain-row kind-${kind}`;
  if (row.iso) bubble.title = row.iso;
  const tagText = { heard: "JobBot", think: "Clank" }[kind];
  if (tagText) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = tagText;
    bubble.appendChild(tag);
  }
  const prefix = { act: "▸ ", result: "✓ ", error: "✕ " }[kind] || "";
  bubble.appendChild(document.createTextNode(prefix + (row.text || "")));
  return bubble;
}

function renderBrain(messages) {
  const box = $("brain-messages");
  const stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "brain-empty";
    empty.textContent = "Nothing heard yet — JobBot speaks once a job starts.";
    box.replaceChildren(empty);
    return;
  }
  box.replaceChildren(...messages.map(brainRow));
  if (stick) box.scrollTop = box.scrollHeight;
}

async function loadBrain() {
  try {
    const doc = await api("/api/brain/messages");
    const messages = doc.messages || [];
    const last = messages[messages.length - 1] || {};
    const signature = `${messages.length}:${last.t ?? ""}`;
    if (signature === brainSignature) return;
    brainSignature = signature;
    renderBrain(messages);
  } catch {
    // The diary is a nicety; it never earns an error banner.
  }
}

loadBrain();
setInterval(loadBrain, BRAIN_POLL_MS);

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
    // The brain diary has its own panel as a conversation; it stays in
    // /api/logs for other consumers, just not rendered twice here.
    delete logData.brain;
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
  // Success is visible in the game. The sequence number and authority were
  // debugging detail sitting in the middle of the controls, so a good press
  // says nothing and only a failure speaks.
  try {
    const reply = await post("/api/body/command", { verb });
    status.textContent = reply.ok
      ? ""
      : `${verb.replace(/_/g, " ")} rejected: ${reply.error || "unknown error"}`;
  } catch (error) {
    status.textContent = `${verb.replace(/_/g, " ")} failed: ${error.message}`;
  } finally {
    if (button) button.disabled = false;
  }
}

document.addEventListener("click", (event) => {
  const verbButton = event.target.closest("button[data-verb]");
  if (verbButton) bodyVerb(verbButton.dataset.verb, verbButton);
});

// The overlay polls on its own timer: it must draw whether the preview is a live depth
// frame, a VNC stream, or nothing at all yet.
startOverlay();
