// Recordings: one composite video per run, with that run's diary as its clock.
//
// Read-only over three routes the dashboard already serves:
//   /api/recordings                 the runs, newest first
//   /api/recordings/<id>/video      the file, seekable with Range
//   /api/brain/messages?run=<id>    the diary of that run
//
// The point of this page is the pairing. The diary does not sit beside the
// video as a list to read: a row appears when playback reaches its moment, so
// what is on screen and what is in the text are the same instant. The rail
// carries a tick per event, which is how you see a decision coming.

const $ = (id) => document.getElementById(id);

const list = $("recording-list");
const video = $("recording-video");
const note = $("recording-note");
const meta = $("recording-meta");
const diaryBox = $("run-diary");
const diaryCount = $("diary-count");
const title = $("recording-title");
const count = $("recording-count");
const hint = $("recording-hint");
const playToggle = $("play-toggle");
const rail = $("rec-rail");
const railFill = $("rec-rail-fill");
const railHead = $("rec-rail-head");
const ticks = $("rec-ticks");
const timeLabel = $("rec-time");
const durationLabel = $("rec-duration");
const speedButton = $("speed");

// How far behind the playhead a row is still "why am I seeing this": a decision
// lands a moment before its effect, so the row is allowed to arrive slightly
// early rather than a beat late.
const REVEAL_LEAD_S = 0.25;
const SPEEDS = [1, 2, 4, 0.5];
const COMPOSING_POLL_MS = 5000;

let rows = [];
let selected = null;
let startedAt = 0;      // epoch seconds of the recorder's clock, for the diary
let diary = [];         // every row of the selected run, sorted by time
let diaryNodes = [];    // the same rows as elements, revealed as time passes
let revealed = -1;      // index of the last revealed row
let speedIndex = 0;
let pollTimer = 0;
let pendingSeek = null;  // `?t=25` opens on that moment, and a diary row can link to one

// ---- formatting -------------------------------------------------------

function mb(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function clock(iso) {
  const when = iso ? new Date(iso) : null;
  if (!when || Number.isNaN(when.getTime())) return "";
  return when.toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function mmss(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const whole = Math.floor(seconds);
  const minutes = Math.floor(whole / 60);
  return `${minutes}:${String(whole % 60).padStart(2, "0")}`;
}

function length() {
  if (Number.isFinite(video.duration) && video.duration > 0) return video.duration;
  const row = rows.find((item) => item.run_id === selected);
  return row && Number.isFinite(row.duration_s) ? row.duration_s : 0;
}

function chip(text, cls) {
  const span = document.createElement("span");
  span.className = `rec-chip ${cls || ""}`.trim();
  span.textContent = text;
  return span;
}

// ---- the run list -----------------------------------------------------

// One row per run. A run that cannot be played is still listed: "this run has
// no video" is information, and hiding it makes the list lie about the run.
function rowElement(row) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "rec-row";
  button.dataset.runId = row.run_id;
  button.setAttribute("aria-pressed", String(row.run_id === selected));

  const head = document.createElement("span");
  head.className = "rec-row-head";
  const id = document.createElement("span");
  id.className = "rec-row-id";
  id.textContent = row.run_id;
  head.appendChild(id);
  head.appendChild(document.createTextNode(clock(row.started_at)));
  button.appendChild(head);

  const facts = document.createElement("span");
  facts.className = "rec-row-facts";
  // `pruned` first: its duration survives while its bytes are gone, so any
  // order that tests `ready` first prints "1:40 · " with no size and no reason.
  if (row.state === "pruned") {
    facts.textContent = "pruned to make room";
  } else if (row.state === "failed") {
    facts.textContent = row.error || "the recording failed";
  } else if (row.state === "recording" || row.state === "composing") {
    facts.textContent = row.state === "recording" ? "recording now…" : "composing…";
  } else if (row.state === "ready") {
    facts.textContent = [mmss(row.duration_s), mb(row.bytes)].filter(Boolean).join(" · ");
  } else {
    facts.textContent = "no recording";
  }
  button.appendChild(facts);

  const chips = document.createElement("span");
  chips.className = "rec-row-chips";
  if (row.state === "ready" && row.has_file && row.audio_peak === 0) {
    // A silent track is the normal state of a run that never heard JobBot, and
    // it is the first thing to rule out when the video "makes no sound".
    chips.appendChild(chip("silent track", "rec-chip-warn"));
  }
  if (row.state === "composing") chips.appendChild(chip("composing…", "rec-chip-wait"));
  if (row.state === "recording") chips.appendChild(chip("live", "rec-chip-wait"));
  if (row.state === "failed" && row.error) chips.appendChild(chip("failed", "rec-chip-bad"));
  if (row.state === "pruned") chips.appendChild(chip("pruned", "rec-chip-bad"));
  if (row.has_diary) chips.appendChild(chip("diary", "rec-chip-quiet"));
  if (chips.childElementCount) button.appendChild(chips);

  if (!row.has_file) button.classList.add("rec-row-nofile");
  button.onclick = () => select(row);
  return button;
}

function renderList(payload) {
  list.replaceChildren();
  const available = Boolean(payload.recording_available);

  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "rec-empty";
    // Recording off and "no runs yet" are different facts, and the operator
    // needs to know which one they are looking at.
    empty.textContent = available
      ? "No runs recorded yet. Start a run on the console and it will appear here."
      : "Recording is off on this machine, so there is no video to watch. Every run still leaves its diary, and the runs below are the ones that have one.";
    list.appendChild(empty);
    return;
  }

  for (const row of rows) list.appendChild(rowElement(row));
  count.textContent = `${rows.length} run${rows.length === 1 ? "" : "s"}`;
  hint.textContent = available ? "" : "recording off";
}

// ---- the diary, as a clock -------------------------------------------

function rowTime(row) {
  return Number.isFinite(row.t) ? Math.max(0, row.t - startedAt) : 0;
}

function renderDiary(messages) {
  diary = messages
    .filter((row) => Number.isFinite(row.t))
    .sort((a, b) => a.t - b.t);
  diaryNodes = [];
  revealed = -1;
  diaryBox.replaceChildren();
  ticks.replaceChildren();

  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "brain-empty";
    empty.textContent = "This run left no diary.";
    diaryBox.appendChild(empty);
    diaryCount.textContent = "";
    return;
  }
  diaryCount.textContent = `${diary.length} events`;

  const total = Math.max(length(), 1);
  for (const row of diary) {
    const at = rowTime(row);
    const bubble = document.createElement("div");
    bubble.className = `brain-row kind-${row.kind || "think"} rec-diary-row rec-hidden`;
    bubble.dataset.at = String(at);

    const stamp = document.createElement("span");
    stamp.className = "rec-diary-at";
    stamp.textContent = mmss(at);
    bubble.appendChild(stamp);

    const tag = { heard: "JobBot", think: "Clank" }[row.kind];
    if (tag) {
      const label = document.createElement("span");
      label.className = "tag";
      label.textContent = tag;
      bubble.appendChild(label);
    }
    const prefix = { act: "▸ ", result: "✓ ", error: "✕ " }[row.kind] || "";
    bubble.appendChild(document.createTextNode(prefix + (row.text || "")));
    bubble.onclick = () => seek(at);
    diaryBox.appendChild(bubble);
    diaryNodes.push(bubble);

    const tick = document.createElement("span");
    tick.className = `rec-tick kind-${row.kind || "think"}`;
    tick.style.left = `${Math.min(100, (at / total) * 100)}%`;
    tick.title = `${mmss(at)} · ${row.text || ""}`;
    ticks.appendChild(tick);
  }
}

// A row is on screen when the video has reached it. Revealing rather than
// re-rendering keeps the scroll where the operator left it and costs nothing.
function revealUpTo(seconds) {
  let index = -1;
  for (let i = diary.length - 1; i >= 0; i -= 1) {
    if (rowTime(diary[i]) <= seconds + REVEAL_LEAD_S) { index = i; break; }
  }
  if (index === revealed) return;
  for (let i = 0; i < diaryNodes.length; i += 1) {
    diaryNodes[i].classList.toggle("rec-hidden", i > index);
    diaryNodes[i].classList.toggle("rec-current", i === index);
  }
  revealed = index;
  if (index >= 0 && !video.paused) {
    diaryNodes[index].scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

// ---- our own playback -------------------------------------------------

function paintRail() {
  const total = length();
  const now = video.currentTime || 0;
  const fraction = total > 0 ? Math.min(1, Math.max(0, now / total)) : 0;
  railFill.style.width = `${fraction * 100}%`;
  railHead.style.left = `${fraction * 100}%`;
  timeLabel.textContent = mmss(now);
  durationLabel.textContent = mmss(total);
  rail.setAttribute("aria-valuemax", String(Math.round(total)));
  rail.setAttribute("aria-valuenow", String(Math.round(now)));
  rail.setAttribute("aria-valuetext", `${mmss(now)} of ${mmss(total)}`);
  revealUpTo(now);
}

function tick() {
  paintRail();
  if (!video.paused && !video.ended) requestAnimationFrame(tick);
}

function seek(seconds) {
  const total = length();
  if (!total) return;
  // Clamped both ways: a diary row can predate the recording (the recorder
  // starts at `running`, the diary at `preparing`) and a browser throws on a
  // negative or beyond-the-end `currentTime`.
  video.currentTime = Math.min(Math.max(0, seconds), total);
  paintRail();
}

function togglePlay() {
  if (!video.getAttribute("src")) return;
  if (video.paused) {
    video.play().then(() => { playToggle.textContent = "❚❚"; playToggle.setAttribute("aria-label", "Pause"); }).catch(() => {
      say("The browser refused to play this file.");
    });
  } else {
    video.pause();
  }
}

function railSeek(event) {
  const box = rail.getBoundingClientRect();
  if (!box.width) return;
  const fraction = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
  seek(fraction * length());
}

playToggle.onclick = togglePlay;
// Clicking the picture plays and pauses: the console's preview does the same,
// and a video without native controls needs somewhere obvious to click.
video.addEventListener("click", togglePlay);

// Pointer drag on the rail: down seeks, move keeps seeking, up releases.
rail.addEventListener("pointerdown", (event) => {
  rail.setPointerCapture(event.pointerId);
  railSeek(event);
  const move = (moveEvent) => railSeek(moveEvent);
  const up = () => {
    rail.removeEventListener("pointermove", move);
    rail.removeEventListener("pointerup", up);
  };
  rail.addEventListener("pointermove", move);
  rail.addEventListener("pointerup", up);
});

rail.addEventListener("keydown", (event) => {
  const step = event.shiftKey ? 30 : 5;
  if (event.key === "ArrowRight") { seek(video.currentTime + step); event.preventDefault(); }
  if (event.key === "ArrowLeft") { seek(video.currentTime - step); event.preventDefault(); }
  if (event.key === " ") { togglePlay(); event.preventDefault(); }
});

speedButton.onclick = () => {
  speedIndex = (speedIndex + 1) % SPEEDS.length;
  video.playbackRate = SPEEDS[speedIndex];
  speedButton.textContent = `${SPEEDS[speedIndex]}×`;
};

video.addEventListener("play", () => {
  playToggle.textContent = "❚❚";
  playToggle.setAttribute("aria-label", "Pause");
  requestAnimationFrame(tick);
});
video.addEventListener("pause", () => {
  playToggle.textContent = "▶";
  playToggle.setAttribute("aria-label", "Play");
  paintRail();
});
video.addEventListener("ended", () => {
  playToggle.textContent = "▶";
  paintRail();
});
video.addEventListener("seeked", paintRail);
video.addEventListener("loadedmetadata", () => {
  paintRail();
  revealUpTo(0);
  // A nudge past zero, because a paused player at exactly 0 can sit on a black
  // frame until it is told to go somewhere.
  const start = pendingSeek !== null ? pendingSeek : 0.05;
  seek(start);
  pendingSeek = null;
});
video.addEventListener("error", () => {
  if (!video.getAttribute("src")) return;
  say("The browser could not play this file. The manifest says it was written; check the dashboard log for the composite.");
});

function say(text) {
  note.textContent = text || "";
}

// ---- selecting a run --------------------------------------------------

async function select(row) {
  selected = row.run_id;
  for (const button of list.querySelectorAll(".rec-row")) {
    button.setAttribute("aria-pressed", String(button.dataset.runId === row.run_id));
  }
  title.textContent = `Run ${row.run_id}`;
  startedAt = row.started_at ? new Date(row.started_at).getTime() / 1000 : 0;
  history.replaceState(null, "", `/recordings?run=${encodeURIComponent(row.run_id)}`);

  meta.textContent = [clock(row.started_at), mmss(row.duration_s), mb(row.bytes)]
    .filter(Boolean).join(" · ");

  if (row.has_file) {
    // Setting `src` is what makes the browser ask with Range; the route answers
    // 206 so a scrub does not download the run again from the start.
    video.src = `/api/recordings/${encodeURIComponent(row.run_id)}/video`;
    video.load();
    say(row.audio_peak === 0 ? "This run's track is silent — nothing was heard." : "");
  } else {
    video.removeAttribute("src");
    video.load();
    say(row.error || (row.state === "pruned"
      ? "This video was pruned to make room. The diary is still here."
      : "This run has no video."));
  }
  playToggle.disabled = !row.has_file;
  paintRail();

  await loadDiary(row.run_id);
}

async function loadDiary(runId) {
  diaryBox.replaceChildren();
  try {
    const response = await fetch(`/api/brain/messages?run=${encodeURIComponent(runId)}`);
    if (!response.ok) throw new Error(String(response.status));
    const doc = await response.json();
    renderDiary(doc.messages || []);
    paintRail();
  } catch {
    renderDiary([]);
  }
}

async function load() {
  try {
    const response = await fetch("/api/recordings");
    if (!response.ok) throw new Error(String(response.status));
    const payload = await response.json();
    rows = payload.recordings || [];
    renderList(payload);
    const params = new URLSearchParams(location.search);
    const moment = Number.parseFloat(params.get("t") || "");
    if (Number.isFinite(moment)) pendingSeek = moment;
    const wanted = params.get("run");
    const first = rows.find((row) => row.run_id === wanted) ||
      rows.find((row) => row.has_file) || rows[0];
    if (first && !selected) await select(first);
    schedulePoll(payload);
  } catch (error) {
    say(`Could not read the recordings: ${error}`);
  }
}

// A run that is still composing becomes playable a minute later. Without this
// the operator sees "composing…" forever and restarts the dashboard, which is
// exactly when a half-finished composite is most likely to be lost.
function schedulePoll(payload) {
  clearTimeout(pollTimer);
  const busy = (payload.recordings || []).some(
    (row) => row.state === "recording" || row.state === "composing"
  );
  if (!busy) return;
  pollTimer = setTimeout(() => { load(); }, COMPOSING_POLL_MS);
}

// ---- theme ------------------------------------------------------------

const themeButton = $("theme-toggle");
themeButton.onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("clank-theme", next); } catch (e) { /* private mode */ }
};

load();
