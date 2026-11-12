// Recordings: one composite video per run, with that run's diary beside it.
//
// Everything here is read-only over three routes the dashboard already serves:
//   /api/recordings                 the runs, newest first
//   /api/recordings/<id>/video      the file, seekable with Range
//   /api/brain/messages?run=<id>    the diary of that run
// The only state is which run is selected, and the diary is the scrubber.

const $ = (id) => document.getElementById(id);

const list = $("recording-list");
const video = $("recording-video");
const note = $("recording-note");
const meta = $("recording-meta");
const diaryBox = $("run-diary");
const title = $("recording-title");
const count = $("recording-count");
const hint = $("recording-hint");

let rows = [];
let selected = null;
let startedAt = 0; // epoch seconds of the recording's own clock, for seeks

function say(text) {
  note.textContent = text || "";
}

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

function durationText(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "";
  const whole = Math.round(seconds);
  const minutes = Math.floor(whole / 60);
  return `${minutes}:${String(whole % 60).padStart(2, "0")}`;
}

function chip(text, cls) {
  const span = document.createElement("span");
  span.className = `rec-chip ${cls || ""}`.trim();
  span.textContent = text;
  return span;
}

// One row per run. A row that cannot be played is still listed: "this run has
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
  if (row.state === "ready" || row.state === "pruned") {
    const parts = [durationText(row.duration_s), mb(row.bytes)].filter(Boolean);
    facts.textContent = parts.join(" · ");
  } else if (row.state === "pruned") {
    facts.textContent = "pruned to make room";
  } else if (row.state === "failed") {
    facts.textContent = row.error || "the recording failed";
  } else {
    facts.textContent = "composing…";
  }
  button.appendChild(facts);

  const chips = document.createElement("span");
  chips.className = "rec-row-chips";
  if (row.state === "ready" && row.has_file && row.audio_peak === 0) {
    // A silent track is the normal state of a run that never heard JobBot, and
    // it is the first thing to rule out when the video "makes no sound".
    chips.appendChild(chip("silent track", "rec-chip-warn"));
  }
  if (row.state === "composing" || row.state === "recording") {
    chips.appendChild(chip("composing…", "rec-chip-wait"));
  }
  if (row.state === "failed" && row.error) {
    chips.appendChild(chip("failed", "rec-chip-bad"));
  }
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
      : "Recording is off on this machine. Every run still leaves its diary; add ffmpeg and set recording.enabled to keep the video too.";
    list.appendChild(empty);
    return;
  }

  for (const row of rows) list.appendChild(rowElement(row));
  count.textContent = `${rows.length} run${rows.length === 1 ? "" : "s"}`;
  hint.textContent = available ? "" : "recording off";
}

function renderDiary(messages) {
  diaryBox.replaceChildren();
  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "brain-empty";
    empty.textContent = "This run left no diary.";
    diaryBox.appendChild(empty);
    return;
  }
  for (const row of messages) {
    const bubble = document.createElement("div");
    bubble.className = `brain-row kind-${row.kind || "think"} rec-diary-row`;
    bubble.title = "Jump the video to this moment";
    const at = document.createElement("span");
    at.className = "rec-diary-at";
    at.textContent = Number.isFinite(row.t) ? durationText(row.t - startedAt) : "";
    bubble.appendChild(at);
    const tag = { heard: "JobBot", think: "Clank" }[row.kind];
    if (tag) {
      const label = document.createElement("span");
      label.className = "tag";
      label.textContent = tag;
      bubble.appendChild(label);
    }
    const prefix = { act: "▸ ", result: "✓ ", error: "✕ " }[row.kind] || "";
    bubble.appendChild(document.createTextNode(prefix + (row.text || "")));
    if (Number.isFinite(row.t)) bubble.onclick = () => seek(row.t);
    diaryBox.appendChild(bubble);
  }
}

// A seek has to be clamped both ways: a diary row can predate the recording
// (the recorder starts at `running`, the diary at `preparing`) and a browser
// will throw on a negative or a beyond-the-end `currentTime`.
function seek(epochSeconds) {
  if (!Number.isFinite(epochSeconds) || !startedAt) return;
  const end = Number.isFinite(video.duration) ? video.duration : 0;
  const at = Math.min(Math.max(0, epochSeconds - startedAt), end);
  video.currentTime = at;
  if (video.paused) video.play().catch(() => { /* autoplay policy: the click was the gesture */ });
}

async function select(row) {
  selected = row.run_id;
  for (const button of list.querySelectorAll(".rec-row")) {
    button.setAttribute("aria-pressed", String(button.dataset.runId === row.run_id));
  }
  title.textContent = `Run ${row.run_id}`;
  startedAt = row.started_at ? new Date(row.started_at).getTime() / 1000 : 0;
  history.replaceState(null, "", `/recordings?run=${encodeURIComponent(row.run_id)}`);

  const facts = [clock(row.started_at), durationText(row.duration_s), mb(row.bytes)]
    .filter(Boolean).join(" · ");
  meta.textContent = facts;

  if (row.has_file) {
    // Setting `src` is what makes the browser ask with Range; the route answers
    // 206 so a scrub does not download the run again from the start.
    video.src = `/api/recordings/${encodeURIComponent(row.run_id)}/video`;
    video.load();
    say(row.audio_peak === 0 ? "This run's track is silent — nothing was heard." : "");
  } else {
    video.removeAttribute("src");
    video.load();
    say(row.error || "This run has no video.");
  }

  await loadDiary(row.run_id);
}

async function loadDiary(runId) {
  diaryBox.replaceChildren();
  try {
    const response = await fetch(`/api/brain/messages?run=${encodeURIComponent(runId)}`);
    if (!response.ok) throw new Error(String(response.status));
    const doc = await response.json();
    renderDiary(doc.messages || []);
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
    const wanted = new URLSearchParams(location.search).get("run");
    const first = rows.find((row) => row.run_id === wanted && row.has_file) ||
      rows.find((row) => row.has_file) || rows[0];
    if (first && !selected) await select(first);
  } catch (error) {
    say(`Could not read the recordings: ${error}`);
  }
}

// The console's theme switch, same key and same behaviour.
const themeButton = $("theme-toggle");
themeButton.onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("clank-theme", next); } catch (e) { /* private mode */ }
};

video.addEventListener("error", () => {
  if (!video.getAttribute("src")) return;
  say("The browser could not play this file. The manifest says it was written; check the dashboard log for the composite.");
});

load();
