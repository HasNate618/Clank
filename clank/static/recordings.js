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
const heartButton = $("heart-run");
const deleteButton = $("delete-run");
const heartedToggle = $("hearted-only");

// How far behind the playhead a row is still "why am I seeing this": a decision
// lands a moment before its effect, so the row is allowed to arrive slightly
// early rather than a beat late.
const REVEAL_LEAD_S = 0.25;
const SPEEDS = [1, 2, 4, 0.5];
const COMPOSING_POLL_MS = 5000;

let rows = [];
let available = true;     // the config switch, not a guess: ffmpeg may be absent
let selected = null;
let onlyHearted = false;  // the `Kept` filter, and `?hearted=1` when it is on
// One arm, shared by the rows and the stage, so two surfaces cannot disagree
// about which run the second press means.
let pendingDelete = null; // {runId, timer}
const ARM_MS = 4000;
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

// ---- icons, hearts and the delete arm ---------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";
const HEART_PATH = "M12 20.1 4.9 13a4.5 4.5 0 0 1 6.4-6.4l.7.7.7-.7A4.5 4.5 0 1 1 19.1 13z";
const TRASH_PATH = "M4.5 7h15M9.5 7V4.6h5V7M6.8 7l.9 12.4h8.6L17.2 7M10.4 10.6v5.8M13.6 10.6v5.8";

function svgIcon(pathData) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "rec-icon");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", pathData);
  svg.appendChild(path);
  return svg;
}

// Filled when kept, outlined when not: the CSS does the fill, this decides which.
function paintHeart(button, hearted, subject) {
  button.setAttribute("aria-pressed", String(hearted));
  button.classList.toggle("is-hearted", hearted);
  button.title = `${hearted ? "Stop keeping" : "Keep"} ${subject}`;
  button.setAttribute("aria-label", button.title);
}

// What a run's `run.json` said, for the runs that never reached a picture. The
// error is a multi-line crash tail; the row has one line to spend.
function firstLine(text, limit = 76) {
  const line = String(text || "").split("\n")[0].trim();
  return line.length > limit ? `${line.slice(0, limit - 1)}…` : line;
}

function noVideoReason(row) {
  if (row.run_live) return "no video · still running";
  if (row.run_error) return `no video · ${firstLine(row.run_error)}`;
  if (row.state === "pruned") return "pruned to make room";
  if (row.state === "failed") return row.error ? firstLine(row.error) : "the recording failed";
  return "no video · the run ended before it recorded";
}

// ---- the run list -----------------------------------------------------

// One row per run. A run that cannot be played is still listed: "this run has
// no video" is information, and hiding it makes the list lie about the run.
//
// The row is a div, not a button: it carries two buttons of its own now, and a
// button inside a button is invalid HTML that the browser silently re-parents.
// `.rec-row-main` is the click target and holds everything the row shows.
function rowElement(row) {
  const item = document.createElement("div");
  item.className = "rec-row";
  item.dataset.runId = row.run_id;
  item.classList.toggle("is-selected", row.run_id === selected);
  if (!row.has_file) item.classList.add("rec-row-nofile");

  const main = document.createElement("button");
  main.type = "button";
  main.className = "rec-row-main";
  main.setAttribute("aria-pressed", String(row.run_id === selected));

  const head = document.createElement("span");
  head.className = "rec-row-head";
  const id = document.createElement("span");
  id.className = "rec-row-id";
  id.textContent = row.run_id;
  head.appendChild(id);
  head.appendChild(document.createTextNode(clock(row.started_at)));
  main.appendChild(head);

  const facts = document.createElement("span");
  facts.className = "rec-row-facts";
  // `pruned` first: its duration survives while its bytes are gone, so any
  // order that tests `ready` first prints "1:40 · " with no size and no reason.
  if (row.state === "pruned") {
    facts.textContent = "pruned to make room";
  } else if (row.state === "failed") {
    facts.textContent = row.error ? firstLine(row.error) : "the recording failed";
  } else if (row.state === "recording" || row.state === "composing") {
    facts.textContent = row.state === "recording" ? "recording now…" : "composing…";
  } else if (row.state === "ready" && !row.has_file) {
    // A manifest that says ready over a file somebody removed: without this the
    // row reads "12.5 s" with no size and no reason, which looks like a bug.
    facts.textContent = "ready · the video file is missing";
  } else if (row.state === "ready") {
    facts.textContent = [mmss(row.duration_s), mb(row.bytes)].filter(Boolean).join(" · ");
  } else {
    facts.textContent = noVideoReason(row);
  }
  main.appendChild(facts);

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
  if (chips.childElementCount) main.appendChild(chips);
  main.onclick = () => select(row);
  item.appendChild(main);

  // Keeping and throwing away, on the row you found it on. The keyboard order is
  // the reading order: the run, then what you can do to it.
  const heart = document.createElement("button");
  heart.type = "button";
  heart.className = "rec-icon-btn rec-row-heart";
  heart.dataset.runId = row.run_id;
  heart.appendChild(svgIcon(HEART_PATH));
  paintHeart(heart, Boolean(row.hearted), `run ${row.run_id}`);
  heart.onclick = () => setHeart(row);
  item.appendChild(heart);

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "rec-icon-btn rec-icon-btn-danger rec-row-delete";
  remove.dataset.runId = row.run_id;
  remove.appendChild(svgIcon(TRASH_PATH));
  remove.title = "Delete this run and its diary";
  remove.setAttribute("aria-label", `Delete run ${row.run_id}`);
  remove.onclick = () => deleteRun(row);
  item.appendChild(remove);

  return item;
}

// The picture keeps the aspect it was recorded at: the composite was 1680x540
// until 2026-11-13 and is 1264x540 after it, and both have to play without a
// re-encode. The row carries the manifest's own numbers, so the page never
// guesses from the current build.
function applyAspect(row) {
  const player = document.querySelector(".rec-player");
  if (!player) return;
  const width = Number(row && row.width);
  const height = Number(row && row.height);
  if (!width || !height) {
    player.style.removeProperty("--rec-aspect");
    return;
  }
  player.style.setProperty("--rec-aspect", `${width} / ${height}`);
}

function visibleRows() {
  return onlyHearted ? rows.filter((row) => row.hearted) : rows;
}

function renderList() {
  list.replaceChildren();
  const shown = visibleRows();

  if (!shown.length) {
    const empty = document.createElement("div");
    empty.className = "rec-empty";
    // Recording off, nothing kept and "no runs yet" are three different facts,
    // and the operator needs to know which one they are looking at.
    if (onlyHearted && rows.length) {
      empty.textContent = "Nothing kept yet. Press the heart on a run to keep it here, and it stays out of the size prune.";
    } else {
      empty.textContent = available
        ? "No runs recorded yet. Start a run on the console and it will appear here."
        : "Recording is off on this machine, so there is no video to watch. Every run still leaves its diary, and the runs below are the ones that have one.";
    }
    list.appendChild(empty);
    count.textContent = "";
    hint.textContent = available ? "" : "recording off";
    return;
  }

  for (const row of shown) list.appendChild(rowElement(row));
  paintCount();
  hint.textContent = available ? "" : "recording off";
  paintDeleteButtons();
}

// ---- keeping and throwing away ----------------------------------------

async function setHeart(row) {
  const on = !row.hearted;
  try {
    const response = await fetch(`/api/recordings/${encodeURIComponent(row.run_id)}/heart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ on }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || String(response.status));
    row.hearted = Boolean(payload.hearted);
  } catch (error) {
    say(`Could not ${on ? "keep" : "unkeep"} this run: ${error}`);
    return;
  }
  for (const button of list.querySelectorAll(".rec-row-heart")) {
    if (button.dataset.runId === row.run_id) {
      paintHeart(button, Boolean(row.hearted), `run ${row.run_id}`);
    }
  }
  if (selected === row.run_id) paintHeart(heartButton, Boolean(row.hearted), "this run");
  // One path out of a heart press, taken by the function that owns the filter:
  // unkeeping the run you are watching while `Kept` is on has to move the
  // selection on, and the count has to be rebuilt either way.
  setFilter(onlyHearted);
}

function paintCount() {
  const kept = rows.filter((row) => row.hearted).length;
  const shown = visibleRows().length;
  count.textContent = onlyHearted
    ? `${shown} of ${rows.length} kept`
    : `${rows.length} run${rows.length === 1 ? "" : "s"}${kept ? ` · ${kept} kept` : ""}`;
}

// Arming is shared and singular: the second press means the run the first press
// named, wherever it is pressed. Four seconds is the whole memory it has.
function paintDeleteButtons() {
  const armed = pendingDelete ? pendingDelete.runId : null;
  for (const button of list.querySelectorAll(".rec-row-delete")) {
    const on = button.dataset.runId === armed;
    button.classList.toggle("is-armed", on);
    button.title = on
      ? "Press again to delete this run and its diary"
      : "Delete this run and its diary";
    button.setAttribute("aria-label", on
      ? `Press again to delete run ${button.dataset.runId}`
      : `Delete run ${button.dataset.runId}`);
  }
  const on = armed !== null && armed === selected;
  deleteButton.classList.toggle("is-armed", on);
  deleteButton.title = on
    ? "Press again to delete this run and its diary"
    : "Delete this run and its diary";
  deleteButton.querySelector(".rec-icon-label").textContent = on ? "Sure?" : "Delete";
}

function disarm() {
  if (!pendingDelete) return;
  clearTimeout(pendingDelete.timer);
  pendingDelete = null;
  paintDeleteButtons();
}

async function deleteRun(row) {
  if (pendingDelete && pendingDelete.runId === row.run_id) {
    clearTimeout(pendingDelete.timer);
    pendingDelete = null;
    paintDeleteButtons();
    await reallyDelete(row);
    return;
  }
  disarm();
  pendingDelete = {
    runId: row.run_id,
    timer: setTimeout(() => { pendingDelete = null; paintDeleteButtons(); }, ARM_MS),
  };
  paintDeleteButtons();
  // The second press is the one that costs something, so it names the loss.
  say(`Press Delete again to remove run ${row.run_id} and its diary.`);
}

async function reallyDelete(row) {
  try {
    const response = await fetch(`/api/recordings/${encodeURIComponent(row.run_id)}`, {
      method: "DELETE",
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || String(response.status));
  } catch (error) {
    // A 409 here is the answer, not a failure: the run is still being recorded
    // or encoded, and the server's sentence says which.
    say(`Could not delete this run: ${error}`);
    return;
  }
  say(`Deleted run ${row.run_id}.`);
  if (selected === row.run_id) clearSelection();
  await load();
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

function paintStageActions(row) {
  const has = Boolean(row);
  heartButton.disabled = !has;
  deleteButton.disabled = !has;
  if (has) paintHeart(heartButton, Boolean(row.hearted), "this run");
  paintDeleteButtons();
}

function clearSelection() {
  selected = null;
  video.pause();
  video.removeAttribute("src");
  video.load();
  title.textContent = "Pick a run";
  meta.textContent = "";
  applyAspect(null);
  diaryBox.replaceChildren();
  ticks.replaceChildren();
  diaryCount.textContent = "";
  playToggle.disabled = true;
  say("");
  paintStageActions(null);
  writeUrl();
}

// One place writes the URL, so the filter and the selection cannot erase each
// other; `?t=` is a one-shot deep link and is consumed by the seek.
function writeUrl() {
  const params = new URLSearchParams();
  if (onlyHearted) params.set("hearted", "1");
  if (selected) params.set("run", selected);
  const query = params.toString();
  history.replaceState(null, "", query ? `/recordings?${query}` : "/recordings");
}

async function select(row) {
  selected = row.run_id;
  disarm();
  for (const item of list.querySelectorAll(".rec-row")) {
    const mine = item.dataset.runId === row.run_id;
    item.classList.toggle("is-selected", mine);
    const main = item.querySelector(".rec-row-main");
    if (main) main.setAttribute("aria-pressed", String(mine));
  }
  title.textContent = `Run ${row.run_id}`;
  startedAt = row.started_at ? new Date(row.started_at).getTime() / 1000 : 0;
  paintStageActions(row);
  writeUrl();

  meta.textContent = [clock(row.started_at), mmss(row.duration_s), mb(row.bytes)]
    .filter(Boolean).join(" · ");

  applyAspect(row);
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
    available = Boolean(payload.recording_available);
    renderList();
    const params = new URLSearchParams(location.search);
    const moment = Number.parseFloat(params.get("t") || "");
    if (Number.isFinite(moment)) pendingSeek = moment;
    const wanted = params.get("run");
    const shown = visibleRows();
    const first = shown.find((row) => row.run_id === wanted) ||
      shown.find((row) => row.has_file) || shown[0];
    if (first && !selected) await select(first);
    else if (!first && selected) clearSelection();
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

// ---- the Kept filter --------------------------------------------------

function paintFilter() {
  heartedToggle.setAttribute("aria-pressed", String(onlyHearted));
  heartedToggle.classList.toggle("is-on", onlyHearted);
  heartedToggle.title = onlyHearted
    ? "Show every run again"
    : "Show only the runs you kept";
}

function setFilter(on) {
  onlyHearted = on;
  paintFilter();
  writeUrl();
  renderList();
  // A filter that removed the run being watched has to say so by moving on,
  // rather than leaving the picture of a run the list no longer offers.
  const shown = visibleRows();
  if (selected && !shown.some((row) => row.run_id === selected)) {
    const next = shown.find((row) => row.has_file) || shown[0];
    if (next) select(next);
    else clearSelection();
  }
}

heartedToggle.onclick = () => setFilter(!onlyHearted);

// ---- theme ------------------------------------------------------------

const themeButton = $("theme-toggle");
themeButton.onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("clank-theme", next); } catch { /* private mode */ }
};

// The stage's own pair, acting on the selected run. They are the same functions
// the rows call, so there is one behaviour and two places to reach it.
heartButton.onclick = () => {
  const row = rows.find((candidate) => candidate.run_id === selected);
  if (row) setHeart(row);
};

deleteButton.onclick = () => {
  const row = rows.find((candidate) => candidate.run_id === selected);
  if (row) deleteRun(row);
};

onlyHearted = new URLSearchParams(location.search).get("hearted") === "1";
paintFilter();
load();
