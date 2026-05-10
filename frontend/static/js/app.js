/**
 * app.js —Frontend Logic
 *
 * Two-channel communication with backend:
 *   1. SSE  /api/stream  → instant word commits and clears
 *   2. Poll /api/state   → smooth UI updates every 150 ms
 *
 * State machine reflected in the UI mirrors the Python segmenter:
 *   IDLE → SIGNING → HOLD → (commit) → IDLE
 */

"use strict";

// ── Config ────────────────────────────────────────────────────────────────────
const POLL_MS         = 150;    // state poll interval
const FLASH_DURATION  = 1800;   // ms to show commit flash
const CONF_THRESHOLD  = 0.72;   // must match backend
const MARGIN_THRESHOLD = 0.18;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $connDot     = document.querySelector(".status-dot");
const $connLabel   = document.getElementById("conn-label");
const $stateBadge  = document.getElementById("state-badge");
const $stateIcon   = document.getElementById("state-icon");
const $stateLabel  = document.getElementById("state-label");
const $handDot     = document.getElementById("hand-dot");
const $handLabel   = document.getElementById("hand-label");
const $wordDisplay = document.getElementById("word-display");
const $confBar     = document.getElementById("conf-bar");
const $confValue   = document.getElementById("conf-value");
const $marginBar   = document.getElementById("margin-bar");
const $marginValue = document.getElementById("margin-value");
const $holdGroup   = document.getElementById("hold-group");
const $holdBar     = document.getElementById("hold-bar");
const $holdValue   = document.getElementById("hold-value");
const $flash       = document.getElementById("commit-flash");
const $flashWord   = document.getElementById("flash-word");
const $sentence    = document.getElementById("sentence-display");
const $vocabChips  = document.getElementById("vocab-chips");

// ── Local state ───────────────────────────────────────────────────────────────
let _sentence       = [];
let _currentWord    = "";
let _connected      = false;
let _flashTimer     = null;
let _chipMap        = {};   // word → chip element

// ── Vocabulary chips (loaded once) ───────────────────────────────────────────
fetch("/api/config")
  .then(r => r.json())
  .then(cfg => {
    cfg.labels.forEach(label => {
      const chip = document.createElement("span");
      chip.className   = "chip";
      chip.textContent = label.replace(/_/g, " ");
      chip.dataset.word = label;
      $vocabChips.appendChild(chip);
      _chipMap[label] = chip;
    });
  })
  .catch(() => {});

// ── SSE — instant commits ─────────────────────────────────────────────────────
function connectSSE() {
  const es = new EventSource("/api/stream");

  es.onopen = () => {
    setConnected(true);
  };

  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "commit") {
      _sentence = msg.sentence;
      renderSentence();
      showFlash(msg.word, msg.conf);
      highlightChip(msg.word);
    } else if (msg.type === "clear") {
      _sentence = [];
      renderSentence();
    }
    // "ping" — ignore
  };

  es.onerror = () => {
    setConnected(false);
    es.close();
    setTimeout(connectSSE, 3000);   // reconnect
  };
}

// ── Polling — smooth bar & state updates ─────────────────────────────────────
function poll() {
  fetch("/api/state")
    .then(r => r.json())
    .then(s => {
      setConnected(true);
      updateState(s);
    })
    .catch(() => setConnected(false));
}

setInterval(poll, POLL_MS);
connectSSE();
poll();   // immediate first poll

// ── State update ──────────────────────────────────────────────────────────────
function updateState(s) {
  // ── Hand indicator
  const hasHand = s.hand;
  $handDot.className = "hand-dot" + (hasHand ? " active" : "");
  $handLabel.textContent = hasHand ? "Hand detected" : "No hand";

  // ── Segmenter state badge
  const seg = s.seg_state || "IDLE";
  const stateMap = {
    IDLE:    { label: "Idle",     cls: "idle",    icon: "◯" },
    SIGNING: { label: "Signing",  cls: "signing", icon: "◉" },
    HOLD:    { label: `Confirming ${Math.round(s.hold_pct * 100)}%`,
                                  cls: "hold",    icon: "◕" },
  };
  const st = stateMap[seg] || stateMap.IDLE;
  $stateBadge.className = "state-badge " + st.cls;
  $stateIcon.textContent  = st.icon;
  $stateLabel.textContent = st.label;

  // ── Word display
  const word = s.word || "";
  const wordDisplay = word.replace(/_/g, " ") || "—";
  if ($wordDisplay.textContent !== wordDisplay) {
    $wordDisplay.textContent = wordDisplay;
  }
  $wordDisplay.className = "word-display " +
    (seg === "SIGNING" ? "signing" : seg === "HOLD" ? "hold" : !word ? "dim" : "");

  // ── Confidence bar
  const confPct = Math.round((s.conf || 0) * 100);
  $confBar.style.width = confPct + "%";
  $confBar.style.background = confPct >= 72 ? "var(--accent)" :
                              confPct >= 50 ? "var(--warn)" : "var(--danger)";
  $confValue.textContent = confPct + "%";

  // ── Margin bar
  const marginPct = Math.round((s.margin || 0) * 100);
  $marginBar.style.width = marginPct + "%";
  $marginBar.style.background = marginPct >= 18 ? "var(--info)" : "var(--warn)";
  $marginValue.textContent = marginPct + "%";

  // ── Hold progress bar
  if (seg === "HOLD") {
    $holdGroup.style.display = "block";
    const hPct = Math.round((s.hold_pct || 0) * 100);
    $holdBar.style.width  = hPct + "%";
    $holdValue.textContent = hPct + "%";
  } else {
    $holdGroup.style.display = "none";
  }

  // ── Active chip highlight (current word in vocab)
  const w = s.word || "";
  Object.entries(_chipMap).forEach(([label, el]) => {
    el.classList.toggle("active", label === w && seg === "SIGNING");
  });
}

// ── Sentence rendering ────────────────────────────────────────────────────────
function renderSentence() {
  if (_sentence.length === 0) {
    $sentence.innerHTML =
      '<span class="sentence-placeholder">Start signing to build a sentence…</span>';
    return;
  }
  $sentence.innerHTML = _sentence.map((w, i) => {
    const isNew  = i === _sentence.length - 1;
    const cls    = isNew ? "sentence-word" : "sentence-word old";
    const display = w.replace(/_/g, " ");
    return `<span class="${cls}">${display}</span>`;
  }).join(" ");
}

// ── Commit flash ──────────────────────────────────────────────────────────────
function showFlash(word, conf) {
  const display = word.replace(/_/g, " ");
  $flashWord.textContent = `${display}  (${Math.round(conf * 100)}%)`;
  $flash.classList.add("visible");
  clearTimeout(_flashTimer);
  _flashTimer = setTimeout(() => $flash.classList.remove("visible"), FLASH_DURATION);
}

// ── Chip highlight ────────────────────────────────────────────────────────────
function highlightChip(word) {
  const chip = _chipMap[word];
  if (!chip) return;
  // Briefly pulse the chip to green even if it was already active
  chip.style.boxShadow = "0 0 0 2px var(--accent)";
  setTimeout(() => chip.style.boxShadow = "", 600);
}

// ── Clear ─────────────────────────────────────────────────────────────────────
function clearSentence() {
  fetch("/api/clear", { method: "POST" }).catch(() => {});
  // Optimistic update — SSE will confirm
  _sentence = [];
  renderSentence();
}

// ── Connection indicator ──────────────────────────────────────────────────────
function setConnected(ok) {
  if (ok === _connected) return;
  _connected = ok;
  $connDot.className   = "status-dot " + (ok ? "ok" : "error");
  $connLabel.textContent = ok ? "Connected" : "Reconnecting…";
}
