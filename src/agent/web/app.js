"use strict";

// The server decides what is allowed; this page only mirrors its control state.
const $ = (id) => document.getElementById(id);
const TOKEN_KEY = "so101_operator_token";
const CONTROL_UI_VERSION = "cell-grid-v1";
const TOOL_NAMES = {
  get_state: "상태 확인", observe_scene: "카메라 관찰", describe_places: "장소 이름 확인",
  pick_block: "블록 집기", place_at_slot: "칸에 놓기", place_on_table: "테이블에 놓기",
  place_here: "여기 내려놓기", move_block_to_slot: "칸으로 옮기기", move_block_to_table: "테이블로 옮기기",
  shift_block: "블록 조금 옮기기", move_arm: "팔 움직이기", return_to_home: "home 복귀",
  open_gripper: "그리퍼 열기", run_task1: "미션 1", run_task2: "미션 2", run_task3: "미션 3(수집)",
  recover_and_home: "복귀(그리퍼 열기+home+그리퍼 닫기)",
  rotate_gripper: "그리퍼 돌리기", pick_here: "여기서 집기",
  place_at_cell: "칸 좌표에 놓기", move_block_to_cell: "칸 좌표로 옮기기", move_to_cell: "칸 좌표 위로 이동",
};
const STATE_TEXT = { idle: "대기", busy: "실행 중", stopping: "정지 중", stopped: "비상정지됨", homing: "home 복귀 중" };

let token = null;
try { token = sessionStorage.getItem(TOKEN_KEY); } catch (_) { /* storage unavailable */ }
let control = { state: "idle" };
let isOperator = false;
let directPending = false;
let manualTools = new Set();
let keyboardConfig = null;
const manualAllowed = name => manualTools.has(name);
const MANUAL_BUTTON_TOOLS = {
  'manual-home':'return_to_home', 'roll-ccw':'rotate_gripper', 'roll-cw':'rotate_gripper',
  'pick-here':'pick_here', 'place-here':'place_here', 'open-gripper':'open_gripper',
  'go-cell':'move_to_cell', 'move-pixel':'move_to_pixel', 'place-pixel':'place_at_pixel',
};
let streamingBubble = null;
const toolChips = new Map();
let events = null;
let recognition = null;
let recognizing = false;

// ── toast (manual-control refusals only; chat stays for the LLM) ─────
function showToast(message, kind) {
  const container = $("toast-container");
  const el = document.createElement("div");
  el.className = "toast" + (kind === "bad" ? " bad" : "");
  el.textContent = message;
  container.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 250);
  }, 2200);
}

function saveToken(value) {
  token = value;
  try { value ? sessionStorage.setItem(TOKEN_KEY, value) : sessionStorage.removeItem(TOKEN_KEY); } catch (_) {}
}

async function api(path, body) {
  const headers = {
    "Content-Type": "application/json",
    "X-SO101-Control-Version": CONTROL_UI_VERSION,
  };
  if (token) headers["X-Operator-Token"] = token;
  const response = await fetch(path, { method: "POST", headers, body: JSON.stringify(body || {}), keepalive: path === "/api/stop" || path === "/api/keyboard/stop" || path === "/api/keyboard/release", signal:path.startsWith("/api/keyboard/")?AbortSignal.timeout(2000):undefined });
  let data = {};
  try { data = await response.json(); } catch (_) {}
  if (response.status === 403 && path !== "/api/lease") { isOperator = false; acquireLease(); }
  return { status: response.status, data };
}

// ── chat rendering ──────────────────────────────────────────────────
function addMessage(kind, text) {
  const div = document.createElement("div");
  div.className = `msg msg-${kind}`;
  div.textContent = text;
  $("chat").appendChild(div);
  $("chat").scrollTop = $("chat").scrollHeight;
  return div;
}

// Safe, dependency-free Markdown for assistant messages.  Build DOM nodes
// instead of assigning innerHTML: model output is untrusted text even when it
// looks like Markdown.
function appendInlineMarkdown(parent, source) {
  const token = /(`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|~~[^~\n]+~~|\*[^*\n]+\*|_[^_\n]+_|\[[^\]\n]+\]\(https?:\/\/[^)\s]+\)|\n)/g;
  let cursor = 0;
  for (const match of source.matchAll(token)) {
    if (match.index > cursor) parent.appendChild(document.createTextNode(source.slice(cursor, match.index)));
    const value = match[0];
    let node;
    if (value === "\n") {
      node = document.createElement("br");
    } else if (value.startsWith("`")) {
      node = document.createElement("code");
      node.textContent = value.slice(1, -1);
    } else if (value.startsWith("**") || value.startsWith("__")) {
      node = document.createElement("strong");
      node.textContent = value.slice(2, -2);
    } else if (value.startsWith("~~")) {
      node = document.createElement("del");
      node.textContent = value.slice(2, -2);
    } else if (value.startsWith("*") || value.startsWith("_")) {
      node = document.createElement("em");
      node.textContent = value.slice(1, -1);
    } else {
      const parts = /^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/.exec(value);
      node = document.createElement("a");
      node.textContent = parts[1];
      node.href = parts[2];
      node.target = "_blank";
      node.rel = "noopener noreferrer";
    }
    parent.appendChild(node);
    cursor = match.index + value.length;
  }
  if (cursor < source.length) parent.appendChild(document.createTextNode(source.slice(cursor)));
}

function renderMarkdown(element, source) {
  element.replaceChildren();
  const lines = String(source || "").replace(/\r\n?/g, "\n").split("\n");
  let paragraph = [];

  function flushParagraph() {
    if (!paragraph.length) return;
    const p = document.createElement("p");
    appendInlineMarkdown(p, paragraph.join("\n"));
    element.appendChild(p);
    paragraph = [];
  }

  for (let i = 0; i < lines.length;) {
    const line = lines[i];
    const fence = /^```\s*([^\s`]*)\s*$/.exec(line);
    if (fence) {
      flushParagraph();
      const codeLines = [];
      i += 1;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) codeLines.push(lines[i++]);
      if (i < lines.length) i += 1;
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      if (fence[1]) code.className = `language-${fence[1].replace(/[^a-zA-Z0-9_-]/g, "")}`;
      code.textContent = codeLines.join("\n");
      pre.appendChild(code);
      element.appendChild(pre);
      continue;
    }
    const heading = /^(#{1,6})\s+(.+)$/.exec(line);
    if (heading) {
      flushParagraph();
      const h = document.createElement(`h${heading[1].length}`);
      appendInlineMarkdown(h, heading[2]);
      element.appendChild(h);
      i += 1;
      continue;
    }
    const unordered = /^\s*[-*+]\s+(.+)$/.exec(line);
    const ordered = /^\s*\d+[.)]\s+(.+)$/.exec(line);
    if (unordered || ordered) {
      flushParagraph();
      const list = document.createElement(ordered ? "ol" : "ul");
      const matcher = ordered ? /^\s*\d+[.)]\s+(.+)$/ : /^\s*[-*+]\s+(.+)$/;
      while (i < lines.length) {
        const item = matcher.exec(lines[i]);
        if (!item) break;
        const li = document.createElement("li");
        appendInlineMarkdown(li, item[1]);
        list.appendChild(li);
        i += 1;
      }
      element.appendChild(list);
      continue;
    }
    const quote = /^>\s?(.*)$/.exec(line);
    if (quote) {
      flushParagraph();
      const blockquote = document.createElement("blockquote");
      const quoteLines = [];
      while (i < lines.length) {
        const quoted = /^>\s?(.*)$/.exec(lines[i]);
        if (!quoted) break;
        quoteLines.push(quoted[1]);
        i += 1;
      }
      appendInlineMarkdown(blockquote, quoteLines.join("\n"));
      element.appendChild(blockquote);
      continue;
    }
    if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
      flushParagraph();
      element.appendChild(document.createElement("hr"));
      i += 1;
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
    } else {
      paragraph.push(line);
    }
    i += 1;
  }
  flushParagraph();
}

function addMarkdownMessage(text) {
  const div = addMessage("bot", "");
  div._markdownSource = text || "";
  renderMarkdown(div, div._markdownSource);
  return div;
}

function describeArgs(args) {
  const parts = Object.entries(args || {}).map(([k, v]) => `${k}=${typeof v === "number" ? Math.round(v * 10) / 10 : v}`);
  return parts.length ? `(${parts.join(", ")})` : "";
}

function toolCall(event) {
  const chip = document.createElement("div");
  chip.className = "tool";
  const name = document.createElement("span");
  name.className = "tool-name";
  name.textContent = `🔧 ${TOOL_NAMES[event.name] || event.name} ${describeArgs(event.arguments)}${event.direct ? " · 직접" : ""}`;
  const detail = document.createElement("span");
  detail.className = "tool-detail";
  detail.textContent = "실행 중…";
  chip.append(name, detail);
  $("chat").appendChild(chip);
  $("chat").scrollTop = $("chat").scrollHeight;
  toolChips.set(event.id, chip);
  streamingBubble = null;
}

function toolResult(event) {
  let chip = toolChips.get(event.id);
  if (!chip) { toolCall({ id: event.id, name: event.name, arguments: {} }); chip = toolChips.get(event.id); }
  const result = event.result || {};
  chip.classList.add(result.ok ? "ok" : "fail");
  const seconds = typeof result.elapsed_s === "number" ? ` · ${result.elapsed_s}s` : "";
  chip.querySelector(".tool-detail").textContent = `${result.ok ? "✓" : "✗ " + result.reason} ${result.detail || ""}${seconds}`;
}

function handleToolResult(event) {
  if (!event.direct) { toolResult(event); return; }
  // manual control panel: no chat chip. A refusal (hit a limit, nothing to
  // grab, etc.) gets a toast; a success just lets the camera speak for itself.
  const result = event.result || {};
  if (!result.ok) {
    showToast(result.detail || result.reason || "동작을 수행할 수 없습니다.", "bad");
  }
}

function handleEvent(event) {
  switch (event.type) {
    case "control_state": applyControl(event); break;
    case "reset": $("chat").textContent = ""; toolChips.clear(); streamingBubble = null;
      addMessage("system", "새 대화를 시작합니다. 무엇을 옮겨 드릴까요?"); break;
    case "user": addMessage("user", event.text); streamingBubble = null; break;
    case "text_delta":
      if (!streamingBubble) streamingBubble = addMarkdownMessage("");
      streamingBubble._markdownSource = (streamingBubble._markdownSource || "") + event.text;
      renderMarkdown(streamingBubble, streamingBubble._markdownSource);
      $("chat").scrollTop = $("chat").scrollHeight;
      break;
    case "assistant_text":
      if (streamingBubble) {
        streamingBubble._markdownSource = event.text || "";
        renderMarkdown(streamingBubble, streamingBubble._markdownSource);
      } else {
        addMarkdownMessage(event.text);
      }
      streamingBubble = null;
      break;
    case "tool_call": if (!event.direct) toolCall(event); break;
    case "tool_result": handleToolResult(event); break;
    case "turn_end": streamingBubble = null; break;
    case "stop_pressed": if (event.effective) addMessage("system", "비상정지 요청됨"); break;
    case "keyboard_jog_end": showToast(event.message,"bad"); break;
    case "error": addMessage("error", event.message); break;
    default: break;
  }
}

// ── control state ───────────────────────────────────────────────────
function applyControl(snapshot) {
  control = snapshot;
  document.dispatchEvent(new CustomEvent("robot-control-state",{detail:snapshot}));
  const badge = $("state-badge");
  badge.textContent = STATE_TEXT[snapshot.state] || snapshot.state;
  badge.className = "badge " + (snapshot.state === "idle" ? "badge-idle"
    : (snapshot.state === "stopped" || snapshot.state === "stopping") ? "badge-stopped" : "badge-busy");
  const idle = snapshot.state === "idle" && isOperator;
  const manualControls = [
    $("input"), $("send"), $("mic"), $("reset"), $("jog-step"), $("manual-home"),
    $("roll-ccw"), $("roll-cw"), $("roll-step"), $("pick-here"), $("place-here"), $("open-gripper"),
    $("cell-x"), $("cell-y"), $("go-cell"),
    ...document.querySelectorAll(".jog-btn"),
  ];
  for (const el of manualControls) {
    el.disabled = !idle;
  }
  for (const [id, tool] of Object.entries(MANUAL_BUTTON_TOOLS)) {
    const button=$(id);
    if (!button) continue;
    button.disabled = !idle || !manualAllowed(tool);
    button.title = manualAllowed(tool) ? '' : '이 서버에서는 지원하지 않습니다. 보정 전용 절차를 사용하세요.';
  }
  for (const button of document.querySelectorAll('.jog-btn')) button.disabled=!idle || !manualAllowed('move_arm');
  $("stopped-banner").hidden = !(snapshot.state === "stopped" || snapshot.state === "homing" || snapshot.state === "stopping");
  $("home-button").disabled = !(snapshot.state === "stopped" && isOperator);
  $("home-button").textContent = snapshot.state === "homing" ? "복귀 중…" : "다시 시도(그리퍼 열기+home)";
  if (snapshot.message) $("stopped-message").textContent = snapshot.message;
  enablePixelButtons();
  if (!idle && recognizing && recognition) { try { recognition.abort(); } catch (_) {} }
  $("input").placeholder = idle ? "예: 노란 블록을 적재 구역 좌상단으로 옮겨줘"
    : snapshot.state === "stopped" ? "비상정지 상태입니다. 자동 복귀가 실패했으니 다시 시도해 주세요." : "로봇이 동작 중입니다…";
}

// ── lease & events ──────────────────────────────────────────────────
async function acquireLease() {
  const { status, data } = await api("/api/lease");
  if (status === 200 && data.token) {
    saveToken(data.token);
    isOperator = true;
    $("lock").hidden = true;
  } else {
    isOperator = false;
    $("lock").hidden = false;
    setTimeout(acquireLease, 5000);
  }
  applyControl(control);
  connectEvents();
}

function connectEvents() {
  if (events) events.close();
  events = new EventSource(`/api/events${token && isOperator ? `?token=${encodeURIComponent(token)}` : ""}`);
  events.onmessage = (message) => { try { handleEvent(JSON.parse(message.data)); } catch (err) { console.error(err); } };
  events.onerror = () => { $("state-badge").textContent = "서버 재연결 중…"; };
}

// ── commands ────────────────────────────────────────────────────────
async function send(text) {
  text = (text || "").trim();
  if (!text || control.state !== "idle") return;
  $("input").value = "";
  const { status, data } = await api("/api/chat", { text });
  if (status === 409) addMessage("system", "로봇이 아직 동작 중입니다.");
  else if (status >= 400 && status !== 403) addMessage("error", data.error || `오류 ${status}`);
}

$("composer").addEventListener("submit", (e) => { e.preventDefault(); send($("input").value); });
function pressStop() {
  api("/api/stop");
}
$("stop").addEventListener("click", pressStop);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !e.repeat) pressStop(); });
$("home-button").addEventListener("click", () => api("/api/home"));
$("lock-retry").addEventListener("click", acquireLease);
$("reset").addEventListener("click", () => api("/api/reset"));

async function directRequest(path, body) {
  if (directPending || control.state !== "idle" || !isOperator) return;
  if (!manualAllowed(path === "/api/jog" ? "move_arm" : body.tool)) return;
  directPending = true;
  try {
    const { status, data } = await api(path, body);
    if (status !== 202 && status !== 403) {
      showToast(data.error || `명령을 받지 못했습니다 (HTTP ${status}).`, "bad");
    }
  } catch (error) {
    showToast(`서버에 명령을 보내지 못했습니다: ${error.message}`, "bad");
  } finally {
    directPending = false;
  }
}

function manual(tool, arguments_) {
  directRequest("/api/manual", { tool, arguments: arguments_ || {} });
}

const JOG = { forward: [1, 0, 0], back: [-1, 0, 0], left: [0, 1, 0], right: [0, -1, 0], up: [0, 0, 1], down: [0, 0, -1] };
for (const button of document.querySelectorAll(".jog-btn")) {
  button.addEventListener("click", () => {
    const step = Number($("jog-step").value);
    const [f, l, u] = JOG[button.dataset.jog];
    directRequest("/api/jog", { forward_mm: f * step, left_mm: l * step, up_mm: u * step });
  });
}
$("roll-ccw").addEventListener("click", () => manual("rotate_gripper", { delta_deg: -Number($("roll-step").value) }));
$("roll-cw").addEventListener("click", () => manual("rotate_gripper", { delta_deg: Number($("roll-step").value) }));

$("go-cell").addEventListener("click", () => {
  const x = Number($("cell-x").value);
  const y = Number($("cell-y").value);
  if (!Number.isInteger(x) || !Number.isInteger(y)) {
    showToast("칸 좌표는 정수 x와 y로 입력하세요 (예: 3, 4).", "bad");
    return;
  }
  manual("move_to_cell", { x, y });
});

$("manual-home").addEventListener("click", () => manual("return_to_home"));
$("pick-here").addEventListener("click", () => manual("pick_here"));
$("place-here").addEventListener("click", () => manual("place_here"));
$("open-gripper").addEventListener("click", () => manual("open_gripper"));

// ── voice (browser built-in Web Speech API) ────────────────────────
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
function voiceStatus(text, kind) {
  const status = $("voice-status");
  status.textContent = text;
  status.className = kind || "";
}

if (Recognition) {
  recognition = new Recognition();
  recognition.lang = "ko-KR";
  recognition.interimResults = true;
  recognition.continuous = true;
  recognition.maxAlternatives = 1;
  let finalText = "";
  let latestText = "";
  let heardAudio = false;
  let heardSpeech = false;
  recognition.onstart = () => {
    recognizing = true;
    finalText = "";
    latestText = "";
    heardAudio = false;
    heardSpeech = false;
    $("mic").classList.add("listening");
    $("mic").textContent = "■";
    $("mic").title = "음성 입력 종료";
    voiceStatus("마이크 연결 중…", "listening");
  };
  recognition.onaudiostart = () => {
    heardAudio = true;
    voiceStatus("마이크 입력 확인됨 · 말한 뒤 ■를 누르세요", "listening");
  };
  recognition.onspeechstart = () => {
    heardSpeech = true;
    voiceStatus("말소리 감지됨 · 인식 중…", "listening");
  };
  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) finalText += event.results[i][0].transcript;
      else interim += event.results[i][0].transcript;
    }
    latestText = finalText + interim;
    $("input").value = latestText;
    voiceStatus(`인식 중: ${latestText}`, "listening");
  };
  recognition.onnomatch = () => voiceStatus("음성을 인식하지 못했습니다. 다시 말해 주세요.", "error");
  recognition.onerror = (event) => {
    if (event.error === "aborted") return;
    const explanations = {
      "no-speech": "말소리가 감지되지 않았습니다.",
      "audio-capture": "사용 가능한 마이크를 찾지 못했습니다.",
      "not-allowed": "마이크 또는 음성 인식 권한이 차단됐습니다.",
      "service-not-allowed": "브라우저 음성 인식 서비스가 차단됐습니다.",
      "network": "브라우저 음성 인식 서비스에 연결하지 못했습니다.",
    };
    const detail = explanations[event.error] || event.error;
    voiceStatus(`음성 인식 실패: ${detail}`, "error");
    showToast(`음성 인식 실패: ${detail}`, "bad");
  };
  recognition.onend = () => {
    recognizing = false;
    $("mic").classList.remove("listening");
    $("mic").textContent = "🎤";
    $("mic").title = "음성 입력";
    const transcript = (finalText || latestText).trim();
    if (!transcript) {
      if (!$("voice-status").classList.contains("error")) {
        const detail = !heardAudio
          ? "마이크 입력이 시작되지 않았습니다. 사이트 권한과 입력 장치를 확인하세요."
          : !heardSpeech
            ? "마이크는 연결됐지만 말소리를 감지하지 못했습니다. 입력 장치·음소거·볼륨을 확인하세요."
            : "말소리는 감지했지만 브라우저 인식 서비스가 텍스트를 반환하지 않았습니다.";
        voiceStatus(detail, "error");
      }
      return;
    }
    $("input").value = transcript;
    voiceStatus(`인식 완료: ${transcript}`);
    if ($("voice-autosend").checked && control.state === "idle") send(transcript);
  };
} else {
  voiceStatus("이 브라우저는 음성 인식 미지원", "error");
}

$("mic").addEventListener("click", () => {
  if (!Recognition || !recognition) {
    showToast("이 브라우저는 내장 음성 인식을 지원하지 않습니다. 최신 Chrome 또는 Edge를 사용해 주세요.", "bad");
    voiceStatus("이 브라우저는 음성 인식 미지원", "error");
    return;
  }
  if (recognizing) { recognition.stop(); return; }
  try { recognition.start(); } catch (error) { showToast(`음성 인식을 시작하지 못했습니다: ${error.message}`, "bad"); }
});

// ── camera + clickable places ───────────────────────────────────────
function selectedCamera() {
  return new URLSearchParams(location.search).get('camera') === 'wrist' ? 'wrist' : 'shoulder';
}
function cameraUrl(config) {
  return "/api/camera/video?camera=" + selectedCamera();
}

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function insertText(text) {
  if ($("input").disabled) return;
  const input = $("input");
  input.value = input.value ? `${input.value} ${text}` : text;
  input.focus();
}

const LAYER_KEY = "so101_places_layer";

// The 15 named points and the board cells cover the same ground, so showing
// both at once is just clutter: one layer at a time, the operator picks.
function storedLayer(fallback) {
  try {
    const stored = localStorage.getItem(LAYER_KEY);
    if (stored === "regions" || stored === "grid") return stored;
  } catch (_) { /* storage unavailable */ }
  return fallback === "grid" ? "grid" : "regions";
}

function buildGridLayer(grid) {
  const layer = svgEl("g", { id: "grid-layer" });
  const hover = $("grid-hover");
  for (const cell of grid.cells) {
    const label = `(${cell.x}, ${cell.y})`;
    const quad = svgEl("polygon", {
      class: "grid-cell" + (cell.in_zone ? " in-zone" : ""),
      points: cell.corners_px.map((p) => p.join(",")).join(" "),
    });
    const title = svgEl("title", {});
    // the cells are ~30px across, so the coordinate is shown on hover
    // rather than drawn into every square
    title.textContent = cell.in_zone ? `${label} · 적재 구역` : label;
    quad.appendChild(title);
    if (!cell.in_zone) {
      quad.addEventListener("click", () => insertText(`격자 ${label}`));
      quad.addEventListener("pointerenter", () => { hover.textContent = label; });
      quad.addEventListener("pointerleave", () => { hover.textContent = ""; });
    }
    layer.appendChild(quad);
  }
  return layer;
}

function wireLayerToggle(grid, gridLayer, regionLayer) {
  const radios = document.querySelectorAll('input[name="places-layer"]');
  if (!gridLayer) {
    // nothing to switch to: keep the named points and hide the control
    $("camera-tools").hidden = true;
    return;
  }
  const show = (layer) => {
    gridLayer.setAttribute("class", layer === "grid" ? "" : "hidden-layer");
    regionLayer.setAttribute("class", layer === "regions" ? "" : "hidden-layer");
    if (layer !== "grid") $("grid-hover").textContent = "";
  };
  const initial = storedLayer(grid.default_layer);
  for (const radio of radios) {
    radio.checked = radio.value === initial;
    radio.addEventListener("change", () => {
      if (!radio.checked) return;
      show(radio.value);
      try { localStorage.setItem(LAYER_KEY, radio.value); } catch (_) { /* ignore */ }
    });
  }
  show(initial);
}

function drawPlaces(places) {
  const svg=$('overlay');
  svg.replaceChildren();
  svg.setAttribute('viewBox',`0 0 ${places.image_size.join(' ')}`);
}

let pixelTarget=null;
let pixelSelectionGeneration=0;
function enablePixelButtons() {
  const enabled=Boolean(pixelTarget) && selectedCamera()==='shoulder' && control?.state==='idle' && isOperator;
  $('move-pixel').disabled=!enabled || !manualAllowed('move_to_pixel');
  $('place-pixel').disabled=!enabled || !manualAllowed('place_at_pixel');
  $('pixel-to-chat').disabled=!enabled;
}
// Hover ring follows the pointer; the selected ring stays in image coordinates.
const hoverRing=document.createElement('span');
hoverRing.className='pixel-hover-ring';hoverRing.hidden=true;
hoverRing.setAttribute('aria-hidden','true');
$('camera-wrap').append(hoverRing);
let pixelPreview=null;
function cameraPixel(event) {
  const image=$('camera'),rect=image.getBoundingClientRect();
  return {u:Math.floor((event.clientX-rect.left)*image.naturalWidth/rect.width),
    v:Math.floor((event.clientY-rect.top)*image.naturalHeight/rect.height),
    width:image.naturalWidth,height:image.naturalHeight};
}
async function checkPixel(params,signal) {
  const response=await fetch('/api/pixel-target?'+new URLSearchParams(params),{signal,cache:'no-store'});
  const data=await response.json();
  if(!response.ok)throw Error(data.error||'선택할 수 없는 위치입니다.');
  return data.target;
}
// Same geometric gate as resolve_pixel; this preview never requests IK or motion.
function previewPixel(params, rules=pixelPreview) {
  if(!rules)return 'pending';
  const {u,v,width,height}=params;
  if(rules.camera_name!=='shoulder'||width!==rules.image_size[0]||height!==rules.image_size[1]
    ||!Number.isInteger(u)||!Number.isInteger(v)||u<0||v<0||u>=width||v>=height)return 'invalid';
  const H=rules.H,den=H[2][0]*u+H[2][1]*v+H[2][2];
  if(!Number.isFinite(den)||den===0)return 'invalid';
  const x=(H[0][0]*u+H[0][1]*v+H[0][2])/den;
  const y=(H[1][0]*u+H[1][1]*v+H[1][2])/den;
  if(![x,y,rules.z_mm].every(Number.isFinite))return 'invalid';
  const dx=x-rules.base_xy_mm[0],dy=y-rules.base_xy_mm[1];
  const angle=Math.atan2(dy,dx)*180/Math.PI,radius=Math.hypot(dx,dy);
  let outer=rules.radius_mm;
  const profile=rules.radius_by_angle_mm||[];
  if(profile.length) {
    let interpolated=profile[profile.length-1][1];
    if(angle<profile[0][0])interpolated=profile[0][1];
    else for(let i=1;i<profile.length;i++) {
      if(angle<profile[i][0]) {
        const [a,r]=profile[i-1],[b,t]=profile[i];
        interpolated=r+(t-r)*(angle-a)/(b-a);break;
      }
    }
    outer=Math.min(outer,interpolated);
  }
  return angle<rules.angle_min_deg||angle>rules.angle_max_deg||radius>outer
    ||radius>outer-rules.edge_margin_mm||radius<rules.min_radius_mm
    ||(Math.abs(dy)<=rules.keepout_half_width_mm&&dx<=rules.keepout_depth_mm)
    ?'invalid':'valid';
}
function checkHoverPixel(event) {
  hoverRing.dataset.state=previewPixel(cameraPixel(event));
}
function updateHoverRing(event) {
  const image=$('camera'),wrap=$('camera-wrap');
  const visible=selectedCamera()==='shoulder' && event.pointerType!=='touch' && image.naturalWidth>0;
  hoverRing.hidden=!visible;wrap.classList.toggle('pixel-pointer-active',visible);
  if(!visible){hideHoverRing();return;}
  checkHoverPixel(event);
  const rect=wrap.getBoundingClientRect();
  hoverRing.style.left=`${event.clientX-rect.left}px`;
  hoverRing.style.top=`${event.clientY-rect.top}px`;
  hoverRing.style.width=`${21*rect.width/image.naturalWidth}px`;
  hoverRing.style.height=`${21*rect.height/image.naturalHeight}px`;
  hoverRing.style.borderWidth=`${3*rect.width/image.naturalWidth}px`;
}
function hideHoverRing() {
  hoverRing.hidden=true;$('camera-wrap').classList.remove('pixel-pointer-active');
}
$('camera-wrap').addEventListener('pointermove',updateHoverRing);
$('camera-wrap').addEventListener('pointerenter',updateHoverRing);
$('camera-wrap').addEventListener('pointerleave',hideHoverRing);
window.addEventListener('blur',hideHoverRing);
document.addEventListener('visibilitychange',()=>{if(document.hidden)hideHoverRing();});
$('camera-wrap').addEventListener('click',async event=>{
  if(selectedCamera()!=='shoulder')return;
  const image=$('camera');
  if(!image.naturalWidth || !image.naturalHeight)return;
  const params=cameraPixel(event),{u,v}=params;
  const generation=++pixelSelectionGeneration;
  pixelTarget=null;enablePixelButtons();
  const overlay=$('pixel-target-overlay');overlay.replaceChildren();
  overlay.setAttribute('viewBox',`0 0 ${image.naturalWidth} ${image.naturalHeight}`);
  const mark=svgEl('circle',{cx:u,cy:v,r:9,class:'pixel-selected-ring'});
  mark.dataset.state='pending';overlay.append(mark);
  $('pixel-target-status').textContent=`픽셀 (${u}, ${v}) 확인 중…`;
  try {
    const target=await checkPixel(params,AbortSignal.timeout(5000));
    if(generation!==pixelSelectionGeneration)return;
    pixelTarget=target;mark.dataset.state='valid';
    $('pixel-target-status').textContent=`픽셀 (${u}, ${v}) · X ${pixelTarget.x_mm.toFixed(1)} / Y ${pixelTarget.y_mm.toFixed(1)} / Z ${pixelTarget.z_mm.toFixed(2)} mm · 블록 윗면 고정. 실행 시 IK 검사`;
    enablePixelButtons();
  } catch(error) {
    if(generation!==pixelSelectionGeneration)return;
    mark.dataset.state='invalid';$('pixel-target-status').textContent=error.message;
  }
});
function pixelArguments() {
  const {u,v,calibration_id}=pixelTarget;return {u,v,calibration_id};
}
$('move-pixel').addEventListener('click',()=>{if(pixelTarget)manual('move_to_pixel',pixelArguments());});
$('place-pixel').addEventListener('click',()=>{if(pixelTarget)manual('place_at_pixel',pixelArguments());});
$('pixel-to-chat').addEventListener('click',()=>{
  if(!pixelTarget)return;
  $('chat-tab').click();insertText(`헤드캠 선택 픽셀 (u=${pixelTarget.u}, v=${pixelTarget.v}), calibration_id=${pixelTarget.calibration_id}`);
});

async function loadConfig() {
  const response = await fetch("/api/config");
  const config = await response.json();
  pixelPreview=config.pixel_preview||null;
  keyboardConfig=config.keyboard_jog||null;
  manualTools = new Set(config.manual_tools || []);
  applyControl(control);
  $("model-badge").textContent = `${config.provider} · ${config.model}`;
  const selected=selectedCamera();
  $('camera-select').value=selected;
  $('camera-select').addEventListener('change', e => {
    const url=new URL(location.href);url.searchParams.set('camera',e.target.value);
    try {sessionStorage.setItem('so101-draft', $('input').value);} catch (_) {}
    location.assign(url.href);
  });
  try {const draft=sessionStorage.getItem('so101-draft');if(draft!==null){$('input').value=draft;sessionStorage.removeItem('so101-draft');}} catch (_) {}
  document.body.dataset.camera=selected;
  if(selected==='wrist')$('pixel-target-status').textContent='픽셀 위치 선택은 헤드캠에서만 가능합니다.';
  const img = $("camera");
  img.onerror = () => { $("camera-missing").hidden = false; };
  img.onload = () => { $("camera-missing").hidden = true; };
  img.src = cameraUrl(config);
  const toolsUrl = new URL(config.camera_base_url);
  if (["127.0.0.1", "localhost"].includes(toolsUrl.hostname) &&
      !["127.0.0.1", "localhost"].includes(location.hostname)) {
    toolsUrl.hostname = location.hostname;
    if (location.protocol === "https:") { toolsUrl.protocol = "https:"; toolsUrl.port = ""; }
  }
  $("camera-tools-link").href = toolsUrl.origin + "/";
  if (selected === "shoulder" && config.places && config.places.image_size) {
    $("camera-wrap").style.aspectRatio = `${config.places.image_size[0]} / ${config.places.image_size[1]}`;
    drawPlaces(config.places);
  }
  if (selected === 'shoulder') await window.startCameraOverlay(config);
  else {
    $('overlay').setAttribute('hidden',''); $('perception-overlay').hidden=true;
    $('detection-details').hidden=false;
    $('detection-list').textContent='손목캠에는 헤드캠 보정·검출 결과를 겹쳐 표시하지 않습니다. 헤드캠으로 전환해 확인하세요.';
    document.querySelectorAll('[data-camera-layer]').forEach(b=>b.disabled=true);
  }
}

async function pollHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    const badge = $("camera-badge");
    badge.textContent = health.camera_ok ? "카메라 정상" : "카메라 끊김";
    badge.className = "badge " + (health.camera_ok ? "badge-idle" : "badge-stopped");
  } catch (_) {
    $("camera-badge").textContent = "서버 응답 없음";
  }
  setTimeout(pollHealth, 3000);
}

loadConfig();
acquireLease();
pollHealth();

// Workspace layout: preserve original controls and their event handlers.
(() => {
  const $ = id => document.getElementById(id);
  document.querySelectorAll('[data-command-tab]').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('[data-command-tab]').forEach(b => {
      const selected=b===button;b.setAttribute('aria-selected',String(selected));$(b.dataset.commandTab).hidden=!selected;
    });
  }));
  $('command-chat').setAttribute('role','tabpanel');
  $('command-chat').setAttribute('aria-labelledby','chat-tab');
  $('overlay').addEventListener('click', e => {
    if(e.target.closest('.slot-cell,.region-dot,.grid-cell')) $('chat-tab').click();
  });
  const controls=document.querySelector('.perception-controls');
  controls.prepend($('camera-tools'));
  const settings=document.createElement('details');settings.className='overlay-settings';
  const label=document.createElement('summary');label.textContent='표시 설정';settings.append(label);
  const options=document.createElement('div');options.className='overlay-options';settings.append(options);
  controls.append(settings);
  options.append(document.querySelector('.layer-buttons'));
  options.append(document.querySelector('.color-filter'),$('camera-tools-link'));
  $('diagnostic-detections').append($('candidate-hint'),$('fk-status'),$('detection-details'));
  document.querySelectorAll('[data-diagnostic-tab]').forEach(button => button.addEventListener('click', () => {
    const tab=button.dataset.diagnosticTab;document.body.dataset.diagnosticTab=tab;
    document.querySelectorAll('[data-diagnostic-tab]').forEach(b=>b.setAttribute('aria-selected',String(b===button)));
    $('robot-diagnostics').hidden=['detections','history'].includes(tab);
    $('diagnostic-detections').hidden=tab!=='detections';$('diagnostic-history').hidden=tab!=='history';
    if(tab==='detections') {
      const detail=document.querySelector('[data-camera-layer="details"]');
      if(detail.getAttribute('aria-pressed')!=='true')detail.click();
    }
  }));
  document.body.dataset.diagnosticTab='joints';
  const history=$('diagnostic-history');
  new MutationObserver(() => {
    const tools=[...$('chat').querySelectorAll('.tool,.msg-error')];
    if(!tools.length)return;
    history.replaceChildren(...tools.map(e=>e.cloneNode(true)));
  }).observe($('chat'),{childList:true,subtree:true,characterData:true});
})();

// Diagnostic reads are serialized on the server's existing robot worker.
(() => {
  const panel = document.getElementById('robot-diagnostics');
  const body = document.getElementById('telemetry-body');
  const status = document.getElementById('telemetry-status');
  const fmt = v => typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(1)) : v == null ? '—' : String(v);
  const xyz = v => v ? v.map(fmt).join(' / ') + ' mm' : '—';
  const add = (parent, tag, text) => {const e=document.createElement(tag);e.textContent=text;parent.append(e);return e;};
  function table(parent, headings, rows) {
    const wrap=add(parent,'div','');wrap.className='telemetry-scroll';
    const t=add(wrap,'table',''); const h=add(t,'thead','');const tr=add(h,'tr','');
    headings.forEach(x=>add(tr,'th',x));const b=add(t,'tbody','');
    rows.forEach(row=>{const r=add(b,'tr','');row.forEach(x=>add(r,'td',x));});
  }
  async function poll() {
    if (!panel.open || document.hidden) {setTimeout(poll,1000);return;}
    try {
      const r=await fetch('/api/telemetry',{cache:'no-store',signal:AbortSignal.timeout(4000)});
      if(!r.ok)throw Error('HTTP '+r.status);
      const d=await r.json(); const age=d.sampled_at ? (Date.now()/1000-d.sampled_at) : null;
      status.textContent=age==null ? '최초 측정 대기 중' : `측정 ${new Date(d.sampled_at*1000).toLocaleTimeString()} (${age.toFixed(1)}초 전) · 조회 간격 1초 · ${d.pending?'조회 대기':'수신 완료'} · ${d.control?.state ?? ''} · ${d.control?.busy_with ?? '대기'}`;
      status.classList.toggle('telemetry-stale',age==null||age>3);
      const expanded=[...body.querySelectorAll('details')].map(e=>e.open);
      const scrolls=[...body.querySelectorAll('.telemetry-scroll')].map(e=>e.scrollLeft);
      body.replaceChildren();
      if(d.error)add(body,'p','조회 오류: '+d.error);
      add(body,'p','현재 FK X / Y / Z: '+xyz(d.fk));
      add(body,'p','모터 목표값의 FK: '+xyz(d.target_fk));
      add(body,'p','목표 − 현재 FK: '+xyz(d.fk_delta)+' · 거리 '+fmt(d.fk_distance)+' mm');
      add(body,'p','FK는 모델 계산값이며, 목표는 현재 모터 레지스터 값입니다. 작업의 최종 목표나 실제 손끝 관측값과 다를 수 있습니다.');
      document.getElementById('fk-summary').textContent='현재 FK  '+xyz(d.fk)+'   ·   목표 차이 '+fmt(d.fk_distance)+' mm';
      const motors=d.motors ?? [];
      table(body,['관절','단위','현재','모터 목표','차이','최소','최대'],motors.map(m=>[m.name,m.unit,...[m.current,m.target,m.error,m.minimum,m.maximum].map(fmt)]));
      body.querySelector('.telemetry-scroll').dataset.diagnosticSection='joints';
      body.querySelectorAll('.telemetry-scroll tbody tr').forEach((row,i)=>{
        const m=motors[i];
        if(m.current!=null && (m.current<m.minimum || m.current>m.maximum)) {
          row.classList.add('joint-out-of-range');row.title='현재 값이 모터 보정 범위를 벗어났습니다';
        }
      });
      const details=add(body,'details','');details.dataset.diagnosticSection='motors';add(details,'summary','인코더 · 부하 · 온도 · 전압 원시값 · 토크');
      table(details,['모터','현재 tick','목표 tick','부하 raw','온도 °C','전압 raw','토크'],motors.map(m=>[m.name,...[m.encoder,m.goal_encoder,m.load,m.temperature,m.voltage_raw].map(fmt),m.torque==null?'—':m.torque?'ON':'OFF']));
      add(details,'p','전압은 단위 변환 전 레지스터 값입니다. 부하 raw는 토크(N·m)가 아닙니다.');
      const cal=add(body,'details','');cal.dataset.diagnosticSection='calibration';add(cal,'summary','보정 · 범위 · 안전 제한');
      add(cal,'p','파일: '+(d.calibration_file??'—'));
      add(cal,'p','모터 보정 일치: '+(d.calibration_match==null?'미확인':d.calibration_match?'일치':'불일치'));
      add(cal,'p','틱당 상대 목표 제한: '+JSON.stringify(d.max_relative_target??null)+' · 각도 최소/최대는 모터 보정 범위 환산값입니다.');
      table(cal,['모터','ID','Homing offset','최소 tick','최대 tick','URDF 최소 °','URDF 최대 °'],motors.map(m=>[m.name,m.id,m.calibration?.homing_offset,m.calibration?.range_min,m.calibration?.range_max,...(m.urdf_limits_deg??[null,null])].map(fmt)));
      const g=motors.find(m=>m.name==='gripper');
      add(body,'p',`세션 파지 상태: ${d.grasp?.held ? '보유 판정 기록 있음' : '보유 판정 기록 없음'} · 그리퍼 개방 ${fmt(g?.current)} / 100 · 부하 ${fmt(g?.load)} raw · 판정 모드 ${d.grasp?.mode??'—'}`);
      add(body,'p',`파지 기준: 위치 > ${fmt(d.grasp?.position_threshold)}, |부하| ≥ ${fmt(d.grasp?.load_threshold)}. ${d.grasp?.note??''}`);
      if(d.control?.message)add(body,'p','제어 메시지: '+d.control.message);
      if(d.errors?.length)add(body,'p','읽기 오류: '+d.errors.join(', '));
      [...body.querySelectorAll('details')].forEach(e=>e.open=true);
      [...body.querySelectorAll('.telemetry-scroll')].forEach((e,i)=>e.scrollLeft=scrolls[i]??0);
    } catch(e) {status.textContent='상태 조회 실패 · 기존 표는 마지막 측정값: '+e.message;status.classList.add('telemetry-stale');}
    setTimeout(poll,1000);
  }
  poll();
})();


// Held translation keys publish the latest direction; other actions stay discrete.
(() => {
  const toggle=$('keyboard-toggle'),status=$('keyboard-status');
  const held=new Set();
  const directions={KeyW:[1,0,0],KeyS:[-1,0,0],KeyA:[0,1,0],KeyD:[0,-1,0],KeyJ:[0,0,1],KeyK:[0,0,-1]};
  const actions={KeyQ:'#roll-ccw',KeyE:'#roll-cw',KeyG:'#pick-here',KeyP:'#place-here',KeyO:'#open-gripper',KeyH:'#manual-home'};
  let armed=false,session=null,brakingSession=null,starting=false,timer=null,inFlight=false,epoch=0,seq=0;
  const editable=target=>target instanceof Element && Boolean(target.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"]),[role="textbox"]'));
  const vector=()=>{const v=[0,0,0];for(const key of held)directions[key]?.forEach((n,i)=>v[i]+=n);return v;};
  function stopStream(graceful=false) {
    ++epoch;clearTimeout(timer);timer=null;
    const old=session;session=null;
    if(!graceful&&brakingSession) {
      api('/api/keyboard/stop',{session_id:brakingSession}).catch(()=>{});brakingSession=null;
    }
    if(old) {
      if(graceful)brakingSession=old;
      api(graceful?'/api/keyboard/release':'/api/keyboard/stop',{session_id:old}).catch(()=>{});
    }
  }
  function setArmed(value) {
    stopStream();held.clear();armed=value;
    toggle.setAttribute('aria-pressed',String(armed));
    toggle.textContent=armed?'키보드 끄기':'키보드 켜기';
    status.textContent=armed?`활성 · WASD / J·K 누르는 동안 이동 · 떼면 감속 (설정 ${keyboardConfig?.speed_mm_s} mm/s)`:'꺼짐 · 버튼을 눌러 활성화';
    document.querySelector('.keyboard-control').classList.toggle('keyboard-armed',armed);
  }
  async function sendDirection() {
    clearTimeout(timer);
    if(!session||inFlight)return;
    const id=session,v=vector();
    if(!armed||!isOperator||!v.some(Boolean)){stopStream();return;}
    inFlight=true;
    try {
      const {status:code,data}=await api('/api/keyboard/update',{session_id:id,seq:++seq,vector:v});
      if(session!==id)return;
      if(code!==200)throw Error(data.error||'연속 이동이 종료되었습니다.');
    } catch(error) {
      if(session===id){setArmed(false);showToast(error.message,'bad');}
    } finally {
      inFlight=false;
      if(session===id)timer=setTimeout(sendDirection,keyboardConfig.heartbeat_s*1000);
    }
  }
  async function startStream() {
    if(starting||session||!vector().some(Boolean))return;
    if(!keyboardConfig||!manualAllowed('move_arm')||control.state!=='idle'||!isOperator||directPending){held.clear();return;}
    starting=true;const generation=epoch;
    try {
      const {status:code,data}=await api('/api/keyboard/start');
      if(code!==202)throw Error(data.error||'연속 이동을 시작할 수 없습니다.');
      // Start only reserves the worker. No motion until this first update.
      if(generation!==epoch||!armed||!vector().some(Boolean)) {
        api('/api/keyboard/stop',{session_id:data.session_id}).catch(()=>{});return;
      }
      session=data.session_id;seq=0;sendDirection();
    } catch(error) {
      if(generation===epoch){setArmed(false);showToast(error.message,'bad');}
    } finally {starting=false;}
  }
  toggle.addEventListener('click',()=>{
    if(!armed&&!keyboardConfig){showToast('연속 키보드 조작은 제어 서버에 변경 사항을 적용한 뒤 사용할 수 있습니다.','bad');return;}
    setArmed(!armed);
  });
  document.addEventListener('keydown',event=>{
    if(event.key==='Escape'){setArmed(false);return;}
    if(!armed)return;
    if(event.isComposing||event.ctrlKey||event.altKey||event.metaKey||event.shiftKey||editable(event.target)) {setArmed(false);return;}
    if(!directions[event.code]&&!actions[event.code])return;
    if(document.hidden||$('direct-controls').hidden){setArmed(false);return;}
    event.preventDefault();
    if(event.repeat||held.has(event.code))return;
    if(directions[event.code]) {
      held.add(event.code);
      if(!vector().some(Boolean)){stopStream(true);return;}
      if(session)sendDirection();else startStream();
    } else {
      held.add(event.code);
      if(session||starting||control.state!=='idle'||!isOperator||directPending)return;
      const button=document.querySelector(actions[event.code]);
      if(button&&!button.disabled)button.click();
    }
  });
  document.addEventListener('keyup',event=>{
    if(!held.delete(event.code)||!directions[event.code])return;
    if(!vector().some(Boolean))stopStream(true);
    else if(session)sendDirection();
    // No restart on key release: a new key press is required after a stop.
  });
  document.addEventListener('robot-control-state',event=>{
    if(event.detail.state==='idle')brakingSession=null;
    if(session&&(event.detail.state!=='busy'||event.detail.busy_with!=='keyboard_jog'))setArmed(false);
  });
  window.addEventListener('blur',()=>setArmed(false));
  window.addEventListener('pagehide',()=>setArmed(false));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)setArmed(false);});
  document.addEventListener('focusin',event=>{if(editable(event.target))setArmed(false);});
  $('chat-tab').addEventListener('click',()=>setArmed(false));
  setArmed(false);
})();

// Theme preference is local to the browser and never changes robot state.
(() => {
  const key='so101-theme',root=document.documentElement;
  const button=document.getElementById('theme-toggle');
  const media=matchMedia('(prefers-color-scheme: dark)');
  function saved() {try{return localStorage.getItem(key);}catch(_){return null;}}
  function apply(mode) {
    const dark=mode==='dark'||(mode!=='light'&&media.matches);
    root.classList.toggle('dark',dark);root.dataset.theme=dark?'dark':'light';
    root.style.colorScheme=dark?'dark':'light';
    button.textContent=dark?'라이트 모드':'다크 모드';
    button.setAttribute('aria-label',dark?'라이트 모드로 전환':'다크 모드로 전환');
    button.setAttribute('aria-pressed',String(dark));
  }
  button.addEventListener('click',()=>{
    const mode=root.dataset.theme==='dark'?'light':'dark';
    try {localStorage.setItem(key,mode);}catch(_){}
    apply(mode);
  });
  media.addEventListener('change',()=>{if(!['light','dark'].includes(saved()))apply(null);});
  window.addEventListener('storage',event=>{if(event.key===key||event.key===null)apply(saved());});
  apply(saved());
})();

// Keep both command tabs aligned with the camera as its size changes.
(() => {
  const camera=document.querySelector('.camera-panel');
  const commands=document.querySelector('.command-panel');
  const syncHeight=()=>commands.style.setProperty('--camera-panel-height',`${camera.getBoundingClientRect().height}px`);
  new ResizeObserver(syncHeight).observe(camera);
  syncHeight();
})();
