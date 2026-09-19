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
  const response = await fetch(path, { method: "POST", headers, body: JSON.stringify(body || {}), keepalive: path === "/api/stop" });
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
    case "error": addMessage("error", event.message); break;
    default: break;
  }
}

// ── control state ───────────────────────────────────────────────────
function applyControl(snapshot) {
  control = snapshot;
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
  $("stopped-banner").hidden = !(snapshot.state === "stopped" || snapshot.state === "homing" || snapshot.state === "stopping");
  $("home-button").disabled = !(snapshot.state === "stopped" && isOperator);
  $("home-button").textContent = snapshot.state === "homing" ? "복귀 중…" : "다시 시도(그리퍼 열기+home)";
  if (snapshot.message) $("stopped-message").textContent = snapshot.message;
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
document.addEventListener("keydown", (e) => { if (e.key === "Escape") pressStop(); });
$("home-button").addEventListener("click", () => api("/api/home"));
$("lock-retry").addEventListener("click", acquireLease);
$("reset").addEventListener("click", () => api("/api/reset"));

async function directRequest(path, body) {
  try {
    const { status, data } = await api(path, body);
    if (status !== 202 && status !== 403) {
      showToast(data.error || `명령을 받지 못했습니다 (HTTP ${status}).`, "bad");
    }
  } catch (error) {
    showToast(`서버에 명령을 보내지 못했습니다: ${error.message}`, "bad");
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
function cameraUrl(config) {
  const url = new URL(config.camera_base_url);
  if (["127.0.0.1", "localhost"].includes(url.hostname) && !["127.0.0.1", "localhost"].includes(location.hostname)) {
    // The venue exposes camera.server at the tailnet hostname's standard
    // HTTPS port while this agent UI may use a different HTTPS port.
    // Avoid mixed-content blocking (https page -> http MJPEG).
    if (location.protocol === "https:") return `https://${location.hostname}${config.mjpeg_path}`;
    url.hostname = location.hostname;
  }
  return url.origin + config.mjpeg_path;
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
  const svg = $("overlay");
  const [w, h] = places.image_size;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const sector = places.sector_px;
  if (sector && sector.arc.length) {
    // base + arc closes the reach envelope into the sector the camera page
    // draws; drawn first so it never sits on top of a clickable place
    const outline = [sector.base, ...sector.arc].map((p) => p.join(",")).join(" ");
    svg.appendChild(svgEl("polygon", { class: "sector-poly", points: outline }));
  }
  const grid = places.grid && places.grid.cells.length ? places.grid : null;
  const gridLayer = grid ? buildGridLayer(grid) : null;
  if (gridLayer) svg.appendChild(gridLayer);
  if (places.zone_polygon_px.length) {
    svg.appendChild(svgEl("polygon", { class: "zone-poly", points: places.zone_polygon_px.map((p) => p.join(",")).join(" ") }));
  }
  const zone = places.zone_polygon_px;
  const cellHalf = zone.length === 4 ? Math.max(18, Math.abs(zone[1][0] - zone[0][0]) / 7) : 30;
  for (const slot of places.slots) {
    const [x, y] = slot.px;
    const cell = svgEl("rect", { class: "slot-cell", x: x - cellHalf, y: y - cellHalf * 0.8, width: cellHalf * 2, height: cellHalf * 1.6, rx: 6 });
    const title = svgEl("title", {});
    title.textContent = slot.korean;
    cell.appendChild(title);
    cell.addEventListener("click", () => insertText(`적재 구역 ${slot.korean}`));
    svg.appendChild(cell);
    const label = svgEl("text", { class: "slot-text", x, y: y + 8, "text-anchor": "middle" });
    label.textContent = slot.korean;
    svg.appendChild(label);
  }
  const regionLayer = svgEl("g", { id: "regions-layer" });
  for (const region of places.regions) {
    const [x, y] = region.px;
    if (!(x >= 0 && y >= 0 && x <= w && y <= h)) continue;
    const dot = svgEl("circle", { class: "region-dot", cx: x, cy: y, r: 11 });
    const title = svgEl("title", {});
    title.textContent = `부채꼴 ${region.korean}`;
    dot.appendChild(title);
    dot.addEventListener("click", () => insertText(`부채꼴 ${region.korean}`));
    regionLayer.appendChild(dot);
  }
  svg.appendChild(regionLayer);
  wireLayerToggle(grid, gridLayer, regionLayer);
}

async function loadConfig() {
  const response = await fetch("/api/config");
  const config = await response.json();
  $("model-badge").textContent = `${config.provider} · ${config.model}`;
  const img = $("camera");
  img.onerror = () => { $("camera-missing").hidden = false; };
  img.onload = () => { $("camera-missing").hidden = true; };
  img.src = cameraUrl(config);
  if (config.places && config.places.image_size) {
    $("camera-wrap").style.aspectRatio = `${config.places.image_size[0]} / ${config.places.image_size[1]}`;
    drawPlaces(config.places);
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
