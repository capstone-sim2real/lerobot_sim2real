/* Camera observation only: this module never sends robot commands. */
(() => {
  "use strict";
  const KEY = "so101-camera-layers-v1";
  let staleMs, pollMs, retryMs;
  const defaults = {master: true, places: true, boxes: true, centers: true, axes: true,
    candidates: false, rejects: false, fk: false, details: false, color: ""};
  let flags = {...defaults};
  try {
    const stored = JSON.parse(localStorage.getItem(KEY));
    for (const key of Object.keys(defaults)) {
      if (typeof stored?.[key] === typeof defaults[key]) flags[key] = stored[key];
    }
  } catch (_) { /* private browsing or invalid stored preferences */ }
  const $ = id => document.getElementById(id);
  let source = null, packet = null, reference = null, lastFresh = 0, referenceFresh = 0;
  let generation = 0, refAbort = null, timer = null, config = null, message = "검출 연결 중…";
  let disposed = false, ready = false, fkSupported = true, placesCompatible = true;
  let fallbackSize = [1280, 720], nextConfigAttempt = 0;
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(flags)); } catch (_) {} };
  const needsDetections = () => flags.master &&
    ["boxes", "centers", "axes", "candidates", "rejects", "details"].some(k => flags[k]);
  const age = () => lastFresh ? performance.now() - lastFresh : Infinity;
  function closeEvents() {
    source?.close(); source = null; packet = null; lastFresh = 0;
  }
  function syncEvents() {
    if (!ready || disposed || document.hidden || !needsDetections()) { closeEvents(); return; }
    if (source) return;
    source = new EventSource("/api/camera/events");
    message = "검출 연결 중…";
    source.onmessage = event => {
      try {
        const next = JSON.parse(event.data);
        if (next.error || !next.ready) {
          packet = null; lastFresh = 0; message = "검출 준비 중"; render(); return;
        }
        if (!Array.isArray(next.image_size) || next.image_size[0] !== config.image_size[0] ||
            next.image_size[1] !== config.image_size[1]) {
          packet = null; lastFresh = 0; message = "영상·검출 해상도 불일치"; render(); return;
        }
        if (next.frame_seq !== packet?.frame_seq || next.captured_at !== packet?.captured_at) {
          lastFresh = performance.now();
        }
        packet = next; message = "검출 정상";
      } catch (_) { packet = null; lastFresh = 0; message = "검출 데이터 오류"; }
      render();
    };
    source.onerror = () => {
      packet = null; lastFresh = 0; message = "검출 연결 끊김 · 재연결 중"; render();
    };
  }
  function syncUI() {
    for (const button of document.querySelectorAll("[data-camera-layer]")) {
      const key = button.dataset.cameraLayer;
      button.setAttribute("aria-pressed", String(flags[key]));
      button.disabled = (key !== "master" && !flags.master) || (key === "places" && !placesCompatible);
    }
    $("detection-color").value = flags.color;
    $("detection-color").disabled = !flags.master;
    $("overlay").toggleAttribute("hidden", !(flags.master && flags.places && placesCompatible));
    for (const radio of document.querySelectorAll('input[name="places-layer"]')) {
      radio.disabled = !(flags.master && flags.places && placesCompatible);
    }
    $("camera-tools").classList.toggle("layers-disabled", !(flags.master && flags.places && placesCompatible));
    $("detection-details").hidden = !(flags.master && flags.details);
  }
  function renderDetails(fresh) {
    const panel = $("detection-list"); panel.replaceChildren();
    if (!flags.master || !flags.details) return;
    if (!fresh) { panel.textContent = "최신 검출 결과를 기다리고 있습니다."; return; }
    const pointText = p => Array.isArray(p) ? p.map(v => Number(v).toFixed(1)).join(", ") + " mm" : "—";
    for (const d of packet.detections || []) {
      if (flags.color && d.color !== flags.color) continue;
      const row = document.createElement("div"); row.className = "detection-row";
      const name = document.createElement("strong");
      name.textContent = window.CameraOverlayDrawing.colors[d.color] || d.color;
      const text = document.createElement("span");
      text.textContent = "중심 " + pointText(d.center_mm) + " · 보정 " + pointText(d.biased_center_mm) +
        " · 파지 방향 " + Number(d.grasp_yaw_deg ?? d.block_angle_deg).toFixed(1) + "°";
      row.append(name, text); panel.append(row);
    }
    if (flags.rejects) for (const d of packet.rejects || []) {
      if (flags.color && d.color !== flags.color) continue;
      const row = document.createElement("div");
      row.textContent = (window.CameraOverlayDrawing.colors[d.color] || d.color) +
        " 제외 · " + (window.CameraOverlayDrawing.reasons[d.reason] || d.reason);
      panel.append(row);
    }
    if (!panel.childNodes.length) panel.textContent = "선택한 색상의 블록이 없습니다.";
  }
  function render() {
    if (!config) return;
    const canvas = $("perception-overlay"), ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const img = $("camera");
    const imageCompatible = !img.naturalWidth ||
      (img.naturalWidth === config.image_size[0] && img.naturalHeight === config.image_size[1]);
    const fresh = age() < staleMs && imageCompatible;
    const fkFresh = performance.now() - referenceFresh < staleMs && reference?.available &&
      Number.isFinite(reference.measured_at) && Date.now() / 1000 - reference.measured_at < staleMs / 1000;
    if (flags.master && imageCompatible) window.CameraOverlayDrawing.drawPerception(
      ctx, fresh ? packet : null, fkFresh ? reference : null, flags, flags.color);
    $("candidate-hint").hidden = !(flags.master && flags.candidates);
    $("fk-status").hidden = !(flags.master && flags.fk);
    $("fk-status").textContent = fkFresh ? "FK 기준점: 블록 평면 투영 · 영상 검출 아님" :
      "FK 기준점 없음 또는 오래된 측정 · 영상 검출 아님";
    renderDetails(fresh);
  }
  async function pollReference() {
    if (disposed || !ready || !fkSupported || document.hidden || !flags.master || !flags.fk) return;
    const current = generation;
    refAbort = new AbortController();
    const timeout = setTimeout(() => refAbort?.abort(), staleMs);
    try {
      const result = await fetch("/api/camera/reference", {signal: refAbort.signal, cache: "no-store"});
      if (result.status === 404) fkSupported = false;
      const next = result.ok ? await result.json() : null;
      if (generation === current) { reference = next; referenceFresh = performance.now(); }
    } catch (_) { reference = null; }
    finally { clearTimeout(timeout); refAbort = null; render(); }
  }
  async function loadOverlayConfig() {
    nextConfigAttempt = performance.now() + retryMs;
    const abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(), staleMs);
    try {
      const result = await fetch("/api/camera/config", {signal: abort.signal});
      if (!result.ok) throw new Error("overlay unavailable");
      const next = await result.json();
      if (!Array.isArray(next.image_size) || next.image_size.length !== 2 ||
          !next.image_size.every(v => Number.isFinite(v) && v > 0)) throw new Error("invalid size");
      if (disposed) return;
      config = next;
      placesCompatible = fallbackSize.every((v, i) => v === next.image_size[i]);
      ready = true;
      $("perception-overlay").width = config.image_size[0];
      $("perception-overlay").height = config.image_size[1];
      $("camera-wrap").style.aspectRatio = config.image_size.join(" / ");
      syncUI(); syncEvents();
    } catch (_) { message = "검출 오버레이 연결 대기 · 원본 영상은 계속 표시"; }
    finally { clearTimeout(timeout); }
  }
  async function tick() {
    if (disposed) return;
    if (!ready && !document.hidden && performance.now() >= nextConfigAttempt) await loadOverlayConfig();
    await pollReference(); render();
    if (!disposed) timer = setTimeout(tick, pollMs);
  }
  window.startCameraOverlay = async uiConfig => {
    staleMs = uiConfig.camera_view.stale_s * 1000;
    pollMs = uiConfig.camera_view.poll_s * 1000;
    retryMs = uiConfig.camera_view.retry_s * 1000;
    for (const button of document.querySelectorAll("[data-camera-layer]")) {
      button.addEventListener("click", () => {
        const key = button.dataset.cameraLayer; flags[key] = !flags[key];
        generation++; reference = null; refAbort?.abort();
        save(); syncUI(); syncEvents(); render();
      });
    }
    $("detection-color").addEventListener("change", event => {
      flags.color = event.target.value; save(); render();
    });
    syncUI();
    fallbackSize = uiConfig.places?.image_size || [1280, 720];
    config = {image_size: fallbackSize};
    await loadOverlayConfig();
    syncEvents(); render(); tick();
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) { generation++; refAbort?.abort(); reference = null; }
      syncEvents(); render();
    });
    window.addEventListener("pagehide", () => {
      disposed = true; clearTimeout(timer); generation++; refAbort?.abort(); closeEvents();
    });
    window.addEventListener("pageshow", event => {
      if (event.persisted) { disposed = false; syncEvents(); tick(); }
    });
  };
})();
