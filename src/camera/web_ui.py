"""Browser UI for a continuous MJPEG image plus client-side canvas overlay."""

from __future__ import annotations

import html


def render_camera_page(
    cameras: list[tuple[str, str]],
    *,
    overlay_cameras: set[str],
) -> bytes:
    primary, *secondary = cameras
    camera_tiles = _camera_tile(
        *primary,
        overlay=primary[0] in overlay_cameras,
        secondary=secondary,
    )
    picker = ""
    if overlay_cameras:
        picker = """
        <label for="tool-menu">tool</label>
        <select id="tool-menu" aria-label="camera tool">
          <option value="overlay" selected>overlay</option>
          <option value="cross">cross calibration</option>
        </select>
        <button id="cross-reset" type="button" hidden>reset cross points</button>
        <label for="overlay-color">overlay</label>
        <select id="overlay-color" aria-label="overlay colour">
          <option value="" selected>all</option>
          <option value="green">green</option>
          <option value="yellow">yellow</option>
          <option value="blue">blue</option>
          <option value="red">red</option>
          <option value="wood">wood</option>
          <option value="none">none</option>
        </select>
        """

    page = _PAGE.replace("__CAMERA_TILES__", camera_tiles).replace(
        "__OVERLAY_PICKER__", picker
    )
    return page.encode("utf-8")


def _camera_tile(
    name: str,
    device: str,
    *,
    overlay: bool,
    secondary: list[tuple[str, str]],
) -> str:
    safe_name = html.escape(name)
    safe_device = html.escape(device)
    canvas = (
        f'<canvas class="camera-overlay" data-camera="{safe_name}" '
        f'aria-label="{safe_name} perception overlay"></canvas>'
        if overlay
        else ""
    )
    details = (
        '<aside class="details" aria-live="polite">Waiting for detections…</aside>'
        if overlay
        else ""
    )
    secondary_views = "\n".join(
        f'''<section class="secondary-camera">
            <div class="stream-stack">
              <img class="camera-image" data-camera="{html.escape(extra_name)}"
                   src="/video/{html.escape(extra_name)}.mjpg" alt="{html.escape(extra_name)} camera stream">
            </div>
            <div class="camera-info"><span>{html.escape(extra_name)} · {html.escape(extra_device)}</span></div>
          </section>'''
        for extra_name, extra_device in secondary
    )
    return f"""
      <section class="camera-layout" data-camera="{safe_name}">
        <div class="camera-view">
          <div class="stream-stack">
            <img class="camera-image" data-camera="{safe_name}"
                 src="/video/{safe_name}.mjpg" alt="{safe_name} camera stream">
            {canvas}
          </div>
          <div class="camera-info">
            <span>{safe_name} · {safe_device}</span>
            <span class="overlay-status"></span>
          </div>
          {secondary_views}
        </div>
        {details}
      </section>
    """


_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SO-101 Cameras</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Arial, sans-serif;
      background: #111;
      color: #f5f5f5;
    }
    body { margin: 0; padding: 24px; }
    main, .topbar { max-width: 1280px; margin-left: auto; margin-right: auto; }
    .topbar {
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      color: #aaa;
      font-size: 13px;
    }
    .camera-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 230px;
      gap: 20px;
      align-items: start;
    }
    .secondary-camera { margin-top: 16px; }
    .stream-stack {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      overflow: hidden;
      background: #000;
    }
    .camera-image, .camera-overlay {
      position: absolute;
      inset: 0;
      display: block;
      width: 100%;
      height: 100%;
    }
    .camera-image { object-fit: contain; }
    .camera-overlay { pointer-events: none; }
    .camera-info {
      padding-top: 8px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: #a8a8a8;
      font-size: 13px;
    }
    .overlay-status { text-align: right; }
    .details {
      display: grid;
      gap: 9px;
      color: #b0b0b0;
      font-size: 13px;
      line-height: 1.3;
    }
    .detail-block { display: grid; gap: 6px; }
    .detail-block + .detail-block { padding-top: 10px; border-top: 1px solid #444; }
    .detail-name { color: #f5f5f5; font-weight: 600; text-transform: capitalize; }
    .detail-row { display: grid; grid-template-columns: 66px 1fr; gap: 8px; }
    .detail-row.c span:first-child { color: #00ffff; }
    .detail-row.b span:first-child { color: #ffa500; }
    .detail-row.retry span:first-child { color: #ff00ff; }
    .detail-row.angle span:first-child { color: #ffff00; }
    .detail-row.reject span:first-child { color: #9aa0a6; }
    select {
      color: #f5f5f5;
      background: #111;
      border: 1px solid #666;
      padding: 5px 7px;
      font: inherit;
    }
    button {
      color: #f5f5f5;
      background: #111;
      border: 1px solid #666;
      padding: 5px 7px;
      font: inherit;
      cursor: pointer;
    }
    @media (max-width: 760px) {
      .camera-layout { grid-template-columns: 1fr; }
      .details { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="topbar">__OVERLAY_PICKER__</div>
  <main>__CAMERA_TILES__</main>
  <script>
    const picker = document.querySelector('#overlay-color');
    const toolMenu = document.querySelector('#tool-menu');
    const crossReset = document.querySelector('#cross-reset');
    const overlays = new Map();
    let gripperReference = null;
    const fetchGripperReference = async () => {
      try {
        const response = await fetch('/gripper-reference.json', {cache: 'no-store'});
        gripperReference = response.ok ? await response.json() : null;
      } catch (_) { gripperReference = null; }
      overlays.forEach(renderDetails);
      setTimeout(fetchGripperReference, 500);
    };
    const shortLabel = {
      'centre': 'B', 'front': 'F', 'back': 'BK', 'left': 'L', 'right': 'R',
      'front-left': 'FL', 'front-right': 'FR',
      'back-left': 'BL', 'back-right': 'BR'
    };

    const pointText = (point) =>
      `(${point.map((value) => Number(value).toFixed(1)).join(', ')}) mm`;

    const addDetailRow = (parent, label, value, style) => {
      const row = document.createElement('div');
      row.className = `detail-row ${style}`;
      const key = document.createElement('span');
      key.textContent = label;
      const text = document.createElement('span');
      text.textContent = value;
      row.append(key, text);
      parent.append(row);
    };

    const selectedColor = () => picker?.value;
    const crossMode = () => toolMenu?.value === 'cross';
    const visibleDetections = (state) => {
      const color = selectedColor();
      if (color === 'none') return [];
      return (state.latest?.detections || []).filter(
        (detection) => !color || detection.color === color
      );
    };

    const visibleRejects = (state) => {
      const color = selectedColor();
      if (color === 'none') return [];
      return (state.latest?.rejects || []).filter(
        (reject) => !color || reject.color === color
      );
    };

    const rejectText = (reject) => {
      const value = {
        area: `${reject.area_mm2} mm²`,
        aspect: reject.aspect,
        fill: reject.fill,
        solidity: reject.solidity
      }[reject.reason];
      return `${reject.color} ✕ ${reject.reason} ${value}`;
    };

    const drawPolyline = (context, points, color, width, close = false) => {
      if (!points || points.length < 2) return;
      context.beginPath();
      context.moveTo(points[0][0], points[0][1]);
      points.slice(1).forEach((point) => context.lineTo(point[0], point[1]));
      if (close) context.closePath();
      context.strokeStyle = color;
      context.lineWidth = width;
      context.stroke();
    };

    const drawCross = (context, point, color, size = 6) => {
      context.beginPath();
      context.moveTo(point[0] - size, point[1]);
      context.lineTo(point[0] + size, point[1]);
      context.moveTo(point[0], point[1] - size);
      context.lineTo(point[0], point[1] + size);
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.stroke();
    };

    const drawText = (context, text, point, color) => {
      context.font = 'bold 13px Arial';
      context.lineWidth = 4;
      context.strokeStyle = '#000';
      context.strokeText(text, point[0], point[1]);
      context.fillStyle = color;
      context.fillText(text, point[0], point[1]);
    };

    const drawDetection = (context, detection) => {
      drawPolyline(context, detection.box_px, '#55ff88', 2.5, true);
      (detection.box_px || []).forEach((point, index) => {
        context.beginPath();
        context.arc(point[0], point[1], 3, 0, Math.PI * 2);
        context.fillStyle = '#55ff88';
        context.fill();
        drawText(context, String(index + 1), [point[0] + 5, point[1] - 5], '#55ff88');
      });

      drawPolyline(context, detection.block_axis_px, '#ffff00', 2);
      drawCross(context, detection.center_px, '#00ffff');
      drawText(context, 'C', [detection.center_px[0] + 9, detection.center_px[1] - 9], '#00ffff');
      drawText(
        context,
        `${Number(detection.block_angle_deg).toFixed(0)}°`,
        [detection.center_px[0] + 10, detection.center_px[1] + 19],
        '#ffff00'
      );

      drawCross(context, detection.biased_center_px, '#ffa500');
      drawText(
        context,
        'B',
        [detection.biased_center_px[0] + 8, detection.biased_center_px[1] + 17],
        '#ffa500'
      );
      (detection.candidates_px || []).slice(1).forEach((candidate) => {
        drawCross(context, candidate.xy, '#ff00ff', 5);
        drawText(
          context,
          shortLabel[candidate.label] || candidate.label,
          [candidate.xy[0] + 7, candidate.xy[1] - 7],
          '#ff00ff'
        );
      });
    };

    const drawReject = (context, reject) => {
      context.save();
      context.setLineDash([6, 4]);
      drawPolyline(context, reject.box_px, '#9aa0a6', 1.5, true);
      context.restore();
      drawText(
        context,
        rejectText(reject),
        [reject.center_px[0] + 8, reject.center_px[1] + 4],
        '#9aa0a6'
      );
    };

    const renderDetails = (state) => {
      const panel = state.layout.querySelector('.details');
      if (!panel) return;
      if (selectedColor() === 'none') {
        panel.textContent = 'Overlay hidden.';
        return;
      }
      const detections = visibleDetections(state);
      panel.replaceChildren();
      if (state.camera === 'shoulder' && gripperReference?.available) {
        const block = document.createElement('div');
        block.className = 'detail-block';
        const title = document.createElement('div');
        title.className = 'detail-name'; title.textContent = 'Gripper · FK XY';
        block.append(title);
        addDetailRow(block, 'XYZ', pointText(gripperReference.xyz_mm), '');
        addDetailRow(block, '표시', '블록 평면 투영 · 영상 검출 아님', '');
        addDetailRow(block, '측정 시각', new Date(gripperReference.measured_at * 1000).toLocaleTimeString(), '');
        panel.append(block);
      }
      detections.forEach((detection) => {
        const block = document.createElement('div');
        block.className = 'detail-block';
        const name = document.createElement('div');
        name.className = 'detail-name';
        name.textContent = detection.color;
        block.append(name);
        addDetailRow(block, 'C', pointText(detection.center_mm), 'c');
        addDetailRow(block, 'angle', `${Number(detection.block_angle_deg).toFixed(1)}°`, 'angle');
        addDetailRow(block, 'B nominal', pointText(detection.biased_center_mm), 'b');
        detection.candidates_mm.slice(1).forEach((candidate) => {
          addDetailRow(
            block,
            shortLabel[candidate.label] || candidate.label,
            pointText(candidate.xy),
            'retry'
          );
        });
        panel.append(block);
      });
      const rejects = visibleRejects(state);
      if (!detections.length && !rejects.length) {
        panel.textContent = 'No matching block.';
        return;
      }
      rejects.forEach((reject) => {
        const block = document.createElement('div');
        block.className = 'detail-block';
        const name = document.createElement('div');
        name.className = 'detail-name';
        name.textContent = `${reject.color} (rejected)`;
        block.append(name);
        addDetailRow(block, 'gate', reject.reason, 'reject');
        addDetailRow(block, 'C', pointText(reject.center_mm), 'reject');
        addDetailRow(block, 'area', `${reject.area_mm2} mm²`, 'reject');
        addDetailRow(
          block,
          'shape',
          `asp ${reject.aspect} · fill ${reject.fill} · sol ${reject.solidity}`,
          'reject'
        );
        panel.append(block);
      });
    };

    const connect = (state) => {
      if (state.source || selectedColor() === 'none') return;
      const source = new EventSource(`/events/${encodeURIComponent(state.camera)}`);
      state.source = source;
      source.onmessage = (event) => {
        state.latest = JSON.parse(event.data);
        const age = Math.max(0, Date.now() / 1000 - Number(state.latest.captured_at || 0));
        state.status.textContent = state.latest.error
          ? `vision error: ${state.latest.error}`
          : `vision ${Number(state.latest.analysis_ms || 0).toFixed(0)}ms · age ${age.toFixed(1)}s`;
        renderDetails(state);
      };
      source.onerror = () => { state.status.textContent = 'vision reconnecting…'; };
    };

    const disconnect = (state) => {
      state.source?.close();
      state.source = null;
      state.latest = null;
      state.status.textContent = 'overlay off';
      state.context.clearRect(0, 0, state.canvas.width, state.canvas.height);
      renderDetails(state);
    };

    const initialize = async (canvas) => {
      const camera = canvas.dataset.camera;
      const layout = canvas.closest('.camera-layout');
      const response = await fetch(`/overlay-config/${encodeURIComponent(camera)}.json`);
      if (!response.ok) throw new Error(`overlay config unavailable for ${camera}`);
      const config = await response.json();
      canvas.width = Number(config.image_size[0]);
      canvas.height = Number(config.image_size[1]);
      const state = {
        camera, canvas, config, layout,
        context: canvas.getContext('2d'),
        status: layout.querySelector('.overlay-status'),
        source: null,
        latest: null
      };
      overlays.set(camera, state);
      connect(state);
      canvas.addEventListener('click', async (event) => {
        if (!crossMode() || camera !== 'shoulder') return;
        const bounds = canvas.getBoundingClientRect();
        const point = [
          (event.clientX - bounds.left) * canvas.width / bounds.width,
          (event.clientY - bounds.top) * canvas.height / bounds.height
        ];
        if (!state.crossPoint) {
          state.crossPoint = point;
          state.status.textContent = 'click the table reference point';
          return;
        }
        const response = await fetch('/tools/cross-calibration/points', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({marker_px: state.crossPoint, reference_px: point})
        });
        const result = await response.json();
        state.crossPoint = null;
        state.status.textContent = result.ready
          ? `cross calibration: ${result.samples.length} points, RMS ${Number(result.rms_mm).toFixed(1)}mm`
          : `cross calibration: ${result.samples.length}/4 points`;
      });
    };

    fetchGripperReference();

    const paint = () => {
      overlays.forEach((state) => {
        const context = state.context;
        context.clearRect(0, 0, state.canvas.width, state.canvas.height);
        if (crossMode() && state.camera === 'shoulder') {
          state.canvas.style.pointerEvents = 'auto';
          if (state.crossPoint) {
            drawCross(context, state.crossPoint, '#ff9800', 10);
            drawText(context, 'cross', [state.crossPoint[0] + 12, state.crossPoint[1] - 12], '#ff9800');
          }
        } else {
          state.canvas.style.pointerEvents = 'none';
        }
        if (selectedColor() === 'none') return;
        const boundary = state.config.workspace_boundary;
        if (boundary) {
          const arc = boundary.points_px;
          drawPolyline(context, arc, '#ff9800', 2.5);
          const middle = arc[Math.floor(arc.length / 2)];
          const boundaryLabel = boundary.label || `${Number(boundary.radius_mm).toFixed(0)} mm`;
          drawText(context, boundaryLabel, [middle[0] + 8, middle[1] - 8], '#ff9800');
          if (boundary.base_px && arc.length >= 2) {
            const first = arc[0];
            const last = arc[arc.length - 1];
            drawPolyline(context, [boundary.base_px, first], '#ff9800', 2.5);
            drawPolyline(context, [boundary.base_px, last], '#ff9800', 2.5);
            drawText(context, `${Number(boundary.angle_min_deg).toFixed(0)}°`, [first[0] + 6, first[1] - 6], '#ff9800');
            drawText(context, `${Number(boundary.angle_max_deg).toFixed(0)}°`, [last[0] + 6, last[1] - 6], '#ff9800');
          }
        }
        const targetZone = state.config.target_zone;
        if (targetZone?.points_px?.length) {
          drawPolyline(
            context,
            [...targetZone.points_px, targetZone.points_px[0]],
            '#e040fb',
            3
          );
          drawText(context, 'excluded zone', targetZone.points_px[0], '#e040fb');
        }
        visibleRejects(state).forEach((reject) => drawReject(context, reject));
        visibleDetections(state).forEach((detection) => drawDetection(context, detection));
        const ref = gripperReference;
        if (state.camera === 'shoulder' && ref?.available && ref.pixel?.every(Number.isFinite)) {
          const age = Math.max(0, Date.now()/1000 - ref.measured_at);
          const point = ref.pixel;
          context.save();
          context.beginPath(); context.arc(point[0], point[1], 13, 0, Math.PI*2);
          context.strokeStyle = '#111'; context.lineWidth = 6; context.stroke();
          context.strokeStyle = '#ffffff'; context.lineWidth = 2; context.stroke();
          drawCross(context, point, '#ffffff', 19);
          drawText(context, `FK XY · ${age.toFixed(0)}s ago`, [point[0]+22, point[1]-18], '#ffffff');
          context.restore();
        }
      });
      requestAnimationFrame(paint);
    };

    picker?.addEventListener('change', () => {
      overlays.forEach((state) => {
        if (selectedColor() === 'none') disconnect(state);
        else {
          connect(state);
          renderDetails(state);
        }
      });
    });
    toolMenu?.addEventListener('change', () => {
      crossReset.hidden = !crossMode();
      overlays.forEach((state) => {
        state.crossPoint = null;
        state.status.textContent = crossMode()
          ? 'click cross centre, then table reference point'
          : '';
      });
    });
    crossReset?.addEventListener('click', async () => {
      const response = await fetch('/tools/cross-calibration/reset', {method: 'POST'});
      if (!response.ok) {
        overlays.forEach((state) => { state.status.textContent = 'cross calibration reset failed'; });
        return;
      }
      overlays.forEach((state) => {
        state.crossPoint = null;
        state.status.textContent = 'cross calibration reset';
      });
    });

    Promise.all(
      Array.from(document.querySelectorAll('canvas.camera-overlay')).map(initialize)
    ).catch((error) => {
      document.querySelectorAll('.overlay-status').forEach((node) => {
        node.textContent = error.message;
      });
    });
    requestAnimationFrame(paint);
  </script>
</body>
</html>
"""
