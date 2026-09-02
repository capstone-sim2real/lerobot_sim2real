"""Browser UI for a continuous MJPEG image plus client-side canvas overlay."""

from __future__ import annotations

import html


def render_camera_page(
    cameras: list[tuple[str, str]],
    *,
    overlay_cameras: set[str],
) -> bytes:
    camera_tiles = "\n".join(
        _camera_tile(name, device, overlay=name in overlay_cameras)
        for name, device in cameras
    )
    picker = ""
    if overlay_cameras:
        picker = """
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


def _camera_tile(name: str, device: str, *, overlay: bool) -> str:
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
    .camera-layout + .camera-layout { margin-top: 24px; }
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
    const overlays = new Map();
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
    };

    const paint = () => {
      overlays.forEach((state) => {
        const context = state.context;
        context.clearRect(0, 0, state.canvas.width, state.canvas.height);
        if (selectedColor() === 'none') return;
        const boundary = state.config.workspace_boundary;
        if (boundary) {
          const arc = boundary.points_px;
          drawPolyline(context, arc, '#ff9800', 2.5);
          const middle = arc[Math.floor(arc.length / 2)];
          drawText(context, `${Number(boundary.radius_mm).toFixed(0)} mm`, [middle[0] + 8, middle[1] - 8], '#ff9800');
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
