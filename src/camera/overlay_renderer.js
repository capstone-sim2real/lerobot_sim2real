/* Geometry-only renderer shared by the camera and agent pages. */
(() => {
  "use strict";
  const point = p => Array.isArray(p) && p.length >= 2 && p.every(Number.isFinite);
  function drawPolyline(ctx, points, color, width, close = false) {
    if (!Array.isArray(points) || points.length < 2 || !points.every(point)) return;
    ctx.beginPath(); ctx.moveTo(...points[0]);
    points.slice(1).forEach(p => ctx.lineTo(...p));
    if (close) ctx.closePath();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
  }
  function drawCross(ctx, p, color, size = 6) {
    if (!point(p)) return;
    ctx.beginPath();
    ctx.moveTo(p[0] - size, p[1]); ctx.lineTo(p[0] + size, p[1]);
    ctx.moveTo(p[0], p[1] - size); ctx.lineTo(p[0], p[1] + size);
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
  }
  function drawText(ctx, text, p, color) {
    if (!point(p)) return;
    ctx.font = "bold 14px sans-serif"; ctx.lineWidth = 4;
    ctx.strokeStyle = "#111"; ctx.strokeText(text, ...p);
    ctx.fillStyle = color; ctx.fillText(text, ...p);
  }
  const offset = p => point(p) ? [p[0] + 10, p[1] - 10] : null;
  const colors = {green: "초록", yellow: "노랑", blue: "파랑", red: "빨강", wood: "나무"};
  const reasons = {area: "면적", aspect: "종횡비", fill: "채움", solidity: "밀집도"};
  function drawPerception(ctx, packet, ref, flags, color) {
    for (const d of packet?.detections || []) {
      if (color && d.color !== color) continue;
      if (flags.boxes) {
        const boxColor = d.in_zone ? "#ff9f43" : "#55ff88";
        const label = (colors[d.color] || d.color) + (d.in_zone ? " · 영역 안" : "");
        drawPolyline(ctx, d.box_px, boxColor, 2.5, true);
        drawText(ctx, label, offset(d.center_px), boxColor);
      }
      if (flags.centers) drawCross(ctx, d.center_px, "#00ffff");
      if (flags.axes) drawPolyline(ctx, d.grasp_axis_px || d.block_axis_px, "#ffff00", 3);
      if (flags.candidates) {
        drawCross(ctx, d.biased_center_px, "#ffa500", 9);
        drawText(ctx, "보정", offset(d.biased_center_px), "#ffa500");
        for (const c of (d.candidates_px || []).slice(1)) drawCross(ctx, c.xy, "#ff70ff", 5);
      }
    }
    if (flags.rejects) for (const d of packet?.rejects || []) {
      if (color && d.color !== color) continue;
      ctx.save(); ctx.setLineDash([6, 4]);
      drawPolyline(ctx, d.box_px, "#bbb", 2, true); ctx.restore();
      drawText(ctx, "제외: " + (reasons[d.reason] || d.reason), offset(d.center_px), "#ddd");
    }
    if (flags.fk && ref?.available) {
      drawCross(ctx, ref.pixel, "#fff", 18);
      drawText(ctx, "FK 투영", offset(ref.pixel), "#fff");
    }
  }
  window.CameraOverlayDrawing = {drawPolyline, drawCross, drawText, drawPerception, colors, reasons};
})();
