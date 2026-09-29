"""State, observation and target lookup; never moves the arm."""
from __future__ import annotations

import math
import time
from dataclasses import replace

from session.arm_session import CameraError
from session.results import ObservationImage


class ObserveMixin:
    @property
    def limits(self):
        return self.cfg.agent.primitives

    def state_dict(self, *, read_robot=True):
        state = super().state_dict(read_robot=read_robot)
        state.update(observation_id=self.observation_id,
                     joints=self.s.robot.read_joints() if read_robot else None,
                     arm_position_source="joint_feedback_and_URDF_FK_not_visual_TCP",
                     contact_confirmed=self._contact,
                     grasp_failed=self._grasp_failed, collection=self.collection.status())
        return state

    def _fail(self, action, detail, reason="precondition"):
        return self._result(False, action, reason, detail, retry_advice="retry_ok")

    def select_pixel_target(self, u, v, calibration_id, angle_deg=None):
        """An explicit operator address, not a claim of a CV detection."""
        action, t0 = "select_pixel_target", time.monotonic()
        target, failure = self._pixel_point(action, t0, u, v, calibration_id)
        if failure is not None:
            return failure
        if angle_deg is not None and (not math.isfinite(angle_deg) or not -180 <= angle_deg <= 180):
            return self._fail(action, "angle_deg must be finite and in [-180, 180]", "invalid_arguments")
        observed = self.observe_scene()
        if not observed.ok:
            return observed
        from perception.detector import BlockDetection
        from perception.scene import build_scene
        xy = (target["x_mm"], target["y_mm"])
        side = self.cfg.agent.calibration_clearance.block_side_mm
        # Associate only a footprint containing the selected point, never a
        # distant nearest colour. Retain the exact requested XY either way.
        matches = []
        for block in self._observed_scene.all():
            a = math.radians(block.angle_deg)
            dx, dy = xy[0] - block.center_mm[0], xy[1] - block.center_mm[1]
            if max(abs(dx*math.cos(a)+dy*math.sin(a)),
                   abs(-dx*math.sin(a)+dy*math.cos(a))) <= side / 2:
                matches.append(block)
        matched = min(matches, key=lambda b: math.dist(b.center_mm, xy)) if matches else None
        color = matched.color if matched is not None else "selected"
        angle = angle_deg if angle_deg is not None else (matched.angle_deg if matched else 0.0)
        a = math.radians(angle)
        box = [(xy[0] + side/2*(i*math.cos(a)-j*math.sin(a)),
                xy[1] + side/2*(i*math.sin(a)+j*math.cos(a)))
               for i, j in ((-1,-1),(1,-1),(1,1),(-1,1))]
        detection = BlockDetection(color, xy, side*side, 1.0, 1.0, 1.0, box, angle)
        detections = [b.detection for b in self._observed_scene.all() if b.color != color]
        detections.append(detection)
        scene = build_scene(detections, detections, self.s.calib, self.s.slot_centres,
                            snap_radius_mm=self.cfg.agent.slot_snap_radius_mm,
                            frame_seq=self._observed_scene.frame_seq,
                            captured_at=self._observed_scene.captured_at)
        self._observed_scene = scene
        self._objects = {f"{b.color}_1": b for b in scene.all()}
        # Stable alias lets every object-addressed primitive use the same target.
        self._objects["selected_1"] = scene.find(color)
        return self._result(True, action, "ok", t0=t0,
                            object_id="selected_1", observation_id=self.observation_id,
                            target=target, matched_color=matched.color if matched else None,
                            target_source="user_selected_pixel", angle_deg=angle,
                            geometry_assumed=matched is None,
                            assumed_block_side_mm=side if matched is None else None)

    def _run_with_source(self, source, action, fn):
        """Bind a composite to an explicit address, preserving its retry path."""
        if not isinstance(source, dict):
            return self._fail(action, "source must be a pixel or observed object address", "invalid_arguments")
        try:
            if set(source) == {"object_id", "observation_id"}:
                block = self._object(**source)
                from session.pixel_target import calibration_id
                u, v = self.s.calib.board_to_pixel([block.center_mm])[0]
                pixel = dict(u=int(round(u)), v=int(round(v)),
                             calibration_id=calibration_id(self.s.calib), angle_deg=block.angle_deg)
            elif set(source) in ({"u", "v", "calibration_id"}, {"u", "v", "calibration_id", "angle_deg"}):
                pixel = dict(source)
            else:
                raise ValueError("Use object_id+observation_id or u+v+calibration_id")
        except (ValueError, TypeError) as exc:
            return self._fail(action, str(exc), "invalid_arguments")
        selected = self.select_pixel_target(**pixel)
        if not selected.ok:
            return selected
        color = self._objects["selected_1"].color
        previous = self._transfer_source
        self._transfer_source = pixel
        try:
            return fn(color)
        finally:
            self._transfer_source = previous

    def _observe_target(self, color):
        if self._transfer_source is not None:
            return self.select_pixel_target(**self._transfer_source)
        observed = self.observe_scene()
        if (observed.ok and self._observed_scene is not None
                and self._observed_scene.find(color) is None):
            return self.observe_scene(_window=True)
        return observed

    def observe_scene(self, *, _window=False):
        # Recovery and standalone tools can arrive here before their servo tail
        # settles. Record that tail before blocking on camera/CV work.
        self.collection.settle_for_sequence_pause()
        try:
            scene = (self.s.observe_window() if _window
                     else self.s.observe(after=self.s._clock()))
        except CameraError as exc:
            return self._fail("observe_scene", str(exc), "camera_stale" if exc.stale else "camera_unreachable")
        self._invalidate_pick()
        self._observed_scene = scene
        self.observation_id += 1
        self._observed_at = time.monotonic()
        self._objects = {f"{b.color}_1": b for b in [*scene.outside.values(), *scene.inside.values()]}
        learned_placement = None
        if self._pending_placement is not None and self.s.arm_at_home():
            color, target_xy, expected_slot = self._pending_placement
            landed = scene.find(color)
            cfg = self.cfg.agent.place_correction
            if (landed is not None and landed.in_zone
                    and (expected_slot is None or landed.slot_index == expected_slot)
                    and cfg.enabled and cfg.learn):
                measured_xy = landed.center_mm
                accepted = self.place_correction.observe(
                    target_xy, measured_xy,
                    base_xy_mm=self.s.base_xy,
                    frame=self.cfg.agent.relative.frame,
                )
                learned_placement = {
                    "accepted": accepted,
                    "color": color,
                    "target_xy_mm": list(target_xy),
                    "measured_xy_mm": list(measured_xy),
                    "miss_mm": round(math.dist(target_xy, measured_xy), 1),
                    "correction": self.place_correction.as_dict(),
                }
            self._pending_placement = None
        images = ()
        snapshot = self.s.last_snapshot
        if snapshot is not None:
            import cv2  # optional runtime dependency
            frame = snapshot.frame
            if frame.shape[1] > self.limits.image_max_width:
                scale = self.limits.image_max_width / frame.shape[1]
                frame = cv2.resize(frame, (self.limits.image_max_width, round(frame.shape[0] * scale)))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.limits.image_jpeg_quality])
            if not ok:
                return self._fail("observe_scene", "JPEG encoding failed", "camera_unreachable")
            images = (ObservationImage(encoded.tobytes(), "shoulder", snapshot.frame_seq, snapshot.captured_at),)
        result = self._result(True, "observe_scene", "ok", observation_id=self.observation_id,
                              frame_seq=scene.frame_seq, captured_at=scene.captured_at,
                              objects=[dict(object_id=key, **self._block_dict(b)) for key, b in self._objects.items()],
                              image_available=bool(images), arm_occlusion_possible=not self.s.arm_at_home(),
                              learned_placement=learned_placement,
                              limitations=["one detection per colour; IDs are observation-scoped",
                                            "planar CV does not measure stack height or visual TCP",
                                            "missing detection is not proof of absence"])
        return replace(result, images=images)

    def _object(self, object_id, observation_id):
        if observation_id != self.observation_id or time.monotonic() - self._observed_at > self.limits.target_max_age_s:
            raise ValueError("Target observation expired; observe_scene again")
        if object_id not in self._objects:
            raise ValueError("Object is not in this observation")
        return self._objects[object_id]

    def _held_check(self):
        if self.s.held is not None:
            gripper_pos = self.s.robot.read_joints()["gripper"]
            # VERIFY required both position and load when the block was first
            # picked. During transport the gripper load can relax with arm
            # posture even while the block remains visibly between the jaws.
            # A fully closed jaw position still detects an actual loss.
            # Only position is used here: do not block on averaged load
            # samples while the previous motor command is still settling.
            if gripper_pos <= self.cfg.sensing.gripper_empty_closed_max:
                self._grasp_failed = True
                self._contact = False
                self._stack_drop_ready = False
                self._zone_drop_ready = False
                return False
        return not self._grasp_failed
