"""Picking and arm moves: pick_block, relative/pixel/cell moves, gripper, home."""

from __future__ import annotations

import logging
import time
from dataclasses import replace

from control.grasp import plan_grasp_attempts, run_grasp_attempts
from control.sensing import check_grasp
from control.task1_transport import over_ik_gate, place_tilt_deg
from fsm.states import RunContext, StateName
from fsm.task1 import corrected_pick_xy, far_reach_tilt_deg
from perception.select import SelectionResult
from session.arm_session import PICK_TILT_KEY, HeldBlock, LastPick
from session.relative import clamp_vector, offset_xy, vector_norm
from session.results import SkillResult
from session.skills.base import XY, _xy

logger = logging.getLogger(__name__)


class MotionMixin:
    def pick_block(
        self,
        color: str,
        forward_mm: float = 0.0,
        left_mm: float = 0.0,
        relative_to_last: bool = False,
    ) -> SkillResult:
        action, t0, s, cfg = "pick_block", time.monotonic(), self.s, self.cfg
        if color not in self.colors:
            return self._result(False, action, "invalid_arguments", f"모르는 색입니다: {color}", t0=t0)

        if s.held is not None:
            if s.held.color == color and (relative_to_last or forward_mm or left_mm):
                self._put_back()
            else:
                return self._result(
                    False, action, "already_holding",
                    f"이미 {self._label(s.held.color)} 블록을 들고 있습니다. 먼저 내려놓아야 합니다.",
                    retry_advice="do_not_retry", t0=t0,
                )

        base_forward = base_left = 0.0
        if relative_to_last:
            if s.last_pick is None or s.last_pick.color != color:
                return self._result(
                    False, action, "precondition",
                    f"직전에 {color} 블록을 집은 기록이 없어 '더' 보정을 할 수 없습니다.",
                    retry_advice="do_not_retry", t0=t0,
                )
            base_forward, base_left = s.last_pick.forward_mm, s.last_pick.left_mm
        (offset_forward, offset_left), clamped = clamp_vector(
            (base_forward + forward_mm, base_left + left_mm), cfg.agent.relative.max_pick_offset_mm
        )

        rounds = max(1, cfg.fsm.max_retries_per_block)
        attempts = 0
        last_note = ""
        for round_index in range(rounds):
            scene = self._observe_or_fail(action, t0)
            if isinstance(scene, SkillResult):
                return scene
            target = scene.find(color)
            if target is None:
                if round_index == 0:
                    return self._result(
                        False, action, "not_detected",
                        f"카메라에서 {color} 블록을 찾지 못했습니다.",
                        retry_advice="ask_operator", t0=t0,
                    )
                last_note = "not_detected"
                break

            pick_xy = corrected_pick_xy(target.center_mm, s.base_xy, cfg)
            pick_xy = offset_xy(
                pick_xy, offset_forward, offset_left,
                frame=cfg.agent.relative.frame, base_xy_mm=s.base_xy,
            )
            ctx = RunContext(fsm=cfg.fsm)
            ctx.target_id = color
            ctx.extras["selection"] = SelectionResult(
                replace(target.detection, center_mm=pick_xy), color, 1, [target.detection]
            )
            ctx.extras[PICK_TILT_KEY] = far_reach_tilt_deg(pick_xy, s.base_xy, cfg)

            attempts += 1
            next_state = s.pick_state.step(ctx)
            last_note = ctx.last_note
            if next_state is StateName.VERIFY:
                check = check_grasp(s.robot, cfg.sensing)
                if check.grasped:
                    held = ctx.extras["ik_pick_attempt"]
                    s.held = HeldBlock(color, held, target.center_mm, target.in_zone, held.xy_mm)
                    s.last_pick = LastPick(color, offset_forward, offset_left)
                    s.last_block_color = color
                    return self._result(
                        True, action, "held",
                        f"{color} 블록을 집었습니다" + (
                            f" (파지 보정 앞 {offset_forward:+.0f}mm, 왼쪽 {offset_left:+.0f}mm)."
                            if offset_forward or offset_left else "."
                        ),
                        t0=t0, rounds_used=round_index + 1, grasp_label=held.label,
                        from_zone=target.in_zone,
                        detected=self._block_dict(target),
                        pick_offset_mm={"forward": offset_forward, "left": offset_left},
                        offset_clamped=clamped or None,
                    )
                s.motion.open_gripper()
                last_note = "verify_empty"
            elif "unreachable" in last_note:
                return self._result(
                    False, action, "unreachable",
                    f"{color} 블록이 팔이 닿을 수 있는 범위 밖입니다. 블록을 로봇 쪽으로 옮겨 주세요.",
                    retry_advice="do_not_retry", t0=t0, internal_retries_exhausted=True,
                    detected=self._block_dict(target),
                )
            elif "motion_timeout" in last_note:
                return self._result(
                    False, action, "motion_timeout",
                    "팔이 명령한 자세를 따라가지 못했습니다. 기계적 문제일 수 있어 멈춥니다.",
                    retry_advice="ask_operator", t0=t0,
                )

        s.go_home()
        if last_note == "not_detected":
            return self._result(
                False, action, "not_detected",
                f"재접근 도중 {color} 블록이 보이지 않게 되었습니다.",
                retry_advice="ask_operator", t0=t0, attempts_used=attempts,
            )
        return self._result(
            False, action, "grasp_empty",
            f"{color} 블록을 {attempts}회 접근했지만 집지 못했습니다. 매 접근마다 그리퍼를 90도 "
            f"돌려 재시도했고 home에서 다시 접근했습니다. 블록이 넘어졌거나 다른 블록에 붙어 있을 수 있습니다.",
            retry_advice="ask_operator", t0=t0, attempts_used=attempts,
            internal_retries_exhausted=True,
        )

    # ── place ────────────────────────────────────────────────────────

    def _no_block(self, action: str, t0: float) -> SkillResult:
        return self._result(False, action, "no_block_held", "들고 있는 블록이 없습니다.",
                            retry_advice="do_not_retry", t0=t0)

    # ── arm ──────────────────────────────────────────────────────────

    def move_arm(self, forward_mm: float = 0.0, left_mm: float = 0.0, up_mm: float = 0.0, *, _playback=None) -> SkillResult:
        action, t0, s, cfg = "move_arm", time.monotonic(), self.s, self.cfg
        rel = cfg.agent.relative
        norm = vector_norm(forward_mm, left_mm, up_mm)
        if norm == 0.0:
            return self._result(False, action, "invalid_arguments", "이동 거리가 0입니다.", t0=t0)
        if norm > rel.max_jog_mm:
            return self._result(
                False, action, "limit_exceeded",
                f"한 번에 {rel.max_jog_mm:.0f}mm까지만 움직일 수 있습니다 (요청 {norm:.0f}mm). 나눠서 요청하세요.",
                retry_advice="do_not_retry", t0=t0,
            )
        joints = s.robot.read_joints()
        x, y, z = s.ik.forward_position_mm(joints)
        tx, ty = offset_xy((x, y), forward_mm, left_mm, frame=rel.frame, base_xy_mm=s.base_xy)
        return self._fly_to_xy(action, t0, joints, (x, y, z), (tx, ty), z + up_mm, _playback=_playback)

    def _fly_to_xy(self, action: str, t0: float, joints, from_xyz, target_xy: XY,
                   tz: float, *, _playback=None) -> SkillResult:
        """Take the gripper to one xy at height ``tz``, gated like a jog.

        Shared by ``move_arm`` (a bounded relative vector) and
        ``move_to_cell`` (a board address). The height window, workspace
        gate and IK-error gate are the same either way -- only how the
        target was named differs, and only ``move_arm`` caps the distance.
        """
        s, cfg = self.s, self.cfg
        rel = cfg.agent.relative
        x, y, z = from_xyz
        tx, ty = target_xy
        entered = False
        # A pose reached by an earlier jog may settle a few mm past the exact
        # window edge (the same IK/arrival tolerance that gates every move
        # here) -- both the "do we need to re-enter" check and the final
        # acceptance gate below must tolerate that, or a lateral move
        # (up_mm=0, so tz == z) that lands just outside the strict window
        # would be judged already-inside by the first check yet rejected by
        # a second, stricter one, permanently refusing every jog from there.
        slack = rel.jog_max_ik_error_mm
        lo, hi = rel.jog_min_z_mm - slack, rel.jog_max_z_mm + slack
        if not lo <= z <= hi:
            # e.g. from home, whose tool frame sits near table height
            tz = min(max(tz, rel.jog_min_z_mm), rel.jog_max_z_mm)
            entered = True
        elif not lo <= tz <= hi:
            return self._result(
                False, action, "height_limit",
                f"높이는 {rel.jog_min_z_mm:.0f}~{rel.jog_max_z_mm:.0f}mm 범위에서만 움직일 수 있습니다.",
                retry_advice="do_not_retry", t0=t0, from_mm={"x": x, "y": y, "z": z},
            )
        if not s.in_workspace((tx, ty)):
            return self._verdict_failure(action, "out_of_workspace", t0, target=_xy((tx, ty)))
        # keep the jaws (and a held block) turned as they are now; tip outward
        # at far reach exactly as placements do
        tilt = place_tilt_deg((tx, ty), s.base_xy, cfg)
        result = s.ik.solve_holding_wrist_roll(tx, ty, tz, joints["wrist_roll"], radial_tilt_deg=tilt)
        if over_ik_gate(result, cfg) or result.position_error_mm > rel.jog_max_ik_error_mm:
            return self._result(
                False, action, "ik_gate",
                f"그 위치로는 팔을 정확히 보낼 수 없습니다 (IK 오차 {result.position_error_mm:.0f}mm). "
                "로봇에 더 가깝거나 더 높은 곳으로 요청하세요.",
                retry_advice="do_not_retry", t0=t0, target={"x": tx, "y": ty, "z": tz},
                ik_error_mm=result.position_error_mm,
            )
        if _playback is None:
            s.player.move_to(result.joints, max_step=1.0, tol=cfg.motion.transit_arrival_tol)
        else:
            _playback(result.joints)
        if s.held is not None:
            s.held.over_xy_mm = (tx, ty)
        rx, ry, rz = s.arm_position_mm()
        detail = "팔을 움직였습니다."
        if entered:
            detail = f"먼저 작업 높이({tz:.0f}mm)로 올린 뒤 움직였습니다."
        return self._result(True, action, "moved", detail, t0=t0,
                            from_mm={"x": x, "y": y, "z": z}, target={"x": tx, "y": ty, "z": tz},
                            reached={"x": rx, "y": ry, "z": rz}, entered_jog_height=entered or None)

    def _pixel_point(self, action, t0, u, v, calibration_id):
        from session.pixel_target import resolve_pixel, calibration_id as fingerprint
        from perception.homography import PlaneCalibration
        try:
            live = PlaneCalibration.load(self.cfg.perception.calibration_path)
            if fingerprint(live) != fingerprint(self.s.calib):
                raise ValueError("보정 파일이 변경되었습니다. 제어 서버를 다시 시작하세요.")
            target = resolve_pixel(self.cfg, self.s.calib, u, v, calibration_id)
            return target, None
        except (ValueError, OSError) as exc:
            return None, self._result(False, action, "invalid_arguments", str(exc), t0=t0,
                                      retry_advice="do_not_retry")

    def move_to_pixel(self, u: int, v: int, calibration_id: str) -> SkillResult:
        """Lift, traverse, then descend to the calibrated one-block top plane."""
        action, t0, s, cfg = "move_to_pixel", time.monotonic(), self.s, self.cfg
        target, failure = self._pixel_point(action, t0, u, v, calibration_id)
        if failure is not None:
            return failure
        if s.held is not None:
            return self._result(False, action, "already_holding",
                "블록을 들고 있습니다. 선택 위치에 놓기를 사용하세요.", t0=t0)
        joints = s.robot.read_joints()
        x,y,z = s.ik.forward_position_mm(joints)
        tx,ty,tz = target['x_mm'],target['y_mm'],target['z_mm']
        height = max(z, cfg.agent.relative.jog_min_z_mm)
        if height > cfg.agent.relative.jog_max_z_mm:
            return self._result(False, action, "height_limit", "현재 높이가 작업 범위를 벗어났습니다.", t0=t0)
        plans=[]
        # Plan all three poses before any command is sent. No global jog limits change.
        for px,py,pz in ((x,y,height),(tx,ty,height),(tx,ty,tz)):
            result=s.ik.solve_holding_wrist_roll(px,py,pz,joints['wrist_roll'],
                radial_tilt_deg=place_tilt_deg((px,py),s.base_xy,cfg))
            if over_ik_gate(result,cfg) or result.position_error_mm > cfg.agent.relative.jog_max_ik_error_mm:
                return self._result(False,action,"ik_gate","이 위치의 접근 경로가 IK 검사를 통과하지 못했습니다.",t0=t0)
            plans.append(result)
        for plan in plans:
            s.player.move_to(plan.joints,max_step=1.0,tol=cfg.motion.transit_arrival_tol)
        return self._result(True,action,"moved","선택한 픽셀의 블록 윗면 높이로 이동했습니다.",
                            t0=t0,target=target,reached=list(s.arm_position_mm()))

    def move_to_cell(self, x: int, y: int) -> SkillResult:
        """Fly the gripper over one chessboard cell, at the current height.

        Unlike ``move_arm`` this is an address, not a nudge, so the jog
        distance cap does not apply -- the far edge of the board is 300mm
        from the near edge and must be one move. Every other gate (height
        window, workspace sector, IK error) is the jog's.
        """
        action, t0, s = "move_to_cell", time.monotonic(), self.s
        point, failure = self._cell_point(action, t0, x, y)
        if failure is not None:
            return failure
        joints = s.robot.read_joints()
        from_xyz = s.ik.forward_position_mm(joints)
        result = self._fly_to_xy(action, t0, joints, from_xyz, point, from_xyz[2])
        if result.ok:
            result.data["cell"] = {"x": int(x), "y": int(y)}
            result.detail = f"({x}, {y}) 칸 위로 이동했습니다."
        return result

    def rotate_gripper(self, delta_deg: float) -> SkillResult:
        """Spin the jaws about their own axis without moving x/y/z.

        wrist_roll is the last joint before the gripper, so commanding it
        alone (``interpolate`` only touches joints present in the goal) turns
        the jaws in place -- no IK solve needed. No absolute range is
        enforced: lerobot's own max_relative_target clamp and the per-tick
        step limit already bound real motion, and a target past the physical
        stop simply times out (reported as motion_timeout), so a per-call
        delta cap is the only refusal needed here.
        """
        action, t0, s, cfg = "rotate_gripper", time.monotonic(), self.s, self.cfg
        if delta_deg == 0.0:
            return self._result(False, action, "invalid_arguments", "회전 각도가 0입니다.", t0=t0)
        limit = cfg.agent.relative.max_gripper_roll_deg
        if abs(delta_deg) > limit:
            return self._result(
                False, action, "limit_exceeded",
                f"한 번에 {limit:.0f}도까지만 돌릴 수 있습니다 (요청 {delta_deg:+.0f}도). 나눠서 요청하세요.",
                retry_advice="do_not_retry", t0=t0,
            )
        current = s.robot.read_joints()["wrist_roll"]
        target = current + delta_deg
        s.player.move_to({"wrist_roll": target}, max_step=1.0, tol=cfg.motion.transit_arrival_tol)
        reached = s.robot.read_joints()["wrist_roll"]
        return self._result(True, action, "moved", f"그리퍼를 {delta_deg:+.0f}도 돌렸습니다.", t0=t0,
                            from_deg=current, target_deg=target, reached_deg=reached)

    def pick_here(self) -> SkillResult:
        """Manual 'claw machine' grab: attempt a grasp at wherever the arm
        is right now, with no colour/target lookup -- a photo taken with the
        arm already positioned there would just show its own gripper. Uses
        the same grasp planning/execution (centre attempt, the 90-degree
        gripper-roll retry, check_grasp verification) pick_block drives.
        """
        action, t0, s, cfg = "pick_here", time.monotonic(), self.s, self.cfg
        if s.held is not None:
            return self._result(
                False, action, "already_holding",
                f"이미 {self._label(s.held.color)}을(를) 들고 있습니다. 먼저 내려놓으세요.",
                retry_advice="do_not_retry", t0=t0,
            )
        x, y, _z = s.arm_position_mm()
        if not s.in_workspace((x, y)):
            return self._verdict_failure(action, "out_of_workspace", t0, target=_xy((x, y)))
        plan = plan_grasp_attempts(s.ik, cfg, x, y, s.grasp_z_mm, log=logger.info)
        if not any(a.reachable for a in plan.attempts):
            return self._result(
                False, action, "unreachable",
                "이 위치에서는 아래로 내려가 집을 수 없습니다. 조금 더 로봇 쪽으로 옮겨서 다시 시도하세요.",
                retry_advice="do_not_retry", t0=t0, internal_retries_exhausted=True,
            )
        held = run_grasp_attempts(s.player, s.robot, cfg, plan, log=logger.info)
        if held is None:
            return self._result(
                False, action, "grasp_empty",
                "그리퍼를 닫아봤지만 아무것도 집지 못했습니다. 그리퍼를 90도 돌려 한 번 더 시도했습니다.",
                retry_advice="retry_ok", t0=t0,
            )
        s.held = HeldBlock(color=None, attempt=held, picked_xy_mm=(x, y),
                           from_zone=s.in_zone((x, y)), over_xy_mm=held.xy_mm)
        s.last_block_color = None
        return self._result(True, action, "held", "무언가를 집었습니다.", t0=t0, grasp_label=held.label)

    def return_to_home(self, *, post_release: bool = False) -> SkillResult:
        t0 = time.monotonic()
        lifted, at_home = (self.s.return_home_safely(post_release=True) if post_release
                           else self.s.return_home_safely())
        return self._result(at_home, "return_to_home", "ok" if at_home else "motion_timeout",
                            "home으로 복귀했습니다." if at_home else "home 자세에 도달하지 못했습니다.",
                            t0=t0, lifted_first=lifted or None, arm_at_home=at_home)

    def recover_and_home(self) -> SkillResult:
        """STOP/fault recovery: open the jaws, then return home with them open.

        The service clears cancellation before queuing this skill. A new STOP
        during recovery must remain effective at the next robot bus write.
        """
        t0, s = time.monotonic(), self.s
        released = s.held.color if s.held else None
        open_error = None
        try:
            s.motion.open_gripper()
        except Exception as exc:  # noqa: BLE001 - still try to get home
            open_error = str(exc)
            logger.warning("recover_and_home: open_gripper failed: %s", exc)
        else:
            s.held = None
            if released is not None:
                s.last_block_color = released
        # Explicit STOP recovery uses the measured-joint trajectory directly.
        # The normal low-hover preflight may be unsatisfiable after an
        # interrupted move; the operator requested home even from that pose.
        s.motion.go_home(include_gripper=False)
        lifted, at_home = False, s.arm_at_home()
        ok = at_home and open_error is None
        detail = (
            "그리퍼를 열고 home으로 복귀했습니다." if ok else
            "그리퍼 열기 또는 home 복귀를 완료하지 못했습니다."
        )
        return self._result(
            ok, "recover_and_home", "ok" if ok else "motion_timeout", detail,
            t0=t0, released=released if open_error is None else None,
            lifted_first=lifted or None, arm_at_home=at_home, gripper_open=open_error is None,
            open_error=open_error,
        )

    def open_gripper(self) -> SkillResult:
        t0, s = time.monotonic(), self.s
        released = s.held.color if s.held else None
        s.motion.open_gripper()
        if s.held is not None:
            s.last_block_color = s.held.color
            s.held = None
        return self._result(True, "open_gripper", "released" if released else "ok",
                            f"그리퍼를 열었습니다{f' ({released} 블록을 놓음)' if released else ''}.",
                            t0=t0, released=released)
