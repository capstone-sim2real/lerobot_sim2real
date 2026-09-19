"""Experimental split grasp; never imported by mission runners.

Offsets are residual commands relative to a frozen production plan. The LLM
can name a detected colour and bounded relative offsets, never joint targets.
"""
from __future__ import annotations
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

from control.grasp import plan_grasp_attempts, attempt_grasp, GraspOutcome
from control.sensing import check_grasp, ContactMonitor
from control.trajectory import interpolate
from control.task1_transport import over_ik_gate
from fsm.task1 import corrected_pick_xy, far_reach_tilt_deg
from session.arm_session import HeldBlock
from session.relative import offset_xy
from session.results import SkillResult
from session.primitives import PrimitiveSkills


class CalibrationSkills(PrimitiveSkills):
    def __init__(self, session, output_dir):
        super().__init__(session)
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.trial = None
        self.attempt = None
        self.baseline = None
        self.descent_ready = False
        self.forward = self.left = 0.0
        (self.output / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2, default=str))

    def _record(self, phase, **data):
        joints = self.s.robot.read_joints()
        row = dict(time=time.time(), trial=self.trial, phase=phase,
                   q_measured=joints, fk_measured_mm=self.s.ik.forward_position_mm(joints), **data)
        with (self.output / "measurements.jsonl").open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return row

    def calibration_wrist_clearance_pose(self,x,y):
        """Neutral wrist, lift, then existing gated cell move; no grasp."""
        self.attempt=None
        self.descent_ready=False
        if self.s.held is not None:
            return SkillResult(False,"calibration_wrist_clearance_pose","already_holding")
        q=self.s.robot.read_joints()
        if abs(q["wrist_roll"])>self.cfg.motion.arrival_tol:
            neutral=super().rotate_gripper(-q["wrist_roll"])
            if not neutral.ok:
                return neutral
        self.s.lift_in_place()
        action="calibration_wrist_clearance_pose"
        t0=time.monotonic()
        point,failure=self._cell_point(action,t0,x,y)
        if failure is not None:
            return failure
        q=self.s.robot.read_joints()
        xyz=self.s.ik.forward_position_mm(q)
        from control.grasp import highest_reachable_hover
        yaw=self.s.ik.yaw_for_wrist_roll_deg(*point,xyz[2],q["wrist_roll"])
        height=highest_reachable_hover(self.s.ik,*point,self.s.grasp_z_mm,self.cfg,yaw_deg=yaw)
        result=self._fly_to_xy(action,t0,q,xyz,point,height)
        result.action="calibration_wrist_clearance_pose"
        self._record("wrist_clearance_pose",cell=[x,y],result=result.to_envelope())
        return result

    def calibration_probe_positive(self):
        """Advance toward URDF upper bound by at most one configured jog span."""
        import xml.etree.ElementTree as ET
        root=ET.parse(self.s.ik._project_root / self.s.ik._cfg.urdf_path).getroot()
        limit=root.find("./joint[@name='wrist_roll']/limit")
        ceiling=math.degrees(float(limit.get("upper")))
        current=self.s.robot.read_joints()["wrist_roll"]
        cfg=self.cfg.agent.calibration_clearance
        old=cfg.wrist_probe_target_deg
        cfg.wrist_probe_target_deg=min(ceiling,current+self.cfg.agent.relative.max_gripper_roll_deg)
        try:
            result=self.calibration_retry_wrist()
            result.action="calibration_probe_positive"
            result.data["model_upper_bound_deg"]=ceiling
            return result
        finally:
            cfg.wrist_probe_target_deg=old

    def calibration_retry_wrist(self):
        """User-requested fixed-angle test in free space; no block approach."""
        self.attempt=None
        self.descent_ready=False
        if self.s.held is not None:
            return SkillResult(False,"calibration_retry_wrist","already_holding")
        self.s.lift_in_place()
        robot=self.s.robot
        q=robot.read_joints()
        xyz=self.s.ik.forward_position_mm(q)
        if xyz[2]<self.cfg.agent.relative.jog_min_z_mm:
            return SkillResult(False,"calibration_retry_wrist","height_limit",data={"fk_mm":xyz})
        if self.cfg.agent.calibration_clearance.wrist_probe_close_gripper:
            self.s.player.set_gripper(self.cfg.sensing.gripper_close_pos)
        target=self.cfg.agent.calibration_clearance.wrist_probe_target_deg
        if abs(target-q["wrist_roll"])>self.cfg.agent.relative.max_gripper_roll_deg:
            return SkillResult(False,"calibration_retry_wrist","limit_exceeded")
        baseline=abs(robot.read_loads()["wrist_roll"])
        samples=[];reason=None
        started=time.monotonic()
        for step in interpolate({"wrist_roll":q["wrist_roll"]},{"wrist_roll":target},self.cfg.motion.descent_step_per_tick):
            if time.monotonic()-started>self.cfg.motion.move_timeout_s:
                reason="timeout";break
            robot.send_joints(step)
            if self.cfg.motion.fps>0:time.sleep(1/self.cfg.motion.fps)
            actual=robot.read_joints()["wrist_roll"]
            load=robot.read_loads()["wrist_roll"]
            samples.append({"time":time.time(),"command_deg":step["wrist_roll"],"actual_deg":actual,"load":load})
            if abs(load)-baseline>=self.cfg.sensing.contact_load_delta:
                reason="load_increase";break
            if abs(actual-step["wrist_roll"])>self.cfg.motion.descent_max_lag:
                reason="tracking_lag";break
        actual=robot.read_joints()["wrist_roll"]
        robot.send_joints({"wrist_roll":actual})
        self.trial=str(time.time_ns())
        self._record("wrist_probe",target_deg=target,start_deg=q["wrist_roll"],baseline_load=baseline,
                     stop_reason=reason,samples=samples)
        reached=reason is None and abs(actual-target)<=self.cfg.motion.arrival_tol
        return SkillResult(reached,"calibration_retry_wrist","ok" if reached else "grasp_blocked",
                           data={"target_deg":target,"actual_deg":actual,"stop_reason":reason,
                                 "max_abs_load":max((abs(r["load"]) for r in samples),default=baseline),
                                 "elapsed_probe_s":round(time.monotonic()-started,2)})

    def return_to_home(self):
        self.descent_ready = False
        self.attempt = None
        return super().return_to_home()

    def calibration_continuous(self, color):
        """Legacy name uses the same clearance gate; never bypass it."""
        return self.calibration_pick_guarded(color)

    def calibration_pick_guarded(self, color):
        """One continuous guarded attempt; caller records cameras automatically.

        No retry: any load/lag stop leaves jaws open for evidence review.
        This is an experimental load-increase guard, not a validated contact sensor.
        """
        result=self.calibration_prepare(color)
        if not result.ok and result.data.get("stop_reason")=="hover_not_settled":
            result=self.calibration_correct_hover(dry_run=False)
        if result.ok:
            result=self.calibration_descend_guarded()
        if result.ok:
            result=self.calibration_close_lift()
        result.data["trial"] = self.trial
        result.data["failed_phase"] = None if result.ok else result.action
        result.action = "calibration_pick_guarded"
        self._record("guarded_pick_result", color=color, result=result.to_envelope())
        return result

    def calibration_clearance_status(self):
        """Observe without moving; rank known targets and expose unseen obstacles."""
        from session.calibration_clearance import clearance_check
        scene = self.s.observe(after=time.time())
        blocks = {b.color: b.center_mm for b in scene.all()}
        cfg = self.cfg.agent.calibration_clearance
        missing = sorted(set(cfg.expected_colors) - set(blocks))
        ranked = []
        for color, xy in blocks.items():
            check = clearance_check([(*xy, self.s.grasp_z_mm)],
                                    {k:v for k,v in blocks.items() if k != color}, cfg)
            ranked.append(dict(color=color, **check))
        ranked.sort(key=lambda r: (not r["clear"], -(r["margin_mm"] or 0)))
        return SkillResult(not missing, "calibration_clearance_status",
                           "scene_incomplete" if missing else "ok",
                           data={"missing_colors":missing,"ranked":ranked})

    def _clearance_gate(self, scene, color, attempt):
        import numpy as np
        from session.calibration_jaw_geometry import JawGeometry
        from control.ik import ARM_JOINTS
        cfg = self.cfg.agent.calibration_clearance
        blocks = {b.color: b.center_mm for b in scene.all()}
        missing = sorted(set(cfg.expected_colors) - set(blocks))
        if missing:
            return {"clear":False,"missing_colors":missing,"reason":"scene_incomplete"}
        obstacles = {b.color:{"xy":b.center_mm,"box":b.detection.box_mm} for b in scene.all() if b.color != color}
        ik=self.s.ik
        geometry=JawGeometry(ik._project_root / ik._cfg.urdf_path)
        k=ik._load_kinematics()
        q=self.s.robot.read_joints()
        def pose(joints):
            return k.forward_kinematics(np.array([joints[j] for j in ARM_JOINTS])).copy()
        poses=[pose(q)]
        for goal in (attempt.hover.joints,attempt.grasp.joints):
            for step in interpolate(q,goal,self.cfg.motion.descent_step_per_tick):
                poses.append(pose({**q,**step}))
            q={**q,**goal}
        self._approach_joints=None
        endpoint=geometry.check([poses[-1]],obstacles,cfg)
        # A colliding endpoint rejects every route; avoid scanning those routes.
        if not endpoint["clear"]:
            return {**endpoint,"grasp_pose_check":endpoint,"reason":"neighbour_clearance"}
        result=geometry.check(poses,obstacles,cfg)
        result["grasp_pose_check"]=endpoint
        if not result["clear"] and result["grasp_pose_check"]["clear"]:
            from control.grasp import highest_reachable_hover
            start=self.s.robot.read_joints()
            x,y,z=ik.forward_position_mm(start)
            height=highest_reachable_hover(ik,x,y,self.s.grasp_z_mm,self.cfg,yaw_deg=attempt.yaw_deg)
            high=ik.solve(x,y,height,yaw_deg=attempt.yaw_deg)
            if not over_ik_gate(high,self.cfg):
                q=start;route=[pose(q)]
                for goal in (high.joints,attempt.hover.joints,attempt.grasp.joints):
                    for step in interpolate(q,goal,self.cfg.motion.descent_step_per_tick):
                        route.append(pose({**q,**step}))
                    q={**q,**goal}
                alternate=geometry.check(route,obstacles,cfg)
                result["high_approach_check"]=alternate
                if alternate["clear"]:
                    result["clear"]=True
                    self._approach_joints=high.joints
        result["reason"]="ok" if result["clear"] else "neighbour_clearance"
        return result

    def calibration_prepare_visible(self, color, x, y):
        """Use existing gated observation motion, then gate the entire pick.

        Observation does not open or descend toward a block. Existing lift,
        workspace, IK and bounded trajectory limits apply before fresh vision.
        """
        self.attempt=None
        self.descent_ready=False
        if self.s.held is not None:
            return SkillResult(False,"calibration_prepare_visible","already_holding")
        self.s.lift_in_place()
        moved=super().move_to_cell(x,y)
        if not moved.ok:
            return moved
        scene=self.s.observe(after=time.time())
        result=self.calibration_prepare(color,_scene=scene)
        result.action="calibration_prepare_visible"
        return result

    def calibration_prepare(self, color, _scene=None):
        self.descent_ready = False
        self.attempt = None
        self.baseline = None
        self.trial = None
        if self.s.held is not None:
            return SkillResult(False, "calibration_prepare", "already_holding")
        scene = self.s.home_and_observe() if _scene is None else _scene
        if scene.find("red") is None and self.s._scene_fn is None:
            # Fresh second pass separates a red block joined to thinner tape;
            # keeps all shape/colour gates and restores production config.
            original=self.cfg.perception
            try:
                self.cfg.perception=replace(original,morph_kernel_px=self.cfg.agent.calibration_clearance.red_separation_kernel_px)
                scene=self.s.observe(after=time.time())
            finally:
                self.cfg.perception=original
        block = scene.find(color)
        if block is None:
            return SkillResult(False, "calibration_prepare", "not_detected")
        missing=sorted(set(self.cfg.agent.calibration_clearance.expected_colors)-{b.color for b in scene.all()})
        if missing:
            return SkillResult(False,"calibration_prepare","scene_incomplete",data={"missing_colors":missing})
        xy = corrected_pick_xy(block.center_mm, self.s.base_xy, self.cfg)
        plan = plan_grasp_attempts(self.s.ik, self.cfg, *xy, self.s.grasp_z_mm,
                                  block_angle_deg=block.angle_deg,
                                  radial_tilt_deg=far_reach_tilt_deg(xy, self.s.base_xy, self.cfg))
        # Additional bounded offset candidates retain the primary jaw orientation.
        # Nominal candidates remain first, so clear original grasps are unchanged.
        primary=plan.attempts[0]
        extra=[]
        for forward,left in self.cfg.agent.calibration_clearance.trial_offsets_mm:
            if not all(math.isfinite(v) for v in (forward,left)) or math.hypot(forward,left)>self.cfg.agent.relative.max_pick_offset_mm:
                raise ValueError("Invalid calibration trial offset")
            shifted=offset_xy(primary.xy_mm,forward,left,frame=self.cfg.agent.relative.frame,base_xy_mm=self.s.base_xy)
            yaw=primary.yaw_deg
            if yaw is None:yaw=self.s.ik.neutral_yaw_deg(*primary.xy_mm,primary.grasp_z_mm)
            from control.grasp import highest_reachable_hover
            height=highest_reachable_hover(self.s.ik,*shifted,primary.grasp_z_mm,self.cfg,yaw_deg=yaw,radial_tilt_deg=plan.radial_tilt_deg)
            kw=dict(yaw_deg=yaw,radial_tilt_deg=plan.radial_tilt_deg)
            hover=self.s.ik.solve(*shifted,height,**kw)
            grasp=self.s.ik.solve(*shifted,primary.grasp_z_mm,**kw)
            extra.append(replace(primary,label=f"trial_f{forward:+g}_l{left:+g}",xy_mm=shifted,
                                 offset_mm=(forward,left),hover=hover,hover_z_mm=height,grasp=grasp,
                                 reachable=not any(over_ik_gate(v,self.cfg) for v in (hover,grasp))))
        considered=[]
        a=None
        for candidate in [*plan.attempts,*extra]:
            if candidate.label == "roll_90" and self.cfg.agent.calibration_clearance.rotate_retry_bias:
                primary=plan.attempts[0]
                primary_yaw=primary.yaw_deg
                if primary_yaw is None:
                    primary_yaw=self.s.ik.neutral_yaw_deg(*primary.xy_mm,primary.grasp_z_mm)
                angle=math.radians(candidate.yaw_deg-primary_yaw)
                dx,dy=primary.xy_mm[0]-xy[0],primary.xy_mm[1]-xy[1]
                rotated=(xy[0]+math.cos(angle)*dx-math.sin(angle)*dy,
                         xy[1]+math.sin(angle)*dx+math.cos(angle)*dy)
                if math.dist(rotated,candidate.xy_mm)>self.cfg.agent.relative.max_pick_offset_mm:
                    continue
                kw=dict(yaw_deg=candidate.yaw_deg,radial_tilt_deg=plan.radial_tilt_deg)
                hover=self.s.ik.solve(*rotated,candidate.hover_z_mm,**kw)
                grasp=self.s.ik.solve(*rotated,candidate.grasp_z_mm,**kw)
                candidate=replace(candidate,xy_mm=rotated,hover=hover,grasp=grasp,
                                  reachable=not any(over_ik_gate(v,self.cfg) for v in (hover,grasp)),
                                  label="roll_90_rotated_bias")
            if any(v.joints["wrist_roll"]<self.cfg.agent.calibration_clearance.wrist_roll_min_deg for v in (candidate.hover,candidate.grasp)):
                considered.append({"label":candidate.label,"clear":False,"reason":"limit_exceeded"})
                continue
            if not candidate.reachable or not self.s.in_workspace(candidate.xy_mm):
                continue
            clearance=self._clearance_gate(scene,color,candidate)
            considered.append(dict(label=candidate.label,**clearance))
            if clearance["clear"]:
                a=candidate
                break
        if a is None:
            reason=considered[-1]["reason"] if considered else "ik_gate"
            return SkillResult(False,"calibration_prepare",reason,retry_advice="do_not_retry",
                               data={"candidates":considered})
        self.clearance_scene = scene
        self.trial = str(time.time_ns())
        self.baseline = a
        self.plan = plan
        self.color = color
        self.forward = self.left = 0.0
        self._record("planned", color=color, cv_xy_mm=block.center_mm,
                     corrected_xy_mm=xy, baseline=asdict(a), tilt_deg=plan.radial_tilt_deg,
                     frame_seq=scene.frame_seq, captured_at=scene.captured_at,
                     clearance_candidates=considered)
        if self._approach_joints is not None:
            self.s.player.move_to(self._approach_joints,max_step=self.cfg.motion.descent_step_per_tick,
                                  tol=self.cfg.motion.transit_arrival_tol)
        self.s.motion.open_gripper()
        self.s.player.move_to(a.hover.joints, max_step=1.0, tol=self.cfg.motion.transit_arrival_tol)
        error, settled = self.s.player.settle(a.hover.joints, tol=self.cfg.motion.grasp_hover_arrival_tol,
                             timeout_s=self.cfg.motion.grasp_hover_settle_s)
        if not settled:
            self._record("prepare_shortfall",q_command=a.hover.joints,settle_error_deg=error,
                         loads=self.s.robot.read_loads())
            return SkillResult(False,"calibration_prepare","grasp_blocked",retry_advice="do_not_retry",
                               data={"stop_reason":"hover_not_settled","settle_error_deg":error})
        self.attempt = a
        self._record("prepared", q_command=a.hover.joints)
        return SkillResult(True, "calibration_prepare", "ok", data={
            "trial":self.trial, "cv_xy_mm":block.center_mm,
            "baseline_xy_mm":a.xy_mm,"hover_z_mm":a.hover_z_mm})

    def calibration_correct_hover(self, dry_run=True):
        """One bounded measured tracking correction, above the block only."""
        self.descent_ready=False
        a=self.baseline
        if a is None or self.s.held is not None or self.attempt is not None:
            return SkillResult(False,"calibration_correct_hover","precondition")
        robot,cfg=self.s.robot,self.cfg
        current=robot.read_joints()
        current_z=self.s.ik.forward_position_mm(current)[2]
        if current_z < a.grasp_z_mm+cfg.agent.calibration_clearance.obstacle_height_mm:
            return SkillResult(False,"calibration_correct_hover","precondition")
        errors={j:a.hover.joints[j]-current[j] for j in a.hover.joints}
        bound=cfg.agent.calibration_clearance.hover_correction_max_deg
        if max(map(abs,errors.values()))>cfg.motion.descent_max_lag:
            return SkillResult(False,"calibration_correct_hover","limit_exceeded")
        errors={j:max(-bound,min(bound,e)) for j,e in errors.items()}
        goal={j:a.hover.joints[j]+e for j,e in errors.items()}
        if self.s.ik.forward_position_mm(goal)[2]<current_z:
            return SkillResult(False,"calibration_correct_hover","precondition")
        if goal["wrist_roll"]<cfg.agent.calibration_clearance.wrist_roll_min_deg:
            return SkillResult(False,"calibration_correct_hover","limit_exceeded")
        check=self._clearance_gate(self.clearance_scene,self.color,
                                  replace(a,hover=replace(a.hover,joints=goal)))
        if not check["clear"] or self._approach_joints is not None:
            # This short correction executes only the direct route.
            return SkillResult(False,"calibration_correct_hover","neighbour_clearance",data=check)
        if dry_run:
            return SkillResult(True,"calibration_correct_hover","ok",
                               data={"dry_run":True,"correction_deg":errors})
        self.attempt=None
        baseline=robot.read_loads();samples=[];reason=None
        deadline=time.monotonic()+cfg.motion.move_timeout_s
        for command in interpolate(current,goal,cfg.motion.descent_step_per_tick):
            robot.send_joints(command)
            if cfg.motion.fps>0:time.sleep(1/cfg.motion.fps)
            q=robot.read_joints();loads=robot.read_loads()
            samples.append(dict(q=q,loads=loads,command=command))
            if any(abs(loads[j])-abs(baseline[j])>=cfg.sensing.contact_load_delta for j in cfg.sensing.contact_joints):
                reason="load_increase"
            elif max(abs(q[j]-command[j]) for j in goal)>cfg.motion.descent_max_lag:
                reason="tracking_lag"
            elif time.monotonic()>deadline:
                reason="timeout"
            if reason:break
        q=robot.read_joints()
        error=max(abs(q[j]-a.hover.joints[j]) for j in goal)
        if reason is None and error>cfg.motion.grasp_hover_arrival_tol:
            reason="hover_not_settled"
        if reason:
            robot.send_joints({j:q[j] for j in goal})
        else:
            self.attempt=a
        self._record("hover_tracking_correction",correction_deg=errors,q_command=goal,
                     nominal_error_deg=error,stop_reason=reason,samples=samples)
        return SkillResult(reason is None,"calibration_correct_hover","ok" if reason is None else "grasp_blocked",
                           data={"stop_reason":reason,"nominal_error_deg":error,"correction_deg":errors})

    def calibration_adjust(self, forward_mm=0.0, left_mm=0.0):
        self.descent_ready = False
        if self.attempt is None or self.s.held is not None:
            return SkillResult(False, "calibration_adjust", "precondition")
        f, l = self.forward + forward_mm, self.left + left_mm
        rel = self.cfg.agent.relative
        if not all(math.isfinite(v) for v in (forward_mm,left_mm,f,l)):
            return SkillResult(False, "calibration_adjust", "invalid_arguments")
        if math.hypot(forward_mm,left_mm)>rel.max_jog_mm or math.hypot(f,l)>rel.max_pick_offset_mm:
            return SkillResult(False, "calibration_adjust", "limit_exceeded")
        # Fixed baseline axes and command, not a sum of measured FK jogs.
        xy = offset_xy(self.baseline.xy_mm, f, l, frame=rel.frame, base_xy_mm=self.s.base_xy)
        if not self.s.in_workspace(xy):
            return SkillResult(False, "calibration_adjust", "out_of_workspace")
        a = self.baseline
        yaw = a.yaw_deg
        if yaw is None:
            yaw = self.s.ik.neutral_yaw_deg(*a.xy_mm, a.grasp_z_mm)
        kw = dict(yaw_deg=yaw, radial_tilt_deg=self.plan.radial_tilt_deg)
        hover = self.s.ik.solve(*xy, a.hover_z_mm, **kw)
        grasp = self.s.ik.solve(*xy, a.grasp_z_mm, **kw)
        if any(over_ik_gate(x,self.cfg) for x in (hover,grasp)):
            return SkillResult(False, "calibration_adjust", "ik_gate")
        adjusted = replace(a,xy_mm=xy,hover=hover,grasp=grasp,yaw_deg=yaw)
        clearance = self._clearance_gate(self.clearance_scene, self.color, adjusted)
        if not clearance["clear"]:
            return SkillResult(False, "calibration_adjust", clearance["reason"], data=clearance)
        self._record("adjust_requested", residual_mm=[f,l], target=asdict(adjusted))
        self.attempt = None  # failed/cancelled motion must not leave a usable grasp
        self.s.player.move_to(hover.joints,max_step=1.0,tol=self.cfg.motion.transit_arrival_tol)
        error, settled = self.s.player.settle(hover.joints,tol=self.cfg.motion.grasp_hover_arrival_tol,
                             timeout_s=self.cfg.motion.grasp_hover_settle_s)
        if not settled:
            self._record("adjust_shortfall",q_command=hover.joints,settle_error_deg=error,
                         loads=self.s.robot.read_loads())
            return SkillResult(False,"calibration_adjust","grasp_blocked",retry_advice="do_not_retry",
                               data={"stop_reason":"hover_not_settled","settle_error_deg":error})
        self.attempt = adjusted
        self.forward,self.left=f,l
        self._record("adjusted", q_command=hover.joints)
        return SkillResult(True,"calibration_adjust","moved",data={
            "residual_forward_left_mm":[f,l],"target_xy_mm":xy})

    def calibration_grasp(self):
        if self.attempt is None or self.s.held is not None:
            return SkillResult(False,"calibration_grasp","precondition")
        a,self.attempt=self.attempt,None
        self._record("grasp_requested",attempt=asdict(a),baseline_xy_mm=self.baseline.xy_mm,
                     residual_base_xy_mm=[x-y for x,y in zip(a.xy_mm,self.baseline.xy_mm)])
        outcome,check=attempt_grasp(self.s.player,self.s.robot,self.cfg,a)
        self._record("closed",outcome=outcome.value,check=asdict(check) if check else None)
        held=outcome is GraspOutcome.HELD
        if held:
            self.s.held=HeldBlock(color=self.color,attempt=a,picked_xy_mm=self.plan.detected_xy_mm,
                                 from_zone=self.s.in_zone(self.plan.detected_xy_mm),over_xy_mm=a.xy_mm)
            self.s.player.move_to(a.hover.joints,max_step=1.0,tol=self.cfg.motion.transit_arrival_tol)
            check=check_grasp(self.s.robot,self.cfg.sensing)
            held=check.grasped
            if not held:
                self.s.held=None
            self._record("lifted",check=asdict(check),sensor_held=held,visual_confirmed=False)
        reason="held" if held else ("grasp_blocked" if outcome is GraspOutcome.BLOCKED else "grasp_empty")
        return SkillResult(held,"calibration_grasp",reason,data={"trial":self.trial,
            "sensor_held":held,"visual_confirmed":False,
            "target_xy_mm":a.xy_mm,"residual_forward_left_mm":[self.forward,self.left]})

    def calibration_descend_guarded(self):
        """Load-guarded descent only; never close or retry after contact."""
        self.descent_ready = False
        if self.attempt is None or self.s.held is not None:
            return SkillResult(False, "calibration_descend_guarded", "precondition")
        a, self.attempt = self.attempt, None
        goal = a.grasp.joints
        robot, cfg = self.s.robot, self.cfg
        monitor = ContactMonitor(robot, cfg.sensing)
        baseline = monitor.start()
        self._record("guard_start", baseline_loads=baseline, attempt=asdict(a),
                     contact_load_delta=cfg.sensing.contact_load_delta)
        samples = []
        start = robot.read_joints()
        deadline = time.monotonic() + cfg.motion.move_timeout_s
        reason = None
        def sample(command):
            current = robot.read_joints()
            reading = monitor.check()
            lag = max(abs(current[j]-command[j]) for j in goal)
            samples.append(dict(time=time.time(), q=current, command=command,
                                loads=reading.loads, deltas=reading.deltas, lag=lag))
            # Experimental load increase guard; sign reversal or unloading alone is not contact.
            magnitude_delta = {j:max(0.0, abs(v)-abs(baseline[j])) for j,v in reading.loads.items()}
            samples[-1]["magnitude_deltas"] = magnitude_delta
            if any(v >= cfg.sensing.contact_load_delta for v in magnitude_delta.values()):
                return current, "load_increase"
            if lag > cfg.motion.descent_max_lag:
                return current, "tracking_lag"
            return current, None
        def tick():
            if cfg.motion.fps > 0:
                time.sleep(1.0/cfg.motion.fps)
        def finish(ok, why):
            # Release a deeper outstanding target without disabling torque.
            current = robot.read_joints()
            robot.send_joints(goal if ok else {j:current[j] for j in goal})
            self._record("guard_end", reason=why, samples=samples)
            if ok:
                self.attempt = a
                self.descent_ready = True
            return SkillResult(ok, "calibration_descend_guarded", "ok" if ok else "grasp_blocked",
                               retry_advice=None if ok else "do_not_retry", data={"stop_reason":why,
                "samples":len(samples), "max_load_delta":max(
                    (max(v["deltas"].values()) for v in samples), default=0),
                "gripper_closed":False, "trial":self.trial})
        current, reason = sample(start)
        if reason:
            return finish(False, reason)
        for step in interpolate(start, goal, cfg.motion.descent_step_per_tick):
            if time.monotonic() > deadline:
                return finish(False, "timeout")
            robot.send_joints(step)
            tick()
            current, reason = sample(step)
            if reason:
                return finish(False, reason)
        settle_end = min(deadline, time.monotonic()+cfg.motion.descent_settle_s)
        while max(abs(current[j]-goal[j]) for j in goal) > cfg.motion.arrival_tol:
            if time.monotonic() >= settle_end:
                return finish(False, "shortfall")
            robot.send_joints(goal)
            tick()
            current, reason = sample(goal)
            if reason:
                return finish(False, reason)
        return finish(True, "depth_reached")

    def calibration_close_lift(self):
        """Close after guarded descent; sensor verification gates lifting."""
        if not self.descent_ready or self.attempt is None or self.s.held is not None:
            return SkillResult(False,"calibration_close_lift","precondition")
        self.descent_ready = False
        a, self.attempt = self.attempt, None
        self.s.player.set_gripper(self.cfg.sensing.gripper_close_pos)
        check = check_grasp(self.s.robot,self.cfg.sensing)
        self._record("guard_close",check=asdict(check))
        if not check.grasped:
            self.s.player.set_gripper(self.cfg.sensing.gripper_open_pos)
            return SkillResult(False,"calibration_close_lift","grasp_empty")
        self.s.held = HeldBlock(color=self.color,attempt=a,picked_xy_mm=self.plan.detected_xy_mm,
                               from_zone=self.s.in_zone(self.plan.detected_xy_mm),over_xy_mm=a.xy_mm)
        self.s.player.move_to(a.hover.joints,max_step=self.cfg.motion.descent_step_per_tick,
                             tol=self.cfg.motion.transit_arrival_tol)
        check = check_grasp(self.s.robot,self.cfg.sensing)
        if not check.grasped:
            self.s.held = None
        self._record("guard_lift",check=asdict(check))
        return SkillResult(check.grasped,"calibration_close_lift",
                           "held" if check.grasped else "grasp_empty",data={"visual_confirmed":False})

    def calibration_descend_step(self, down_mm):
        """One bounded approach segment, then return for external camera inspection.

        Loads are evidence only here, not a contact classifier. Existing joint
        tracking and CancellableRobotIO guards remain active. No automatic close.
        """
        self.descent_ready = False
        if self.attempt is None or self.s.held is not None:
            return SkillResult(False,"calibration_descend_step","precondition")
        if not math.isfinite(down_mm) or not 0 < down_mm <= self.cfg.agent.relative.max_jog_mm:
            return SkillResult(False,"calibration_descend_step","limit_exceeded")
        a, self.attempt = self.attempt, None
        robot = self.s.robot
        start = robot.read_joints()
        initial = self.s.ik.forward_position_mm(start)
        goal = a.grasp.joints
        steps = interpolate(start,goal,self.cfg.motion.descent_step_per_tick)
        commands = []
        previous = start
        for step in steps:
            xyz = self.s.ik.forward_position_mm({**start,**step})
            if initial[2]-xyz[2] > down_mm:
                lo,hi = 0.0,1.0
                # Numerical bisection only, not a hardware-tuning constant.
                for _ in range(24):
                    mid=(lo+hi)/2
                    partial={j:previous[j]+mid*(step[j]-previous[j]) for j in goal}
                    z=self.s.ik.forward_position_mm({**start,**partial})[2]
                    if initial[2]-z > down_mm: hi=mid
                    else: lo=mid
                commands.append({j:previous[j]+lo*(step[j]-previous[j]) for j in goal})
                break
            if math.dist(initial[:2],xyz[:2]) > self.cfg.agent.relative.max_jog_mm:
                return SkillResult(False,"calibration_descend_step","limit_exceeded")
            commands.append(step)
            previous=step
        samples=[]
        reason=None
        deadline=time.monotonic()+self.cfg.motion.move_timeout_s
        for step in commands:
            if time.monotonic()>deadline:
                reason="timeout";break
            robot.send_joints(step)
            if self.cfg.motion.fps>0:time.sleep(1/self.cfg.motion.fps)
            measured=robot.read_joints()
            loads=robot.read_loads()
            lag=max(abs(measured[j]-step[j]) for j in goal)
            samples.append(dict(time=time.time(),q=measured,loads=loads,lag=lag))
            if lag>self.cfg.motion.descent_max_lag:
                reason="tracking_lag";break
        current=robot.read_joints()
        robot.send_joints({j:current[j] for j in goal})
        actual=self.s.ik.forward_position_mm(current)
        if reason is None:
            self.attempt=a
            self.descent_ready=max(abs(current[j]-goal[j]) for j in goal)<=self.cfg.motion.arrival_tol
        self._record("camera_step",requested_down_mm=down_mm,start_fk_mm=initial,
                     end_fk_mm=actual,samples=samples,stop_reason=reason)
        return SkillResult(reason is None,"calibration_descend_step","ok" if reason is None else "grasp_blocked",
                           data={"fk_before_mm":initial,"fk_after_mm":actual,"depth_ready":self.descent_ready,
                                 "gripper_closed":False,"visual_check_required":True,"stop_reason":reason})
