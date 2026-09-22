"""Existing web UI with calibration-only manual tools; no external model calls."""
import argparse
import json
import logging
import time
from pathlib import Path
from config import load_config
from agent.server import API_KEY_ENV, EventHub, create_app, load_env_file, make_skills_factory
from agent.service import AgentService
from agent.tools import ToolDef, _obj, _mm
from agent.provider.types import ToolSpec
from agent.provider import build_provider
from session.cancel import CancelToken
from session.grasp_calibration import CalibrationSkills


def definitions(cfg):
    return [
        ToolDef(ToolSpec("calibration_probe_positive","Incremental positive wrist range probe, bounded by URDF and load stop.",_obj({})),
                lambda sk,a:sk.calibration_probe_positive()),
        ToolDef(ToolSpec("calibration_wrist_clearance_pose","Neutral wrist and gated outward observation cell; no rotation test yet.",_obj({"x":{"type":"integer"},"y":{"type":"integer"}},["x","y"])),
                lambda sk,a:sk.calibration_wrist_clearance_pose(**a)),
        ToolDef(ToolSpec("calibration_retry_wrist","Fixed requested wrist angle test after lift, with load guard.",_obj({})),
                lambda sk,a:sk.calibration_retry_wrist()),
        ToolDef(ToolSpec("calibration_clearance_status","Read-only scene completeness and target clearance ranking.",_obj({})),
                lambda sk,a:sk.calibration_clearance_status()),
        ToolDef(ToolSpec("calibration_pick_guarded","One continuous guarded pick, automatic recording; no retry.",
                        _obj({"color":{"type":"string","enum":list(cfg.perception.color_prototypes)}},["color"])),
                lambda sk,a:sk.calibration_pick_guarded(**a)),
        ToolDef(ToolSpec("calibration_descend_step","Bounded descent then pause for camera inspection; never close.",
                        _obj({"down_mm":_mm("Positive downward distance",cfg.agent.relative.max_jog_mm)},["down_mm"])),
                lambda sk,a:sk.calibration_descend_step(**a)),
        ToolDef(ToolSpec("calibration_descend_guarded","Descend with load guard; leave jaws open.",_obj({})),
                lambda sk,a:sk.calibration_descend_guarded()),
        ToolDef(ToolSpec("calibration_close_lift","Close and lift after guarded descent and visual check.",_obj({})),
                lambda sk,a:sk.calibration_close_lift()),
        ToolDef(ToolSpec("calibration_prepare_visible","Observe from a gated board cell, then approach CV target.",
                        _obj({"color":{"type":"string","enum":list(cfg.perception.color_prototypes)},
                              "x":{"type":"integer"},"y":{"type":"integer"}},["color","x","y"])),
                lambda sk,a:sk.calibration_prepare_visible(**a)),
        ToolDef(ToolSpec("calibration_continuous","Production pick_block with zero added offset, no observation pause.",
                        _obj({"color":{"type":"string","enum":list(cfg.perception.color_prototypes)}},["color"])),
                lambda sk,a:sk.calibration_continuous(**a)),
        ToolDef(ToolSpec("calibration_prepare","Home, detect named block, approach baseline hover.",
                        _obj({"color":{"type":"string","enum":list(cfg.perception.color_prototypes)}},["color"])),
                lambda sk,a:sk.calibration_prepare(**a)),
        ToolDef(ToolSpec("calibration_correct_hover","Bounded hover tracking correction; dry-run first.",
                        _obj({"dry_run":{"type":"boolean"}})),
                lambda sk,a:sk.calibration_correct_hover(**a)),
        ToolDef(ToolSpec("calibration_adjust","Bounded residual in fixed baseline axes.",
                        _obj({k:_mm("Relative mm",cfg.agent.relative.max_jog_mm)
                              for k in ("forward_mm","left_mm")})),
                lambda sk,a:sk.calibration_adjust(**a)),
        ToolDef(ToolSpec("calibration_grasp","One attempt, lift and verify; no automatic retries.",_obj({})),
                lambda sk,a:sk.calibration_grasp()),
    ]


# Generic descent/pick/rotation intentionally stay out: calibration tools own
# their load/clearance checks. Jog retains the standard height/workspace/IK gates.
WEB_MANUAL_TOOLS = frozenset({
    "move_arm", "move_to_cell", "open_gripper", "return_to_home",
    "place_here", "place_on_table", "place_at_cell", "place_at_slot", "place_at_pixel",
})


def configure_manual_tools(service, cfg):
    from dataclasses import replace

    from agent.calibration_manual_tools import build_calibration_manual_tools
    defs = definitions(cfg) + build_calibration_manual_tools(cfg)
    primitive_names = frozenset(service.registry._tools)
    service.registry._tools.update({d.spec.name: d for d in defs})
    allowed = primitive_names | WEB_MANUAL_TOOLS | frozenset(d.spec.name for d in defs)
    tools = {}
    for name, definition in service.registry._tools.items():
        if name not in allowed:
            continue
        if name in WEB_MANUAL_TOOLS or name in primitive_names or name.startswith("calibration_"):
            run = definition.run
            def manual_run(skills, arguments, run=run, name=name):
                # A jog/open/release invalidates the earlier visually checked
                # calibration pose, including when the ensuing command fails.
                if name not in primitive_names and hasattr(skills, "_invalidate_pick"):
                    skills._invalidate_pick()
                if not name.startswith("calibration_"):
                    skills.attempt = None
                    skills.descent_ready = False
                return run(skills, arguments)
            definition = replace(definition, run=manual_run)
        tools[name] = definition
    service.registry._tools = tools
    service.MANUAL_TOOLS = frozenset(tools)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--port",type=int,default=8109)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--set",action="append",default=[],dest="overrides")
    parser.add_argument("--provider", choices=["anthropic", "openai", "gemini", "fake"])
    parser.add_argument("--model")
    parser.add_argument("--env-file")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    load_env_file(args.env_file)
    output=Path(args.output).resolve()
    if args.dry_run:
        from agent_helpers import make_skills
        sk,_,_=make_skills({"green":(180,0)})
        sk.s.cfg.agent.calibration_clearance.expected_colors=["green"]
        cal=CalibrationSkills(sk.s,output)
        assert cal.calibration_prepare("green").ok
        base=cal.baseline.xy_mm
        assert cal.calibration_adjust(left_mm=2).ok
        assert cal.calibration_adjust(left_mm=999).reason=="limit_exceeded"
        cal.s.cancel.set()
        from session.cancel import Cancelled
        try:
            cal.calibration_adjust(left_mm=1)
        except Cancelled:
            pass
        else:
            raise AssertionError("STOP failed")
        assert cal.attempt is None
        cal.s.cancel.clear()
        assert cal.calibration_prepare("green").ok
        result=cal.calibration_grasp()
        assert result.ok
        print(json.dumps({"simulation_only":True,"baseline":base,"result":result.to_envelope()}))
        return
    cfg=load_config(str(root/"src/configs/default.yaml"),overrides=[
        "camera.auto_start=false",
        "agent.lock_path=/home/ehdrms/lerobot_sim2real/local_operations/robot.lock"] + args.overrides)
    logging.basicConfig(level=logging.INFO)
    provider_name = args.provider or cfg.agent.provider
    if args.model:
        cfg.agent.models[provider_name] = args.model
    required = API_KEY_ENV.get(provider_name, [])
    if required and not any(__import__("os").environ.get(name) for name in required):
        parser.error(f"{' / '.join(required)} is required for provider {provider_name}")
    try:
        provider = build_provider(cfg.agent, provider=provider_name)
    except (ImportError, ValueError) as exc:
        parser.error(f"cannot initialize provider {provider_name}: {exc}")
    logging.info("calibration server provider=%s model=%s", provider.name, provider.model)
    hub=EventHub()
    def service_builder(publish):
        raw_publish = publish
        publish = lambda event: raw_publish({**event, "emitted_monotonic_ns": time.monotonic_ns()})
        cancel=CancelToken()
        def build_calibration(session):
            from session.calibration_joint_limit import CalibrationJointLimitIO
            from session.cancel import CancellableRobotIO
            assert isinstance(session.robot,CancellableRobotIO)
            session.robot._inner=CalibrationJointLimitIO(session.robot._inner,cfg.agent.calibration_clearance)
            return CalibrationSkills(session,output)
        skills_factory=make_skills_factory(cfg,cancel,sim=False,skills_builder=build_calibration)
        service=AgentService(cfg,provider=provider,
                             skills_factory=skills_factory,cancel=cancel,publish=publish,
                             transcript_dir=str(output/"transcripts"))
        configure_manual_tools(service, cfg)
        return service
    app=create_app(cfg,service_builder,hub)
    import uvicorn
    server=uvicorn.Server(uvicorn.Config(app,host="0.0.0.0",port=args.port,timeout_graceful_shutdown=5))
    app.state.uvicorn_server=server
    server.run()


if __name__=="__main__":
    main()
