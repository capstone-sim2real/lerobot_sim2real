"""AgentService threads: chat, busy lock, STOP during a slow skill, home, jog."""

import threading
import time

from agent.control import ControlState
from agent.provider.fake import ScriptedProvider
from agent.provider.types import TextDelta, ToolCall, ToolCallEvent, TurnEnd
from agent.service import AgentService, needs_fresh_scene
from config import AppConfig
from session.cancel import Cancelled, CancelToken
from session.results import SkillResult


class Skills:
    """Stand-in for session.skills.Skills; the slow skill honours the token."""

    def __init__(self, cancel):
        self.cancel = cancel
        self.started = threading.Event()
        self.homed = 0
        self.at_home = True
        self.s = self

    @property
    def slot_centres(self):
        return []

    def state_dict(self):
        return {}

    def close(self):
        pass

    def close_gripper(self):
        self.started.set()
        for _ in range(200):
            if self.cancel.is_set():
                raise Cancelled("stop")
            time.sleep(0.01)
        return SkillResult(True, "close_gripper", "ok")

    def get_state(self):
        return SkillResult(True, "get_state", "ok")

    def observe_scene(self, include_zone=True):
        return SkillResult(
            True,
            "observe_scene",
            "ok",
            data={
                "blocks_outside": [{"color": "red", "x_mm": 100.0, "y_mm": 0.0}],
                "blocks_inside": [{"color": "blue", "slot": "top-left"}] if include_zone else None,
            },
        )

    def move_relative(self, forward_mm=0, left_mm=0, up_mm=0):
        return SkillResult(True, "move_relative", "moved")

    def rotate_gripper(self, delta_deg):
        return SkillResult(True, "rotate_gripper", "moved")

    def pick_here(self):
        return SkillResult(True, "pick_here", "held")

    def place_here(self):
        return SkillResult(True, "place_here", "released")

    def open_gripper(self):
        return SkillResult(True, "open_gripper", "ok")

    def recover_and_home(self):
        self.homed += 1
        return SkillResult(self.at_home, "recover_and_home", "ok" if self.at_home else "motion_timeout",
                           data={"arm_at_home": self.at_home})


def _service(script, cfg=None):
    cfg = cfg or AppConfig()
    cancel = CancelToken()
    holder = {}
    events = []

    def factory():
        holder["skills"] = Skills(cancel)
        return holder["skills"]

    service = AgentService(cfg, provider=ScriptedProvider(script), skills_factory=factory, cancel=cancel,
                           publish=events.append, system_prompt="sys", transcript_dir="")
    service._worker.start()
    service.started = True
    return service, holder["skills"], events


def test_manual_jog_and_gripper_language_do_not_force_home_observation():
    assert not needs_fresh_scene("오른쪽으로 20mm 이동해")
    assert not needs_fresh_scene("그리퍼를 45도 돌려줘")
    assert needs_fresh_scene("안에 있는 거 전부 밖으로 꺼내줘")
    assert needs_fresh_scene("빨간 블록을 오른쪽으로 20mm 옮겨줘")


def test_manual_recovery_is_the_only_recovery_path():
    cfg = AppConfig()
    service, skills, _events = _service([[ToolCallEvent(ToolCall("a", "close_gripper", {})), TurnEnd("tool_use")]], cfg)
    token = service.acquire_lease(None)
    service.chat(token, "미션 1")
    assert skills.started.wait(5)
    service.stop()
    service.wait_idle()
    # no automatic recovery: stays STOPPED until the operator clicks it
    assert service.gate.state is ControlState.STOPPED and skills.homed == 0
    assert service.home(token)[0] == 202
    service.wait_idle()
    assert service.gate.state is ControlState.IDLE and skills.homed == 1
    service.shutdown()
