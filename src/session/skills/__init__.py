"""High-level arm skills for the LLM agent. Every method returns a SkillResult.

These compose the mission code; they do not reimplement it:

- ``pick_block`` drives the production ``CvIkPickState`` (centre attempt plus
  the one 90-degree gripper-roll retry live inside ``run_grasp_attempts``),
  applies the same ``corrected_pick_xy``/``far_reach_tilt_deg`` Task 1 applies,
  gates on ``check_grasp`` exactly like VERIFY, and
  repeats home -> observe -> pick up to ``fsm.max_retries_per_block`` times,
  which is what SELECT -> PICK -> SELECT does in the FSM.
- Every release uses ``control.task1_transport``'s ``solve_place_point``,
  ``carry_waypoints``, ``fly_carry`` and ``release_at`` -- the motions Task 1
  flies.
- ``run_task`` runs the unmodified FSM flows on this session's robot.

A skill never raises for an expected outcome. ``Cancelled`` (STOP) and
``TimeoutError`` (the arm did not track) propagate to the tool layer, which
turns them into fault results.
"""

from __future__ import annotations

from session.skills.base import SkillsBase
from session.skills.mission import MissionMixin
from session.skills.motion import MotionMixin
from session.skills.place import PlaceMixin


class Skills(MissionMixin, MotionMixin, PlaceMixin, SkillsBase):
    pass
