"""Cooperative STOP for the arm, raised only at tick boundaries.

Nothing in ``control/`` or ``fsm/`` knows about this module. Every arm
command already passes through ``BaseRobotIO.send_joints`` (the same seam
``data.episode_recorder.RecordingRobotIO`` records at), so a decorator there
bounds cancel latency in every motion path without a hook in the control code.
Loops that poll the camera without commanding the arm are covered by wrapping
their injected ``perceive`` callable with :func:`guard`.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

from control.robot_io import BaseRobotIO

T = TypeVar("T")


class Cancelled(Exception):
    """Raised at a tick boundary when STOP was pressed.

    Deliberately not RuntimeError/OSError/ValueError: ``fsm/task1.py`` catches
    that triple around ``perceive()`` and would swallow it into an endless
    poll. Deliberately not KeyboardInterrupt: the unwind point must be
    deterministic -- between two bus writes, never inside one. Same contract
    as ``data.episode_recorder.StopRecording``.
    """


class CancelToken:
    """Thread-safe STOP flag. The web thread sets it; the robot thread checks it."""

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def event(self) -> threading.Event:
        """The underlying event, for ``RecordingRobotIO(stop_event=...)``."""
        return self._event

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()

    def raise_if_set(self) -> None:
        if self._event.is_set():
            raise Cancelled("stop requested")


def guard(token: CancelToken, fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap a polling callable so STOP ends the poll at its next call."""

    def guarded(*args: Any, **kwargs: Any) -> T:
        token.raise_if_set()
        return fn(*args, **kwargs)

    return guarded


class CancellableRobotIO(BaseRobotIO):
    """Raise :class:`Cancelled` at the next bus WRITE once the token is set.

    Reads pass through untouched, so a stopped arm can still be observed and
    the safe-return path (which clears the token first) can still measure it.
    """

    def __init__(self, inner: BaseRobotIO, token: CancelToken):
        self._inner = inner
        self._token = token
        self.joint_names = inner.joint_names

    @property
    def inner(self) -> BaseRobotIO:
        return self._inner

    def connect(self) -> None:
        self._inner.connect()

    def disconnect(self) -> None:
        self._inner.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._inner.is_connected

    def read_joints(self) -> dict[str, float]:
        return self._inner.read_joints()

    def read_observation(self) -> dict[str, Any]:
        return self._inner.read_observation()

    def send_joints(self, positions: dict[str, float]) -> dict[str, float]:
        self._token.raise_if_set()
        return self._inner.send_joints(positions)

    def read_loads(self) -> dict[str, int]:
        return self._inner.read_loads()

    def set_torque(self, enabled: bool) -> None:
        self._inner.set_torque(enabled)

    def __getattr__(self, name: str) -> Any:
        # e.g. So101RobotIO.robot for the ACT policy client
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)
