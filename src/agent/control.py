"""Who may command the arm, and when. Pure logic, no web framework.

States::

    IDLE --chat/jog--> BUSY --turn ends--> IDLE
    BUSY --STOP--> STOPPING --turn thread unwound--> STOPPED
    BUSY --robot fault--> STOPPED
    STOPPED --[home]--> HOMING --home verified--> IDLE
    HOMING --not home / STOP--> STOPPED

Only IDLE accepts a new command. STOPPED accepts only the home request. The
web UI greys buttons out from the broadcast state, but this class is what
actually refuses.

All connected operators share one session token. The state machine still
serializes robot work: only IDLE accepts a command, so multiple browsers may
control the arm but cannot execute motions concurrently. STOP needs no token:
stopping the arm is never gated behind permissions.
"""

from __future__ import annotations

import secrets
import threading
import time
from enum import Enum
from typing import Callable


class ControlState(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    HOMING = "homing"


class ControlGate:
    def __init__(
        self,
        *,
        on_stop: Callable[[], None],
        lease_grace_s: float,
        lease_idle_timeout_s: float,
        on_change: Callable[[dict], None] = lambda _snapshot: None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._lock = threading.RLock()
        self._on_stop = on_stop
        self._on_change = on_change
        self._clock = clock
        self._grace_s = lease_grace_s
        self._idle_timeout_s = lease_idle_timeout_s
        self.state = ControlState.IDLE
        self.busy_with: str | None = None
        self.message: str | None = None
        self._token: str | None = None
        self._lease_since: float | None = None
        self._last_activity: float | None = None
        self._disconnected_since: float | None = None
        self._operator_connections = 0

    # ── snapshot ─────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state.value,
                "busy_with": self.busy_with,
                "message": self.message,
                "lease_held": self._token is not None,
            }

    def _changed(self) -> None:
        self._on_change(self.snapshot())

    # ── lease ────────────────────────────────────────────────────────

    def acquire_lease(self, token: str | None = None) -> str | None:
        """Join the shared operator session, creating it when necessary."""
        with self._lock:
            now = self._clock()
            if self._token is not None:
                # Every browser gets the same capability. A stale token from
                # an earlier session is replaced by the current one.
                self._last_activity = now
                self._disconnected_since = None
                return self._token
            self._token = secrets.token_urlsafe(18)
            self._lease_since = now
            self._last_activity = now
            self._disconnected_since = None
            self._operator_connections = 0
            self._changed()
            return self._token

    def release_lease(self, token: str) -> bool:
        with self._lock:
            if token != self._token or self.state is not ControlState.IDLE:
                return False
            self._drop_lease()
            return True

    def force_release(self) -> None:
        with self._lock:
            if self.state in (ControlState.BUSY, ControlState.HOMING):
                self._request_stop_locked("조작 권한이 회수되어 정지했습니다.")
            self._drop_lease()

    def _drop_lease(self) -> None:
        self._token = None
        self._lease_since = None
        self._last_activity = None
        self._disconnected_since = None
        self._operator_connections = 0
        self._changed()

    def check(self, token: str | None) -> bool:
        with self._lock:
            return token is not None and token == self._token

    def lease_since(self) -> float | None:
        return self._lease_since

    def operator_connected(self, token: str | None) -> None:
        with self._lock:
            if self.check(token):
                self._operator_connections += 1
                self._disconnected_since = None

    def operator_disconnected(self, token: str | None) -> None:
        with self._lock:
            if self.check(token):
                self._operator_connections = max(0, self._operator_connections - 1)
                if self._operator_connections == 0:
                    self._disconnected_since = self._clock()

    def tick(self) -> None:
        """Called about once a second: expire abandoned or idle leases."""
        with self._lock:
            if self._token is None:
                return
            now = self._clock()
            if (
                self._operator_connections == 0
                and self._disconnected_since is not None
                and now - self._disconnected_since > self._grace_s
            ):
                if self.state in (ControlState.BUSY, ControlState.HOMING):
                    # nobody is left to press STOP for this motion
                    self._request_stop_locked("조작자 연결이 끊겨 정지했습니다.")
                self._drop_lease()
                return
            if (
                self.state is ControlState.IDLE
                and self._last_activity is not None
                and now - self._last_activity > self._idle_timeout_s
            ):
                self._drop_lease()

    # ── commands ─────────────────────────────────────────────────────

    def try_begin(self, token: str | None, what: str) -> bool:
        with self._lock:
            if not self.check(token) or self.state is not ControlState.IDLE:
                return False
            self.state = ControlState.BUSY
            self.busy_with = what
            self.message = None
            self._last_activity = self._clock()
            self._changed()
            return True

    def finish(self, *, robot_fault: bool, message: str | None = None) -> None:
        with self._lock:
            if self.state is ControlState.BUSY:
                self.state = ControlState.STOPPED if robot_fault else ControlState.IDLE
            elif self.state is ControlState.STOPPING:
                self.state = ControlState.STOPPED
            else:
                return
            self.busy_with = None
            self.message = message
            self._last_activity = self._clock()
            self._changed()

    def request_stop(self) -> bool:
        with self._lock:
            return self._request_stop_locked("비상정지 버튼이 눌렸습니다.")

    def _request_stop_locked(self, message: str) -> bool:
        if self.state not in (ControlState.BUSY, ControlState.HOMING):
            return False
        self._on_stop()
        self.state = ControlState.STOPPING
        self.message = message
        self._changed()
        return True

    def try_begin_home(self, token: str | None) -> bool:
        with self._lock:
            if not self.check(token) or self.state is not ControlState.STOPPED:
                return False
            return self._begin_home_locked()

    def begin_auto_home(self) -> bool:
        with self._lock:
            if self.state is not ControlState.STOPPED:
                return False
            return self._begin_home_locked()

    def _begin_home_locked(self) -> bool:
        self.state = ControlState.HOMING
        self.busy_with = "recover_and_home"
        self.message = None
        self._last_activity = self._clock()
        self._changed()
        return True

    def finish_home(self, arm_at_home: bool, message: str | None = None) -> None:
        with self._lock:
            if self.state not in (ControlState.HOMING, ControlState.STOPPING):
                return
            ok = arm_at_home and self.state is ControlState.HOMING
            self.state = ControlState.IDLE if ok else ControlState.STOPPED
            self.busy_with = None
            self.message = message if not ok else None
            self._last_activity = self._clock()
            self._changed()
