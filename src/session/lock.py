"""Advisory single-owner lock on the robot serial bus.

The Feetech bus is one CH340 serial device, and pyserial does not reliably
take an exclusive lock on it: two processes can both open it and interleave
half-packets, which shows up as read timeouts and motors twitching toward
stale targets. Every robot-owning entry point that takes this lock refuses to
start while another holds it. Advisory only -- a process that never asks is
not stopped.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path


class RobotBusBusy(RuntimeError):
    pass


class RobotBusLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = None

    @property
    def held(self) -> bool:
        return self._fh is not None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.seek(0)
            holder = fh.read().strip()
            fh.close()
            try:
                info = json.loads(holder)
                who = (
                    f"pid {info.get('pid')} ({' '.join(info.get('argv', []))}) "
                    f"since {info.get('started_at')}"
                )
            except (ValueError, AttributeError):
                who = "another process"
            raise RobotBusBusy(
                f"robot bus already held by {who}. Stop so101-run/so101-collect/so101-agent first."
            ) from None
        fh.seek(0)
        fh.truncate()
        fh.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "argv": [os.path.basename(sys.argv[0]), *sys.argv[1:]],
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
        )
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.seek(0)
            self._fh.truncate()
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    def holder(self) -> str | None:
        """Best-effort description of the current holder, for /api/health."""
        try:
            text = self.path.read_text().strip()
        except OSError:
            return None
        return text or None

    def __enter__(self) -> "RobotBusLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
