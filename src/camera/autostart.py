"""Start ``so101-camera`` automatically when nothing is already serving it.

camera.server is meant to be the sole owner of ``/dev/video*`` (see
docs/architecture.md); everything else -- so101-run, so101-collect,
so101-agent -- only ever reads frames from it over HTTP. This module lets
those callers spawn it themselves instead of requiring an operator to start
``so101-camera`` in a separate terminal first. A health check gates the
spawn, so a server started by someone else is left alone and never gets a
second owner racing it for the device.
"""

from __future__ import annotations

import atexit
import logging
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

logger = logging.getLogger("camera.autostart")


def base_url_of(url: str) -> str:
    """``scheme://host:port`` from any camera.server URL (snapshot/video/health)."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def camera_healthy(base_url: str, *, timeout_s: float = 1.0) -> bool:
    try:
        with urlopen(f"{base_url}/health", timeout=timeout_s):  # nosec B310
            return True
    except (URLError, OSError, ValueError):
        return False


class ManagedCamera:
    """A so101-camera process this call started; stop it when done with it."""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        atexit.register(self.stop)

    def stop(self) -> None:
        if self._proc.poll() is not None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5.0)


def ensure_camera_server(
    base_url: str,
    *,
    extra_args: list[str] | None = None,
    startup_timeout_s: float = 15.0,
) -> ManagedCamera | None:
    """Start so101-camera in the background if nothing answers ``base_url``.

    Returns a handle to stop the process this call started, or ``None`` when
    an existing server already answered the health check -- that one may be
    shared with other tools (the agent's browser feed, another runner) and
    is left running.
    """
    if camera_healthy(base_url):
        return None
    logger.info("so101-camera not detected at %s; starting it", base_url)
    proc = subprocess.Popen(  # nosec B603 - fixed module, no shell, args are our own config
        [sys.executable, "-m", "camera.server", *(extra_args or [])],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + startup_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"so101-camera exited immediately (code {proc.returncode}); "
                "run `so101-camera` by hand to see why"
            )
        if camera_healthy(base_url):
            return ManagedCamera(proc)
        time.sleep(0.3)
    proc.terminate()
    raise RuntimeError(
        f"so101-camera did not become healthy at {base_url} within {startup_timeout_s:g}s"
    )
