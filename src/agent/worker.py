"""The single thread that owns the arm session for the server's lifetime.

Not a ThreadPoolExecutor: the session must be constructed AND closed on this
thread (TopDownIK's os.chdir, the serial bus owner), and an executor has no
on-thread shutdown hook.
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
_SENTINEL = object()


class RobotWorker:
    def __init__(self, factory: Callable[[], Any], *, name: str = "so101-robot"):
        self._factory = factory
        self._queue: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._resource: Any = None
        self._thread = threading.Thread(target=self._main, name=name, daemon=True)

    @property
    def resource(self) -> Any:
        return self._resource

    def start(self, timeout_s: float | None = None) -> None:
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise TimeoutError("robot worker did not start in time")
        if self._error is not None:
            raise self._error

    def _main(self) -> None:
        try:
            self._resource = self._factory()
        except BaseException as exc:  # noqa: BLE001 - reported to start()
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            fn, future = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(self._resource))
            except BaseException as exc:  # noqa: BLE001 - delivered to the caller
                future.set_exception(exc)
        close = getattr(self._resource, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.exception("closing the robot session failed")

    def submit(self, fn: Callable[[Any], T]) -> "Future[T]":
        if not self._thread.is_alive():
            raise RuntimeError("robot worker is not running")
        future: Future = Future()
        self._queue.put((fn, future))
        return future

    def run(self, fn: Callable[[Any], T], timeout_s: float | None = None) -> T:
        return self.submit(fn).result(timeout=timeout_s)

    def stop(self, timeout_s: float = 60.0) -> None:
        if self._thread.is_alive():
            self._queue.put(_SENTINEL)
            self._thread.join(timeout_s)
