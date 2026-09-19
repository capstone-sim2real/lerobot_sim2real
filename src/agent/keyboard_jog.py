"""Latest-input keyboard stream. No queued moves; all bus work stays on RobotWorker."""
import math
import threading
import time
import secrets

from .jog_ramp import JogRamp


class JogReleased(Exception):
    pass


class KeyboardJog:
    def __init__(self, cfg, *, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.id = secrets.token_urlsafe(18)
        self.lock = threading.RLock()
        self.seq = -1
        self.vector = (0., 0., 0.)
        self.expires = clock() + cfg.keyboard_timeout_s
        self.closed = False
        self.releasing = False

    def update(self, seq, vector):
        with self.lock:
            if self.closed or self.releasing:
                return False
            if self.clock() >= self.expires:
                self.closed = True
                return False
            if seq <= self.seq:
                return False
            self.seq = seq
            norm = math.hypot(*vector)
            self.vector = tuple(v / max(1., norm) for v in vector)
            self.expires = self.clock() + self.cfg.keyboard_timeout_s
            if norm == 0:
                self.release()
            return True

    def release(self):
        with self.lock:
            if not self.closed and not self.releasing:
                self.releasing = True
                self.vector = (0., 0., 0.)
                self.expires = self.clock() + self.cfg.keyboard_release_timeout_s

    def close(self):
        with self.lock:
            self.closed = True

    def current(self):
        with self.lock:
            if self.closed or self.clock() >= self.expires:
                self.closed = True
                raise JogReleased()
            return self.vector

    def run(self, skills, cancel):
        from .jog_executor import JogExecutor

        return JogExecutor(self, skills, cancel).run()
