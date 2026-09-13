"""Snapshot worker never accesses serial; the tracking thread supplies joints."""

import json, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from config import SessionToolsConfig
from tools.session_io import snapshot_bytes


class LiveCapture:
    def __init__(
        self, request_path, snapshot_url, max_delta, *, snapshot_timeout_s=None
    ):
        self.snapshot_timeout_s = (
            SessionToolsConfig().snapshot_timeout_s
            if snapshot_timeout_s is None
            else snapshot_timeout_s
        )
        self.request_path = Path(request_path)
        self.url = snapshot_url
        self.max_delta = max_delta
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None

    def snapshot(self):
        return snapshot_bytes(self.url, self.snapshot_timeout_s)

    def poll(self, joints):
        if self.pending:
            item = self.pending
            item["max_delta"] = max(
                item["max_delta"],
                max(abs(joints[j] - item["before"][j]) for j in joints),
            )
            if not item["future"].done():
                return
            prefix = item["prefix"]
            try:
                data = item["future"].result()
                if item["max_delta"] > self.max_delta:
                    raise ValueError("Arm moved during capture; hold still and retry")
                with prefix.with_suffix(".jpg").open("xb") as out:
                    out.write(data)
                payload = {
                    "name": item["name"],
                    "time": time.time(),
                    "joints": item["before"],
                    "joints_after": dict(joints),
                    "max_joint_delta": item["max_delta"],
                    "image": prefix.with_suffix(".jpg").name,
                    "pixel_pending": True,
                }
                with prefix.with_suffix(".json").open("x") as out:
                    json.dump(payload, out, indent=2)
                result = {"ok": True, "record": str(prefix.with_suffix(".json"))}
            except Exception as e:
                result = {"ok": False, "error": str(e)}
            prefix.with_name(prefix.name + "_result.json").write_text(
                json.dumps(result, indent=2)
            )
            print("CAPTURE", json.dumps(result), flush=True)
            self.pending = None
        if self.request_path.exists():
            claimed = self.request_path.with_suffix(".processing")
            self.request_path.rename(claimed)
            try:
                request = json.loads(claimed.read_text())
                name = request["name"]
                if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                    raise ValueError("Invalid point name")
                folder = Path(request["output_dir"])
                folder.mkdir(parents=True, exist_ok=True)
                prefix = folder / (name.lower() + "_live")
                if (
                    prefix.with_suffix(".json").exists()
                    or prefix.with_suffix(".jpg").exists()
                ):
                    raise ValueError("Point already exists; choose a retry name")
                self.pending = {
                    "future": self.pool.submit(self.snapshot),
                    "before": dict(joints),
                    "max_delta": 0.0,
                    "name": name,
                    "prefix": prefix,
                }
            except Exception as e:
                print("CAPTURE_REJECTED", str(e), flush=True)
            finally:
                claimed.unlink()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
