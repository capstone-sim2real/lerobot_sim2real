"""Call one so101-panel manual tool and save camera snapshots and SSE events as evidence.

python -m tools.agent_tool_call move_to_cell --arguments '{"x": 2, "y": 3}' --output var/evidence
"""

import argparse
import json
import queue
import threading
import time
import urllib.request
from pathlib import Path

from config import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tool")
    ap.add_argument("--arguments", default="{}")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8109")
    ap.add_argument("--camera-url", default="http://127.0.0.1:8090")
    args = ap.parse_args()
    cfg = load_config("src/configs/default.yaml")
    out = Path(args.output).resolve() / str(time.time_ns())
    out.mkdir(parents=True)
    headers = {"Content-Type": "application/json", "x-so101-control-version": "cell-grid-v1"}

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.base_url + path, data=data, headers=headers)
        return json.load(urllib.request.urlopen(req, timeout=10))

    token = request("/api/lease", {})["token"]
    if not token:
        raise RuntimeError("Operator lease unavailable")
    headers["x-operator-token"] = token
    dispatch_ns = None
    ready = threading.Event()
    results = queue.Queue()
    stream = urllib.request.urlopen(
        urllib.request.Request(args.base_url + "/api/events", headers=headers),
        timeout=cfg.agent.tool_timeout_s,
    )

    def read():
        try:
            with (out / "events.jsonl").open("w") as f:
                for raw in stream:
                    if not raw.startswith(b"data: "):
                        continue
                    event = json.loads(raw[6:])
                    f.write(json.dumps(event) + "\n")
                    f.flush()
                    ready.set()
                    if (
                        event.get("type") == "tool_result"
                        and event.get("name") == args.tool
                        and dispatch_ns is not None
                        and event.get("emitted_monotonic_ns", 0) >= dispatch_ns
                    ):
                        results.put(event)
        except Exception as exc:
            results.put({"stream_error": str(exc)})

    threading.Thread(target=read, daemon=True).start()
    if not ready.wait(10):
        raise TimeoutError("SSE not ready")

    def snapshots(phase):
        for camera in ("shoulder", "wrist"):
            b = urllib.request.urlopen(f"{args.camera_url}/snapshot/{camera}.jpg", timeout=5).read()
            (out / (phase + "-" + camera + ".jpg")).write_bytes(b)

    capture_done = threading.Event()

    def capture():
        i = 0
        while not capture_done.is_set():
            started = time.monotonic()
            try:
                snapshots("during-%04d" % i)
            except Exception as exc:
                with (out / "camera-errors.txt").open("a") as f:
                    f.write(str(exc) + "\n")
            i += 1
            capture_done.wait(max(0, cfg.agent.camera_view.poll_s - (time.monotonic() - started)))

    capture_thread = threading.Thread(target=capture, daemon=True)
    try:
        snapshots("before")
        capture_thread.start()
        dispatch_ns = time.monotonic_ns()
        sent = (
            request("/api/home", {})
            if args.tool == "recover_and_home"
            else request("/api/manual", {"tool": args.tool, "arguments": json.loads(args.arguments)})
        )
        if not sent.get("accepted"):
            raise RuntimeError(str(sent))
        result = results.get(timeout=cfg.agent.tool_timeout_s)
        if "stream_error" in result:
            raise RuntimeError(str(result))
        (out / "result.json").write_text(json.dumps(result, indent=2))
        snapshots("after")
        print(json.dumps({"evidence": str(out), **result}))
    except BaseException:
        request("/api/stop", {})
        raise
    finally:
        capture_done.set()
        if capture_thread.is_alive():
            capture_thread.join(timeout=cfg.agent.camera_view.read_timeout_s)
        request("/api/lease/release", {})


if __name__ == "__main__":
    main()
