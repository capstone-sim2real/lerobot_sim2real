"""Read-only browser for finalized LeRobot episode metadata and videos."""
from __future__ import annotations

import json
import re
from pathlib import Path

_RUN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}\Z")
_VIDEO_KEY = re.compile(r"observation\.images\.[a-zA-Z0-9_]+\Z")


def _run(root: Path, run_id: str) -> Path:
    if not _RUN.fullmatch(run_id):
        raise ValueError("Invalid dataset id")
    path = root / run_id
    if not path.is_dir() or path.is_symlink():
        raise FileNotFoundError(run_id)
    return path


def list_runs(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    result = []
    for path in sorted(root.iterdir(), reverse=True):
        if not _RUN.fullmatch(path.name) or not path.is_dir() or path.is_symlink():
            continue
        info_path = path / "meta/info.json"
        if not info_path.is_file():
            continue
        try:
            info = json.loads(info_path.read_text())
            summary_path = path / "agent_collection_summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        except (OSError, ValueError):
            continue
        result.append({"id": path.name, "episodes": info.get("total_episodes", 0),
                       "frames": info.get("total_frames", 0), "fps": info.get("fps"),
                       "discard_reasons": summary.get("discard_reasons", {}),
                       "recording": summary.get("recording", False)})
    return result


def list_episodes(root: Path, run_id: str) -> list[dict]:
    run = _run(root, run_id)
    info = json.loads((run / "meta/info.json").read_text())
    features = info.get("features", {})
    video_keys = [key for key, value in features.items() if value.get("dtype") == "video" and _VIDEO_KEY.fullmatch(key)]
    # Parquet is imported only for this page; server startup and robot control do not depend on it.
    import pyarrow.parquet as pq
    episodes = []
    for path in sorted((run / "meta/episodes").glob("chunk-*/file-*.parquet")) if (run / "meta/episodes").is_dir() else []:
        for row in pq.read_table(path).to_pylist():
            index = int(row["episode_index"])
            videos = []
            for key in video_keys:
                try:
                    chunk = int(row[f"videos/{key}/chunk_index"])
                    file = int(row[f"videos/{key}/file_index"])
                    start = float(row[f"videos/{key}/from_timestamp"])
                    end = float(row[f"videos/{key}/to_timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                video_path = run / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
                if video_path.is_file():
                    videos.append({"key": key, "url": f"/api/episodes/{run_id}/{index}/video/{key}",
                                   "start_s": start, "end_s": end})
            episodes.append({"index": index, "frames": int(row["length"]),
                             "duration_s": round(int(row["length"]) / info["fps"], 2),
                             "tasks": row.get("tasks", []), "videos": videos})
    return sorted(episodes, key=lambda item: item["index"])


def video_file(root: Path, run_id: str, episode_index: int, key: str) -> Path:
    if not _VIDEO_KEY.fullmatch(key):
        raise ValueError("Invalid video key")
    run = _run(root, run_id)
    for episode in list_episodes(root, run_id):
        if episode["index"] != episode_index:
            continue
        for video in episode["videos"]:
            if video["key"] == key:
                # The metadata reader validated the key and file indices.
                import pyarrow.parquet as pq
                for path in sorted((run / "meta/episodes").glob("chunk-*/file-*.parquet")):
                    for row in pq.read_table(path).to_pylist():
                        if int(row["episode_index"]) == episode_index:
                            chunk = int(row[f"videos/{key}/chunk_index"])
                            file = int(row[f"videos/{key}/file_index"])
                            return run / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
    raise FileNotFoundError("Episode video unavailable")
