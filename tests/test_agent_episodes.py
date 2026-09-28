import json

import pytest

from agent.episodes import list_episodes, list_runs, video_file


def test_episode_browser_uses_saved_metadata_and_video_segment(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    root = tmp_path / "agent"
    run_id = "so101_task3_20260929_050154_582bc819"
    run = root / run_id
    (run / "meta/episodes/chunk-000").mkdir(parents=True)
    (run / "meta/info.json").write_text(json.dumps({"fps": 30, "total_episodes": 2,
        "total_frames": 75, "features": {"observation.images.top": {"dtype": "video"}}}))
    (run / "agent_collection_summary.json").write_text(json.dumps({"discard_reasons": {"timing_gap": 1}}))
    pq.write_table(pa.Table.from_pylist([
        {"episode_index": 0, "length": 30, "videos/observation.images.top/chunk_index": 0,
         "videos/observation.images.top/file_index": 0, "videos/observation.images.top/from_timestamp": 0.,
         "videos/observation.images.top/to_timestamp": 1.},
        {"episode_index": 1, "length": 45, "videos/observation.images.top/chunk_index": 0,
         "videos/observation.images.top/file_index": 0, "videos/observation.images.top/from_timestamp": 1.,
         "videos/observation.images.top/to_timestamp": 2.5},
    ]), run / "meta/episodes/chunk-000/file-000.parquet")
    video = run / "videos/observation.images.top/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    assert list_runs(root)[0]["discard_reasons"] == {"timing_gap": 1}
    episodes = list_episodes(root, run_id)
    assert [episode["index"] for episode in episodes] == [0, 1]
    assert episodes[1]["videos"][0]["start_s"] == 1.
    assert video_file(root, run_id, 1, "observation.images.top") == video
    with pytest.raises(ValueError):
        video_file(root, run_id, 1, "../secret")
