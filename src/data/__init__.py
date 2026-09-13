"""Dataset recording for imitation learning (Task 3).

Nothing here is imported by the Task 1 / Task 2 control path, and the lerobot
dependency is loaded lazily inside :class:`LeRobotEpisodeSink` so the test
suite keeps running without lerobot, torch, or placo (AGENTS.md §13).
"""

from data.episode_recorder import (
    EpisodeRecorder,
    EpisodeSink,
    RecordingRobotIO,
    StopRecording,
)

__all__ = [
    "EpisodeRecorder",
    "EpisodeSink",
    "RecordingRobotIO",
    "StopRecording",
]
