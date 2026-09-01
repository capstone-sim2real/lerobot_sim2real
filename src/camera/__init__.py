"""Single-owner camera service and HTTP snapshot client for SO-101."""

from .client import DEFAULT_SHOULDER_SNAPSHOT_URL, fetch_snapshot

__all__ = ["DEFAULT_SHOULDER_SNAPSHOT_URL", "fetch_snapshot"]
