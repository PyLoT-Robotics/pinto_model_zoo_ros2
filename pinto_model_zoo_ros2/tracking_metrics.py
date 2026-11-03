"""Summary statistics for tracking evaluation. Pure Python, no ROS or GPU dependency."""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

P95_FRACTION = 0.95


@dataclass(frozen=True)
class FrameRow:
    stamp_sec: float
    n_detections: int
    n_tracks: int
    target_track_id: Optional[int]
    target_depth_mm: Optional[float]
    detect_ms: float
    reid_ms: float
    total_ms: float


@dataclass(frozen=True)
class TimingStats:
    mean: float
    p95: float


@dataclass(frozen=True)
class Summary:
    frames: int
    frames_with_target: int
    target_switches: int
    detect_ms: TimingStats
    reid_ms: TimingStats
    total_ms: TimingStats


def timing_stats(values: Sequence[float]) -> TimingStats:
    """Mean and nearest-rank 95th percentile; zeros for no values."""
    if not values:
        return TimingStats(mean=0.0, p95=0.0)
    ordered = sorted(values)
    rank = math.ceil(P95_FRACTION * len(ordered))
    return TimingStats(mean=sum(ordered) / len(ordered), p95=ordered[rank - 1])


def count_target_switches(target_track_ids: Sequence[Optional[int]]) -> int:
    """Count changes of target id against the previous frame that had a target.

    A change across a gap (target lost, then a different id) also counts.
    """
    switches = 0
    previous_id: Optional[int] = None
    for track_id in target_track_ids:
        if track_id is None:
            continue
        if previous_id is not None and track_id != previous_id:
            switches += 1
        previous_id = track_id
    return switches


def summarize(rows: Sequence[FrameRow]) -> Summary:
    target_ids = [row.target_track_id for row in rows]
    return Summary(
        frames=len(rows),
        frames_with_target=sum(1 for track_id in target_ids if track_id is not None),
        target_switches=count_target_switches(target_ids),
        detect_ms=timing_stats([row.detect_ms for row in rows]),
        reid_ms=timing_stats([row.reid_ms for row in rows]),
        total_ms=timing_stats([row.total_ms for row in rows]),
    )
