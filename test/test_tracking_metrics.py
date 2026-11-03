from pinto_model_zoo_ros2.tracking_metrics import (
    FrameRow,
    count_target_id_changes,
    summarize,
    timing_stats,
)


def make_row(target_track_id, total_ms=10.0):
    return FrameRow(
        stamp_sec=0.0,
        n_detections=1,
        n_tracks=1,
        target_track_id=target_track_id,
        target_depth_mm=None if target_track_id is None else 1500.0,
        detect_ms=total_ms / 2,
        reid_ms=total_ms / 4,
        total_ms=total_ms,
    )


def test_id_changes_ignore_gaps_with_same_id():
    assert count_target_id_changes([1, 1, None, None, 1]) == 0


def test_id_changes_count_id_change_directly_and_across_gap():
    assert count_target_id_changes([1, 2, None, 3, 3]) == 2


def test_summarize_counts_frames_and_target_frames():
    summary = summarize([make_row(1), make_row(None), make_row(2)])

    assert summary.frames == 3
    assert summary.frames_with_target == 2
    assert summary.target_id_changes == 1


def test_timing_stats_mean_and_nearest_rank_p95():
    stats = timing_stats([float(value) for value in range(1, 21)])

    assert stats.mean == 10.5
    assert stats.p95 == 19.0


def test_summarize_empty_rows_gives_zeros():
    summary = summarize([])

    assert summary.frames == 0
    assert summary.total_ms.mean == 0.0
