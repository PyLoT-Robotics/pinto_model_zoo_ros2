import numpy as np

from pinto_model_zoo_ros2.person_tracking import (
    CandidateFilters,
    PersonTracker,
    PersonTrackingPipeline,
    TargetSelector,
    TrackedPerson,
    bbox_depth_mm,
)
from pinto_model_zoo_ros2.wholebody49 import Detection, intersection_over_union


def body(x1=100.0, y1=100.0, x2=200.0, y2=300.0):
    return Detection(0, 0.9, x1, y1, x2, y2)


def unit(*values):
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def update_one(tracker, box):
    return tracker.update([(box, 1000.0, unit(1, 0))])


def person(track_id, depth_mm, embedding):
    return TrackedPerson(track_id, body(), depth_mm, embedding)


def test_intersection_over_union_is_public():
    assert intersection_over_union(body(), body()) == 1.0


def test_bbox_depth_uses_center_median_then_whole_box_then_none():
    depth = np.zeros((90, 90), dtype=np.uint16)
    box = Detection(0, 0.9, 0, 0, 90, 90)
    depth[30:60, 30:60] = 2000
    assert bbox_depth_mm(depth, box, 100, 8000) == 2000.0
    depth[30:60, 30:60] = 0
    depth[:10, :10] = 3000
    assert bbox_depth_mm(depth, box, 100, 8000) == 3000.0
    assert bbox_depth_mm(np.zeros((90, 90), np.uint16), box, 100, 8000) is None


def test_tracker_keeps_id_for_same_box_and_issues_new_monotonic_ids():
    tracker = PersonTracker(0.3, 2)
    first = update_one(tracker, body())[0].track_id
    assert update_one(tracker, body())[0].track_id == first
    far = body(500, 100, 600, 300)
    ids = {p.track_id for p in tracker.update([(body(), 1.0, unit(1, 0)), (far, 1.0, unit(0, 1))])}
    assert ids == {first, first + 1}


def test_tracker_recovers_within_max_lost_frames_only():
    tracker = PersonTracker(0.3, 2)
    first = update_one(tracker, body())[0].track_id
    assert tracker.update([]) == ()
    assert tracker.update([]) == ()
    assert update_one(tracker, body())[0].track_id == first
    for _ in range(3):
        tracker.update([])
    assert update_one(tracker, body())[0].track_id != first


def test_selector_locks_nearest_then_follows_and_rejects_swapped_id():
    selector = TargetSelector(0.5, 7.0, 0.1)
    target_embedding, other_embedding = unit(1, 0), unit(0, 1)
    nearest = person(1, 1000, target_embedding)
    assert selector.update([nearest, person(2, 3000, other_embedding)], 0.0) is nearest
    assert selector.events.locked == 1
    # ids swapped: id 1 now looks like the other person -> rejected, id 2 reacquired
    swapped = [person(1, 1000, other_embedding), person(2, 3000, target_embedding)]
    assert selector.update(swapped, 1.0).track_id == 2
    assert (selector.events.rejected_by_reid, selector.events.reacquired_by_reid) == (1, 1)


def test_selector_returns_none_until_timeout_then_relocks():
    selector = TargetSelector(0.5, 7.0, 0.1)
    selector.update([person(1, 1000, unit(1, 0))], 0.0)
    stranger = person(2, 2000, unit(0, 1))
    assert selector.update([stranger], 1.0) is None
    assert selector.update([stranger], 8.0) is stranger
    assert (selector.events.released_by_timeout, selector.events.locked) == (1, 2)


def test_selector_release_relocks_nearest():
    selector = TargetSelector(0.5, 7.0, 0.1)
    selector.update([person(1, 1000, unit(1, 0))], 0.0)
    selector.release()
    assert selector.update([person(2, 500, unit(0, 1))], 1.0).track_id == 2


def test_candidate_filters_reject_small_wide_and_depthless():
    filters = CandidateFilters()
    assert filters.accepts(body(), 1000.0)
    assert not filters.accepts(body(), None)
    assert not filters.accepts(body(0, 0, 40, 60), 1000.0)
    assert not filters.accepts(body(0, 0, 300, 120), 1000.0)


class FakeDetector:
    def detect(self, image):
        return (body(), Detection(1, 0.9, 0, 0, 10, 10))


class FakeEncoder:
    def encode(self, crops):
        return np.tile(unit(1, 0), (len(crops), 1))


def test_pipeline_fills_tracks_target_and_timings():
    pipeline = PersonTrackingPipeline(
        FakeDetector(), FakeEncoder(), PersonTracker(0.3, 5),
        TargetSelector(0.5, 7.0, 0.1), CandidateFilters(),
    )
    depth = np.full((480, 640), 2000, dtype=np.uint16)
    result = pipeline.process(np.zeros((480, 640, 3), np.uint8), depth, 1.5)
    assert len(result.detections) == 2 and len(result.tracks) == 1
    assert result.target.track_id == result.tracks[0].track_id
    assert result.stamp_sec == 1.5 and result.total_ms >= result.detect_ms >= 0
