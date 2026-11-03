"""Person tracking with ReID-gated target locking, without any ROS dependency."""

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from pinto_model_zoo_ros2.wholebody49 import Detection, intersection_over_union

BODY_CLASS_ID = 0
CENTER_REGION_DIVISOR = 3
MILLISECONDS_PER_SECOND = 1000.0


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    box: Detection
    depth_mm: float
    embedding: np.ndarray


@dataclass
class TrackerEvents:
    locked: int = 0
    rejected_by_reid: int = 0
    reacquired_by_reid: int = 0
    released_by_timeout: int = 0


@dataclass(frozen=True)
class FrameResult:
    stamp_sec: float
    detections: Tuple[Detection, ...]
    tracks: Tuple[TrackedPerson, ...]
    target: Optional[TrackedPerson]
    detect_ms: float
    reid_ms: float
    total_ms: float


@dataclass(frozen=True)
class CandidateFilters:
    """Same rules as the existing pan-tilt person tracking node."""

    min_bbox_height_px: int = 100
    min_bbox_area_px: int = 2000
    max_width_to_height_ratio: float = 2.0
    depth_min_mm: float = 100.0
    depth_max_mm: float = 8000.0

    def accepts(self, box: Detection, depth_mm: Optional[float]) -> bool:
        width = box.x2 - box.x1
        height = box.y2 - box.y1
        return (
            depth_mm is not None
            and height >= self.min_bbox_height_px
            and width * height >= self.min_bbox_area_px
            and width <= self.max_width_to_height_ratio * height
        )


def bbox_depth_mm(
    depth_mm_image: np.ndarray, box: Detection, min_mm: float, max_mm: float
) -> Optional[float]:
    """Median valid depth of the box center third, else of the whole box, else None."""
    image_height, image_width = depth_mm_image.shape[:2]
    left = max(int(box.x1), 0)
    top = max(int(box.y1), 0)
    right = min(int(box.x2), image_width)
    bottom = min(int(box.y2), image_height)
    margin_x = (right - left) // CENTER_REGION_DIVISOR
    margin_y = (bottom - top) // CENTER_REGION_DIVISOR
    regions = (
        depth_mm_image[top + margin_y:bottom - margin_y, left + margin_x:right - margin_x],
        depth_mm_image[top:bottom, left:right],
    )
    for region in regions:
        valid_values = region[(region >= min_mm) & (region <= max_mm)]
        if valid_values.size > 0:
            return float(np.median(valid_values))
    return None


@dataclass
class _Track:
    box: Detection
    lost_frames: int = 0


class PersonTracker:
    """Greedy IoU association that keeps ids stable across short disappearances."""

    def __init__(self, iou_threshold: float, max_lost_frames: int) -> None:
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames
        self._next_track_id = 1
        self._tracks = {}

    def update(
        self, detections: Sequence[Tuple[Detection, float, np.ndarray]]
    ) -> Tuple[TrackedPerson, ...]:
        """Take (body box, depth_mm, embedding) per detection; return only visible tracks."""
        pairs = sorted(
            (
                (intersection_over_union(track.box, box), track_id, detection_index)
                for track_id, track in self._tracks.items()
                for detection_index, (box, _, _) in enumerate(detections)
            ),
            reverse=True,
        )
        track_id_by_detection_index = {}
        for iou, track_id, detection_index in pairs:
            if iou < self._iou_threshold:
                break
            if track_id in track_id_by_detection_index.values():
                continue
            if detection_index in track_id_by_detection_index:
                continue
            track_id_by_detection_index[detection_index] = track_id

        matched_track_ids = set(track_id_by_detection_index.values())
        self._age_unmatched_tracks(matched_track_ids)

        visible = []
        for detection_index, (box, depth_mm, embedding) in enumerate(detections):
            track_id = track_id_by_detection_index.get(detection_index)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
            self._tracks[track_id] = _Track(box)
            visible.append(TrackedPerson(track_id, box, depth_mm, embedding))
        return tuple(visible)

    def _age_unmatched_tracks(self, matched_track_ids: set) -> None:
        for track_id in list(self._tracks):
            if track_id in matched_track_ids:
                continue
            track = self._tracks[track_id]
            if track.lost_frames >= self._max_lost_frames:
                del self._tracks[track_id]
            else:
                self._tracks[track_id] = _Track(track.box, track.lost_frames + 1)


def _normalized(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


class TargetSelector:
    """Locks one person and re-checks its ReID similarity on every frame."""

    def __init__(
        self, reid_similarity_threshold: float, lost_timeout_sec: float, ema_alpha: float
    ) -> None:
        self._similarity_threshold = reid_similarity_threshold
        self._lost_timeout_sec = lost_timeout_sec
        self._ema_alpha = ema_alpha
        self._target_embedding: Optional[np.ndarray] = None
        self._lost_since_sec: Optional[float] = None
        self.events = TrackerEvents()
        self.target_track_id: Optional[int] = None

    def release(self) -> None:
        self._target_embedding = None
        self._lost_since_sec = None
        self.target_track_id = None

    def update(
        self, candidates: Sequence[TrackedPerson], now_sec: float
    ) -> Optional[TrackedPerson]:
        if self._target_embedding is None:
            return self._lock_nearest(candidates)

        locked_person = self._find_locked_person(candidates)
        if locked_person is not None:
            if self._similarity(locked_person) >= self._similarity_threshold:
                return self._follow(locked_person)
            self.events.rejected_by_reid += 1
            self.target_track_id = None

        if self._lost_since_sec is None:
            self._lost_since_sec = now_sec
        best_person = max(candidates, key=self._similarity, default=None)
        if best_person is not None and self._similarity(best_person) >= self._similarity_threshold:
            self.events.reacquired_by_reid += 1
            return self._follow(best_person)
        if now_sec - self._lost_since_sec >= self._lost_timeout_sec:
            self.events.released_by_timeout += 1
            self.release()
            return self._lock_nearest(candidates)
        return None

    def _find_locked_person(self, candidates: Sequence[TrackedPerson]) -> Optional[TrackedPerson]:
        return next(
            (person for person in candidates if person.track_id == self.target_track_id), None
        )

    def _similarity(self, person: TrackedPerson) -> float:
        return float(np.dot(_normalized(person.embedding), self._target_embedding))

    def _follow(self, person: TrackedPerson) -> TrackedPerson:
        blended = (1.0 - self._ema_alpha) * self._target_embedding + (
            self._ema_alpha * _normalized(person.embedding)
        )
        self._target_embedding = _normalized(blended)
        self.target_track_id = person.track_id
        self._lost_since_sec = None
        return person

    def _lock_nearest(self, candidates: Sequence[TrackedPerson]) -> Optional[TrackedPerson]:
        if not candidates:
            return None
        nearest = min(candidates, key=lambda person: person.depth_mm)
        self._target_embedding = _normalized(nearest.embedding)
        self.target_track_id = nearest.track_id
        self._lost_since_sec = None
        self.events.locked += 1
        return nearest


class PersonTrackingPipeline:
    """detector.detect(image) -> Detections, encoder.encode(crops) -> (N, D) embeddings."""

    def __init__(
        self,
        detector,
        encoder,
        tracker: PersonTracker,
        selector: TargetSelector,
        filters: CandidateFilters,
    ) -> None:
        self._detector = detector
        self._encoder = encoder
        self._tracker = tracker
        self._selector = selector
        self._filters = filters

    def process(
        self, image_bgr: np.ndarray, depth_mm_image: np.ndarray, stamp_sec: float
    ) -> FrameResult:
        total_start = time.perf_counter()
        detections = tuple(self._detector.detect(image_bgr))
        detect_end = time.perf_counter()

        candidates = self._candidate_boxes(detections, depth_mm_image)
        embeddings = self._encode_crops(image_bgr, [box for box, _ in candidates])
        reid_end = time.perf_counter()

        tracks = self._tracker.update(
            [(box, depth_mm, embedding) for (box, depth_mm), embedding in zip(candidates, embeddings)]
        )
        target = self._selector.update(tracks, stamp_sec)
        total_end = time.perf_counter()
        return FrameResult(
            stamp_sec=stamp_sec,
            detections=detections,
            tracks=tracks,
            target=target,
            detect_ms=(detect_end - total_start) * MILLISECONDS_PER_SECOND,
            reid_ms=(reid_end - detect_end) * MILLISECONDS_PER_SECOND,
            total_ms=(total_end - total_start) * MILLISECONDS_PER_SECOND,
        )

    def _candidate_boxes(self, detections, depth_mm_image):
        candidates = []
        for box in detections:
            if box.class_id != BODY_CLASS_ID:
                continue
            depth_mm = bbox_depth_mm(
                depth_mm_image, box, self._filters.depth_min_mm, self._filters.depth_max_mm
            )
            if self._filters.accepts(box, depth_mm):
                candidates.append((box, depth_mm))
        return candidates

    def _encode_crops(self, image_bgr, boxes):
        if not boxes:
            return []
        image_height, image_width = image_bgr.shape[:2]
        crops = [
            image_bgr[
                max(int(box.y1), 0):min(int(box.y2), image_height),
                max(int(box.x1), 0):min(int(box.x2), image_width),
            ]
            for box in boxes
        ]
        return list(self._encoder.encode(crops))
