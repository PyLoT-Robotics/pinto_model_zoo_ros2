"""Evaluate person tracking on a recorded rosbag2 (mcap) without touching ROS.

Reads the bag directly with rosbag2_py: no rclpy.init, no publisher, no `ros2 bag play`.
--start / --end are seconds since the first message in the bag.
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from pinto_model_zoo_ros2.person_tracking import (
    CandidateFilters,
    FrameResult,
    PersonTracker,
    PersonTrackingPipeline,
    TargetSelector,
)
from pinto_model_zoo_ros2.reid import PersonReidEncoder
from pinto_model_zoo_ros2.tracking_metrics import FrameRow, Summary, summarize
from pinto_model_zoo_ros2.wholebody49 import ScoreThresholds, Wholebody49Detector

NANOSECONDS_PER_SECOND = 1_000_000_000
# Same defaults as the person_tracker node.
OBJECT_SCORE_THRESHOLD = 0.50
IOU_MATCH_THRESHOLD = 0.3
MAX_LOST_FRAMES = 30
REID_SIMILARITY_THRESHOLD = 0.5
LOST_TIMEOUT_SEC = 7.0
EMA_ALPHA = 0.1
SYNC_SLOP_SEC = 0.05
MAX_RATE_HZ = 15.0
REID_MODEL_FILE = "osnet_ain_x1_0_p_unified_aug_n.onnx"
BYTES_PER_DEPTH_PIXEL = 2

TARGET_COLOR_BGR = (0, 255, 0)
OTHER_COLOR_BGR = (160, 160, 160)
CSV_COLUMNS = (
    "stamp_sec", "n_detections", "n_tracks", "target_track_id",
    "target_depth_mm", "detect_ms", "reid_ms", "total_ms",
)  # fmt: skip


def stamp_sec_of(message) -> float:
    return message.header.stamp.sec + message.header.stamp.nanosec / NANOSECONDS_PER_SECOND


def image_to_bgr(message) -> np.ndarray:
    if message.encoding != "rgb8":
        raise ValueError(f"color image must be rgb8: {message.encoding!r}")
    rgb = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    return cv2.cvtColor(rgb[:, : message.width * 3].reshape(message.height, message.width, 3), cv2.COLOR_RGB2BGR)


def depth_to_mm(message) -> np.ndarray:
    if message.encoding != "16UC1":
        raise ValueError(f"depth image must be 16UC1: {message.encoding!r}")
    pixels_per_row = message.step // BYTES_PER_DEPTH_PIXEL
    depth = np.frombuffer(message.data, dtype=np.uint16).reshape(message.height, pixels_per_row)
    return depth[:, : message.width]


def read_messages(
    bag_directory: str, topics: List[str], start_sec: float, end_sec: Optional[float]
) -> Iterator[Tuple[str, object]]:
    """Yield (topic, deserialized message) for messages inside the time window."""
    # Imported here so the pure-Python parts (and their tests) need no ROS install.
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_options = rosbag2_py.StorageOptions(uri=bag_directory, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    type_names = {info.name: info.type for info in reader.get_all_topics_and_types()}
    message_types = {topic: get_message(type_names[topic]) for topic in topics}

    bag_start_ns = rosbag2_py.Info().read_metadata(bag_directory, "mcap").starting_time.nanoseconds
    while reader.has_next():
        topic, raw_data, timestamp_ns = reader.read_next()
        elapsed_sec = (timestamp_ns - bag_start_ns) / NANOSECONDS_PER_SECOND
        if end_sec is not None and elapsed_sec > end_sec:
            return
        if elapsed_sec >= start_sec:
            yield topic, deserialize_message(raw_data, message_types[topic])


def synchronized_frames(
    messages: Iterator[Tuple[str, object]], image_topic: str, depth_topic: str
) -> Iterator[Tuple[object, object]]:
    """Pair each image with the depth message nearest in stamp, within SYNC_SLOP_SEC.

    Streams in bag order, holding only images that are newer than the latest depth.
    """
    previous_depth = None
    held_images: List[object] = []
    last_processed_sec = float("-inf")

    def pair_for(image, depth_candidates):
        nonlocal last_processed_sec
        image_sec = stamp_sec_of(image)
        candidates = [depth for depth in depth_candidates if depth is not None]
        if not candidates or image_sec - last_processed_sec < 1.0 / MAX_RATE_HZ:
            return None
        nearest = min(candidates, key=lambda depth: abs(stamp_sec_of(depth) - image_sec))
        if abs(stamp_sec_of(nearest) - image_sec) > SYNC_SLOP_SEC:
            return None
        last_processed_sec = image_sec
        return image, nearest

    for topic, message in messages:
        if topic == image_topic:
            held_images.append(message)
            continue
        if topic != depth_topic:
            continue
        depth_sec = stamp_sec_of(message)
        ready = [image for image in held_images if stamp_sec_of(image) <= depth_sec]
        held_images = [image for image in held_images if stamp_sec_of(image) > depth_sec]
        for image in ready:
            pair = pair_for(image, (previous_depth, message))
            if pair is not None:
                yield pair
        previous_depth = message
    for image in held_images:
        pair = pair_for(image, (previous_depth,))
        if pair is not None:
            yield pair


def draw_result(image_bgr: np.ndarray, result: FrameResult) -> np.ndarray:
    canvas = image_bgr.copy()
    target_id = result.target.track_id if result.target is not None else None
    for person in result.tracks:
        is_target = person.track_id == target_id
        color = TARGET_COLOR_BGR if is_target else OTHER_COLOR_BGR
        top_left = (int(person.box.x1), int(person.box.y1))
        bottom_right = (int(person.box.x2), int(person.box.y2))
        cv2.rectangle(canvas, top_left, bottom_right, color, 4 if is_target else 1)
        label = f"id {person.track_id}" + (" TARGET" if is_target else "")
        cv2.putText(canvas, label, (top_left[0], max(top_left[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  # fmt: skip
    return canvas


def frame_row_of(result: FrameResult) -> FrameRow:
    return FrameRow(
        stamp_sec=result.stamp_sec,
        n_detections=len(result.detections),
        n_tracks=len(result.tracks),
        target_track_id=result.target.track_id if result.target else None,
        target_depth_mm=result.target.depth_mm if result.target else None,
        detect_ms=result.detect_ms,
        reid_ms=result.reid_ms,
        total_ms=result.total_ms,
    )


def build_pipeline(model_directory: Path, model_variant: str):
    selector = TargetSelector(REID_SIMILARITY_THRESHOLD, LOST_TIMEOUT_SEC, EMA_ALPHA)
    pipeline = PersonTrackingPipeline(
        Wholebody49Detector(model_directory, model_variant, ScoreThresholds(object_score=OBJECT_SCORE_THRESHOLD)),
        PersonReidEncoder(model_directory, REID_MODEL_FILE),
        PersonTracker(IOU_MATCH_THRESHOLD, MAX_LOST_FRAMES),
        selector,
        CandidateFilters(),
    )  # fmt: skip
    return pipeline, selector


def write_outputs(output_directory: Path, rows: List[FrameRow], summary: Summary, events) -> None:
    with open(output_directory / "frames.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    summary_json = {**asdict(summary), "events": asdict(events)}
    (output_directory / "summary.json").write_text(json.dumps(summary_json, indent=2))
    print(json.dumps(summary_json, indent=2))


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_directory")
    parser.add_argument("--image-topic", default="/d455/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/d455/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--start", type=float, default=0.0, help="seconds since bag start")
    parser.add_argument("--end", type=float, default=None, help="seconds since bag start")
    parser.add_argument("--model-variant", default="s")
    parser.add_argument("--model-directory", type=Path, default=Path("/models"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    arguments = parse_arguments(argv)
    arguments.output.mkdir(parents=True, exist_ok=True)
    pipeline, selector = build_pipeline(arguments.model_directory, arguments.model_variant)

    messages = read_messages(
        arguments.bag_directory, [arguments.image_topic, arguments.depth_topic],
        arguments.start, arguments.end,
    )  # fmt: skip
    rows: List[FrameRow] = []
    video_writer = None
    try:
        for image_message, depth_message in synchronized_frames(
            messages, arguments.image_topic, arguments.depth_topic
        ):
            image_bgr = image_to_bgr(image_message)
            result = pipeline.process(image_bgr, depth_to_mm(depth_message), stamp_sec_of(image_message))
            rows.append(frame_row_of(result))
            if video_writer is None:
                height, width = image_bgr.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(arguments.output / "tracking.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), MAX_RATE_HZ, (width, height),
                )  # fmt: skip
            video_writer.write(draw_result(image_bgr, result))
    finally:
        if video_writer is not None:
            video_writer.release()
    write_outputs(arguments.output, rows, summarize(rows), selector.events)


if __name__ == "__main__":
    main()
