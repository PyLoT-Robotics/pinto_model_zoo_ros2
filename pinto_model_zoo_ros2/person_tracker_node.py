"""Track people with Wholebody49 + ReID and publish the locked target as Detection2D."""

from pathlib import Path
from typing import Iterable, Optional

import cv2
import message_filters
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from pinto_model_zoo_ros2.person_tracking import (
    CandidateFilters,
    Intrinsics,
    PersonTracker,
    PersonTrackingPipeline,
    TargetSelector,
    TrackedPerson,
)
from pinto_model_zoo_ros2.reid import PersonReidEncoder
from pinto_model_zoo_ros2.wholebody49 import ScoreThresholds, Wholebody49Detector

DEPTH_ENCODING = "16UC1"
MILLIMETERS_PER_METER = 1000.0
TRACK_COLOR_BGR = (0, 255, 0)
TARGET_COLOR_BGR = (0, 0, 255)


def to_person_message(
    person: TrackedPerson, header, intrinsics: Optional[Intrinsics]
) -> Detection2D:
    """Camera optical frame position [m]; X and Y stay 0 without intrinsics."""
    box = person.box
    center_x = (box.x1 + box.x2) / 2.0
    center_y = (box.y1 + box.y2) / 2.0
    depth_m = person.depth_mm / MILLIMETERS_PER_METER

    message = Detection2D()
    message.header = header
    message.id = str(person.track_id)
    message.bbox.center.position.x = center_x
    message.bbox.center.position.y = center_y
    message.bbox.size_x = box.x2 - box.x1
    message.bbox.size_y = box.y2 - box.y1
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = "body"
    hypothesis.hypothesis.score = box.score
    position = hypothesis.pose.pose.position
    if intrinsics is not None:
        position.x = (center_x - intrinsics.cx) * depth_m / intrinsics.fx
        position.y = (center_y - intrinsics.cy) * depth_m / intrinsics.fy
    position.z = depth_m
    message.results.append(hypothesis)
    return message


def draw_tracks(image_bgr, tracks: Iterable[TrackedPerson], target: Optional[TrackedPerson]):
    canvas = image_bgr.copy()
    for person in tracks:
        is_target = target is not None and person.track_id == target.track_id
        color = TARGET_COLOR_BGR if is_target else TRACK_COLOR_BGR
        top_left = (int(person.box.x1), int(person.box.y1))
        bottom_right = (int(person.box.x2), int(person.box.y2))
        cv2.rectangle(canvas, top_left, bottom_right, color, 3 if is_target else 2)
        cv2.putText(canvas, f"id {person.track_id}", top_left, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    return canvas


class PersonTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("person_tracker")
        model_directory = Path(self.declare_parameter("model_directory", "/models").value)
        variant = self.declare_parameter("model_variant", "s").value
        reid_model_file = self.declare_parameter(
            "reid_model_file", "osnet_ain_x1_0_p_unified_aug_n.onnx").value
        thresholds = ScoreThresholds(
            object_score=self.declare_parameter("object_score_threshold", 0.50).value)
        filters = CandidateFilters(
            min_bbox_height_px=self.declare_parameter("min_bbox_height_px", 100).value,
            min_bbox_area_px=self.declare_parameter("min_bbox_area_px", 2000).value,
            depth_min_mm=self.declare_parameter("depth_min_mm", 100).value,
            depth_max_mm=self.declare_parameter("depth_max_mm", 8000).value,
        )
        tracker = PersonTracker(
            iou_threshold=self.declare_parameter("iou_match_threshold", 0.3).value,
            max_lost_frames=self.declare_parameter("max_lost_frames", 30).value,
        )
        # Field calibration knob: lighting and clothing shift usable similarity.
        self._selector = TargetSelector(
            reid_similarity_threshold=self.declare_parameter("reid_similarity_threshold", 0.5).value,
            lost_timeout_sec=self.declare_parameter("lost_timeout_sec", 7.0).value,
            ema_alpha=self.declare_parameter("target_embedding_ema_alpha", 0.1).value,
        )
        detector = Wholebody49Detector(model_directory, variant, thresholds)
        encoder = PersonReidEncoder(model_directory, reid_model_file)
        self._pipeline = PersonTrackingPipeline(detector, encoder, tracker, self._selector, filters)
        self._minimum_interval_sec = 1.0 / self.declare_parameter("max_rate_hz", 15.0).value
        self._last_processed_stamp_sec: Optional[float] = None
        self._intrinsics: Optional[Intrinsics] = None
        self._bridge = CvBridge()

        self._tracks_publisher = self.create_publisher(Detection2DArray, "~/tracks", 10)
        self._target_publisher = self.create_publisher(Detection2D, "~/target", 10)
        self._debug_image_publisher = self.create_publisher(Image, "~/tracking_image", 1)
        self.create_service(Trigger, "~/release_target", self._on_release_target)
        self.create_subscription(
            CameraInfo, "camera_info", self._on_camera_info, qos_profile_sensor_data)
        image_subscriber = message_filters.Subscriber(
            self, Image, "image", qos_profile=qos_profile_sensor_data)
        depth_subscriber = message_filters.Subscriber(
            self, Image, "depth", qos_profile=qos_profile_sensor_data)
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [image_subscriber, depth_subscriber], queue_size=10,
            slop=self.declare_parameter("sync_slop_sec", 0.05).value)
        self._synchronizer.registerCallback(self._on_synchronized_images)

        provider = detector.active_provider
        if provider == "CUDAExecutionProvider":
            self.get_logger().info(f"Loaded variant {variant!r} on {provider}")
        else:
            self.get_logger().warning(f"GPU unavailable: variant {variant!r} is running on {provider}")

    def _on_camera_info(self, message: CameraInfo) -> None:
        self._intrinsics = Intrinsics(fx=message.k[0], fy=message.k[4], cx=message.k[2], cy=message.k[5])

    def _on_release_target(self, _request, response):
        self._selector.release()
        response.success = True
        response.message = "target released"
        return response

    def _on_synchronized_images(self, image_message: Image, depth_message: Image) -> None:
        stamp_sec = image_message.header.stamp.sec + image_message.header.stamp.nanosec * 1e-9
        if (self._last_processed_stamp_sec is not None
                and stamp_sec - self._last_processed_stamp_sec < self._minimum_interval_sec):
            return
        if depth_message.encoding != DEPTH_ENCODING:
            self.get_logger().error(
                f"Depth must be {DEPTH_ENCODING}, got {depth_message.encoding!r}",
                throttle_duration_sec=5.0)
            return
        try:
            image_bgr = self._bridge.imgmsg_to_cv2(image_message, "bgr8")
            depth_mm_image = self._bridge.imgmsg_to_cv2(depth_message, DEPTH_ENCODING)
        except CvBridgeError as error:
            self.get_logger().error(f"Cannot convert images: {error}", throttle_duration_sec=5.0)
            return
        self._last_processed_stamp_sec = stamp_sec

        if self._intrinsics is None:
            self.get_logger().warning("camera_info not received: X and Y are 0", once=True)
        result = self._pipeline.process(image_bgr, depth_mm_image, stamp_sec)
        header = image_message.header

        tracks_message = Detection2DArray()
        tracks_message.header = header
        tracks_message.detections = [
            to_person_message(person, header, self._intrinsics) for person in result.tracks]
        self._tracks_publisher.publish(tracks_message)
        if result.target is not None:
            self._target_publisher.publish(to_person_message(result.target, header, self._intrinsics))

        if self._debug_image_publisher.get_subscription_count() > 0:
            debug_message = self._bridge.cv2_to_imgmsg(
                draw_tracks(image_bgr, result.tracks, result.target), "bgr8")
            debug_message.header = header
            self._debug_image_publisher.publish(debug_message)


def main() -> None:
    rclpy.init()
    node = PersonTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
