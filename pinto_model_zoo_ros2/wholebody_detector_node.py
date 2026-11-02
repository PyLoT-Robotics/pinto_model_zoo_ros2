"""Publish DEIMv2-Wholebody49 detections as vision_msgs/Detection2DArray."""

from pathlib import Path
from typing import Iterable

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from pinto_model_zoo_ros2.wholebody49 import (
    ATTRIBUTE_CLASS_IDS,
    KEYPOINT_CLASS_IDS,
    Detection,
    ScoreThresholds,
    Wholebody49Detector,
)

DEBUG_BOX_COLOR_BGR = (0, 255, 0)


def to_detection_message(detection: Detection, header) -> Detection2D:
    message = Detection2D()
    message.header = header
    message.bbox.center.position.x = (detection.x1 + detection.x2) / 2.0
    message.bbox.center.position.y = (detection.y1 + detection.y2) / 2.0
    message.bbox.size_x = detection.x2 - detection.x1
    message.bbox.size_y = detection.y2 - detection.y1
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = detection.class_name
    hypothesis.hypothesis.score = detection.score
    message.results.append(hypothesis)
    return message


def draw_detections(image_bgr, detections: Iterable[Detection]):
    canvas = image_bgr.copy()
    for detection in detections:
        top_left = (int(detection.x1), int(detection.y1))
        bottom_right = (int(detection.x2), int(detection.y2))
        if detection.class_id in KEYPOINT_CLASS_IDS:
            center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
            cv2.circle(canvas, center, 3, DEBUG_BOX_COLOR_BGR, -1)
        elif detection.class_id not in ATTRIBUTE_CLASS_IDS:
            cv2.rectangle(canvas, top_left, bottom_right, DEBUG_BOX_COLOR_BGR, 2)
            cv2.putText(canvas, detection.class_name, top_left, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, DEBUG_BOX_COLOR_BGR, 1, cv2.LINE_AA)
    return canvas


class WholebodyDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("wholebody_detector")
        model_directory = self.declare_parameter("model_directory", "/models").value
        variant = self.declare_parameter("model_variant", "s").value
        # Field calibration knobs: lighting at the venue shifts usable scores.
        thresholds = ScoreThresholds(
            object_score=self.declare_parameter("object_score_threshold", 0.50).value,
            attribute_score=self.declare_parameter("attribute_score_threshold", 0.75).value,
            keypoint_score=self.declare_parameter("keypoint_score_threshold", 0.50).value,
        )
        self._detector = Wholebody49Detector(Path(model_directory), variant, thresholds)
        self._bridge = CvBridge()
        self._detections_publisher = self.create_publisher(Detection2DArray, "detections", 10)
        self._debug_image_publisher = self.create_publisher(Image, "detections_image", 1)
        self.create_subscription(Image, "image", self._on_image, qos_profile_sensor_data)

        provider = self._detector.active_provider
        if provider == "CUDAExecutionProvider":
            self.get_logger().info(f"Loaded variant {variant!r} on {provider}")
        else:
            self.get_logger().warning(f"GPU unavailable: variant {variant!r} is running on {provider}")

    def _on_image(self, image_message: Image) -> None:
        try:
            image_bgr = self._bridge.imgmsg_to_cv2(image_message, "bgr8")
        except CvBridgeError as error:
            self.get_logger().error(f"Cannot convert {image_message.encoding!r}: {error}",
                                    throttle_duration_sec=5.0)
            return
        detections = self._detector.detect(image_bgr)

        array_message = Detection2DArray()
        array_message.header = image_message.header
        array_message.detections = [
            to_detection_message(detection, image_message.header) for detection in detections
        ]
        self._detections_publisher.publish(array_message)

        if self._debug_image_publisher.get_subscription_count() > 0:
            debug_message = self._bridge.cv2_to_imgmsg(draw_detections(image_bgr, detections), "bgr8")
            debug_message.header = image_message.header
            self._debug_image_publisher.publish(debug_message)


def main() -> None:
    rclpy.init()
    node = WholebodyDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
