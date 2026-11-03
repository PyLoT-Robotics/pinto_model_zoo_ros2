"""Publish face boxes with cosine scores against a directory-backed face gallery."""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from pinto_model_zoo_ros2.face_recognition import FaceEncoder, FaceGallery, crop_face
from pinto_model_zoo_ros2.wholebody49 import Detection, ScoreThresholds, Wholebody49Detector

FACE_CLASS_NAME = "face"
DEBUG_BOX_COLOR_BGR = (0, 255, 0)


def to_face_message(
    detection: Detection, scores: Tuple[Tuple[str, float], ...], header
) -> Detection2D:
    message = Detection2D()
    message.header = header
    message.bbox.center.position.x = (detection.x1 + detection.x2) / 2.0
    message.bbox.center.position.y = (detection.y1 + detection.y2) / 2.0
    message.bbox.size_x = detection.x2 - detection.x1
    message.bbox.size_y = detection.y2 - detection.y1
    message.id = ""
    for name, score in scores:
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = name
        hypothesis.hypothesis.score = score
        message.results.append(hypothesis)
    return message


def draw_faces(image_bgr: np.ndarray, detections, scores_per_face) -> np.ndarray:
    canvas = image_bgr.copy()
    for detection, scores in zip(detections, scores_per_face):
        top_left = (int(detection.x1), int(detection.y1))
        bottom_right = (int(detection.x2), int(detection.y2))
        label = f"{scores[0][0]} {scores[0][1]:.2f}" if scores else FACE_CLASS_NAME
        cv2.rectangle(canvas, top_left, bottom_right, DEBUG_BOX_COLOR_BGR, 2)
        cv2.putText(canvas, label, top_left, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, DEBUG_BOX_COLOR_BGR, 1, cv2.LINE_AA)
    return canvas


class FaceRecognizerNode(Node):
    def __init__(self) -> None:
        super().__init__("face_recognizer")
        model_directory = Path(self.declare_parameter("model_directory", "/models").value)
        variant = self.declare_parameter("model_variant", "n").value
        face_model_file = self.declare_parameter("face_model_file", "LVFace-T_Glint360K.onnx").value
        face_score_threshold = self.declare_parameter("face_score_threshold", 0.50).value
        gallery_directory = self.declare_parameter("gallery_directory", "/faces").value
        rate_hz = self.declare_parameter("rate_hz", 5.0).value
        if rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be positive: {rate_hz}")

        self._detector = Wholebody49Detector(
            model_directory, variant, ScoreThresholds(object_score=face_score_threshold)
        )
        self._encoder = FaceEncoder(model_directory, face_model_file)
        self._gallery = FaceGallery(Path(gallery_directory), self._embed_gallery_image)
        self._bridge = CvBridge()
        self._latest_image: Optional[Image] = None
        self._faces_publisher = self.create_publisher(Detection2DArray, "~/faces", 10)
        self._debug_image_publisher = self.create_publisher(Image, "~/faces_image", 1)
        self.create_subscription(Image, "image", self._on_image, qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self._on_timer)
        self.get_logger().info(
            f"Detector on {self._detector.active_provider}, encoder on {self._encoder.active_provider}"
        )

    def _faces_in(self, image_bgr: np.ndarray) -> Tuple[Detection, ...]:
        return tuple(
            detection for detection in self._detector.detect(image_bgr)
            if detection.class_name == FACE_CLASS_NAME
        )

    def _embed_gallery_image(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        faces = sorted(self._faces_in(image_bgr),
                       key=lambda face: (face.x2 - face.x1) * (face.y2 - face.y1), reverse=True)
        face_image = next(
            (crop for crop in (crop_face(image_bgr, face) for face in faces) if crop is not None),
            None,
        )
        if face_image is None:
            self.get_logger().warning("No usable face in gallery image; using the whole image")
            face_image = image_bgr
        return self._encoder.encode([face_image])[0]

    def _on_image(self, image_message: Image) -> None:
        self._latest_image = image_message

    def _has_subscribers(self) -> bool:
        return (self._faces_publisher.get_subscription_count() > 0
                or self._debug_image_publisher.get_subscription_count() > 0)

    def _recognize(self, image_bgr: np.ndarray):
        if self._gallery.refresh():
            self.get_logger().info(f"Gallery: {list(self._gallery.names)}")
        crops = [(face, crop_face(image_bgr, face)) for face in self._faces_in(image_bgr)]
        faces = tuple(face for face, crop in crops if crop is not None)
        scores_per_face = [()] * len(faces)
        if faces and self._gallery.names:
            embeddings = self._encoder.encode([crop for _, crop in crops if crop is not None])
            scores_per_face = [tuple(scores) for scores in self._gallery.match(embeddings)]
        return faces, scores_per_face

    def _on_timer(self) -> None:
        image_message, self._latest_image = self._latest_image, None
        if image_message is None or not self._has_subscribers():
            return
        try:
            image_bgr = self._bridge.imgmsg_to_cv2(image_message, "bgr8")
        except CvBridgeError as error:
            self.get_logger().error(f"Cannot convert {image_message.encoding!r}: {error}",
                                    throttle_duration_sec=5.0)
            return
        try:
            faces, scores_per_face = self._recognize(image_bgr)
        except Exception as error:  # One bad frame or gallery file must not kill the node.
            self.get_logger().error(f"Dropped a cycle: {error!r}", throttle_duration_sec=5.0)
            return

        array_message = Detection2DArray()
        array_message.header = image_message.header
        array_message.detections = [
            to_face_message(face, scores, image_message.header)
            for face, scores in zip(faces, scores_per_face)
        ]
        self._faces_publisher.publish(array_message)

        if self._debug_image_publisher.get_subscription_count() > 0:
            debug_message = self._bridge.cv2_to_imgmsg(
                draw_faces(image_bgr, faces, scores_per_face), "bgr8")
            debug_message.header = image_message.header
            self._debug_image_publisher.publish(debug_message)


def main() -> None:
    rclpy.init()
    node = FaceRecognizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
