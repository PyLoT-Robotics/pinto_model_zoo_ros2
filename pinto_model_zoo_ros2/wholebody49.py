"""DEIMv2-Wholebody49 inference without any ROS dependency."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

CLASS_NAMES = (
    "body", "adult", "child", "male", "female",
    "body_with_wheelchair", "body_with_crutches",
    "head", "front", "right_front", "right_side", "right_back",
    "back", "left_back", "left_side", "left_front",
    "face", "eye", "nose", "mouth", "ear",
    "collarbone", "shoulder", "shoulder_left", "shoulder_right", "solar_plexus",
    "elbow", "elbow_left", "elbow_right", "wrist", "wrist_left", "wrist_right",
    "hand", "hand_left", "hand_right", "abdomen",
    "hip_joint", "hip_joint_left", "hip_joint_right",
    "knee", "knee_left", "knee_right", "ankle", "ankle_left", "ankle_right",
    "foot", "foot_left", "foot_right", "bone",
)
ATTRIBUTE_CLASS_IDS = frozenset({1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15})
KEYPOINT_CLASS_IDS = frozenset(
    {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44}
)
BONE_CLASS_ID = 48
KEYPOINT_NMS_IOU = 0.20
INPUT_SIZE = 640

MODEL_FILE_NAMES = {
    "n": "deimv2_hgnetv2_n_wholebody49_ins_s08_maskhead64x2_center_1240query.onnx",
    "s": "deimv2_dinov3_s_wholebody49_ins_s08_maskhead256x3_center_1240query.onnx",
    "x": "deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query.onnx",
}
# The upstream demo normalizes DINOv3 variants only.
VARIANTS_WITH_IMAGENET_NORMALIZATION = frozenset({"s", "x"})
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]


@dataclass(frozen=True)
class ScoreThresholds:
    """Defaults match the upstream demo."""

    object_score: float = 0.50
    attribute_score: float = 0.75
    keypoint_score: float = 0.50

    def for_class(self, class_id: int) -> float:
        if class_id in ATTRIBUTE_CLASS_IDS:
            return self.attribute_score
        if class_id in KEYPOINT_CLASS_IDS:
            return self.keypoint_score
        return self.object_score


def preprocess(image_bgr: np.ndarray, should_normalize: bool) -> np.ndarray:
    resized = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    channels_first = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    if should_normalize:
        channels_first = (channels_first - IMAGENET_MEAN) / IMAGENET_STD
    return channels_first[np.newaxis]


def intersection_over_union(first: Detection, second: Detection) -> float:
    width = min(first.x2, second.x2) - max(first.x1, second.x1)
    height = min(first.y2, second.y2) - max(first.y1, second.y1)
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
    second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
    return intersection / (first_area + second_area - intersection)


def _suppress_duplicate_keypoints(detections: List[Detection]) -> List[Detection]:
    kept: List[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        is_duplicate = detection.class_id in KEYPOINT_CLASS_IDS and any(
            other.class_id == detection.class_id
            and intersection_over_union(detection, other) > KEYPOINT_NMS_IOU
            for other in kept
        )
        if not is_duplicate:
            kept.append(detection)
    return kept


def decode(
    label_xyxy_score: np.ndarray,
    image_width: int,
    image_height: int,
    thresholds: ScoreThresholds,
) -> Tuple[Detection, ...]:
    """Convert one batch item of shape (queries, 6) with normalized boxes to pixels."""
    detections: List[Detection] = []
    for label, x1, y1, x2, y2, score in label_xyxy_score:
        class_id = int(label)
        if class_id == BONE_CLASS_ID or not 0 <= class_id < len(CLASS_NAMES):
            continue
        if score < thresholds.for_class(class_id):
            continue
        left = float(np.clip(x1, 0.0, 1.0)) * image_width
        top = float(np.clip(y1, 0.0, 1.0)) * image_height
        right = float(np.clip(x2, 0.0, 1.0)) * image_width
        bottom = float(np.clip(y2, 0.0, 1.0)) * image_height
        if right <= left or bottom <= top:
            continue
        detections.append(Detection(class_id, float(score), left, top, right, bottom))
    return tuple(_suppress_duplicate_keypoints(detections))


class Wholebody49Detector:
    def __init__(self, model_directory: Path, variant: str, thresholds: ScoreThresholds) -> None:
        if variant not in MODEL_FILE_NAMES:
            raise ValueError(f"model_variant must be one of {sorted(MODEL_FILE_NAMES)}: {variant!r}")
        model_path = Path(model_directory) / MODEL_FILE_NAMES[variant]
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}. Run scripts/download_models.py.")

        import onnxruntime  # Deferred so decode() and its tests need no GPU packages.

        onnxruntime.preload_dlls()
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._should_normalize = variant in VARIANTS_WITH_IMAGENET_NORMALIZATION
        self._thresholds = thresholds

    @property
    def active_provider(self) -> str:
        return self._session.get_providers()[0]

    def detect(self, image_bgr: np.ndarray) -> Tuple[Detection, ...]:
        height, width = image_bgr.shape[:2]
        input_tensor = preprocess(image_bgr, self._should_normalize)
        label_xyxy_score = self._session.run(None, {self._input_name: input_tensor})[0][0]
        return decode(label_xyxy_score, width, height, self._thresholds)
