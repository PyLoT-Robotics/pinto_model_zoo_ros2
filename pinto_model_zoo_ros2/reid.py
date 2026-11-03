"""OSNet-AIN person re-identification without any ROS dependency."""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from pinto_model_zoo_ros2.wholebody49 import IMAGENET_MEAN, IMAGENET_STD

REID_INPUT_HEIGHT, REID_INPUT_WIDTH = 256, 128
EMBEDDING_DIMENSION = 512
ZERO_NORM_EPSILON = 1e-12


def preprocess_crops(crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
    """Return a (N, 3, 256, 128) float32 RGB batch normalized with ImageNet mean/std."""
    resized_crops = [
        cv2.cvtColor(
            cv2.resize(crop, (REID_INPUT_WIDTH, REID_INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB,
        )
        for crop in crops_bgr
    ]
    if not resized_crops:
        return np.empty((0, 3, REID_INPUT_HEIGHT, REID_INPUT_WIDTH), dtype=np.float32)
    batch = np.stack(resized_crops).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return (batch - IMAGENET_MEAN) / IMAGENET_STD


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize the last axis to unit length; zero vectors stay zero instead of becoming NaN."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, ZERO_NORM_EPSILON)


def cosine_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query (D,) against a gallery (M, D); returns (M,)."""
    return l2_normalize(gallery) @ l2_normalize(query)


class PersonReidEncoder:
    def __init__(self, model_directory: Path, file_name: str) -> None:
        model_path = Path(model_directory) / file_name
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}. Run scripts/download_models.py.")

        import onnxruntime  # Deferred so the pure functions and their tests need no GPU packages.

        onnxruntime.preload_dlls()
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    @property
    def active_provider(self) -> str:
        return self._session.get_providers()[0]

    def encode(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """Return L2-normalized embeddings of shape (N, 512)."""
        if not crops_bgr:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        input_tensor = preprocess_crops(crops_bgr)
        embeddings = self._session.run(None, {self._input_name: input_tensor})[0]
        return l2_normalize(embeddings).astype(np.float32)
