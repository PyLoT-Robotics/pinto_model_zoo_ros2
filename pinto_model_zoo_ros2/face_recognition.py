"""LVFace embeddings and a directory-backed face gallery without any ROS dependency."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

FACE_INPUT_SIZE = 112
GALLERY_SUFFIXES = (".jpg", ".png")

FileSignature = Tuple[int, int]  # (mtime_ns, size)


def preprocess_faces(faces_bgr: Sequence[np.ndarray]) -> np.ndarray:
    """Resize to 112x112, convert to RGB and scale to [-1, 1] (upstream demo without landmarks)."""
    resized = [
        cv2.cvtColor(
            cv2.resize(face, (FACE_INPUT_SIZE, FACE_INPUT_SIZE), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB,
        )
        for face in faces_bgr
    ]
    batch = np.stack(resized).astype(np.float32).transpose(0, 3, 1, 2)
    return (batch / 255.0 - 0.5) / 0.5


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


class FaceEncoder:
    def __init__(self, model_directory: Path, file_name: str) -> None:
        model_path = Path(model_directory) / file_name
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}. Run scripts/download_models.py.")

        import onnxruntime  # Deferred so preprocessing and its tests need no GPU packages.

        onnxruntime.preload_dlls()
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    @property
    def active_provider(self) -> str:
        return self._session.get_providers()[0]

    def encode(self, faces_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """Return L2-normalized embeddings of shape (N, D)."""
        features = self._session.run(None, {self._input_name: preprocess_faces(faces_bgr)})[0]
        return l2_normalize(features)


class FaceGallery:
    """Registers every *.jpg|*.png in a directory; the file stem is the person's name."""

    def __init__(
        self, directory: Path, embed: Callable[[np.ndarray], Optional[np.ndarray]]
    ) -> None:
        self._directory = Path(directory)
        self._embed = embed
        self._signatures: Dict[str, FileSignature] = {}
        self._embedding_by_name: Dict[str, np.ndarray] = {}
        self.names: Tuple[str, ...] = ()
        self.embeddings = np.empty((0, 0), dtype=np.float32)

    def refresh(self) -> bool:
        """Re-read changed, added or removed files. Return True when the gallery changed."""
        current = self._scan()
        removed = set(self._signatures) - set(current)
        changed = [name for name, (_, signature) in current.items()
                   if self._signatures.get(name) != signature]
        is_updated = bool(removed)
        for name in removed:
            del self._signatures[name]
            del self._embedding_by_name[name]
        for name in changed:
            path, signature = current[name]
            embedding = self._embed_file(path)
            if embedding is None:
                # Not recorded, so the next refresh retries (e.g. file still being written).
                continue
            self._signatures[name] = signature
            self._embedding_by_name[name] = embedding
            is_updated = True
        if is_updated:
            self._rebuild()
        return is_updated

    def match(self, face_embeddings: np.ndarray) -> List[List[Tuple[str, float]]]:
        """For each face, return (name, cosine) over the whole gallery, best first."""
        if not self.names:
            return []
        scores = np.asarray(face_embeddings) @ self.embeddings.T
        return [
            sorted(zip(self.names, map(float, row)), key=lambda pair: pair[1], reverse=True)
            for row in scores
        ]

    def _scan(self) -> Dict[str, Tuple[Path, FileSignature]]:
        files: Dict[str, Tuple[Path, FileSignature]] = {}
        if not self._directory.is_dir():
            return files
        for path in sorted(self._directory.iterdir()):
            if path.suffix.lower() in GALLERY_SUFFIXES and path.is_file():
                status = path.stat()
                # Same stem with two suffixes: the later one in sort order wins.
                files[path.stem] = (path, (status.st_mtime_ns, status.st_size))
        return files

    def _embed_file(self, path: Path) -> Optional[np.ndarray]:
        image = cv2.imread(str(path))
        return None if image is None else self._embed(image)

    def _rebuild(self) -> None:
        self.names = tuple(sorted(self._embedding_by_name))
        if self.names:
            self.embeddings = np.stack([self._embedding_by_name[name] for name in self.names])
        else:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
