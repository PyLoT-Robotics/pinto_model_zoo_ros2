import numpy as np
import pytest

from pinto_model_zoo_ros2.reid import (
    PersonReidEncoder,
    cosine_similarity,
    l2_normalize,
    preprocess_crops,
)


def test_preprocess_crops_shape_channel_order_and_normalization():
    blue_crop = np.zeros((40, 30, 3), dtype=np.uint8)
    blue_crop[..., 0] = 255
    batch = preprocess_crops([blue_crop, blue_crop])
    assert batch.shape == (2, 3, 256, 128) and batch.dtype == np.float32
    assert abs(batch[0, 2, 0, 0] - (1.0 - 0.406) / 0.225) < 1e-5
    assert abs(batch[0, 0, 0, 0] - (0.0 - 0.485) / 0.229) < 1e-5


def test_preprocess_crops_empty_gives_empty_batch():
    assert preprocess_crops([]).shape == (0, 3, 256, 128)


def test_l2_normalize_keeps_zero_vector_finite():
    normalized = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
    assert np.all(np.isfinite(normalized))
    assert np.allclose(normalized, [[0.0, 0.0], [0.6, 0.8]])


def test_cosine_similarity_matches_dot_product_of_normalized_vectors():
    rng = np.random.default_rng(0)
    query = rng.normal(size=8).astype(np.float32)
    gallery = rng.normal(size=(5, 8)).astype(np.float32)
    expected = l2_normalize(gallery) @ l2_normalize(query)
    assert np.allclose(cosine_similarity(query, gallery), expected)
    assert np.allclose(cosine_similarity(query, query[np.newaxis]), [1.0])


def test_encoder_reports_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        PersonReidEncoder(tmp_path, "missing.onnx")
