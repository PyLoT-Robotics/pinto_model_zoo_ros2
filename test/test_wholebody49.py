import numpy as np

from pinto_model_zoo_ros2.wholebody49 import (
    CLASS_NAMES,
    MODEL_FILE_NAMES,
    ScoreThresholds,
    decode,
    preprocess,
)

BODY, MALE, SHOULDER, BONE = 0, 3, 22, 48


def row(class_id, score, x1=0.1, y1=0.1, x2=0.5, y2=0.9):
    return [class_id, x1, y1, x2, y2, score]


def run_decode(rows):
    return decode(np.array(rows, dtype=np.float32), 200, 100, ScoreThresholds())


def test_class_table_matches_upstream():
    assert len(CLASS_NAMES) == 49
    assert CLASS_NAMES[BODY] == "body" and CLASS_NAMES[BONE] == "bone"


def test_boxes_are_scaled_to_pixels_and_clipped():
    detection = run_decode([row(BODY, 0.9, x1=-0.1, x2=0.5)])[0]
    assert detection.class_name == "body"
    assert (detection.x1, detection.y2) == (0.0, np.float32(0.9).item() * 100)
    assert detection.x2 == 100.0


def test_attributes_use_stricter_threshold_than_objects():
    assert [d.class_id for d in run_decode([row(BODY, 0.6), row(MALE, 0.6)])] == [BODY]
    assert len(run_decode([row(MALE, 0.8)])) == 1


def test_bone_and_invalid_labels_and_empty_boxes_are_dropped():
    assert run_decode([row(BONE, 0.99), row(99, 0.99), row(BODY, 0.99, x1=0.5, x2=0.5)]) == ()


def test_only_duplicate_keypoints_are_suppressed():
    keypoints = run_decode([row(SHOULDER, 0.9), row(SHOULDER, 0.7)])
    assert [d.score for d in keypoints] == [np.float32(0.9).item()]
    assert len(run_decode([row(BODY, 0.9), row(BODY, 0.7)])) == 2


def test_preprocess_shape_channel_order_and_normalization():
    blue_image = np.zeros((48, 64, 3), dtype=np.uint8)
    blue_image[..., 0] = 255
    raw = preprocess(blue_image, should_normalize=False)
    assert raw.shape == (1, 3, 640, 640)
    assert raw[0, 2].min() == 1.0 and raw[0, 0].max() == 0.0
    normalized = preprocess(blue_image, should_normalize=True)
    assert abs(normalized[0, 0, 0, 0] - (0.0 - 0.485) / 0.229) < 1e-5


def test_every_variant_has_a_model():
    assert set(MODEL_FILE_NAMES) == {"n", "s", "x"}
