import os

import cv2
import numpy as np

from pinto_model_zoo_ros2.face_recognition import FaceGallery, crop_face, preprocess_faces
from pinto_model_zoo_ros2.wholebody49 import Detection


def write_image(path, value):
    cv2.imwrite(str(path), np.full((20, 30, 3), value, dtype=np.uint8))


def embed_by_brightness(image):
    return np.array([float(image.mean()), 1.0], dtype=np.float32)


def advance_mtime(path, seconds=10):
    status = path.stat()
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + seconds * 10**9))


def test_preprocess_faces_shape_range_and_rgb_order():
    bgr = np.zeros((50, 40, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue in BGR
    batch = preprocess_faces([bgr, bgr])
    assert batch.shape == (2, 3, 112, 112) and batch.dtype == np.float32
    assert batch[0, 2].min() == 1.0  # RGB: blue is the last channel, (255/255-0.5)/0.5
    assert batch[0, 0].max() == -1.0


def test_gallery_uses_stem_and_detects_add_overwrite_remove(tmp_path):
    gallery = FaceGallery(tmp_path, embed_by_brightness)
    assert gallery.refresh() is False and gallery.match(np.ones((1, 2))) == []

    write_image(tmp_path / "alice.jpg", 100)
    assert gallery.refresh() is True and gallery.names == ("alice",)
    assert gallery.refresh() is False

    before = gallery.embeddings.copy()
    write_image(tmp_path / "alice.jpg", 200)
    advance_mtime(tmp_path / "alice.jpg")
    assert gallery.refresh() is True
    assert gallery.embeddings[0, 0] != before[0, 0]

    (tmp_path / "alice.jpg").unlink()
    assert gallery.refresh() is True and gallery.names == ()


def test_gallery_retries_unreadable_file_on_next_refresh(tmp_path):
    (tmp_path / "bob.png").write_bytes(b"partial")
    gallery = FaceGallery(tmp_path, embed_by_brightness)
    assert gallery.refresh() is False and gallery.names == ()

    write_image(tmp_path / "bob.png", 50)
    assert gallery.refresh() is True and gallery.names == ("bob",)


def test_gallery_retries_when_embed_returns_none(tmp_path):
    write_image(tmp_path / "carol.jpg", 80)
    results = iter([None, np.array([1.0, 0.0], dtype=np.float32)])
    gallery = FaceGallery(tmp_path, lambda image: next(results))
    assert gallery.refresh() is False
    assert gallery.refresh() is True and gallery.names == ("carol",)


def test_match_returns_all_names_best_first_per_face(tmp_path):
    write_image(tmp_path / "a.jpg", 10)
    write_image(tmp_path / "b.jpg", 10)
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    names = iter(["a", "b"])
    gallery = FaceGallery(tmp_path, lambda image: np.array(vectors[next(names)], dtype=np.float32))
    gallery.refresh()
    result = gallery.match(np.array([[0.2, 0.9], [1.0, 0.0]], dtype=np.float32))
    assert [name for name, _ in result[0]] == ["b", "a"]
    assert [name for name, _ in result[1]] == ["a", "b"]
    assert result[1][0][1] == 1.0


def test_crop_face_rounds_outward_and_drops_empty_boxes():
    image = np.zeros((50, 40, 3), dtype=np.uint8)
    sub_pixel = Detection(16, 0.9, 10.2, 5.0, 10.8, 20.0)
    assert crop_face(image, sub_pixel).shape == (15, 1, 3)  # ceil/floor keeps one column
    outside = Detection(16, 0.9, 45.0, 5.0, 46.0, 20.0)
    assert crop_face(image, outside) is None
    preprocess_faces([crop_face(image, sub_pixel)])  # used to raise cv2.error on empty crops


def test_preprocess_faces_maps_midpoint_value():
    batch = preprocess_faces([np.full((8, 8, 3), 128, dtype=np.uint8)])
    assert np.allclose(batch, (128 / 255.0 - 0.5) / 0.5)


def test_newer_of_jpg_and_png_with_same_stem_wins(tmp_path):
    write_image(tmp_path / "dan.png", 10)
    write_image(tmp_path / "dan.jpg", 200)
    advance_mtime(tmp_path / "dan.jpg")
    gallery = FaceGallery(tmp_path, embed_by_brightness)
    gallery.refresh()
    assert gallery.embeddings[0, 0] > 150


def test_overwrite_unreadable_keeps_previous_embedding_until_readable(tmp_path):
    write_image(tmp_path / "eve.jpg", 100)
    gallery = FaceGallery(tmp_path, embed_by_brightness)
    gallery.refresh()
    old_value = gallery.embeddings[0, 0]

    (tmp_path / "eve.jpg").write_bytes(b"partial")  # overwrite in progress
    advance_mtime(tmp_path / "eve.jpg")
    assert gallery.refresh() is False
    assert gallery.names == ("eve",) and gallery.embeddings[0, 0] == old_value

    write_image(tmp_path / "eve.jpg", 220)
    advance_mtime(tmp_path / "eve.jpg", seconds=20)
    assert gallery.refresh() is True and gallery.embeddings[0, 0] > old_value
