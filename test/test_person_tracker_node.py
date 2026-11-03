import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("vision_msgs")
pytest.importorskip("cv_bridge")
pytest.importorskip("message_filters")

from std_msgs.msg import Header

from pinto_model_zoo_ros2.person_tracker_node import should_process_frame, to_person_message
from pinto_model_zoo_ros2.person_tracking import Intrinsics, TrackedPerson
from pinto_model_zoo_ros2.wholebody49 import Detection


def make_person():
    box = Detection(class_id=0, score=0.9, x1=300.0, y1=100.0, x2=340.0, y2=300.0)
    return TrackedPerson(track_id=7, box=box, depth_mm=2000.0, embedding=np.ones(4))


def test_to_person_message_projects_center_to_camera_coordinates():
    header = Header(frame_id="camera_color_optical_frame")
    message = to_person_message(make_person(), header, Intrinsics(fx=400.0, fy=400.0, cx=320.0, cy=240.0))
    position = message.results[0].pose.pose.position
    assert message.id == "7" and message.header.frame_id == "camera_color_optical_frame"
    assert message.results[0].hypothesis.class_id == "body"
    assert position.z == pytest.approx(2.0)
    assert position.x == pytest.approx(0.0)
    assert position.y == pytest.approx((200.0 - 240.0) * 2.0 / 400.0)
    assert message.bbox.size_x == 40.0 and message.bbox.size_y == 200.0


def test_to_person_message_without_intrinsics_keeps_only_depth():
    position = to_person_message(make_person(), Header(), None).results[0].pose.pose.position
    assert (position.x, position.y, position.z) == (0.0, 0.0, 2.0)


def test_should_process_frame_applies_rate_limit_and_resets_on_backwards_stamp():
    assert should_process_frame(10.0, None, 0.1)
    assert not should_process_frame(10.05, 10.0, 0.1)
    assert should_process_frame(10.25, 10.0, 0.1)
    assert should_process_frame(3.0, 10.0, 0.1)
