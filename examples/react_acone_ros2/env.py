"""ROS2 environment for the Acone / ARX X5 bimanual robot.

Subscribes to the three camera topics and the two follower-arm state topics, and drives the robot by publishing joint
targets on the leader-arm command topics (`arx5_arm_msg/RobotStatus`). A background thread re-publishes the latest
target at `send_hz`, which the ARX driver expects.

Requires a sourced ROS2 workspace with `arx5_arm_msg` (e.g. `source /opt/ros/humble/setup.bash` and the ARX X5
workspace's `install/setup.bash`).
"""

import dataclasses
import logging
import threading
import time

from arx5_arm_msg.msg import RobotStatus
import cv2
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image
from typing_extensions import override

logger = logging.getLogger(__name__)

# Joint targets per arm: 6 joints + 1 gripper.
JOINTS_PER_ARM = 7


@dataclasses.dataclass(frozen=True)
class Topics:
    camera_top: str = "/camera/camera_h/color/image_rect_raw"
    camera_wrist_left: str = "/camera/camera_l/color/image_rect_raw"
    camera_wrist_right: str = "/camera/camera_r/color/image_rect_raw"
    left_arm_state: str = "/arm_slave_l_status"
    right_arm_state: str = "/arm_slave_r_status"
    left_arm_command: str = "/arm_master_l_status"
    right_arm_command: str = "/arm_master_r_status"


def _image_to_rgb(msg: Image) -> np.ndarray:
    channels = 1 if msg.encoding == "mono8" else 3
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    if msg.encoding == "bgr8":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


class AconeRos2Environment(_environment.Environment):
    """`openpi_client` environment backed by ROS2 topics. Safe to use from several threads."""

    def __init__(self, prompt: str, *, topics: Topics | None = None, send_hz: float = 150.0, timeout_s: float = 30.0):
        topics = topics or Topics()
        if not rclpy.ok():
            rclpy.init()
        self._prompt = prompt
        self._send_period = 1.0 / send_hz
        self._timeout_s = timeout_s
        self._node = Node("openpi_react_acone")

        # Callbacks replace these references atomically; readers take a snapshot of the references.
        self._images: dict[str, np.ndarray | None] = {"top": None, "wrist_left": None, "wrist_right": None}
        self._joints: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._command: np.ndarray | None = None
        self._command_lock = threading.Lock()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        for name, topic in (
            ("top", topics.camera_top),
            ("wrist_left", topics.camera_wrist_left),
            ("wrist_right", topics.camera_wrist_right),
        ):
            self._node.create_subscription(Image, topic, self._image_callback(name), qos)
        for side, topic in (("left", topics.left_arm_state), ("right", topics.right_arm_state)):
            self._node.create_subscription(RobotStatus, topic, self._joint_callback(side), qos)
        self._publishers = {
            "left": self._node.create_publisher(RobotStatus, topics.left_arm_command, qos),
            "right": self._node.create_publisher(RobotStatus, topics.right_arm_command, qos),
        }

        self._running = True
        self._threads = [
            threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True),
            threading.Thread(target=self._send_loop, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _image_callback(self, name: str):
        def callback(msg: Image) -> None:
            self._images[name] = _image_to_rgb(msg)

        return callback

    def _joint_callback(self, side: str):
        def callback(msg: RobotStatus) -> None:
            self._joints[side] = np.asarray(msg.joint_pos[:JOINTS_PER_ARM], dtype=np.float32)

        return callback

    def _send_loop(self) -> None:
        while self._running:
            with self._command_lock:
                command = self._command
            if command is not None:
                for side, joints in (("left", command[:JOINTS_PER_ARM]), ("right", command[JOINTS_PER_ARM:])):
                    msg = RobotStatus()
                    msg.joint_pos = joints.astype(np.float64).tolist()
                    self._publishers[side].publish(msg)
            time.sleep(self._send_period)

    def _wait_for_sensors(self) -> None:
        deadline = time.monotonic() + self._timeout_s
        while any(v is None for v in (*self._images.values(), *self._joints.values())):
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for camera and arm state topics.")
            time.sleep(0.1)

    @override
    def reset(self) -> None:
        # The robot is reset by the operator; stop commanding and wait until all sensors are streaming.
        with self._command_lock:
            self._command = None
        self._wait_for_sensors()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        images = dict(self._images)
        left, right = self._joints["left"], self._joints["right"]

        def process(image: np.ndarray) -> np.ndarray:
            image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
            return np.transpose(image, (2, 0, 1))

        return {
            "observation.state.arm.position": np.concatenate([left[:6], right[:6]]),
            "observation.state.effector.position": np.array([left[6], right[6]], dtype=np.float32),
            "observation.images.camera_top": process(images["top"]),
            "observation.images.camera_wrist_left": process(images["wrist_left"]),
            "observation.images.camera_wrist_right": process(images["wrist_right"]),
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        # [left arm (6), left gripper, right arm (6), right gripper]
        command = np.asarray(action["actions"], dtype=np.float32)
        with self._command_lock:
            self._command = command

    def close(self) -> None:
        self._running = False
        self._node.destroy_node()
