"""Policy transforms for the Acone / ARX X5 bimanual robot.

The robot has two 6-DoF arms with parallel grippers and three cameras (top, left wrist, right wrist). Raw data uses
LeRobot-style keys:
- observation.state.arm.position: [left arm (6), right arm (6)]
- observation.state.effector.position: [left gripper, right gripper]
- action.arm.position / action.effector.position: same layout as the state

The model sees state and actions as [left arm (6), left gripper, right arm (6), right gripper] (14 dims).
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_acone_example() -> dict:
    """Creates a random input example for the Acone policy."""
    return {
        "observation.state.arm.position": np.random.rand(12).astype(np.float32),
        "observation.state.effector.position": np.random.rand(2).astype(np.float32),
        "observation.images.camera_top": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.camera_wrist_left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.camera_wrist_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def react_hold_actions(obs: dict, action_horizon: int) -> dict:
    """REACT warm-up: raw action keys for holding the current pose over `action_horizon` steps."""
    return {
        "action.arm.position": np.tile(np.asarray(obs["observation.state.arm.position"]), (action_horizon, 1)),
        "action.effector.position": np.tile(
            np.asarray(obs["observation.state.effector.position"]), (action_horizon, 1)
        ),
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _interleave(arm: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    # [left arm (6), right arm (6)] + [left gripper, right gripper] -> [left arm, left gripper, right arm, right gripper]
    return np.concatenate([arm[..., :6], gripper[..., :1], arm[..., 6:12], gripper[..., 1:2]], axis=-1)


@dataclasses.dataclass(frozen=True)
class AconeInputs(transforms.DataTransformFn):
    """Converts Acone inputs to the model format. Used for both training and inference."""

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": _interleave(data["observation.state.arm.position"], data["observation.state.effector.position"]),
            "image": {
                "base_0_rgb": _parse_image(data["observation.images.camera_top"]),
                "left_wrist_0_rgb": _parse_image(data["observation.images.camera_wrist_left"]),
                "right_wrist_0_rgb": _parse_image(data["observation.images.camera_wrist_right"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training (and for the REACT warm-up).
        if "action.arm.position" in data:
            inputs["actions"] = _interleave(data["action.arm.position"], data["action.effector.position"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AconeOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the Acone action format. Used for inference only."""

    def __call__(self, data: dict) -> dict:
        # [left arm (6), left gripper, right arm (6), right gripper]
        return {"actions": np.asarray(data["actions"][:, :14])}
