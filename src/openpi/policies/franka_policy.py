"""Policy transforms for the dual-arm Franka Research 3 setup with end-effector control.

Raw data uses LeRobot-style keys:
- observation.state.end.position: [left end effector (N), right end effector (N)], N = 9 by default
  (x, y, z, roll_sin, roll_cos, pitch_sin, pitch_cos, yaw_sin, yaw_cos)
- observation.state.effector.position: [left gripper, right gripper]
- action.end.position / action.effector.position: same layout as the state

The model sees state and actions as [left ee (N), left gripper, right ee (N), right gripper] (20 dims for N = 9).
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy."""
    return {
        "observation.state.end.position": np.random.rand(18).astype(np.float32),
        "observation.state.effector.position": np.random.rand(2).astype(np.float32),
        "observation.images.camera_top": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation.images.camera_wrist_left": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation.images.camera_wrist_right": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def react_hold_actions(obs: dict, action_horizon: int) -> dict:
    """REACT warm-up: raw action keys for holding the current pose over `action_horizon` steps."""
    return {
        "action.end.position": np.tile(np.asarray(obs["observation.state.end.position"]), (action_horizon, 1)),
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


def _interleave(end: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    # [left ee, right ee] + [left gripper, right gripper] -> [left ee, left gripper, right ee, right gripper]
    arm_dof = end.shape[-1] // 2
    return np.concatenate(
        [end[..., :arm_dof], gripper[..., :1], end[..., arm_dof : 2 * arm_dof], gripper[..., 1:2]], axis=-1
    )


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """Converts Franka inputs to the model format. Used for both training and inference."""

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": _interleave(data["observation.state.end.position"], data["observation.state.effector.position"]),
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
        if "action.end.position" in data:
            inputs["actions"] = _interleave(data["action.end.position"], data["action.effector.position"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the Franka action format. Used for inference only."""

    # [left ee (9), left gripper, right ee (9), right gripper]
    action_dim: int = 20

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
