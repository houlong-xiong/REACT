from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models_pytorch import pi0_pytorch as _pi0_pytorch
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    # REACT helpers (PyTorch only). The rolling buffer and scheduling live in `openpi.react`; these methods only
    # move data between the robot format and the model space (normalized, padded actions).

    @property
    def react_layout(self) -> dict[str, Any]:
        """Rolling-buffer layout of a REACT model: num_blocks (K), block_size (S), action_dim and device."""
        self._check_react()
        config = self._model.config
        return {
            "num_blocks": config.react_num_blocks,
            "block_size": config.react_block_size,
            "action_dim": config.action_dim,
            "device": self._pytorch_device,
        }

    def react_embed(self, obs: dict) -> _pi0_pytorch.PrefixCache:
        """Transform `obs` and run the VLM prefix pass. The result conditions `react_denoise`."""
        self._check_react()
        inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
        return self._model.embed_observation(_model.Observation.from_dict(inputs))

    def react_denoise(self, prefix: _pi0_pytorch.PrefixCache, x_t: torch.Tensor) -> torch.Tensor:
        """Apply one REACT update (a single Euler step) to the model-space rolling buffer `x_t`."""
        self._check_react()
        return self._model.react_denoise_step(prefix, x_t)

    def react_decode(self, actions: torch.Tensor, prefix: _pi0_pytorch.PrefixCache) -> dict:
        """Convert model-space `actions` (1, n, action_dim) to the robot format with the output transforms.

        The state of the observation behind `prefix` is used by state-dependent transforms (e.g. delta actions).
        """
        outputs = {"state": prefix.state, "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        return self._output_transform(outputs)

    def react_encode_actions(self, obs: dict) -> torch.Tensor:
        """Convert the raw action chunk contained in `obs` to model space, shape (1, action_horizon, action_dim).

        `obs` must contain the action keys expected by the input transforms (e.g. "actions"). REACT uses this to
        initialize the rolling buffer from the robot's reset pose.
        """
        inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
        if "actions" not in inputs:
            raise ValueError("The input transforms did not produce 'actions'; add the raw action keys to `obs`.")
        return torch.from_numpy(np.asarray(inputs["actions"], dtype=np.float32)).to(self._pytorch_device)[None, ...]

    def _check_react(self) -> None:
        if not self._is_pytorch_model:
            raise NotImplementedError("REACT inference is only supported for PyTorch models.")
        if not getattr(self._model.config, "react_enabled", False):
            raise ValueError("The model was not trained with REACT (react_enabled=False).")

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
