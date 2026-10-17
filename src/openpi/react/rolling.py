"""REACT rolling denoising.

REACT keeps an H-step rolling action buffer x = [B_0, ..., B_{K-1}] of K blocks of S actions (H = K * S). Block k
sits at flow time tau_k = (k + 1) / K. Each control update runs one VLM pass on the latest observation and a single
Euler step of size 1/K on the whole buffer, so B_0 becomes clean and executable. The buffer is then shifted left by
one block and fresh noise is appended at the tail:

    x_tilde = x - v(x, tau, o) / K
    x <- [x_tilde_1, ..., x_tilde_{K-1}, eps]

Every emitted block is therefore refined under K consecutive observations. At the start of an episode the buffer is
initialized from the reset pose, B_k = (1 - tau_k) * s_0 + tau_k * eps, and K - 1 warm-up updates are run on the first
observation before the first block is emitted.
"""

from collections.abc import Callable
import logging
import time
from typing import Literal

import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi.policies import policy as _policy

logger = logging.getLogger(__name__)

# Observation key that clients can set (to any truthy value) to start a new episode on the same connection.
RESET_KEY = "react_reset"

# Maps a raw observation to the raw action keys of a "hold still" action chunk of length `action_horizon`.
HoldActionsFn = Callable[[dict, int], dict]


def hold_state_actions(obs: dict, action_horizon: int) -> dict:
    """Default `HoldActionsFn` for observations whose "state" uses the same layout as the actions (e.g. ALOHA)."""
    state = np.asarray(obs["state"])
    return {"actions": np.tile(state, (action_horizon, 1))}


class RollingBuffer:
    """The model-space REACT buffer x = [B_0, ..., B_{K-1}], shape (1, K * S, action_dim)."""

    def __init__(self, *, num_blocks: int, block_size: int, action_dim: int, device: torch.device | str):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.x: torch.Tensor | None = None

    @property
    def action_horizon(self) -> int:
        return self.num_blocks * self.block_size

    def reset(self, hold_actions: torch.Tensor | None = None) -> None:
        """Initialize the buffer for a new episode.

        With `hold_actions` (model-space, shape (1, H, action_dim)), block k is (1 - tau_k) * hold + tau_k * eps with
        tau_k = (k + 1) / K, i.e. the reset pose noised to the staircase. Otherwise the buffer is pure noise.
        """
        noise = self._noise(self.action_horizon)
        if hold_actions is None:
            self.x = noise
            return
        block_index = torch.arange(self.action_horizon, device=self.device) // self.block_size
        tau = ((block_index + 1).to(torch.float32) / self.num_blocks)[None, :, None]
        self.x = (1 - tau) * hold_actions.to(self.device, torch.float32) + tau * noise

    def roll(self, denoised: torch.Tensor) -> torch.Tensor:
        """Consume the output of one REACT update: return its executable front block and shift the buffer."""
        front = denoised[:, : self.block_size].clone()
        self.x = torch.cat([denoised[:, self.block_size :], self._noise(self.block_size)], dim=1)
        return front

    def _noise(self, length: int) -> torch.Tensor:
        return torch.randn(1, length, self.action_dim, dtype=torch.float32, device=self.device)


class ReactPolicy(_base_policy.BasePolicy):
    """Synchronous REACT inference ("REACT w/o DD" in the paper).

    Each `infer` call runs one VLM pass on `obs` and one REACT update, and returns the next block of S executable
    actions (`actions` has shape (S, robot_action_dim)). Clients should execute all S actions before calling again,
    e.g. by wrapping the policy in `openpi_client.action_chunk_broker.ActionChunkBroker(policy, action_horizon=S)`.

    The first call after `reset()` (or a call with `obs[RESET_KEY]` set) warms the buffer up on that observation.
    """

    def __init__(
        self,
        policy: _policy.Policy,
        *,
        warmup: Literal["hold", "noise"] = "hold",
        hold_actions_fn: HoldActionsFn = hold_state_actions,
    ):
        layout = policy.react_layout
        self._policy = policy
        self._warmup_mode = warmup
        self._hold_actions_fn = hold_actions_fn
        self._buffer = RollingBuffer(
            num_blocks=layout["num_blocks"],
            block_size=layout["block_size"],
            action_dim=layout["action_dim"],
            device=layout["device"],
        )
        self._num_updates = 0

    @property
    def block_size(self) -> int:
        return self._buffer.block_size

    @property
    def num_blocks(self) -> int:
        return self._buffer.num_blocks

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        obs = dict(obs)
        if obs.pop(RESET_KEY, False):
            self.reset()

        start_time = time.monotonic()
        prefix = self._policy.react_embed(obs)
        embed_time = time.monotonic()
        if self._buffer.x is None:
            self.warmup(obs, prefix)
        block = self._buffer.roll(self._policy.react_denoise(prefix, self._buffer.x))
        outputs = self._policy.react_decode(block, prefix)
        self._num_updates += 1
        end_time = time.monotonic()

        outputs["policy_timing"] = {
            "infer_ms": (end_time - start_time) * 1000,
            "vlm_ms": (embed_time - start_time) * 1000,
        }
        outputs["react"] = {"update": self._num_updates, "block_size": self.block_size}
        return outputs

    def warmup(self, obs: dict, prefix) -> None:
        """Initialize the buffer and run K - 1 updates on the first observation, discarding their front blocks."""
        hold = None
        if self._warmup_mode == "hold":
            hold = self._policy.react_encode_actions({**obs, **self._hold_actions_fn(obs, self._buffer.action_horizon)})
        self._buffer.reset(hold)
        for _ in range(self.num_blocks - 1):
            self._buffer.roll(self._policy.react_denoise(prefix, self._buffer.x))
        self._num_updates = 0
        logger.info("REACT buffer warmed up (%s, %d updates)", self._warmup_mode, self.num_blocks - 1)

    @override
    def reset(self) -> None:
        self._buffer.x = None

    @property
    def metadata(self) -> dict:
        return {
            **self._policy.metadata,
            "react": {"num_blocks": self.num_blocks, "block_size": self.block_size},
        }
