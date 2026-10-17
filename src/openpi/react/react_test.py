import dataclasses

import numpy as np
from openpi_client.runtime import environment as _environment
import pytest
import torch
from torch import nn

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.policies import acone_policy
from openpi.policies import franka_policy
from openpi.policies import policy as _policy
from openpi.react import dual_decoupling
from openpi.react import rolling
from openpi.training import config as _config

ACTION_HORIZON = 10
NUM_BLOCKS = 5
BLOCK_SIZE = ACTION_HORIZON // NUM_BLOCKS
MAX_TOKEN_LEN = 16


def make_model(**overrides) -> pi0_pytorch.PI0Pytorch:
    """A tiny, deterministic REACT model for CPU tests."""
    kwargs = {
        "pi05": True,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "action_horizon": ACTION_HORIZON,
        "max_token_len": MAX_TOKEN_LEN,
        "react_enabled": True,
        "react_num_blocks": NUM_BLOCKS,
        "pytorch_compile_mode": None,
        "dtype": "float32",
    }
    kwargs.update(overrides)
    torch.manual_seed(0)
    model = pi0_pytorch.PI0Pytorch(pi0_config.Pi0Config(**kwargs))
    # The dummy Gemma variant keeps the full SigLIP tower; shrink it and match the projector to the dummy width.
    paligemma = model.paligemma_with_expert.paligemma
    encoder = paligemma.model.vision_tower.vision_model.encoder
    encoder.layers = encoder.layers[:1]
    projector = paligemma.model.multi_modal_projector
    projector.linear = nn.Linear(projector.linear.in_features, _gemma.get_config(kwargs["paligemma_variant"]).width)
    return model.eval()


def make_observation(batch_size: int = 1, seed: int = 0) -> _model.Observation:
    gen = torch.Generator().manual_seed(seed)
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    return _model.Observation(
        images={k: torch.rand(batch_size, 3, 224, 224, generator=gen) * 2 - 1 for k in keys},
        image_masks={k: torch.ones(batch_size, dtype=torch.bool) for k in keys},
        state=torch.rand(batch_size, 32, generator=gen),
        tokenized_prompt=torch.randint(0, 1000, (batch_size, MAX_TOKEN_LEN), generator=gen),
        tokenized_prompt_mask=torch.ones(batch_size, MAX_TOKEN_LEN, dtype=torch.bool),
    )


def make_obs_dict(seed: int = 0) -> dict:
    """A raw observation that is already in model format (the test policy has no transforms)."""
    rng = np.random.default_rng(seed)
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    return {
        "image": {k: rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8) for k in keys},
        "image_mask": {k: np.True_ for k in keys},
        "state": rng.random(32, dtype=np.float32),
        "tokenized_prompt": rng.integers(0, 1000, size=(MAX_TOKEN_LEN,), dtype=np.int32),
        "tokenized_prompt_mask": np.ones(MAX_TOKEN_LEN, dtype=bool),
    }


def make_policy(device: str = "cpu") -> _policy.Policy:
    return _policy.Policy(make_model(), is_pytorch=True, pytorch_device=device)


def test_config_validation():
    with pytest.raises(ValueError, match="divisible"):
        pi0_config.Pi0Config(react_enabled=True, action_horizon=50, react_num_blocks=7)
    with pytest.raises(ValueError, match="react_enabled"):
        pi0_config.Pi0Config(react_block_causal_mask=True)
    assert _config.get_config("pi05_react_aloha").model.react_block_size == 10


def test_react_time_is_a_staircase():
    model = make_model()
    time = model.react_time(2, "cpu")
    expected = torch.tensor([0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0])
    assert time.shape == (2, ACTION_HORIZON)
    torch.testing.assert_close(time[0], expected)
    torch.testing.assert_close(time[1], expected)


@pytest.mark.parametrize("pi05", [True, False])
def test_per_position_time_matches_scalar_time(pi05):
    model = make_model(pi05=pi05)
    prefix = model.embed_observation(make_observation())
    x_t = torch.randn(1, ACTION_HORIZON, 32)
    time = torch.full((1,), 0.3)

    v_scalar = model.denoise_step(prefix.state, prefix.prefix_pad_masks, prefix.past_key_values, x_t, time)
    v_per_position = model.denoise_step(
        prefix.state, prefix.prefix_pad_masks, prefix.past_key_values, x_t, time[:, None].expand(1, ACTION_HORIZON)
    )
    torch.testing.assert_close(v_scalar, v_per_position, atol=1e-5, rtol=1e-5)


def test_training_uses_the_staircase():
    model = make_model().train()
    observation = make_observation(batch_size=2)
    actions = torch.randn(2, ACTION_HORIZON, 32)
    noise = torch.randn(2, ACTION_HORIZON, 32)

    torch.manual_seed(1)
    loss = model(observation, actions, noise=noise)
    torch.manual_seed(1)
    expected = model(observation, actions, noise=noise, time=model.react_time(2, "cpu"))

    assert loss.shape == (2, ACTION_HORIZON, 32)
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(loss, expected)


def test_react_step_is_one_euler_step():
    model = make_model()
    prefix = model.embed_observation(make_observation())
    x_t = torch.randn(1, ACTION_HORIZON, 32)
    v_t = model.denoise_step(
        prefix.state, prefix.prefix_pad_masks, prefix.past_key_values, x_t, model.react_time(1, "cpu")
    )
    torch.testing.assert_close(model.react_denoise_step(prefix, x_t), x_t - v_t / NUM_BLOCKS)


@pytest.mark.parametrize("block_causal", [True, False])
def test_block_causal_mask(block_causal):
    model = make_model(react_block_causal_mask=block_causal)
    prefix = model.embed_observation(make_observation())
    x_t = torch.randn(1, ACTION_HORIZON, 32)
    perturbed = x_t.clone()
    perturbed[:, 2 * BLOCK_SIZE :] += 1.0

    v_t = model.react_denoise_step(prefix, x_t)
    v_perturbed = model.react_denoise_step(prefix, perturbed)
    earlier_blocks_unchanged = torch.allclose(v_t[:, : 2 * BLOCK_SIZE], v_perturbed[:, : 2 * BLOCK_SIZE], atol=1e-6)
    assert earlier_blocks_unchanged == block_causal


def test_rolling_buffer():
    buffer = rolling.RollingBuffer(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, action_dim=3, device="cpu")
    hold = torch.ones(1, ACTION_HORIZON, 3)

    torch.manual_seed(0)
    buffer.reset(hold)
    torch.manual_seed(0)
    noise = torch.randn(1, ACTION_HORIZON, 3)
    tau = torch.tensor([0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0])[None, :, None]
    torch.testing.assert_close(buffer.x, (1 - tau) * hold + tau * noise)

    denoised = torch.randn(1, ACTION_HORIZON, 3)
    front = buffer.roll(denoised)
    torch.testing.assert_close(front, denoised[:, :BLOCK_SIZE])
    torch.testing.assert_close(buffer.x[:, :-BLOCK_SIZE], denoised[:, BLOCK_SIZE:])
    assert buffer.x.shape == (1, ACTION_HORIZON, 3)


def test_react_policy():
    policy = make_policy()
    react = rolling.ReactPolicy(policy)
    calls = []
    denoise = policy.react_denoise
    policy.react_denoise = lambda prefix, x_t: calls.append(1) or denoise(prefix, x_t)

    outputs = react.infer(make_obs_dict(0))
    assert outputs["actions"].shape == (BLOCK_SIZE, 32)
    assert len(calls) == NUM_BLOCKS  # K - 1 warm-up updates + 1

    react.infer(make_obs_dict(1))
    assert len(calls) == NUM_BLOCKS + 1

    react.infer({**make_obs_dict(2), rolling.RESET_KEY: True})
    assert len(calls) == 2 * NUM_BLOCKS + 1
    assert react.metadata["react"] == {"num_blocks": NUM_BLOCKS, "block_size": BLOCK_SIZE}


@pytest.mark.parametrize(
    ("make_example", "inputs_cls", "hold_actions"),
    [
        (acone_policy.make_acone_example, acone_policy.AconeInputs, acone_policy.react_hold_actions),
        (franka_policy.make_franka_example, franka_policy.FrankaInputs, franka_policy.react_hold_actions),
    ],
)
def test_hold_actions_repeat_the_state(make_example, inputs_cls, hold_actions):
    example = make_example()
    inputs = inputs_cls(model_type=_model.ModelType.PI05)({**example, **hold_actions(example, 4)})
    assert inputs["actions"].shape == (4, inputs["state"].shape[-1])
    np.testing.assert_allclose(inputs["actions"], np.tile(inputs["state"], (4, 1)))


@dataclasses.dataclass(frozen=True)
class _TestPolicySpec(dual_decoupling.PolicySpec):
    def create(self, device: str) -> _policy.Policy:
        return make_policy(device)


class _FakeEnv(_environment.Environment):
    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.actions = []
        self._seed = 0

    def reset(self) -> None:
        self.actions = []

    def is_episode_complete(self) -> bool:
        return len(self.actions) >= self.num_steps

    def get_observation(self) -> dict:
        self._seed += 1
        return make_obs_dict(self._seed)

    def apply_action(self, action: dict) -> None:
        self.actions.append(action["actions"])


def test_dual_decoupling():
    config = dual_decoupling.DualDecouplingConfig(
        vlm_devices=("cpu", "cpu"), dit_device="cpu", camera_fps=100.0, control_hz=100.0
    )
    env = _FakeEnv(num_steps=6 * BLOCK_SIZE)
    with dual_decoupling.DualDecouplingRunner(_TestPolicySpec("test", "unused"), config) as runner:
        for _ in range(2):
            stats = runner.run_episode(env)
            assert len(env.actions) >= env.num_steps
            assert env.actions[0].shape == (32,)
            assert stats["updates"] >= env.num_steps // BLOCK_SIZE
            assert stats["executed"] == len(env.actions)
