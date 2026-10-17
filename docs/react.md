# REACT: Rolling Denoising and Dual Decoupling

This fork of openpi adds **REACT** (*Rolling Denoising and Dual Decoupling for Reactive Robot Control with VLA Models*, CoRL 2026) on top of the PyTorch π0 / π0.5 implementation. REACT changes the training timestep layout and the inference schedule of the action expert. The model architecture stays the same.

## Method in brief

An H-step action chunk is split into K blocks of S actions (H = K·S). Block k is assigned the flow time τ_k = (k+1)/K, so the front block is almost clean and the last one is pure noise.

- **Staircase training.** Every training chunk is noised with this deterministic staircase, so one forward pass supervises all K denoising stages that rolling inference uses (`react_time` in `pi0_pytorch.py`).
- **Rolling denoising.** At inference, the policy keeps a rolling buffer x = [B_0, …, B_{K-1}]. Each update runs one VLM pass on the latest observation and a **single** Euler step x ← x − v/K. The now-clean B_0 is executed, the buffer is shifted left by one block, and fresh noise is appended at the tail. Each executed block has been refined under K consecutive observations.
- **Warm-up.** At the start of an episode, block k is initialized as (1−τ_k)·s_0 + τ_k·ε from the reset pose s_0. K−1 updates then run on the first observation before the first block is emitted.
- **Dual Decoupling (DD).** Sensing, VLM encoding, DiT denoising and execution run as separate streams. One observation is selected every S camera frames. A pool of M VLM workers encodes these into a *latest-ready* cache, which only moves forward in capture time. Each DiT update consumes the freshest embedding. This lifts REACT to asynchronous throughput without changing the weights.

## Installation

Install openpi with PyTorch support as described in the [main README](../README.md). REACT modifies `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` so that adaRMS conditioning can vary per action token. If you copied `transformers_replace` into your environment before, copy it again:

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

REACT is implemented for the PyTorch model only. The JAX model refuses REACT configs.

## Training

REACT is controlled by three fields of `Pi0Config`:

| Field | Meaning |
| --- | --- |
| `react_enabled` | Use staircase training and enable REACT inference. |
| `react_num_blocks` | K. `action_horizon` must be divisible by it; S = `action_horizon / K`. |
| `react_block_causal_mask` | Optional, off by default. Action tokens of block k attend only to blocks 0..k. |

Ready-made configs in `src/openpi/training/config.py` (all π0.5, H = 50, K = 5, S = 10, with the hyperparameters from the paper):

| Config | Data | Batch size |
| --- | --- | --- |
| `pi05_react_aloha` | RoboTwin 2.0 / ALOHA-format LeRobot data | 32 (30k steps) |
| `pi05_react_acone` | Acone / ARX X5 (`acone_policy.py`) | 128 |
| `pi05_react_franka` | Dual-arm Franka Research 3, end-effector actions (`franka_policy.py`) | 256 |

For the real-robot configs the paper trains for 20 epochs. Set `num_train_steps` and the learning-rate schedule to match your dataset size.

```bash
# 1. Convert the pi0.5 base checkpoint to PyTorch and set `pytorch_weight_path` in the config.
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/pi05_base --config_name pi05_react_aloha --output_path /path/to/pi05_base_pytorch

# 2. Compute normalization statistics for your dataset.
uv run scripts/compute_norm_stats.py --config-name pi05_react_aloha

# 3. Train (single node, multiple GPUs).
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pi05_react_aloha --exp_name my_react_run
```

## Inference without Dual Decoupling

`openpi.react.rolling.ReactPolicy` wraps a trained policy and implements the rolling buffer. Each `infer` call returns the next block of **S** actions. `scripts/serve_policy.py` wraps REACT checkpoints in it automatically:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_react_aloha --policy.dir=/path/to/checkpoint
```

Clients execute all S returned actions before requesting the next block, e.g. with `ActionChunkBroker(policy, action_horizon=S)`. The server reports K and S under `metadata["react"]`. A new connection starts a new episode. To start a new episode on the same connection, send an observation with `"react_reset": True`.

The warm-up reads the reset pose from the observation. The default (`--react-hold state`) tiles `obs["state"]`, which works for ALOHA / RoboTwin observations. Use `--react-hold acone` or `--react-hold franka` for those robots, or `--react-warmup noise` to start from pure noise.

## Inference with Dual Decoupling

`openpi.react.dual_decoupling.DualDecouplingRunner` drives an `openpi_client.runtime.environment.Environment` directly:

```python
from openpi.react import dual_decoupling

spec = dual_decoupling.PolicySpec("pi05_react_acone", "/path/to/checkpoint", default_prompt="stack bowls")
config = dual_decoupling.DualDecouplingConfig(
    vlm_devices=("cuda:0", "cuda:0"),  # M = 2 VLM workers, colocated with the DiT
    dit_device="cuda:0",
    camera_fps=30.0,
    control_hz=30.0,
)

if __name__ == "__main__":  # VLM workers are started with "spawn"
    with dual_decoupling.DualDecouplingRunner(spec, config, hold_actions_fn=acone_policy.react_hold_actions) as runner:
        stats = runner.run_episode(env, max_steps=1000)
```

- Every VLM worker is a separate process with its own copy of the policy. Workers can share the DiT's GPU or run on other GPUs. KV caches come back through CUDA IPC.
- The DiT process drops its own VLM weights (`release_dit_vlm`), since it only runs the action expert.
- The next update starts when at most `lead_steps` actions of the current block are left. Each update produces S actions, executed at `control_hz`.
- `run_episode` returns scheduler statistics: embedding age (mean / p95 / max), DiT and VLM latency, reused embeddings, out-of-order completions, dropped observations, and execution holds.
- The environment's `get_observation` and `apply_action` are called from different threads.

## Real-robot example

`examples/react_acone_ros2/` runs both modes on the Acone / ARX X5 robot over ROS2:

```bash
python -m examples.react_acone_ros2.main --checkpoint-dir /path/to/checkpoint --prompt "stack bowls" --mode dd
```

## Code map

| Path | Content |
| --- | --- |
| `src/openpi/models/pi0_config.py` | `react_*` config fields and validation |
| `src/openpi/models_pytorch/pi0_pytorch.py` | `react_time`, staircase training in `forward`, per-position time embedding, block-causal mask, `embed_observation`, `react_denoise_step` |
| `src/openpi/policies/policy.py` | `react_embed`, `react_denoise`, `react_decode`, `react_encode_actions` |
| `src/openpi/react/rolling.py` | `RollingBuffer`, `ReactPolicy` |
| `src/openpi/react/dual_decoupling.py` | `DualDecouplingRunner` |
| `src/openpi/policies/{acone,franka}_policy.py` | Robot transforms |
| `src/openpi/react/react_test.py` | CPU tests (`pytest src/openpi/react`) |
