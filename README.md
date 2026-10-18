# REACT: Rolling Denoising and Dual Decoupling for Reactive Robot Control with VLA Models

**CoRL 2026 (Spotlight)**

Houlong Xiong<sup>1,2\*</sup>, Zhenqi Qiu<sup>2,3\*</sup>, Zechen Wang<sup>2\*</sup>, Suohang Zhang<sup>4</sup>, Yiyu Ren<sup>2</sup>, Wanting Xu<sup>3</sup>, Hongfei Niu<sup>2</sup>, Chengyang He<sup>5</sup>, Ge Sun<sup>5</sup>, Ran Cheng<sup>2</sup>, Qian Zhu<sup>2†</sup>

<sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>PrimeBot, <sup>3</sup>ShanghaiTech University, <sup>4</sup>Zhejiang University, <sup>5</sup>National University of Singapore<br>
<sup>\*</sup>Equal contribution, <sup>†</sup>Corresponding author

Chunked VLA policies trade reactivity for coherence: executing a long chunk ignores new observations, while re-planning often breaks the chunk into independently sampled pieces that switch between action modes. **REACT** keeps the long horizon but rolls the denoising process across control iterations, so every executed block is refined under several consecutive observations:

- **Rolling denoising.** The H-step chunk is a rolling buffer of K blocks at increasing noise levels. Each control update runs one VLM pass on the latest observation and a *single* Euler step on the whole buffer. The front block becomes clean and is executed, the buffer shifts, and fresh noise enters at the tail.
- **Staircase training.** The action expert is trained on exactly the per-block noise levels that rolling inference uses, so one forward pass supervises all K denoising stages.
- **Dual Decoupling.** Sensing, VLM encoding (a pool of VLM workers feeding a latest-ready embedding cache), DiT denoising and action execution run as separate streams. This gives REACT asynchronous throughput without changing the weights.

REACT changes the training timestep layout and the inference schedule of the action expert; the model architecture is unchanged. This repository implements it on top of the PyTorch π₀.₅ model in [openpi](https://github.com/Physical-Intelligence/openpi).

## Installation

REACT needs an NVIDIA GPU. The paper's real-robot deployments, including full REACT with two VLM workers, run on a single RTX 4090.

```bash
git clone --recurse-submodules https://github.com/houlong-xiong/REACT.git
cd REACT
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# PyTorch support: patch transformers (REACT needs the per-token adaRMS change in this copy).
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

See the [openpi README](docs/openpi.md) for system requirements, Docker, and caveats about patching transformers.

## Quick start

REACT is implemented for the PyTorch model and is enabled with `react_enabled=True` and `react_num_blocks` (K) in `Pi0Config`. Full documentation: **[docs/react.md](docs/react.md)**.

**Train.** Ready-made π₀.₅ configs use H = 50, K = 5, S = 10:

| Config | Data |
| --- | --- |
| `pi05_react_aloha` | RoboTwin 2.0 / ALOHA-format LeRobot data |
| `pi05_react_acone` | Acone / ARX X5 bimanual robot |
| `pi05_react_franka` | Dual-arm Franka Research 3, end-effector actions |

```bash
# Convert the pi0.5 base checkpoint to PyTorch, then set `pytorch_weight_path` in the config.
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/pi05_base --config_name pi05_react_aloha --output_path /path/to/pi05_base_pytorch
uv run scripts/compute_norm_stats.py --config-name pi05_react_aloha
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pi05_react_aloha --exp_name my_run
```

**Serve without Dual Decoupling.** REACT checkpoints are served with rolling inference: each request returns the next block of S actions.

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_react_aloha --policy.dir=/path/to/checkpoint
```

**Run with Dual Decoupling.** `openpi.react.dual_decoupling.DualDecouplingRunner` drives an `openpi_client` environment with M VLM workers, which can share the DiT's GPU. `examples/react_acone_ros2/` deploys both modes on the ARX X5 robot over ROS2:

```bash
python -m examples.react_acone_ros2.main --checkpoint-dir /path/to/checkpoint --prompt "stack bowls" --mode dd
```

**Test.** `pytest src/openpi/react` runs the REACT tests on CPU with a tiny model.

## Changes to openpi

This repository is a modified version of [openpi](https://github.com/Physical-Intelligence/openpi) (Apache-2.0), forked at commit `215abfb`; every later commit belongs to REACT.

Modified files:

| File | Change |
| --- | --- |
| `src/openpi/models/pi0_config.py` | `react_*` config fields; the JAX model rejects REACT configs |
| `src/openpi/models_pytorch/pi0_pytorch.py` | Staircase times, per-position time embedding, optional block-causal action mask, `embed_observation`, `react_denoise_step` |
| `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` | adaRMS accepts per-token conditioning |
| `src/openpi/policies/policy.py` | REACT helpers on `Policy`; `PolicyRecorder.reset` forwards to the wrapped policy |
| `src/openpi/serving/websocket_policy_server.py` | Resets the policy on every new connection |
| `src/openpi/training/config.py` | Acone and Franka data configs, REACT train configs |
| `scripts/serve_policy.py` | Serves REACT checkpoints with rolling inference |
| `README.md` | This file; the original is in [docs/openpi.md](docs/openpi.md) |

New files: `src/openpi/react/` (rolling inference, Dual Decoupling, tests), `src/openpi/policies/acone_policy.py`, `src/openpi/policies/franka_policy.py`, `examples/react_acone_ros2/`, `docs/react.md`, `docs/openpi.md`.

Non-REACT inference and training paths produce bit-identical outputs to upstream openpi.

## Acknowledgements

REACT is built on [openpi](https://github.com/Physical-Intelligence/openpi) and the π₀ / π₀.₅ models by [Physical Intelligence](https://www.physicalintelligence.company/). This is an independent research release and is not affiliated with or endorsed by Physical Intelligence.

## License

Apache License 2.0, the same as openpi; see [LICENSE](LICENSE). The Gemma model components are subject to the terms in [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt).

## Citation

```bibtex
@inproceedings{xiong2026react,
  title     = {{REACT}: Rolling Denoising and Dual Decoupling for Reactive Robot Control with {VLA} Models},
  author    = {Xiong, Houlong and Qiu, Zhenqi and Wang, Zechen and Zhang, Suohang and Ren, Yiyu and Xu, Wanting and
               Niu, Hongfei and He, Chengyang and Sun, Ge and Cheng, Ran and Zhu, Qian},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

Please also cite π₀.₅, on which REACT is built:

```bibtex
@article{intelligence2025pi05,
  title   = {{$\pi_{0.5}$}: a Vision-Language-Action Model with Open-World Generalization},
  author  = {Intelligence, Physical and Black, Kevin and Brown, Noah and Darpinian, James and Dhabalia, Karan and
             Driess, Danny and Esmail, Adnan and Equi, Michael and Finn, Chelsea and Fusai, Niccolo and others},
  journal = {arXiv preprint arXiv:2504.16054},
  year    = {2025}
}
```
