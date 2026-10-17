"""Run a REACT policy on the Acone / ARX X5 robot over ROS2.

Two modes, matching the paper:
- `sync`: REACT without Dual Decoupling. Every S control steps the policy runs one VLM pass and one REACT update in
  this process, then the robot executes the S returned actions.
- `dd`: full REACT with Dual Decoupling. VLM workers, the DiT update and action execution run asynchronously.

Example:
    python examples/react_acone_ros2/main.py --config pi05_react_acone --checkpoint-dir /path/to/ckpt \
        --prompt "stack bowls" --mode dd --vlm-devices cuda:0 cuda:0
"""

import dataclasses
import logging
from typing import Literal

from openpi_client import action_chunk_broker
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.react_acone_ros2 import env as _env
from openpi.policies import acone_policy
from openpi.react import dual_decoupling
from openpi.react import rolling


@dataclasses.dataclass
class Args:
    # Training config name and checkpoint directory.
    config: str = "pi05_react_acone"
    checkpoint_dir: str = ""
    prompt: str = "stack bowls"

    mode: Literal["sync", "dd"] = "dd"
    # Initial rolling buffer: the current pose ("hold") or pure noise ("noise").
    warmup: Literal["hold", "noise"] = "hold"

    control_hz: float = 30.0
    camera_fps: float = 30.0
    # Rate at which the latest joint target is re-published to the ARX driver.
    send_hz: float = 150.0

    # Dual Decoupling: devices of the VLM workers (one entry per worker) and of the DiT.
    vlm_devices: tuple[str, ...] = ("cuda:0", "cuda:0")
    dit_device: str = "cuda:0"

    num_episodes: int = 1
    max_episode_steps: int = 1000

    topics: _env.Topics = dataclasses.field(default_factory=_env.Topics)


def run_sync(args: Args, environment: _env.AconeRos2Environment) -> None:
    spec = dual_decoupling.PolicySpec(args.config, args.checkpoint_dir, default_prompt=args.prompt)
    policy = rolling.ReactPolicy(
        spec.create(args.dit_device), warmup=args.warmup, hold_actions_fn=acone_policy.react_hold_actions
    )
    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            action_chunk_broker.ActionChunkBroker(policy, action_horizon=policy.block_size)
        ),
        subscribers=[],
        max_hz=args.control_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )
    runtime.run()


def run_dual_decoupling(args: Args, environment: _env.AconeRos2Environment) -> None:
    spec = dual_decoupling.PolicySpec(args.config, args.checkpoint_dir, default_prompt=args.prompt)
    config = dual_decoupling.DualDecouplingConfig(
        vlm_devices=args.vlm_devices,
        dit_device=args.dit_device,
        camera_fps=args.camera_fps,
        control_hz=args.control_hz,
        warmup=args.warmup,
    )
    with dual_decoupling.DualDecouplingRunner(spec, config, hold_actions_fn=acone_policy.react_hold_actions) as runner:
        for episode in range(args.num_episodes):
            input(f"Episode {episode + 1}/{args.num_episodes}: reset the robot and press Enter to start.")
            stats = runner.run_episode(environment, max_steps=args.max_episode_steps)
            logging.info("Episode %d: %s", episode + 1, stats)


def main(args: Args) -> None:
    environment = _env.AconeRos2Environment(args.prompt, topics=args.topics, send_hz=args.send_hz)
    try:
        if args.mode == "sync":
            run_sync(args, environment)
        else:
            run_dual_decoupling(args, environment)
    finally:
        environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
