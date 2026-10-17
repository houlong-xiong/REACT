# REACT on the Acone / ARX X5 robot (ROS2)

Runs a REACT policy on the Acone / ARX X5 bimanual robot (two 6-DoF arms with grippers, one top and two wrist cameras) directly over ROS2. Model inference happens in this process, so the robot computer needs the GPU.

- `env.py`: an `openpi_client` environment that subscribes to the camera and follower-arm topics and publishes joint targets on the leader-arm command topics (`arx5_arm_msg/RobotStatus`).
- `main.py`: runs REACT without Dual Decoupling (`--mode sync`) or full REACT with Dual Decoupling (`--mode dd`).

## Setup

1. Train a policy with the `pi05_react_acone` config (see [docs/react.md](../../docs/react.md)). The dataset uses the keys described in `src/openpi/policies/acone_policy.py`.
2. Source ROS2 and the ARX X5 workspace that provides `arx5_arm_msg`:

   ```bash
   source /opt/ros/humble/setup.bash
   source /path/to/ARX_X5/ROS2/X5_ws/install/setup.bash
   ```

3. Check the topic names in `env.Topics` (override them with e.g. `--topics.camera-top /my/topic`).

## Run

From the repository root:

```bash
# Full REACT: two VLM workers colocated with the DiT on one GPU.
python -m examples.react_acone_ros2.main --checkpoint-dir /path/to/checkpoint --prompt "stack bowls" \
    --mode dd --vlm-devices cuda:0 cuda:0 --dit-device cuda:0

# REACT without Dual Decoupling.
python -m examples.react_acone_ros2.main --checkpoint-dir /path/to/checkpoint --prompt "stack bowls" --mode sync
```

At the start of every episode the policy warms up its rolling buffer from the current pose and holds that pose until the first block is ready. In `dd` mode the script waits for Enter before each episode and logs the scheduler statistics (embedding age, DiT/VLM latency, holds) at the end.
