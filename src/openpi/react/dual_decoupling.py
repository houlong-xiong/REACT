"""Dual Decoupling: asynchronous REACT deployment ("full REACT" in the paper).

Sensing, VLM encoding, action-expert (DiT) denoising and action execution run as separate streams, so neither the VLM
pass nor block execution sits on the DiT critical path:

    sensing thread   selects one observation every S camera periods  ->  request queue
    M VLM workers    run the prefix pass, send the KV cache back      ->  latest-ready cache
    DiT loop         one REACT update per block, freshest embedding   ->  action queue
    executor thread  applies one action per control period            ->  environment

The latest-ready cache only accepts embeddings captured after the one it holds, so a late worker never regresses the
conditioning. The policy weights are unchanged: VLM workers are separate processes holding their own copy of the
policy and may share the DiT's GPU or run on other GPUs. KV caches are passed through `torch.multiprocessing` queues
(CUDA IPC when on GPU).

Scripts that use this module must guard their entry point with `if __name__ == "__main__":` because the VLM workers
are started with the "spawn" method.
"""

import collections
from collections.abc import Sequence
import contextlib
import dataclasses
import logging
import queue
import threading
import time
from typing import Literal

import numpy as np
from openpi_client.runtime import environment as _environment
import torch
import torch.multiprocessing as mp
from transformers.cache_utils import DynamicCache

from openpi.models_pytorch import pi0_pytorch as _pi0_pytorch
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.react import rolling as _rolling
from openpi.training import config as _config

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    """How each process loads the policy. It is sent to the VLM worker processes, so it must be picklable."""

    # Name of the training config (see `openpi.training.config`).
    config_name: str
    # Checkpoint directory.
    checkpoint_dir: str
    # Prompt to use if the observation has none.
    default_prompt: str | None = None

    def create(self, device: str) -> _policy.Policy:
        return _policy_config.create_trained_policy(
            _config.get_config(self.config_name),
            self.checkpoint_dir,
            default_prompt=self.default_prompt,
            pytorch_device=device,
        )


@dataclasses.dataclass(frozen=True)
class DualDecouplingConfig:
    # Devices of the M VLM workers. Workers may share a device with each other and with the DiT.
    vlm_devices: Sequence[str] = ("cuda:0", "cuda:0")
    # Device of the DiT (action expert), which runs in the calling process.
    dit_device: str = "cuda:0"
    # Camera frame rate f_cam. The sensing loop selects one observation every S camera periods (rate f_cam / S).
    camera_fps: float = 30.0
    # Action execution rate.
    control_hz: float = 30.0
    # The next REACT update starts once at most this many actions of the current block remain to be executed.
    lead_steps: int = 1
    # Initial rolling buffer: the reset pose ("hold") or pure noise ("noise").
    warmup: Literal["hold", "noise"] = "hold"
    # Drop the VLM weights from the DiT process, which only runs the action expert. This leaves memory for
    # colocated VLM workers.
    release_dit_vlm: bool = True
    # How often to log scheduler statistics, in seconds.
    log_interval_s: float = 10.0


@dataclasses.dataclass
class _Embedding:
    """A completed VLM pass as held by the latest-ready cache."""

    seq: int
    capture_time: float
    worker_id: int
    vlm_ms: float
    packed: tuple
    prefix: _pi0_pytorch.PrefixCache | None = None


def _pack_prefix(prefix: _pi0_pytorch.PrefixCache) -> tuple[torch.Tensor, ...]:
    # Stack the per-layer KV tensors so that only four tensors cross the process boundary. Everything is copied:
    # outputs of a function compiled with CUDA graphs are overwritten by its next call.
    layers = prefix.past_key_values.to_legacy_cache()
    keys = torch.stack([k for k, _ in layers])
    values = torch.stack([v for _, v in layers])
    return keys, values, prefix.prefix_pad_masks.clone(), prefix.state.clone()


def _unpack_prefix(packed: tuple, device: torch.device) -> _pi0_pytorch.PrefixCache:
    keys, values, prefix_pad_masks, state = (t.to(device) for t in packed)
    cache = DynamicCache.from_legacy_cache(tuple(zip(keys.unbind(0), values.unbind(0), strict=True)))
    return _pi0_pytorch.PrefixCache(past_key_values=cache, prefix_pad_masks=prefix_pad_masks, state=state)


def _vlm_worker(worker_id: int, spec: PolicySpec, device: str, requests, results, ready, stop) -> None:
    """VLM worker process: observation -> prefix KV cache."""
    logging.basicConfig(level=logging.INFO, format=f"[vlm-{worker_id}] %(message)s", force=True)
    on_cuda = device.startswith("cuda")
    if on_cuda:
        torch.cuda.set_device(device)
    policy = spec.create(device)
    logger.info("Loaded policy on %s", device)
    ready.set()

    while not stop.is_set():
        try:
            request = requests.get(timeout=0.1)
        except queue.Empty:
            continue
        if request is None:
            break
        seq, capture_time, obs = request
        start_time = time.monotonic()
        packed = _pack_prefix(policy.react_embed(obs))
        if on_cuda:
            # The consumer reads these tensors from another process; make sure they are complete.
            torch.cuda.synchronize(device)
        results.put((seq, capture_time, worker_id, (time.monotonic() - start_time) * 1000, packed))


class _ActionQueue:
    """Actions committed by the DiT loop and consumed by the executor."""

    def __init__(self):
        self._actions = collections.deque()
        self._cond = threading.Condition()

    def extend(self, actions) -> None:
        with self._cond:
            self._actions.extend(actions)
            self._cond.notify_all()

    def pop(self) -> np.ndarray | None:
        with self._cond:
            action = self._actions.popleft() if self._actions else None
            self._cond.notify_all()
            return action

    def wait_until_at_most(self, n: int, timeout: float) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: len(self._actions) <= n, timeout=timeout)

    def clear(self) -> None:
        with self._cond:
            self._actions.clear()
            self._cond.notify_all()


@dataclasses.dataclass
class _EpisodeStats:
    updates: int = 0
    executed: int = 0
    # Control periods in which no action was available (after the first action).
    holds: int = 0
    # Observations dropped because all VLM workers were busy.
    dropped_observations: int = 0
    # Completed embeddings that were older than the cached one when they arrived.
    out_of_order: int = 0
    # DiT updates that reused the embedding of the previous update.
    reused_embeddings: int = 0
    embedding_age_ms: list[float] = dataclasses.field(default_factory=list)
    dit_ms: list[float] = dataclasses.field(default_factory=list)
    vlm_ms: list[float] = dataclasses.field(default_factory=list)

    def summary(self) -> dict:
        def describe(values: list[float]) -> dict:
            if not values:
                return {}
            arr = np.asarray(values)
            return {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)), "max": float(arr.max())}

        return {
            "updates": self.updates,
            "executed": self.executed,
            "holds": self.holds,
            "dropped_observations": self.dropped_observations,
            "out_of_order": self.out_of_order,
            "reused_embeddings": self.reused_embeddings,
            "embedding_age_ms": describe(self.embedding_age_ms),
            "dit_ms": describe(self.dit_ms),
            "vlm_ms": describe(self.vlm_ms),
        }


class DualDecouplingRunner:
    """Runs REACT episodes on an `openpi_client` environment with Dual Decoupling.

    The environment must tolerate `get_observation` (sensing thread) and `apply_action` (executor thread) being
    called concurrently. Actions are applied as `{"actions": action}`, one robot action per control period.

    Usage:
        with DualDecouplingRunner(spec, DualDecouplingConfig(...)) as runner:
            stats = runner.run_episode(env, max_steps=1000)
    """

    def __init__(
        self,
        spec: PolicySpec,
        config: DualDecouplingConfig | None = None,
        *,
        hold_actions_fn: _rolling.HoldActionsFn = _rolling.hold_state_actions,
    ):
        config = config or DualDecouplingConfig()
        if not config.vlm_devices:
            raise ValueError("Dual Decoupling needs at least one VLM worker.")
        self._config = config
        self._hold_actions_fn = hold_actions_fn

        self._seq = 0
        self._episode_first_seq = 0
        self._latest: _Embedding | None = None
        self._actions = _ActionQueue()
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._env: _environment.Environment | None = None
        self._stats = _EpisodeStats()

        ctx = mp.get_context("spawn")
        num_workers = len(config.vlm_devices)
        self._requests = ctx.Queue(maxsize=num_workers)
        self._results = ctx.Queue()
        self._workers_stop = ctx.Event()
        ready_events = []
        self._workers = []
        for worker_id, device in enumerate(config.vlm_devices):
            ready = ctx.Event()
            worker = ctx.Process(
                target=_vlm_worker,
                args=(worker_id, spec, device, self._requests, self._results, ready, self._workers_stop),
                name=f"react-vlm-{worker_id}",
                daemon=True,
            )
            worker.start()
            ready_events.append(ready)
            self._workers.append(worker)

        try:
            # The DiT runs in this process. Load it while the workers load theirs.
            self._dit_device = torch.device(config.dit_device)
            self._policy = spec.create(config.dit_device)
            if config.release_dit_vlm:
                _release_vlm_weights(self._policy)
            layout = self._policy.react_layout
            self._buffer = _rolling.RollingBuffer(
                num_blocks=layout["num_blocks"],
                block_size=layout["block_size"],
                action_dim=layout["action_dim"],
                device=config.dit_device,
            )

            for worker, ready in zip(self._workers, ready_events, strict=True):
                while not ready.wait(timeout=1.0):
                    if not worker.is_alive():
                        raise RuntimeError(f"VLM worker {worker.name} exited during startup.")
        except BaseException:
            self.close()
            raise

        logger.info(
            "Dual Decoupling ready: %d VLM workers on %s, DiT on %s, K=%d, S=%d",
            num_workers,
            list(config.vlm_devices),
            config.dit_device,
            self._buffer.num_blocks,
            self._buffer.block_size,
        )

    @property
    def block_size(self) -> int:
        return self._buffer.block_size

    def run_episode(self, env: _environment.Environment, *, max_steps: int | None = None) -> dict:
        """Run one episode until the environment reports completion or `max_steps` actions were executed."""
        self._env = env
        self._stats = _EpisodeStats()
        self._actions.clear()
        self._stop.clear()

        env.reset()
        obs = env.get_observation()
        first = self._warm_up(obs)
        self._error = None
        execution_start = time.monotonic()

        threads = [
            threading.Thread(target=self._guarded(self._sense_loop), name="react-sense", daemon=True),
            threading.Thread(target=self._guarded(self._execute_loop), name="react-execute", daemon=True),
        ]
        for thread in threads:
            thread.start()

        last_log = time.monotonic()
        last_seq = None
        try:
            while not env.is_episode_complete() and (max_steps is None or self._stats.executed < max_steps):
                if self._error is not None:
                    raise RuntimeError("Dual Decoupling thread failed.") from self._error
                if not self._actions.wait_until_at_most(self._config.lead_steps, timeout=0.1):
                    continue

                self._poll_results()
                embedding = self._latest or first
                start_time = time.monotonic()
                if embedding.capture_time >= execution_start:
                    self._stats.embedding_age_ms.append((start_time - embedding.capture_time) * 1000)
                if embedding.seq == last_seq:
                    self._stats.reused_embeddings += 1
                last_seq = embedding.seq

                prefix = self._prefix(embedding)
                block = self._buffer.roll(self._policy.react_denoise(prefix, self._buffer.x))
                actions = self._policy.react_decode(block, prefix)["actions"]
                self._actions.extend(list(actions))
                self._stats.dit_ms.append((time.monotonic() - start_time) * 1000)
                self._stats.updates += 1

                if time.monotonic() - last_log > self._config.log_interval_s:
                    last_log = time.monotonic()
                    logger.info("Dual Decoupling: %s", self._stats.summary())
        finally:
            self._stop.set()
            for thread in threads:
                thread.join(timeout=2.0)
            self._actions.clear()
            self._env = None

        summary = self._stats.summary()
        logger.info("Episode finished: %s", summary)
        return summary

    def _warm_up(self, obs: dict) -> _Embedding:
        """Embed the first observation on every worker (this also compiles them) and fill the rolling buffer."""
        self._drain(self._requests)
        self._drain(self._results)
        self._latest = None
        num_workers = len(self._workers)
        seqs = set()
        for _ in range(num_workers):
            seq = self._next_seq()
            seqs.add(seq)
            self._requests.put((seq, time.monotonic(), obs))
        # Results of requests that were still in flight at the end of the previous episode are ignored.
        self._episode_first_seq = min(seqs)

        first = None
        while seqs:
            try:
                result = _Embedding(*self._results.get(timeout=1.0))
            except queue.Empty:
                if not all(worker.is_alive() for worker in self._workers):
                    raise RuntimeError("A VLM worker exited.") from None
                continue
            if result.seq in seqs:
                seqs.discard(result.seq)
                first = result

        hold = None
        if self._config.warmup == "hold":
            hold = self._policy.react_encode_actions({**obs, **self._hold_actions_fn(obs, self._buffer.action_horizon)})
        self._buffer.reset(hold)
        prefix = self._prefix(first)
        for _ in range(self._buffer.num_blocks - 1):
            self._buffer.roll(self._policy.react_denoise(prefix, self._buffer.x))
        self._latest = first
        return first

    def _guarded(self, loop):
        def run():
            try:
                loop()
            except Exception as e:
                logger.exception("Dual Decoupling thread failed")
                self._error = e
                self._stop.set()

        return run

    def _prefix(self, embedding: _Embedding) -> _pi0_pytorch.PrefixCache:
        if embedding.prefix is None:
            embedding.prefix = _unpack_prefix(embedding.packed, self._dit_device)
            embedding.packed = ()
        return embedding.prefix

    def _poll_results(self) -> None:
        while True:
            try:
                result = _Embedding(*self._results.get_nowait())
            except queue.Empty:
                return
            if result.seq < self._episode_first_seq:
                continue
            self._stats.vlm_ms.append(result.vlm_ms)
            if self._latest is not None and result.capture_time <= self._latest.capture_time:
                self._stats.out_of_order += 1
                continue
            self._latest = result

    @staticmethod
    def _drain(q) -> None:
        with contextlib.suppress(queue.Empty):
            while True:
                q.get_nowait()

    def _sense_loop(self) -> None:
        period = self._buffer.block_size / self._config.camera_fps
        next_time = time.monotonic()
        while not self._stop.is_set():
            obs = self._env.get_observation()
            self._submit((self._next_seq(), time.monotonic(), obs))
            next_time += period
            delay = next_time - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_time = time.monotonic()

    def _submit(self, request: tuple) -> None:
        # Keep only the freshest observations when every worker is busy.
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._requests.get_nowait()
                self._stats.dropped_observations += 1
            with contextlib.suppress(queue.Full):
                self._requests.put_nowait(request)

    def _execute_loop(self) -> None:
        period = 1.0 / self._config.control_hz
        next_time = time.monotonic()
        while not self._stop.is_set():
            action = self._actions.pop()
            if action is not None:
                self._env.apply_action({"actions": action})
                self._stats.executed += 1
            elif self._stats.executed > 0:
                self._stats.holds += 1
            next_time += period
            delay = next_time - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_time = time.monotonic()

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def close(self) -> None:
        self._stop.set()
        self._workers_stop.set()
        for worker in self._workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
        self._workers = []
        # Observations that no worker will read anymore may still be buffered in the queue's feeder thread, which
        # would otherwise block interpreter exit while writing them to the pipe.
        for q in (self._requests, self._results):
            q.cancel_join_thread()
            q.close()

    def __enter__(self) -> "DualDecouplingRunner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _release_vlm_weights(policy: _policy.Policy) -> None:
    """Move the VLM (PaliGemma) weights of the DiT-side policy to the meta device; only the action expert is used."""
    model = policy._model  # noqa: SLF001
    model.paligemma_with_expert.paligemma.to("meta")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
