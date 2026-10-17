"""REACT: rolling denoising for flow-based VLA policies.

- `openpi.react.rolling.ReactPolicy`: synchronous rolling inference ("REACT w/o DD"); a drop-in `BasePolicy` that
  returns one block of S actions per call and can be served with `scripts/serve_policy.py`.
- `openpi.react.dual_decoupling.DualDecouplingRunner`: asynchronous deployment with a pool of VLM workers
  ("full REACT"), driving an `openpi_client` environment directly.

See `docs/react.md`.
"""
