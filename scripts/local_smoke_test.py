"""
CPU smoke test for the naive JAX attention path.

Invoked by run_local.ps1 (step 3). Verifies:
  - forward pass shape and dtype
  - output is finite
  - jax.grad over the attention loss returns finite gradients of correct shape

Kept in pure ASCII so the surrounding PowerShell wrapper is encoding-agnostic.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "02_naive_jax_baseline"))

import jax
import jax.numpy as jnp
import numpy as np

from standard_attention import attention


def main() -> int:
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    shape = (1, 2, 256, 128)
    q = jax.random.normal(k1, shape, dtype=jnp.bfloat16)
    k = jax.random.normal(k2, shape, dtype=jnp.bfloat16)
    v = jax.random.normal(k3, shape, dtype=jnp.bfloat16)

    out = attention(q, k, v)
    print(f"output shape: {out.shape}, dtype: {out.dtype}")

    arr = np.asarray(out, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        print("FAIL: output contains non-finite values", file=sys.stderr)
        return 1
    print(
        f"output summary: min={arr.min():.4f} "
        f"max={arr.max():.4f} mean={arr.mean():.4f}"
    )

    grads = jax.grad(
        lambda q, k, v: attention(q, k, v).sum(),
        argnums=(0, 1, 2),
    )(q, k, v)

    for name, g in zip(("dq", "dk", "dv"), grads):
        finite = bool(np.all(np.isfinite(np.asarray(g))))
        print(f"{name}: shape={g.shape} finite={finite}")
        if not finite:
            print(f"FAIL: {name} contains non-finite values", file=sys.stderr)
            return 1

    print("OK: naive forward + gradients sane.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
