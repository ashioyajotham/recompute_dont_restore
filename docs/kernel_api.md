# Pallas Flash Attention kernel API

This document defines the supported import surface and tensor contract for
downstream integrations. The kernels remain pedagogical research code rather
than a production attention implementation; this API description makes their
current boundary explicit so consumers do not need to import private helpers.

## Installation and imports

Install the repository as a package and import from the top-level module:

```bash
python -m pip install -e .
```

```python
from recompute_dont_restore import (
    BlockSizes,
    MIN_BLOCK_SIZE,
    flash_attention,
    flash_attention_backward,
    flash_attention_forward,
    get_block_sizes,
)
```

Downstream projects that need reproducible behavior should pin a commit rather
than track a moving branch. For example:

```bash
python -m pip install \
  "git+https://github.com/ashioyajotham/recompute_dont_restore.git@<full-commit-sha>"
```

An editable install is useful during development, but it intentionally follows
changes in the local clone. Record both the full upstream commit and whether
that checkout was dirty when publishing results.

## Common tensor contract

The forward and backward launch functions use the following names:

| Tensor | Shape | Meaning |
|---|---|---|
| `q` | `(batch, heads, seq_q, d_k)` | Query |
| `k` | `(batch, heads, seq_kv, d_k)` | Key |
| `v` | `(batch, heads, seq_kv, d_v)` | Value |
| `o` | `(batch, heads, seq_q, d_v)` | Attention output |
| `do` | `(batch, heads, seq_q, d_v)` | Upstream gradient of `o` |
| `dq`, `dk`, `dv` | same shapes as `q`, `k`, `v` | Input gradients |

Inputs must satisfy all of these conditions:

- `q`, `k`, and `v` are rank-4 arrays.
- Batch and head counts match across `q`, `k`, and `v`.
- `q` and `k` have the same head dimension; `k` and `v` have the same
  sequence length.
- Kernel inputs use `jax.numpy.bfloat16` or `jax.numpy.float16`. Use the same
  dtype for `q`, `k`, and `v`; the tested path is BF16.
- `d_k` is a multiple of `MIN_BLOCK_SIZE` (128). The demonstrated downstream
  contract uses `d_k = d_v = 128`; 64-wide heads and other `d_v` values have
  not been validated by this repository.
- `seq_q` is divisible by `block_q`, and `seq_kv` is divisible by `block_kv`.
- Both block sizes are positive multiples of 128.

The implementation accumulates logits, normalizers, and output tiles in
float32 before converting `o`, `dq`, `dk`, and `dv` to the corresponding input
dtypes.

## Differentiable entry point

```python
flash_attention(
    q,
    k,
    v,
    causal=False,
    sm_scale=None,
    block_sizes=None,
) -> o
```

Use `flash_attention` for ordinary JAX model code. It is registered with
`jax.custom_vjp`; `jax.grad` therefore runs the recomputing Pallas backward
kernels without exposing the saved forward state to the caller.

If `sm_scale` is `None`, the implementation uses `1 / sqrt(d_k)`. If
`block_sizes` is `None`, it calls `get_block_sizes(seq_q, d_k, q.dtype)`.
Treat `causal`, `sm_scale`, and `block_sizes` as static configuration when
wrapping this function in a compiled integration.

```python
import jax
import jax.numpy as jnp

from recompute_dont_restore import BlockSizes, flash_attention

shape = (1, 8, 1024, 128)
q = jnp.ones(shape, dtype=jnp.bfloat16)
k = jnp.ones(shape, dtype=jnp.bfloat16)
v = jnp.ones(shape, dtype=jnp.bfloat16)
blocks = BlockSizes(block_q=256, block_kv=256)

def loss(q, k, v):
    return flash_attention(q, k, v, True, None, blocks).sum()

dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
```

## Explicit forward launch

```python
flash_attention_forward(
    q,
    k,
    v,
    *,
    causal=False,
    sm_scale=None,
    block_sizes=None,
) -> (o, m, l)
```

This lower-level entry point is useful when a downstream framework owns the
autograd registration. It returns the output plus the state required by the
explicit backward launch:

- `o` has shape `(batch, heads, seq_q, d_v)` and the dtype of `q`.
- `m` and `l` have shape
  `(batch, heads, seq_q, MIN_BLOCK_SIZE)` and dtype float32.

`m` and `l` carry the online-softmax running maximum and normalizer. Their last
dimension is padded to 128 for the TPU vector layout; they are not a public
log-sum-exp representation. Preserve them unchanged and pass them back to
`flash_attention_backward` with the matching `q`, `k`, `v`, `o`, scale,
causal mode, and block sizes.

## Explicit backward launch

```python
flash_attention_backward(
    q,
    k,
    v,
    o,
    m,
    l,
    do,
    *,
    causal=False,
    sm_scale=None,
    block_sizes=None,
) -> (dq, dk, dv)
```

This function exposes the same recomputing backward implementation used by the
custom VJP. `do` must match the shape of `o`. The forward and backward calls
must use identical static configuration; mixing residuals or configuration
from different calls is unsupported.

```python
from recompute_dont_restore import (
    BlockSizes,
    flash_attention_backward,
    flash_attention_forward,
)

blocks = BlockSizes(256, 256)
o, m, l = flash_attention_forward(
    q, k, v, causal=True, block_sizes=blocks
)
do = jnp.ones_like(o)
dq, dk, dv = flash_attention_backward(
    q, k, v, o, m, l, do, causal=True, block_sizes=blocks
)
```

## Block-size selection

`BlockSizes(block_q, block_kv)` is a named tuple and may be supplied directly.
`get_block_sizes(seq_len, head_dim, dtype)` provides a conservative heuristic
using an 8 MiB VMEM budget. It returns 128-, 256-, or 512-element sequence
tiles, reducing them when the estimated forward working set would exceed that
budget.

The selector receives one sequence length. The default forward path passes
`seq_q`, so for cross-attention (`seq_q != seq_kv`) callers must verify that the
selected `block_kv` divides `seq_kv` or provide an explicit `BlockSizes` value.
Selection is a validity heuristic, not an autotuner or a guarantee of optimal
performance on every TPU generation.

## Causal semantics

With `causal=True`, query position `i` may attend to key positions `j <= i`.
Positions are compared from zero at the start of their respective tensors. For
unequal query and key sequence lengths, this is a top-left-aligned causal mask;
the API does not apply an offset automatically.

## Current execution boundary

- TPU execution uses Pallas TPU primitives and is single-device.
- Shapes and configuration are static for the demonstrated compiled paths.
- The backward dKV kernel loads full-sequence `q`, `do`, `o`, `m`, and `l`
  buffers for each KV tile. At `d_k = d_v = 128` in BF16, the documented 8 MiB
  VMEM budget is exceeded at `seq_q >= 8192`.
- The dQ kernel loads the full K and V sequences and reaches the same budget at
  `seq_kv >= 16384` for `d_k = d_v = 128` in BF16.
- The repository does not promise sharding, ragged sequence lengths, dropout,
  FP8/int8 inputs, or production-level autotuning.

See the root README for the native TPU test commands and the complete list of
known limitations.
