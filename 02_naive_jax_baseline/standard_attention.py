"""
Naive scaled dot-product attention in pure JAX.

This is the baseline we benchmark against. It materializes the full O(n^2)
attention matrix in HBM — the memory wall that Flash Attention is designed
to break through.

Memory breakdown for one forward pass (bfloat16):
  Q, K, V:            3 * batch * heads * seq * d_k * 2 bytes
  Attention matrix:   batch * heads * seq^2 * 2 bytes  <-- the expensive term
  Output O:           batch * heads * seq * d_v * 2 bytes

At seq=16384, heads=8, d_k=128, batch=1 (bfloat16):
  Attention matrix alone: 8 * 16384^2 * 2 = ~4.3 GB
"""

import functools
import math
from typing import Optional

import jax
import jax.numpy as jnp

# Safe large negative — avoids NaN when an entire row is masked.
# Using -inf produces NaN via 0 * -inf in the softmax; a finite large negative
# produces 0 after softmax normalization.
_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)


def scaled_dot_product_attention(
    q: jnp.ndarray,                   # (seq_q, d_k)
    k: jnp.ndarray,                   # (seq_kv, d_k)
    v: jnp.ndarray,                   # (seq_kv, d_v)
    mask: Optional[jnp.ndarray] = None,  # (seq_q, seq_kv) bool, True = attend
    causal: bool = False,
) -> jnp.ndarray:                     # (seq_q, d_v)
    """
    Single-head scaled dot-product attention.

    Does NOT use flash attention; materializes the full (seq_q, seq_kv) logit
    and weight matrices in HBM.
    """
    seq_q, d_k = q.shape
    seq_kv = k.shape[0]
    scale = math.sqrt(d_k)

    # Logits: (seq_q, seq_kv)
    logits = jnp.einsum("id,jd->ij", q, k) / scale

    if causal:
        # Upper triangle (future positions) set to large negative.
        rows = jnp.arange(seq_q)[:, None]
        cols = jnp.arange(seq_kv)[None, :]
        logits = jnp.where(rows >= cols, logits, _MASK_VALUE)

    if mask is not None:
        logits = jnp.where(mask, logits, _MASK_VALUE)

    # Softmax in float32 for numerical stability, even if inputs are bfloat16.
    weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)

    return jnp.einsum("ij,jd->id", weights.astype(v.dtype), v)


# vmap across heads, then across batch.
# Axes: (batch, heads, seq, dim) -> operate on (seq, dim) leaf.
_mha_inner = jax.vmap(
    scaled_dot_product_attention,
    in_axes=(0, 0, 0, None, None),   # over heads
)
_mha = jax.vmap(
    _mha_inner,
    in_axes=(0, 0, 0, None, None),   # over batch
)


@functools.partial(jax.jit, static_argnames=("causal",))
def attention(
    q: jnp.ndarray,                   # (batch, heads, seq_q, d_k)
    k: jnp.ndarray,                   # (batch, heads, seq_kv, d_k)
    v: jnp.ndarray,                   # (batch, heads, seq_kv, d_v)
    mask: Optional[jnp.ndarray] = None,
    causal: bool = False,
) -> jnp.ndarray:                     # (batch, heads, seq_q, d_v)
    """
    Multi-head attention, jit-compiled and vmapped over batch and heads.

    This is the reference implementation used as the correctness oracle for
    all Pallas kernel tests. Use atol=1e-2 when comparing bfloat16 outputs.
    """
    return _mha(q, k, v, mask, causal)
