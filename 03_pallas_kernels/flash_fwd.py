"""
Flash Attention forward pass in JAX Pallas for TPU.

Algorithm summary (per Q tile):
  For each KV tile (j = 0..num_kv_tiles-1):
    1. Compute logits = Q_i @ K_j^T * scale           (block_q x block_kv)
    2. Update running max:  m_new = max(m_prev, rowmax(logits))
    3. Compute unnorm weights: P = exp(logits - m_new) (block_q x block_kv)
    4. Rescale prev acc:    acc = exp(m_prev - m_new) * acc
    5. Update norm:         l_new = exp(m_prev - m_new) * l_prev + rowsum(P)
    6. Accumulate:          acc += P @ V_j
  Output: O_i = acc / l_final, saved (m, l) for backward recomputation.

Memory: O(seq * d) instead of O(seq^2).
  The attention matrix is never written to HBM. VMEM holds only the active
  Q/K/V tiles and the running (m, l, O_acc) state.

Pallas grid:
  (batch, heads, q_tiles, kv_tiles)
  The kv_tiles axis has dimension_semantics="arbitrary" because the online
  softmax carries state from tile j to tile j+1. All other axes are parallel.
"""

import functools
import math
from typing import Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from utils import (
    MIN_BLOCK_SIZE,
    BlockSizes,
    get_block_sizes,
    pallas_interpret_mode,
    validate_shapes,
)


def _flash_fwd_kernel(
    # Input tile refs — sliced from HBM by in_specs
    q_ref,          # (block_q, d_k)   bfloat16
    k_ref,          # (block_kv, d_k)  bfloat16
    v_ref,          # (block_kv, d_v)  bfloat16
    # Output tile refs — written to HBM
    o_ref,          # (block_q, d_v)   bfloat16
    m_ref,          # (block_q, MIN_BLOCK_SIZE) float32 — running max (log-sum-exp)
    l_ref,          # (block_q, MIN_BLOCK_SIZE) float32 — running normalizer
    # VMEM scratch — persist across kv_tile steps for the same q_tile
    m_scratch_ref,  # (block_q, MIN_BLOCK_SIZE) float32
    l_scratch_ref,  # (block_q, MIN_BLOCK_SIZE) float32
    acc_scratch_ref,# (block_q, d_v)   float32
    *,
    causal: bool,
    sm_scale: float,
    block_q: int,
    block_kv: int,
    kv_seq_len: int,
    mask_value: float,
):
    """
    Single kernel invocation for grid point (batch, head, q_tile, kv_tile).

    VMEM scratch tensors persist across kv_tile steps (Mosaic pipelines them
    across the "arbitrary" dimension), allowing online softmax accumulation.
    """
    b = pl.program_id(0)
    h = pl.program_id(1)
    q_tile_idx = pl.program_id(2)
    kv_tile_idx = pl.program_id(3)

    num_kv_tiles = kv_seq_len // block_kv

    # --- Initialize VMEM scratch on the first KV tile ---
    # On subsequent tiles the scratch holds accumulated state from prior tiles.
    @pl.when(kv_tile_idx == 0)
    def _init():
        m_scratch_ref[...] = jnp.full(
            m_scratch_ref.shape, -jnp.inf, jnp.float32
        )
        l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, jnp.float32)
        acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, jnp.float32)

    # --- Load current running state from VMEM ---
    m_prev = m_scratch_ref[...]   # (block_q, MIN_BLOCK_SIZE)
    l_prev = l_scratch_ref[...]   # (block_q, MIN_BLOCK_SIZE)

    # --- Compute QK^T for this tile pair ---
    # Cast to float32 for numerical stability; bfloat16 has only ~3 decimal
    # digits of precision, insufficient for softmax exponentiation.
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)

    # logits: (block_q, block_kv)
    # lax.dot_general with contracting over dim 1 of both = Q @ K^T
    logits = (
        lax.dot_general(
            q, k,
            dimension_numbers=(([1], [1]), ([], [])),
            preferred_element_type=jnp.float32,
        )
        * sm_scale
    )

    # --- Causal mask ---
    # Zero out positions where a query token attends to a future key token.
    if causal:
        q_pos = (q_tile_idx * block_q + jnp.arange(block_q))[:, None]
        kv_pos = (kv_tile_idx * block_kv + jnp.arange(block_kv))[None, :]
        logits = jnp.where(q_pos >= kv_pos, logits, mask_value)

    # --- Online softmax update ---
    # m_curr: (block_q, 1)
    m_curr = jnp.max(logits, axis=1, keepdims=True)
    # m_next: new running max, broadcast to (block_q, MIN_BLOCK_SIZE)
    m_next = jnp.maximum(m_prev, jnp.broadcast_to(m_curr, m_prev.shape))

    # Unnormalized weights for this tile: exp(logits - m_next)
    # Broadcast m_next from (block_q, MIN_BLOCK_SIZE) -> (block_q, block_kv)
    m_next_for_logits = jnp.broadcast_to(m_next[:, :1], logits.shape)
    p = jnp.exp(logits - m_next_for_logits)   # (block_q, block_kv)

    # Rescaling factor for previously accumulated values
    alpha = jnp.exp(m_prev - m_next)   # (block_q, MIN_BLOCK_SIZE)

    # Updated normalizer
    l_next = (
        alpha * l_prev
        + jnp.broadcast_to(jnp.sum(p, axis=1, keepdims=True), l_prev.shape)
    )

    # --- Accumulate output ---
    v = v_ref[...].astype(jnp.float32)
    pv = lax.dot(p, v, preferred_element_type=jnp.float32)   # (block_q, d_v)

    # alpha broadcast: (block_q, MIN_BLOCK_SIZE) -> (block_q, d_v)
    alpha_o = jnp.broadcast_to(alpha[:, :1], acc_scratch_ref.shape)
    acc_scratch_ref[...] = alpha_o * acc_scratch_ref[...] + pv

    # --- Store updated running state to VMEM ---
    m_scratch_ref[...] = m_next
    l_scratch_ref[...] = l_next

    # --- Flush to HBM on the last KV tile ---
    # Normalize the accumulated output by the final l value.
    @pl.when(kv_tile_idx == num_kv_tiles - 1)
    def _flush():
        l_final = l_scratch_ref[...][:, :1]   # (block_q, 1)
        o_ref[...] = (acc_scratch_ref[...] / l_final).astype(o_ref.dtype)
        m_ref[...] = m_scratch_ref[...]
        l_ref[...] = l_scratch_ref[...]


def flash_attention_forward(
    q: jnp.ndarray,    # (batch, heads, seq_q, d_k)
    k: jnp.ndarray,    # (batch, heads, seq_kv, d_k)
    v: jnp.ndarray,    # (batch, heads, seq_kv, d_v)
    *,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    block_sizes: Optional[BlockSizes] = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Flash Attention forward pass.

    Returns:
        o:   (batch, heads, seq_q, d_v)  — attention output
        m:   (batch, heads, seq_q, MIN_BLOCK_SIZE)  — per-row running max
        l:   (batch, heads, seq_q, MIN_BLOCK_SIZE)  — per-row normalizer

    m and l are saved for the backward pass. They encode the log-sum-exp
    without requiring the attention matrix to be stored.
    """
    batch, heads, seq_q, d_k = q.shape
    _, _, seq_kv, d_v = v.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d_k)

    if block_sizes is None:
        block_sizes = get_block_sizes(seq_q, d_k, q.dtype)

    block_q = block_sizes.block_q
    block_kv = block_sizes.block_kv

    validate_shapes(q, k, v, block_q, block_kv)

    num_q_tiles = seq_q // block_q
    num_kv_tiles = seq_kv // block_kv

    # Mask value: large negative, not -inf, to avoid NaN in masked softmax rows.
    mask_value = -0.7 * float(jnp.finfo(jnp.float32).max)

    kernel = functools.partial(
        _flash_fwd_kernel,
        causal=causal,
        sm_scale=sm_scale,
        block_q=block_q,
        block_kv=block_kv,
        kv_seq_len=seq_kv,
        mask_value=mask_value,
    )

    o, m, l = pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct((batch, heads, seq_q, d_v), q.dtype),
            jax.ShapeDtypeStruct(
                (batch, heads, seq_q, MIN_BLOCK_SIZE), jnp.float32
            ),
            jax.ShapeDtypeStruct(
                (batch, heads, seq_q, MIN_BLOCK_SIZE), jnp.float32
            ),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, heads, num_q_tiles, num_kv_tiles),
            in_specs=[
                # Q: indexed by q_tile, constant over kv_tile
                pl.BlockSpec(
                    block_shape=(None, None, block_q, d_k),
                    index_map=lambda b, h, i, j: (b, h, i, 0),
                ),
                # K: indexed by kv_tile, constant over q_tile
                pl.BlockSpec(
                    block_shape=(None, None, block_kv, d_k),
                    index_map=lambda b, h, i, j: (b, h, j, 0),
                ),
                # V: indexed by kv_tile, constant over q_tile
                pl.BlockSpec(
                    block_shape=(None, None, block_kv, d_v),
                    index_map=lambda b, h, i, j: (b, h, j, 0),
                ),
            ],
            out_specs=[
                # O, m, l: indexed by q_tile only — all kv_tiles map to same
                # location; we guard writes with pl.when(kv_tile == last).
                pl.BlockSpec(
                    block_shape=(None, None, block_q, d_v),
                    index_map=lambda b, h, i, j: (b, h, i, 0),
                ),
                pl.BlockSpec(
                    block_shape=(None, None, block_q, MIN_BLOCK_SIZE),
                    index_map=lambda b, h, i, j: (b, h, i, 0),
                ),
                pl.BlockSpec(
                    block_shape=(None, None, block_q, MIN_BLOCK_SIZE),
                    index_map=lambda b, h, i, j: (b, h, i, 0),
                ),
            ],
            scratch_shapes=[
                # VMEM scratch: persists across kv_tile steps for fixed (b,h,q_tile)
                pltpu.VMEM((block_q, MIN_BLOCK_SIZE), jnp.float32),  # m_scratch
                pltpu.VMEM((block_q, MIN_BLOCK_SIZE), jnp.float32),  # l_scratch
                pltpu.VMEM((block_q, d_v), jnp.float32),             # acc_scratch
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",   # batch
                "parallel",   # heads
                "parallel",   # q_tiles — independent
                "arbitrary",  # kv_tiles — sequential: online softmax state
            )
        ),
        interpret=pallas_interpret_mode(),
    )(q, k, v)

    return o, m, l
