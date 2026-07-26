"""
Flash Attention backward pass with recomputation.

The key insight: the backward pass does NOT load a stored attention matrix.
Instead, it recomputes attention weights from Q, K, V and the saved per-row
log-sum-exp values (m, l) that the forward pass wrote. This means the memory
footprint of the combined forward+backward is still O(seq * d), not O(seq^2).

Algorithm (per dK, dV for one KV tile):
  For each Q tile (i = 0..num_q_tiles-1):
    1. Recompute S_ij = Q_i @ K_j^T * scale
    2. Recompute P_ij = exp(S_ij - m_i) / l_i   (no stored attention matrix)
    3. dV_j  += P_ij^T @ dO_i
    4. dP_ij  = dO_i @ V_j^T
    5. delta_i = rowsum(dO_i * O_i)               (D term in FA2 backward)
    6. dS_ij  = P_ij * (dP_ij - delta_i)
    7. dK_j  += dS_ij^T @ Q_i * scale
    8. dQ_i  += dS_ij @ K_j * scale

Two Pallas kernels:
  _flash_bwd_dkv_kernel:  grid over KV tiles, lax.fori_loop over Q tiles
  _flash_bwd_dq_kernel:   grid over Q tiles, lax.fori_loop over KV tiles

Both are wired into flash_attention via jax.custom_vjp.
"""

import functools
import math
from typing import Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from utils import MIN_BLOCK_SIZE, BlockSizes, get_block_sizes, validate_shapes
from flash_fwd import flash_attention_forward


# ---------------------------------------------------------------------------
# dK, dV kernel
# ---------------------------------------------------------------------------

def _flash_bwd_dkv_kernel(
    # Full-sequence input refs (loaded in full for each kv_tile grid step)
    q_ref,      # (seq_q, d_k)     bfloat16
    do_ref,     # (seq_q, d_v)     bfloat16  upstream gradient
    o_ref,      # (seq_q, d_v)     bfloat16  forward output (for delta)
    m_ref,      # (seq_q, MIN_BLOCK_SIZE) float32  saved m from forward
    l_ref,      # (seq_q, MIN_BLOCK_SIZE) float32  saved l from forward
    # KV tile refs
    k_ref,      # (block_kv, d_k)  bfloat16
    v_ref,      # (block_kv, d_v)  bfloat16
    # Output gradient refs
    dk_ref,     # (block_kv, d_k)
    dv_ref,     # (block_kv, d_v)
    *,
    causal: bool,
    sm_scale: float,
    block_q: int,
    block_kv: int,
    q_seq_len: int,
    kv_seq_len: int,
    mask_value: float,
):
    kv_tile_idx = pl.program_id(2)   # grid: (batch, heads, kv_tiles)

    k = k_ref[...].astype(jnp.float32)   # (block_kv, d_k)
    v = v_ref[...].astype(jnp.float32)   # (block_kv, d_v)

    kv_start = kv_tile_idx * block_kv

    def q_body(q_tile_idx, carry):
        dk_acc, dv_acc = carry

        q_start = q_tile_idx * block_q
        q_slice = pl.ds(q_start, block_q)

        # Dynamically load Q tile from full-sequence ref
        q = q_ref[q_slice, :].astype(jnp.float32)      # (block_q, d_k)
        do = do_ref[q_slice, :].astype(jnp.float32)    # (block_q, d_v)
        o = o_ref[q_slice, :].astype(jnp.float32)      # (block_q, d_v)
        m = m_ref[q_slice, :1].astype(jnp.float32)     # (block_q, 1)
        l = l_ref[q_slice, :1].astype(jnp.float32)     # (block_q, 1)

        # Recompute attention logits — no stored attention matrix needed
        logits = (
            lax.dot_general(
                q, k,
                dimension_numbers=(([1], [1]), ([], [])),
                preferred_element_type=jnp.float32,
            )
            * sm_scale
        )  # (block_q, block_kv)

        if causal:
            q_pos = (q_start + jnp.arange(block_q))[:, None]
            kv_pos = (kv_start + jnp.arange(block_kv))[None, :]
            logits = jnp.where(q_pos >= kv_pos, logits, mask_value)

        # Recompute P from saved m, l (the recomputation trick)
        p = jnp.exp(logits - m) / l   # (block_q, block_kv)

        # dV += P^T @ dO
        dv_acc = dv_acc + lax.dot(
            p.T, do, preferred_element_type=jnp.float32
        )  # (block_kv, d_v)

        # delta_i = rowsum(dO * O): contribution to the D term
        delta = jnp.sum(do * o, axis=1, keepdims=True)   # (block_q, 1)

        # dP = dO @ V^T
        dp = lax.dot_general(
            do, v,
            dimension_numbers=(([1], [1]), ([], [])),
            preferred_element_type=jnp.float32,
        )  # (block_q, block_kv)

        # dS = P * (dP - delta)  — from the softmax Jacobian identity
        ds = p * (dp - delta)   # (block_q, block_kv)

        # dK += dS^T @ Q * scale
        dk_acc = dk_acc + lax.dot(
            ds.T, q, preferred_element_type=jnp.float32
        ) * sm_scale  # (block_kv, d_k)

        return dk_acc, dv_acc

    num_q_tiles = q_seq_len // block_q
    dk_init = jnp.zeros((block_kv, k.shape[-1]), jnp.float32)
    dv_init = jnp.zeros((block_kv, v.shape[-1]), jnp.float32)

    dk, dv = lax.fori_loop(0, num_q_tiles, q_body, (dk_init, dv_init))

    dk_ref[...] = dk.astype(dk_ref.dtype)
    dv_ref[...] = dv.astype(dv_ref.dtype)


# ---------------------------------------------------------------------------
# dQ kernel
# ---------------------------------------------------------------------------

def _flash_bwd_dq_kernel(
    # Q tile ref
    q_ref,      # (block_q, d_k)   bfloat16
    do_ref,     # (block_q, d_v)   bfloat16
    m_ref,      # (block_q, MIN_BLOCK_SIZE) float32
    l_ref,      # (block_q, MIN_BLOCK_SIZE) float32
    o_ref,      # (block_q, d_v)   bfloat16
    # Full KV sequence refs
    k_full_ref,  # (seq_kv, d_k)   bfloat16
    v_full_ref,  # (seq_kv, d_v)   bfloat16
    # Output
    dq_ref,     # (block_q, d_k)
    *,
    causal: bool,
    sm_scale: float,
    block_q: int,
    block_kv: int,
    kv_seq_len: int,
    mask_value: float,
):
    q_tile_idx = pl.program_id(2)   # grid: (batch, heads, q_tiles)
    q_start = q_tile_idx * block_q

    q = q_ref[...].astype(jnp.float32)    # (block_q, d_k)
    do = do_ref[...].astype(jnp.float32)  # (block_q, d_v)
    o = o_ref[...].astype(jnp.float32)    # (block_q, d_v)
    m = m_ref[...][..., :1].astype(jnp.float32)   # (block_q, 1)
    l = l_ref[...][..., :1].astype(jnp.float32)   # (block_q, 1)

    # delta: D term, per query row
    delta = jnp.sum(do * o, axis=1, keepdims=True)   # (block_q, 1)

    def kv_body(kv_tile_idx, dq_acc):
        kv_start = kv_tile_idx * block_kv
        kv_slice = pl.ds(kv_start, block_kv)

        k = k_full_ref[kv_slice, :].astype(jnp.float32)  # (block_kv, d_k)
        v = v_full_ref[kv_slice, :].astype(jnp.float32)  # (block_kv, d_v)

        logits = (
            lax.dot_general(
                q, k,
                dimension_numbers=(([1], [1]), ([], [])),
                preferred_element_type=jnp.float32,
            )
            * sm_scale
        )  # (block_q, block_kv)

        if causal:
            q_pos = (q_start + jnp.arange(block_q))[:, None]
            kv_pos = (kv_start + jnp.arange(block_kv))[None, :]
            logits = jnp.where(q_pos >= kv_pos, logits, mask_value)

        p = jnp.exp(logits - m) / l   # (block_q, block_kv)

        dp = lax.dot_general(
            do, v,
            dimension_numbers=(([1], [1]), ([], [])),
            preferred_element_type=jnp.float32,
        )  # (block_q, block_kv)

        ds = p * (dp - delta)   # (block_q, block_kv)

        # dQ += dS @ K * scale
        dq_acc = dq_acc + lax.dot(
            ds, k, preferred_element_type=jnp.float32
        ) * sm_scale  # (block_q, d_k)

        return dq_acc

    num_kv_tiles = kv_seq_len // block_kv
    dq_init = jnp.zeros(q.shape, jnp.float32)
    dq = lax.fori_loop(0, num_kv_tiles, kv_body, dq_init)

    dq_ref[...] = dq.astype(dq_ref.dtype)


# ---------------------------------------------------------------------------
# custom_vjp wiring
# ---------------------------------------------------------------------------

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def flash_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    block_sizes: Optional[BlockSizes] = None,
) -> jnp.ndarray:
    """
    Flash Attention: forward + backward with recomputation.

    Public API. Returns only the output O; m and l are saved internally
    as residuals and used by the backward pass to avoid storing the attention
    matrix across the forward/backward boundary.
    """
    o, _, _ = flash_attention_forward(
        q, k, v,
        causal=causal,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
    )
    return o


def _fwd(q, k, v, causal, sm_scale, block_sizes):
    o, m, l = flash_attention_forward(
        q, k, v,
        causal=causal,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
    )
    return o, (q, k, v, o, m, l)


def _bwd(causal, sm_scale, block_sizes, residuals, do):
    q, k, v, o, m, l = residuals

    batch, heads, seq_q, d_k = q.shape
    _, _, seq_kv, d_v = v.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d_k)
    if block_sizes is None:
        block_sizes = get_block_sizes(seq_q, d_k, q.dtype)

    block_q = block_sizes.block_q
    block_kv = block_sizes.block_kv

    num_q_tiles = seq_q // block_q
    num_kv_tiles = seq_kv // block_kv
    mask_value = -0.7 * float(jnp.finfo(jnp.float32).max)

    # --- dK, dV via Pallas kernel ---
    # Grid: (batch, heads, kv_tiles)
    # For each KV tile, loop over all Q tiles inside the kernel.
    # Q, dO, O, m, l are loaded as full-sequence tiles (seq_q x dim).

    dkv_kernel = functools.partial(
        _flash_bwd_dkv_kernel,
        causal=causal,
        sm_scale=sm_scale,
        block_q=block_q,
        block_kv=block_kv,
        q_seq_len=seq_q,
        kv_seq_len=seq_kv,
        mask_value=mask_value,
    )

    # VMEM LIMITATION: the dKV kernel loads all five full-sequence tensors
    # (Q, dO, O as bfloat16; m, l as float32) into VMEM at once per KV-tile
    # step.  With d_k=d_v=128 and the 8 MB conservative budget from utils.py:
    #
    #   VMEM = seq_q * (d_k*2 + d_v*2 + d_v*2 + 128*4 + 128*4)  bytes
    #        = seq_q * (256 + 256 + 256 + 512 + 512)             bytes
    #        = seq_q * 1792                                       bytes
    #
    #   seq_q=1024  ->  1.84 MB  (fits)
    #   seq_q=4096  ->  7.34 MB  (fits, barely)
    #   seq_q=8192  -> 14.68 MB  *** exceeds 8 MB budget -> spill or OOM ***
    #
    # This is not merely a "production concern" — it is a hard wall that
    # triggers at seq_q >= 8K on any TPU v4 with standard d_k=128.
    # The fix is two-level tiling: an outer q_major loop in the grid so that
    # only block_q_major rows of Q are in VMEM at a time, matching the forward.
    dk, dv = pl.pallas_call(
        dkv_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, heads, num_kv_tiles),
            in_specs=[
                # Full Q sequence per (b, h) — loaded once per kv_tile step
                pl.BlockSpec(
                    block_shape=(seq_q, d_k),
                    index_map=lambda b, h, j: (b, h, 0, 0),
                ),
                pl.BlockSpec(
                    block_shape=(seq_q, d_v),
                    index_map=lambda b, h, j: (b, h, 0, 0),
                ),
                pl.BlockSpec(
                    block_shape=(seq_q, d_v),
                    index_map=lambda b, h, j: (b, h, 0, 0),
                ),
                pl.BlockSpec(
                    block_shape=(seq_q, MIN_BLOCK_SIZE),
                    index_map=lambda b, h, j: (b, h, 0, 0),
                ),
                pl.BlockSpec(
                    block_shape=(seq_q, MIN_BLOCK_SIZE),
                    index_map=lambda b, h, j: (b, h, 0, 0),
                ),
                # KV tile
                pl.BlockSpec(
                    block_shape=(block_kv, d_k),
                    index_map=lambda b, h, j: (b, h, j, 0),
                ),
                pl.BlockSpec(
                    block_shape=(block_kv, d_v),
                    index_map=lambda b, h, j: (b, h, j, 0),
                ),
            ],
            out_specs=[
                pl.BlockSpec(
                    block_shape=(block_kv, d_k),
                    index_map=lambda b, h, j: (b, h, j, 0),
                ),
                pl.BlockSpec(
                    block_shape=(block_kv, d_v),
                    index_map=lambda b, h, j: (b, h, j, 0),
                ),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(q, do, o, m, l, k, v)

    # --- dQ via Pallas kernel ---
    # Grid: (batch, heads, q_tiles)
    # For each Q tile, loop over all KV tiles inside the kernel.
    #
    # VMEM LIMITATION: the dQ kernel loads full K and V sequences into VMEM
    # once per Q-tile step.  With d_k=d_v=128 in bfloat16:
    #
    #   VMEM_KV = seq_kv * (d_k + d_v) * 2  bytes
    #           = seq_kv * 512              bytes
    #
    #   seq_kv=16384 ->  8.39 MB  *** exceeds 8 MB budget ***
    #   seq_kv=32768 -> 16.78 MB
    #
    # The dQ kernel is less aggressive than dKV (only 2 full-seq buffers vs 5),
    # but still hits the wall at seq_kv >= 16K.  The same two-level tiling fix
    # (outer kv_major loop in the grid) resolves both kernels consistently.

    dq_kernel = functools.partial(
        _flash_bwd_dq_kernel,
        causal=causal,
        sm_scale=sm_scale,
        block_q=block_q,
        block_kv=block_kv,
        kv_seq_len=seq_kv,
        mask_value=mask_value,
    )

    dq = pl.pallas_call(
        dq_kernel,
        out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype)],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, heads, num_q_tiles),
            in_specs=[
                pl.BlockSpec(
                    block_shape=(block_q, d_k),
                    index_map=lambda b, h, i: (b, h, i, 0),
                ),
                pl.BlockSpec(
                    block_shape=(block_q, d_v),
                    index_map=lambda b, h, i: (b, h, i, 0),
                ),
                pl.BlockSpec(
                    block_shape=(block_q, MIN_BLOCK_SIZE),
                    index_map=lambda b, h, i: (b, h, i, 0),
                ),
                pl.BlockSpec(
                    block_shape=(block_q, MIN_BLOCK_SIZE),
                    index_map=lambda b, h, i: (b, h, i, 0),
                ),
                pl.BlockSpec(
                    block_shape=(block_q, d_v),
                    index_map=lambda b, h, i: (b, h, i, 0),
                ),
                # Full KV sequence per (b, h) — see VMEM LIMITATION above.
                pl.BlockSpec(
                    block_shape=(seq_kv, d_k),
                    index_map=lambda b, h, i: (b, h, 0, 0),
                ),
                pl.BlockSpec(
                    block_shape=(seq_kv, d_v),
                    index_map=lambda b, h, i: (b, h, 0, 0),
                ),
            ],
            out_specs=[
                pl.BlockSpec(
                    block_shape=(block_q, d_k),
                    index_map=lambda b, h, i: (b, h, i, 0),
                ),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(q, do, m, l, o, k, v)

    return dq, dk, dv


flash_attention.defvjp(_fwd, _bwd)
