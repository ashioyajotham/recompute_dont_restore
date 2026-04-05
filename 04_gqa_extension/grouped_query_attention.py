"""
Grouped Query Attention (GQA) in Pallas.

GQA generalizes Multi-Head Attention (MHA) and Multi-Query Attention (MQA):
  MHA:  num_kv_heads == num_q_heads   (every Q head has its own K, V)
  GQA:  num_kv_heads divides num_q_heads (groups of Q heads share K, V)
  MQA:  num_kv_heads == 1             (all Q heads share one K, V pair)

Used in: Gemma (GQA), LLaMA-3 (GQA), Mistral (GQA), PaLM-2 (MQA).

Implementation approach
-----------------------
The forward kernel body is identical to flash_fwd._flash_fwd_kernel.
The only change is in the BlockSpec index_map for K and V: instead of mapping
head index h -> h, we map h -> h // groups.

This means multiple Q head tiles read from the same KV tile in HBM. Mosaic
handles the shared reads via its prefetch pipeline — no kernel change needed.

The grid is still (batch, num_q_heads, q_tiles, kv_tiles), but K and V have
shape (batch, num_kv_heads, seq_kv, d_k).
"""

import functools
import math
from typing import Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "03_pallas_kernels"))

from utils import MIN_BLOCK_SIZE, BlockSizes, get_block_sizes


def _gqa_fwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    m_ref,
    l_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal: bool,
    sm_scale: float,
    block_q: int,
    block_kv: int,
    kv_seq_len: int,
    mask_value: float,
):
    """
    Kernel body identical to flash_fwd._flash_fwd_kernel.

    The GQA head mapping (h -> h // groups) is entirely in the index_maps
    passed to BlockSpec; the kernel itself is head-dimension agnostic.
    """
    b = pl.program_id(0)
    h = pl.program_id(1)
    q_tile_idx = pl.program_id(2)
    kv_tile_idx = pl.program_id(3)

    num_kv_tiles = kv_seq_len // block_kv

    @pl.when(kv_tile_idx == 0)
    def _init():
        m_scratch_ref[...] = jnp.full(
            m_scratch_ref.shape, -jnp.inf, jnp.float32
        )
        l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, jnp.float32)
        acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, jnp.float32)

    m_prev = m_scratch_ref[...]
    l_prev = l_scratch_ref[...]

    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)

    logits = (
        lax.dot_general(
            q, k,
            dimension_numbers=(([1], [1]), ([], [])),
            preferred_element_type=jnp.float32,
        )
        * sm_scale
    )

    if causal:
        q_pos = (q_tile_idx * block_q + jnp.arange(block_q))[:, None]
        kv_pos = (kv_tile_idx * block_kv + jnp.arange(block_kv))[None, :]
        logits = jnp.where(q_pos >= kv_pos, logits, mask_value)

    m_curr = jnp.max(logits, axis=1, keepdims=True)
    m_next = jnp.maximum(m_prev, jnp.broadcast_to(m_curr, m_prev.shape))

    m_next_for_logits = jnp.broadcast_to(m_curr, logits.shape)
    p = jnp.exp(logits - m_next_for_logits)

    alpha = jnp.exp(m_prev - m_next)
    l_next = (
        alpha * l_prev
        + jnp.broadcast_to(jnp.sum(p, axis=1, keepdims=True), l_prev.shape)
    )

    v = v_ref[...].astype(jnp.float32)
    pv = lax.dot(p, v, preferred_element_type=jnp.float32)

    alpha_o = jnp.broadcast_to(alpha[:, :1], acc_scratch_ref.shape)
    acc_scratch_ref[...] = alpha_o * acc_scratch_ref[...] + pv

    m_scratch_ref[...] = m_next
    l_scratch_ref[...] = l_next

    @pl.when(kv_tile_idx == num_kv_tiles - 1)
    def _flush():
        l_final = l_scratch_ref[...][:, :1]
        o_ref[...] = (acc_scratch_ref[...] / l_final).astype(o_ref.dtype)
        m_ref[...] = m_scratch_ref[...]
        l_ref[...] = l_scratch_ref[...]


def grouped_query_attention(
    q: jnp.ndarray,    # (batch, num_q_heads, seq_q, d_k)
    k: jnp.ndarray,    # (batch, num_kv_heads, seq_kv, d_k)
    v: jnp.ndarray,    # (batch, num_kv_heads, seq_kv, d_v)
    *,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    block_sizes: Optional[BlockSizes] = None,
) -> jnp.ndarray:
    """
    Flash Attention forward pass with Grouped Query Attention head layout.

    num_q_heads must be divisible by num_kv_heads.
    Special cases:
      num_kv_heads == num_q_heads  =>  standard MHA
      num_kv_heads == 1            =>  MQA (Multi-Query Attention)
    """
    batch, num_q_heads, seq_q, d_k = q.shape
    _, num_kv_heads, seq_kv, d_v = v.shape

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )
    groups = num_q_heads // num_kv_heads

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d_k)
    if block_sizes is None:
        block_sizes = get_block_sizes(seq_q, d_k, q.dtype)

    block_q = block_sizes.block_q
    block_kv = block_sizes.block_kv

    num_q_tiles = seq_q // block_q
    num_kv_tiles = seq_kv // block_kv
    mask_value = -0.7 * float(jnp.finfo(jnp.float32).max)

    kernel = functools.partial(
        _gqa_fwd_kernel,
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
            jax.ShapeDtypeStruct((batch, num_q_heads, seq_q, d_v), q.dtype),
            jax.ShapeDtypeStruct(
                (batch, num_q_heads, seq_q, MIN_BLOCK_SIZE), jnp.float32
            ),
            jax.ShapeDtypeStruct(
                (batch, num_q_heads, seq_q, MIN_BLOCK_SIZE), jnp.float32
            ),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, num_q_heads, num_q_tiles, num_kv_tiles),
            in_specs=[
                # Q: indexed by q_head h, q_tile i
                pl.BlockSpec(
                    block_shape=(block_q, d_k),
                    index_map=lambda b, h, i, j: (b, h, i, 0),
                ),
                # K: head index h maps to h // groups — the GQA mapping
                pl.BlockSpec(
                    block_shape=(block_kv, d_k),
                    index_map=lambda b, h, i, j: (b, h // groups, j, 0),
                ),
                # V: same GQA mapping as K
                pl.BlockSpec(
                    block_shape=(block_kv, d_v),
                    index_map=lambda b, h, i, j: (b, h // groups, j, 0),
                ),
            ],
            out_specs=[
                pl.BlockSpec(
                    block_shape=(block_q, d_v),
                    index_map=lambda b, h, i, j: (b, h, i, 0),
                ),
                pl.BlockSpec(
                    block_shape=(block_q, MIN_BLOCK_SIZE),
                    index_map=lambda b, h, i, j: (b, h, i, 0),
                ),
                pl.BlockSpec(
                    block_shape=(block_q, MIN_BLOCK_SIZE),
                    index_map=lambda b, h, i, j: (b, h, i, 0),
                ),
            ],
            scratch_shapes=[
                pltpu.VMEM((block_q, MIN_BLOCK_SIZE), jnp.float32),
                pltpu.VMEM((block_q, MIN_BLOCK_SIZE), jnp.float32),
                pltpu.VMEM((block_q, d_v), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel", "parallel", "parallel", "arbitrary"
            )
        ),
    )(q, k, v)

    return o


def reference_gqa(
    q: jnp.ndarray,    # (batch, num_q_heads, seq_q, d_k)
    k: jnp.ndarray,    # (batch, num_kv_heads, seq_kv, d_k)
    v: jnp.ndarray,    # (batch, num_kv_heads, seq_kv, d_v)
    *,
    causal: bool = False,
) -> jnp.ndarray:
    """
    Reference GQA implementation in pure JAX for correctness testing.

    Expands K and V to num_q_heads by repeating each KV head `groups` times,
    then calls standard attention. Memory-inefficient but correct.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "02_naive_jax_baseline"))
    from standard_attention import attention

    batch, num_q_heads, seq_q, d_k = q.shape
    _, num_kv_heads, seq_kv, d_v = v.shape
    groups = num_q_heads // num_kv_heads

    # Expand: (batch, num_kv_heads, seq_kv, d) -> (batch, num_q_heads, seq_kv, d)
    k_expanded = jnp.repeat(k, groups, axis=1)
    v_expanded = jnp.repeat(v, groups, axis=1)

    return attention(q, k_expanded, v_expanded, causal=causal)
