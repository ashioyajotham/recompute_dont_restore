"""
Grouped Query Attention (GQA) in Pallas.

GQA generalizes Multi-Head Attention (MHA) and Multi-Query Attention (MQA):
  MHA:  num_kv_heads == num_q_heads   (every Q head has its own K, V)
  GQA:  num_kv_heads divides num_q_heads (groups of Q heads share K, V)
  MQA:  num_kv_heads == 1             (all Q heads share one K, V pair)

Used in: Gemma (GQA), LLaMA-3 (GQA), Mistral (GQA), PaLM-2 (MQA).

Implementation approach
-----------------------
The kernel body is IDENTICAL to the MHA kernel in flash_fwd._flash_fwd_kernel.
We import it directly — no copy-paste.

The only thing that differs between MHA and GQA is the BlockSpec index_map
for K and V.  MHA uses:
    index_map=lambda b, h, i, j: (b, h, j, 0)

GQA uses:
    index_map=lambda b, h, i, j: (b, h // groups, j, 0)

That integer division is the complete GQA extension point.  Multiple Q-head
tiles that belong to the same KV group map to the same KV tile in HBM;
Mosaic's prefetch pipeline handles the shared reads with no kernel changes.

The grid still runs over (batch, num_q_heads, q_tiles, kv_tiles).
K and V have shape (batch, num_kv_heads, seq_kv, d_k).
"""

import functools
import math
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "03_pallas_kernels"))

from utils import MIN_BLOCK_SIZE, BlockSizes, get_block_sizes
# Import the kernel directly — GQA uses _flash_fwd_kernel unchanged.
from flash_fwd import _flash_fwd_kernel


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

    The kernel body (_flash_fwd_kernel) is shared with flash_fwd.py.
    The only GQA-specific code is the ``h // groups`` index_map below.
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

    # _flash_fwd_kernel is imported unchanged from flash_fwd.py.
    # All GQA logic lives in the index_maps below, not here.
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
                # Q: indexed by q_head h, q_tile i — identical to MHA.
                pl.BlockSpec(
                    block_shape=(block_q, d_k),
                    index_map=lambda b, h, i, j: (b, h, i, 0),
                ),
                # K: GQA mapping — h // groups selects the KV head.
                # This single lambda is the complete GQA extension point.
                pl.BlockSpec(
                    block_shape=(block_kv, d_k),
                    index_map=lambda b, h, i, j: (b, h // groups, j, 0),
                ),
                # V: same GQA mapping as K.
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
