"""
Block size selection and shape validation for Pallas flash attention kernels.

TPU hardware constraints:
- The TensorCore minimum tile dimension is 128. Every block dimension along
  the sequence axis must be a multiple of 128.
- VMEM (scratchpad) is ~16 MB per TensorCore on TPU v4. Exceeding it causes
  spilling to HBM, defeating the purpose of tiling.
- Input dtype must be bfloat16 or float16; accumulators are always float32.
"""

import math
from typing import NamedTuple

import jax.numpy as jnp

# Hardware constant: TPU TensorCore minimum tile dimension along any axis.
MIN_BLOCK_SIZE: int = 128

# Conservative VMEM budget per kernel invocation (bytes).
# True VMEM capacity is ~16 MB on v4, but we leave headroom for temporaries.
DEFAULT_VMEM_BUDGET: int = 8 * 1024 * 1024  # 8 MB


class BlockSizes(NamedTuple):
    block_q: int
    block_kv: int


def vmem_usage_bytes(
    block_q: int,
    block_kv: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> int:
    """
    Estimate VMEM bytes needed for one kernel step.

    Accounts for:
      - Q tile:          block_q  x head_dim  (dtype)
      - K tile:          block_kv x head_dim  (dtype)
      - V tile:          block_kv x head_dim  (dtype)
      - acc_scratch:     block_q  x head_dim  (float32)
      - m_scratch:       block_q  x MIN_BLOCK_SIZE (float32)
      - l_scratch:       block_q  x MIN_BLOCK_SIZE (float32)
      - logits buffer:   block_q  x block_kv  (float32)
    """
    itemsize = jnp.dtype(dtype).itemsize
    f32 = 4  # float32 itemsize

    qkv = (block_q + 2 * block_kv) * head_dim * itemsize
    acc = block_q * head_dim * f32
    stats = 2 * block_q * MIN_BLOCK_SIZE * f32
    logits = block_q * block_kv * f32

    return qkv + acc + stats + logits


def get_block_sizes(
    seq_len: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.bfloat16,
    vmem_budget: int = DEFAULT_VMEM_BUDGET,
) -> BlockSizes:
    """
    Select block sizes that fit within VMEM.

    Strategy:
      1. Start from a sequence-length-dependent initial guess.
      2. Reduce block_kv first (less impact on throughput than block_q).
      3. Hard minimum: MIN_BLOCK_SIZE for both dimensions.
    """
    if seq_len >= 4096:
        block_q = block_kv = 512
    elif seq_len >= 1024:
        block_q = block_kv = 256
    else:
        block_q = block_kv = 128

    # Snap to multiples of MIN_BLOCK_SIZE
    block_q = _round_down(block_q, MIN_BLOCK_SIZE)
    block_kv = _round_down(block_kv, MIN_BLOCK_SIZE)

    # Reduce until we fit in VMEM
    while (
        vmem_usage_bytes(block_q, block_kv, head_dim, dtype) > vmem_budget
        and block_kv > MIN_BLOCK_SIZE
    ):
        block_kv = max(block_kv // 2, MIN_BLOCK_SIZE)

    while (
        vmem_usage_bytes(block_q, block_kv, head_dim, dtype) > vmem_budget
        and block_q > MIN_BLOCK_SIZE
    ):
        block_q = max(block_q // 2, MIN_BLOCK_SIZE)

    return BlockSizes(block_q=block_q, block_kv=block_kv)


def validate_shapes(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    block_q: int,
    block_kv: int,
) -> None:
    """
    Validate shapes and dtypes before kernel dispatch.

    Raises ValueError with a diagnostic message on any violation.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"Expected 4-D inputs (batch, heads, seq, dim), "
            f"got q={q.shape}, k={k.shape}, v={v.shape}"
        )

    batch, heads, seq_q, d_k = q.shape
    _, kv_heads, seq_kv, d_k2 = k.shape
    _, _, seq_kv2, d_v = v.shape

    if d_k != d_k2:
        raise ValueError(f"Q head dim {d_k} != K head dim {d_k2}")
    if seq_kv != seq_kv2:
        raise ValueError(f"K seq len {seq_kv} != V seq len {seq_kv2}")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Batch size mismatch across Q, K, V")

    if seq_q % block_q != 0:
        raise ValueError(
            f"seq_q={seq_q} must be divisible by block_q={block_q}"
        )
    if seq_kv % block_kv != 0:
        raise ValueError(
            f"seq_kv={seq_kv} must be divisible by block_kv={block_kv}"
        )
    if block_q % MIN_BLOCK_SIZE != 0 or block_q < MIN_BLOCK_SIZE:
        raise ValueError(
            f"block_q={block_q} must be a multiple of MIN_BLOCK_SIZE={MIN_BLOCK_SIZE}"
        )
    if block_kv % MIN_BLOCK_SIZE != 0 or block_kv < MIN_BLOCK_SIZE:
        raise ValueError(
            f"block_kv={block_kv} must be a multiple of MIN_BLOCK_SIZE={MIN_BLOCK_SIZE}"
        )
    if d_k % MIN_BLOCK_SIZE != 0:
        raise ValueError(
            f"head_dim={d_k} must be a multiple of MIN_BLOCK_SIZE={MIN_BLOCK_SIZE}. "
            f"Standard head dims: 64 is not supported; use 128 or 256."
        )

    allowed_dtypes = {jnp.bfloat16, jnp.float16}
    for name, arr in [("q", q), ("k", k), ("v", v)]:
        if arr.dtype not in allowed_dtypes:
            raise ValueError(
                f"{name} dtype {arr.dtype} not supported. Use bfloat16 or float16."
            )


def cdiv(x: int, y: int) -> int:
    """Ceiling division."""
    return (x + y - 1) // y


def next_power_of_2(x: int) -> int:
    """Smallest power of 2 >= x."""
    if x <= 0:
        raise ValueError(f"x must be positive, got {x}")
    return 1 << (x - 1).bit_length()


def _round_down(x: int, multiple: int) -> int:
    return (x // multiple) * multiple


def attention_matrix_bytes(
    seq_len: int,
    heads: int,
    batch: int = 1,
    dtype: jnp.dtype = jnp.bfloat16,
) -> int:
    """
    HBM bytes consumed by a materialized attention matrix.

    This is the term Flash Attention eliminates from peak memory.
    For a single head: seq^2 * itemsize.
    """
    itemsize = jnp.dtype(dtype).itemsize
    return batch * heads * seq_len * seq_len * itemsize
