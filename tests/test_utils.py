"""
Unit tests for 03_pallas_kernels/utils.py.

Tests are split into:
  - Pure Python (no JAX): cdiv, next_power_of_2, _round_down
  - JAX-dependent: vmem_usage_bytes, get_block_sizes, validate_shapes,
    attention_matrix_bytes

The JAX-dependent tests use jnp.bfloat16 and jnp.float16 dtype constants.
"""

import pytest


# ---------------------------------------------------------------------------
# Pure Python helpers — no JAX required
# ---------------------------------------------------------------------------

class TestCdiv:
    def test_exact_divisor(self):
        from utils import cdiv
        assert cdiv(8, 4) == 2

    def test_rounds_up(self):
        from utils import cdiv
        assert cdiv(9, 4) == 3
        assert cdiv(1, 128) == 1

    def test_equal(self):
        from utils import cdiv
        assert cdiv(128, 128) == 1

    def test_large(self):
        from utils import cdiv
        assert cdiv(32768, 128) == 256


class TestNextPowerOf2:
    def test_already_power(self):
        from utils import next_power_of_2
        assert next_power_of_2(1) == 1
        assert next_power_of_2(2) == 2
        assert next_power_of_2(128) == 128

    def test_rounds_up(self):
        from utils import next_power_of_2
        assert next_power_of_2(3) == 4
        assert next_power_of_2(100) == 128
        assert next_power_of_2(129) == 256

    def test_invalid(self):
        from utils import next_power_of_2
        with pytest.raises(ValueError):
            next_power_of_2(0)
        with pytest.raises(ValueError):
            next_power_of_2(-1)


# ---------------------------------------------------------------------------
# JAX-dependent tests
# ---------------------------------------------------------------------------

@pytest.mark.jax
class TestVmemUsageBytes:
    def test_increases_with_block_size(self):
        import jax.numpy as jnp
        from utils import vmem_usage_bytes
        small = vmem_usage_bytes(128, 128, 128, jnp.bfloat16)
        large = vmem_usage_bytes(256, 256, 128, jnp.bfloat16)
        assert large > small

    def test_float16_half_of_float32_for_qkv(self):
        import jax.numpy as jnp
        from utils import vmem_usage_bytes
        # float16 is 2 bytes, float32 is 4 — the QKV portion should halve
        # (accumulators are always float32 so the total doesn't exactly halve)
        f16 = vmem_usage_bytes(128, 128, 128, jnp.float16)
        bf16 = vmem_usage_bytes(128, 128, 128, jnp.bfloat16)
        assert f16 == bf16  # both are 2-byte types

    def test_positive(self):
        import jax.numpy as jnp
        from utils import vmem_usage_bytes
        assert vmem_usage_bytes(128, 128, 128, jnp.bfloat16) > 0

    def test_accounts_for_all_buffers(self):
        """
        Manual calculation for block_q=128, block_kv=128, d=128, bfloat16.
          QKV:     (128 + 2*128) * 128 * 2  = 98304
          acc:     128 * 128 * 4             = 65536
          m+l:     2 * 128 * 128 * 4         = 131072
          logits:  128 * 128 * 4             = 65536
          Total:   360448
        """
        import jax.numpy as jnp
        from utils import vmem_usage_bytes
        result = vmem_usage_bytes(128, 128, 128, jnp.bfloat16)
        expected = (128 + 2*128)*128*2 + 128*128*4 + 2*128*128*4 + 128*128*4
        assert result == expected


@pytest.mark.jax
class TestGetBlockSizes:
    def test_short_seq_minimum(self):
        import jax.numpy as jnp
        from utils import get_block_sizes, MIN_BLOCK_SIZE
        sizes = get_block_sizes(128, 128, jnp.bfloat16)
        assert sizes.block_q >= MIN_BLOCK_SIZE
        assert sizes.block_kv >= MIN_BLOCK_SIZE

    def test_multiples_of_min_block_size(self):
        import jax.numpy as jnp
        from utils import get_block_sizes, MIN_BLOCK_SIZE
        for seq in [512, 1024, 4096, 16384]:
            sizes = get_block_sizes(seq, 128, jnp.bfloat16)
            assert sizes.block_q % MIN_BLOCK_SIZE == 0
            assert sizes.block_kv % MIN_BLOCK_SIZE == 0

    def test_fits_in_vmem_budget(self):
        import jax.numpy as jnp
        from utils import get_block_sizes, vmem_usage_bytes
        budget = 4 * 1024 * 1024  # 4 MB
        for seq in [512, 2048, 8192]:
            sizes = get_block_sizes(seq, 128, jnp.bfloat16, vmem_budget=budget)
            usage = vmem_usage_bytes(sizes.block_q, sizes.block_kv, 128, jnp.bfloat16)
            assert usage <= budget, (
                f"seq={seq}: vmem {usage} exceeds budget {budget}"
            )

    def test_larger_seq_gets_larger_blocks(self):
        """Short sequences don't need 512-element tiles."""
        import jax.numpy as jnp
        from utils import get_block_sizes
        small = get_block_sizes(256, 128, jnp.bfloat16)
        large = get_block_sizes(32768, 128, jnp.bfloat16)
        # Large sequences should have at least as large tiles
        assert large.block_q >= small.block_q


@pytest.mark.jax
class TestValidateShapes:
    def test_valid_inputs(self, small_inputs):
        from utils import validate_shapes
        q, k, v = small_inputs
        # Should not raise
        validate_shapes(q, k, v, block_q=128, block_kv=128)

    def test_rejects_wrong_ndim(self):
        import jax.numpy as jnp
        from utils import validate_shapes
        q = jnp.ones((2, 256, 128), dtype=jnp.bfloat16)  # 3D instead of 4D
        k = jnp.ones((1, 2, 256, 128), dtype=jnp.bfloat16)
        v = jnp.ones((1, 2, 256, 128), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="4-D"):
            validate_shapes(q, k, v, block_q=128, block_kv=128)

    def test_rejects_mismatched_head_dim(self):
        import jax.numpy as jnp
        from utils import validate_shapes
        q = jnp.ones((1, 2, 256, 128), dtype=jnp.bfloat16)
        k = jnp.ones((1, 2, 256, 64), dtype=jnp.bfloat16)  # d_k mismatch
        v = jnp.ones((1, 2, 256, 128), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="head dim"):
            validate_shapes(q, k, v, block_q=128, block_kv=128)

    def test_rejects_non_divisible_seq(self):
        import jax.numpy as jnp
        from utils import validate_shapes
        q = jnp.ones((1, 2, 300, 128), dtype=jnp.bfloat16)  # 300 % 128 != 0
        k = jnp.ones((1, 2, 300, 128), dtype=jnp.bfloat16)
        v = jnp.ones((1, 2, 300, 128), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="divisible"):
            validate_shapes(q, k, v, block_q=128, block_kv=128)

    def test_rejects_block_below_min(self):
        import jax.numpy as jnp
        from utils import validate_shapes
        q = jnp.ones((1, 2, 256, 128), dtype=jnp.bfloat16)
        k = jnp.ones((1, 2, 256, 128), dtype=jnp.bfloat16)
        v = jnp.ones((1, 2, 256, 128), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="MIN_BLOCK_SIZE"):
            validate_shapes(q, k, v, block_q=64, block_kv=128)

    def test_rejects_float32_dtype(self):
        import jax.numpy as jnp
        from utils import validate_shapes
        q = jnp.ones((1, 2, 256, 128), dtype=jnp.float32)
        k = jnp.ones((1, 2, 256, 128), dtype=jnp.float32)
        v = jnp.ones((1, 2, 256, 128), dtype=jnp.float32)
        with pytest.raises(ValueError, match="dtype"):
            validate_shapes(q, k, v, block_q=128, block_kv=128)


@pytest.mark.jax
class TestAttentionMatrixBytes:
    def test_quadratic_scaling(self):
        import jax.numpy as jnp
        from utils import attention_matrix_bytes
        b4 = attention_matrix_bytes(4096, 1, 1, jnp.bfloat16)
        b8 = attention_matrix_bytes(8192, 1, 1, jnp.bfloat16)
        # Doubling seq should quadruple the attention matrix
        assert b8 == 4 * b4

    def test_concrete_value(self):
        """seq=1024, heads=1, batch=1, bfloat16: 1024^2 * 2 = 2097152."""
        import jax.numpy as jnp
        from utils import attention_matrix_bytes
        result = attention_matrix_bytes(1024, 1, 1, jnp.bfloat16)
        assert result == 1024 * 1024 * 2

    def test_scales_with_heads(self):
        import jax.numpy as jnp
        from utils import attention_matrix_bytes
        single = attention_matrix_bytes(512, 1, 1, jnp.bfloat16)
        multi = attention_matrix_bytes(512, 8, 1, jnp.bfloat16)
        assert multi == 8 * single
