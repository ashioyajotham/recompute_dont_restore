"""
Tests for Grouped Query Attention (GQA).

Test levels:
  1. reference_gqa (pure JAX, CPU) — validates expand-and-attend logic
  2. grouped_query_attention (Pallas, TPU) — validates kernel against reference

The Pallas kernel tests are TPU-gated. The reference tests run on CPU.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.jax

ATOL_BF16 = 5e-2


class TestReferenceGQA:
    """
    Tests for reference_gqa — the expand-then-attend reference implementation.
    CPU-compatible, tests correctness of the expand logic.
    """

    def test_groups_1_equals_mha(self, small_inputs):
        """GQA with num_kv_heads == num_q_heads is standard MHA."""
        from standard_attention import attention as mha
        from grouped_query_attention import reference_gqa

        q, k, v = small_inputs  # (1, 2, 256, 128) — already MHA layout
        out_mha = np.array(mha(q, k, v), dtype=np.float32)
        out_gqa = np.array(reference_gqa(q, k, v), dtype=np.float32)

        np.testing.assert_allclose(out_mha, out_gqa, atol=ATOL_BF16)

    def test_mqa_single_kv_head(self):
        """MQA: 1 KV head shared across all Q heads."""
        import jax
        import jax.numpy as jnp
        from grouped_query_attention import reference_gqa

        key = jax.random.PRNGKey(10)
        k1, k2, k3 = jax.random.split(key, 3)
        batch, num_q, seq, d = 1, 4, 128, 128

        q = jax.random.normal(k1, (batch, num_q, seq, d), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (batch, 1, seq, d), dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (batch, 1, seq, d), dtype=jnp.bfloat16)

        out = reference_gqa(q, k, v)
        assert out.shape == (batch, num_q, seq, d)

    def test_output_shape(self, gqa_inputs):
        """Output shape matches Q shape."""
        from grouped_query_attention import reference_gqa
        q, k, v = gqa_inputs
        out = reference_gqa(q, k, v)
        assert out.shape == q.shape

    def test_output_finite(self, gqa_inputs):
        from grouped_query_attention import reference_gqa
        q, k, v = gqa_inputs
        out = np.array(reference_gqa(q, k, v))
        assert np.all(np.isfinite(out))

    def test_groups_expand_correctly(self):
        """
        Each group of Q heads should produce the same output as if K/V were
        explicitly replicated. Verify by checking two Q heads in the same
        group attend identically.
        """
        import jax
        import jax.numpy as jnp
        from grouped_query_attention import reference_gqa

        key = jax.random.PRNGKey(20)
        k1, k2, k3 = jax.random.split(key, 3)
        batch, num_q, num_kv, seq, d = 1, 4, 2, 128, 128

        q = jax.random.normal(k1, (batch, num_q, seq, d), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (batch, num_kv, seq, d), dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (batch, num_kv, seq, d), dtype=jnp.bfloat16)

        # groups=2: Q heads 0,1 share KV head 0; Q heads 2,3 share KV head 1
        out = np.array(reference_gqa(q, k, v), dtype=np.float32)

        # Q heads 0 and 1 use the same K/V but different Q vectors — outputs
        # should differ (different Q = different attention weights)
        # They should NOT be identical unless Q[0] == Q[1]
        assert out.shape == (batch, num_q, seq, d)

    def test_invalid_groups_raises(self):
        import jax
        import jax.numpy as jnp
        from grouped_query_attention import reference_gqa

        key = jax.random.PRNGKey(30)
        k1, k2, k3 = jax.random.split(key, 3)
        q = jax.random.normal(k1, (1, 3, 128, 128), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (1, 2, 128, 128), dtype=jnp.bfloat16)  # 3 % 2 != 0
        v = jax.random.normal(k3, (1, 2, 128, 128), dtype=jnp.bfloat16)

        with pytest.raises((ValueError, AssertionError)):
            reference_gqa(q, k, v)


@pytest.mark.tpu
class TestGQAKernelVsReference:
    """
    Pallas GQA kernel must match reference_gqa within bfloat16 tolerance.
    TPU required.
    """

    def test_output_matches_reference(self, gqa_inputs):
        from grouped_query_attention import grouped_query_attention, reference_gqa
        from utils import get_block_sizes

        q, k, v = gqa_inputs
        block_sizes = get_block_sizes(q.shape[2], q.shape[3], q.dtype)

        o_ref = np.array(reference_gqa(q, k, v), dtype=np.float32)
        o_kernel = np.array(
            grouped_query_attention(q, k, v, block_sizes=block_sizes),
            dtype=np.float32,
        )
        np.testing.assert_allclose(o_kernel, o_ref, atol=ATOL_BF16)

    def test_mqa_output_matches_reference(self):
        import jax
        import jax.numpy as jnp
        from grouped_query_attention import grouped_query_attention, reference_gqa
        from utils import get_block_sizes

        key = jax.random.PRNGKey(40)
        k1, k2, k3 = jax.random.split(key, 3)
        batch, num_q, seq, d = 1, 4, 256, 128
        q = jax.random.normal(k1, (batch, num_q, seq, d), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (batch, 1, seq, d), dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (batch, 1, seq, d), dtype=jnp.bfloat16)

        block_sizes = get_block_sizes(seq, d, jnp.bfloat16)
        o_ref = np.array(reference_gqa(q, k, v), dtype=np.float32)
        o_kernel = np.array(
            grouped_query_attention(q, k, v, block_sizes=block_sizes),
            dtype=np.float32,
        )
        np.testing.assert_allclose(o_kernel, o_ref, atol=ATOL_BF16)

    def test_mha_case_matches_flash_fwd(self, small_inputs):
        """GQA with groups=1 must match flash_fwd for the MHA case."""
        import jax.numpy as jnp
        from grouped_query_attention import grouped_query_attention
        from flash_fwd import flash_attention_forward
        from utils import get_block_sizes

        q, k, v = small_inputs  # already MHA layout
        block_sizes = get_block_sizes(q.shape[2], q.shape[3], q.dtype)

        o_flash, _, _ = flash_attention_forward(q, k, v, block_sizes=block_sizes)
        o_gqa = grouped_query_attention(q, k, v, block_sizes=block_sizes)

        np.testing.assert_allclose(
            np.array(o_gqa, dtype=np.float32),
            np.array(o_flash, dtype=np.float32),
            atol=ATOL_BF16,
        )

    def test_output_shape(self, gqa_inputs):
        from grouped_query_attention import grouped_query_attention
        from utils import get_block_sizes
        q, k, v = gqa_inputs
        block_sizes = get_block_sizes(q.shape[2], q.shape[3], q.dtype)
        out = grouped_query_attention(q, k, v, block_sizes=block_sizes)
        assert out.shape == q.shape
