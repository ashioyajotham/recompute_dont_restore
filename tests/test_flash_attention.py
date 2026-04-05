"""
Integration tests: flash attention (Pallas) vs naive attention (JAX baseline).

All tests in this file require a TPU device. On CPU, the Pallas TPU-specific
primitives (PrefetchScalarGridSpec, pltpu.VMEM, CompilerParams) are not
available.

To run these locally with the Pallas interpret mode (CPU-compatible but slow):
    JAX_INTERPRET_PALLAS=1 pytest tests/test_flash_attention.py

Note: interpret mode does not emulate all TPU hardware constraints (e.g.
MIN_BLOCK_SIZE=128 is not enforced). Test inputs use block sizes valid on real
TPU hardware for portability.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.tpu

# Absolute tolerance for bfloat16 comparisons. bfloat16 has ~3 decimal digits
# of precision so we expect O(1e-2) error vs float32 reference.
ATOL_BF16 = 5e-2


class TestFlashForwardVsNaive:
    """flash_attention_forward output must match naive attention."""

    def test_output_matches_noncausal(self, small_inputs):
        import jax
        import jax.numpy as jnp
        from standard_attention import attention as naive
        from flash_fwd import flash_attention_forward
        from utils import get_block_sizes

        q, k, v = small_inputs
        block_sizes = get_block_sizes(q.shape[2], q.shape[3], q.dtype)

        o_naive = np.array(naive(q, k, v), dtype=np.float32)
        o_flash, _, _ = flash_attention_forward(q, k, v, block_sizes=block_sizes)
        o_flash = np.array(o_flash, dtype=np.float32)

        np.testing.assert_allclose(o_flash, o_naive, atol=ATOL_BF16)

    def test_output_matches_causal(self, small_inputs):
        import jax.numpy as jnp
        from standard_attention import attention as naive
        from flash_fwd import flash_attention_forward
        from utils import get_block_sizes

        q, k, v = small_inputs
        block_sizes = get_block_sizes(q.shape[2], q.shape[3], q.dtype)

        o_naive = np.array(naive(q, k, v, causal=True), dtype=np.float32)
        o_flash, _, _ = flash_attention_forward(
            q, k, v, causal=True, block_sizes=block_sizes
        )
        o_flash = np.array(o_flash, dtype=np.float32)

        np.testing.assert_allclose(o_flash, o_naive, atol=ATOL_BF16)

    def test_output_shape(self, small_inputs):
        from flash_fwd import flash_attention_forward
        from utils import get_block_sizes

        q, k, v = small_inputs
        block_sizes = get_block_sizes(q.shape[2], q.shape[3], q.dtype)
        o, m, l = flash_attention_forward(q, k, v, block_sizes=block_sizes)
        assert o.shape == q.shape

    def test_lse_shape(self, small_inputs):
        """m and l must have shape (batch, heads, seq_q, MIN_BLOCK_SIZE)."""
        import jax.numpy as jnp
        from flash_fwd import flash_attention_forward
        from utils import get_block_sizes, MIN_BLOCK_SIZE

        q, k, v = small_inputs
        block_sizes = get_block_sizes(q.shape[2], q.shape[3], q.dtype)
        _, m, l = flash_attention_forward(q, k, v, block_sizes=block_sizes)

        batch, heads, seq_q, _ = q.shape
        expected_shape = (batch, heads, seq_q, MIN_BLOCK_SIZE)
        assert m.shape == expected_shape, f"m.shape {m.shape} != {expected_shape}"
        assert l.shape == expected_shape, f"l.shape {l.shape} != {expected_shape}"
        assert m.dtype == jnp.float32
        assert l.dtype == jnp.float32

    def test_output_finite(self, small_inputs):
        from flash_fwd import flash_attention_forward
        from utils import get_block_sizes

        q, k, v = small_inputs
        block_sizes = get_block_sizes(q.shape[2], q.shape[3], q.dtype)
        o, m, l = flash_attention_forward(q, k, v, block_sizes=block_sizes)
        for name, arr in [("o", o), ("m", m), ("l", l)]:
            assert np.all(np.isfinite(np.array(arr))), f"{name} contains non-finite values"

    @pytest.mark.parametrize("seq_len", [256, 512, 1024])
    def test_multiple_seq_lengths(self, seq_len):
        import jax
        import jax.numpy as jnp
        from standard_attention import attention as naive
        from flash_fwd import flash_attention_forward
        from utils import get_block_sizes

        key = jax.random.PRNGKey(seq_len)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = (1, 2, seq_len, 128)
        q = jax.random.normal(k1, shape, dtype=jnp.bfloat16)
        k = jax.random.normal(k2, shape, dtype=jnp.bfloat16)
        v = jax.random.normal(k3, shape, dtype=jnp.bfloat16)

        block_sizes = get_block_sizes(seq_len, 128, jnp.bfloat16)
        o_naive = np.array(naive(q, k, v), dtype=np.float32)
        o_flash, _, _ = flash_attention_forward(q, k, v, block_sizes=block_sizes)

        np.testing.assert_allclose(
            np.array(o_flash, dtype=np.float32), o_naive, atol=ATOL_BF16
        )


class TestFlashBackward:
    """
    Gradient tests for flash_attention (custom_vjp).

    Checks that gradients from flash_attention match gradients from naive
    attention within bfloat16 tolerance.
    """

    def _grads_naive(self, q, k, v, causal=False):
        import jax
        from standard_attention import attention as naive

        def loss(q, k, v):
            return naive(q, k, v, causal=causal).sum()

        return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

    def _grads_flash(self, q, k, v, causal=False):
        import jax
        from flash_bwd import flash_attention
        from utils import get_block_sizes

        block_sizes = get_block_sizes(q.shape[2], q.shape[3], q.dtype)

        def loss(q, k, v):
            return flash_attention(q, k, v, causal, None, block_sizes).sum()

        return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

    def test_dq_matches_naive(self, small_inputs):
        q, k, v = small_inputs
        dq_naive, _, _ = self._grads_naive(q, k, v)
        dq_flash, _, _ = self._grads_flash(q, k, v)
        np.testing.assert_allclose(
            np.array(dq_flash, dtype=np.float32),
            np.array(dq_naive, dtype=np.float32),
            atol=ATOL_BF16,
        )

    def test_dk_matches_naive(self, small_inputs):
        q, k, v = small_inputs
        _, dk_naive, _ = self._grads_naive(q, k, v)
        _, dk_flash, _ = self._grads_flash(q, k, v)
        np.testing.assert_allclose(
            np.array(dk_flash, dtype=np.float32),
            np.array(dk_naive, dtype=np.float32),
            atol=ATOL_BF16,
        )

    def test_dv_matches_naive(self, small_inputs):
        q, k, v = small_inputs
        _, _, dv_naive = self._grads_naive(q, k, v)
        _, _, dv_flash = self._grads_flash(q, k, v)
        np.testing.assert_allclose(
            np.array(dv_flash, dtype=np.float32),
            np.array(dv_naive, dtype=np.float32),
            atol=ATOL_BF16,
        )

    def test_causal_gradients_finite(self, small_inputs):
        q, k, v = small_inputs
        dq, dk, dv = self._grads_flash(q, k, v, causal=True)
        for name, g in [("dq", dq), ("dk", dk), ("dv", dv)]:
            assert np.all(np.isfinite(np.array(g))), \
                f"{name} contains non-finite values"

    def test_gradient_shapes(self, small_inputs):
        q, k, v = small_inputs
        dq, dk, dv = self._grads_flash(q, k, v)
        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape
