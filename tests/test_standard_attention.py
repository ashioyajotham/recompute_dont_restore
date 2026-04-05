"""
Functional tests for 02_naive_jax_baseline/standard_attention.py.

These run on CPU with JAX installed — no TPU required. The standard attention
implementation is the correctness oracle for all Pallas kernel tests, so it
must be tested carefully.
"""

import pytest
import numpy as np

pytestmark = pytest.mark.jax


class TestAttentionOutputShape:
    def test_basic_shape(self, small_inputs):
        from standard_attention import attention
        q, k, v = small_inputs
        out = attention(q, k, v)
        assert out.shape == q.shape

    def test_different_kv_seq_len(self):
        """seq_q != seq_kv is valid (cross-attention)."""
        import jax
        import jax.numpy as jnp
        from standard_attention import attention
        key = jax.random.PRNGKey(1)
        k1, k2, k3 = jax.random.split(key, 3)
        q = jax.random.normal(k1, (1, 2, 128, 128), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (1, 2, 256, 128), dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (1, 2, 256, 128), dtype=jnp.bfloat16)
        out = attention(q, k, v)
        assert out.shape == (1, 2, 128, 128)

    def test_output_dtype_matches_input(self, small_inputs):
        from standard_attention import attention
        q, k, v = small_inputs
        out = attention(q, k, v)
        # Softmax is computed in float32 but output is cast back
        # to the input dtype after the weighted sum
        assert out.dtype == q.dtype


class TestCausalMask:
    def test_causal_output_differs_from_noncausal(self, small_inputs):
        from standard_attention import attention
        import jax.numpy as jnp
        q, k, v = small_inputs
        out_full = attention(q, k, v, causal=False)
        out_causal = attention(q, k, v, causal=True)
        # Should produce different results except at position 0
        assert not np.allclose(
            np.array(out_full), np.array(out_causal), atol=1e-3
        )

    def test_causal_first_query_unaffected(self, small_inputs):
        """
        Query position 0 can only attend to key 0 in causal attention.
        Its output should be identical to full attention when seq_kv=1,
        and the same value as non-causal would produce for a 1-token context.
        """
        import jax
        import jax.numpy as jnp
        from standard_attention import attention

        key = jax.random.PRNGKey(2)
        k1, k2, k3 = jax.random.split(key, 3)
        # Single-position sequence: causal == non-causal
        q = jax.random.normal(k1, (1, 1, 128, 128), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (1, 1, 128, 128), dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (1, 1, 128, 128), dtype=jnp.bfloat16)

        out_full = attention(q, k, v, causal=False)
        out_causal = attention(q, k, v, causal=True)
        np.testing.assert_allclose(
            np.array(out_full, dtype=np.float32),
            np.array(out_causal, dtype=np.float32),
            atol=1e-2,
        )

    def test_causal_last_query_unchanged(self, small_inputs):
        """
        The last query sees all keys in both causal and full attention.
        Its output should match.
        """
        import jax.numpy as jnp
        from standard_attention import attention
        q, k, v = small_inputs
        out_full = np.array(attention(q, k, v, causal=False), dtype=np.float32)
        out_causal = np.array(attention(q, k, v, causal=True), dtype=np.float32)
        # Last query position
        np.testing.assert_allclose(
            out_full[:, :, -1, :],
            out_causal[:, :, -1, :],
            atol=1e-2,
        )


class TestAttentionProperties:
    def test_output_is_finite(self, small_inputs):
        from standard_attention import attention
        q, k, v = small_inputs
        out = np.array(attention(q, k, v))
        assert np.all(np.isfinite(out)), "Output contains non-finite values"

    def test_self_attention_shape(self):
        """Q = K = V is the common self-attention case."""
        import jax
        import jax.numpy as jnp
        from standard_attention import attention
        x = jax.random.normal(jax.random.PRNGKey(3), (1, 2, 128, 128), dtype=jnp.bfloat16)
        out = attention(x, x, x)
        assert out.shape == x.shape

    def test_uniform_keys_uniform_weights(self):
        """
        When all keys are identical, every query attends uniformly.
        Output row i = mean of V rows.
        """
        import jax
        import jax.numpy as jnp
        from standard_attention import attention

        key = jax.random.PRNGKey(4)
        seq, d = 128, 128
        q = jax.random.normal(key, (1, 1, seq, d), dtype=jnp.bfloat16)
        k = jnp.ones((1, 1, seq, d), dtype=jnp.bfloat16)
        v = jax.random.normal(jax.random.PRNGKey(5), (1, 1, seq, d), dtype=jnp.bfloat16)

        out = np.array(attention(q, k, v), dtype=np.float32)
        v_mean = np.array(v, dtype=np.float32).mean(axis=2, keepdims=True)
        v_mean = np.broadcast_to(v_mean, out.shape)

        np.testing.assert_allclose(out, v_mean, atol=5e-2)  # bfloat16 tolerance

    def test_matches_numpy_reference(self):
        """
        Compare JAX attention against the NumPy reference implementation
        from test_online_softmax.py for a small float32 input.
        """
        import jax
        import jax.numpy as jnp
        from standard_attention import attention

        rng = np.random.default_rng(42)
        seq, d = 128, 128
        Q_np = rng.standard_normal((seq, d)).astype(np.float32)
        K_np = rng.standard_normal((seq, d)).astype(np.float32)
        V_np = rng.standard_normal((seq, d)).astype(np.float32)

        # NumPy reference (single head, single batch)
        scale = 1.0 / np.sqrt(d)
        logits = Q_np @ K_np.T * scale
        logits -= logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=-1, keepdims=True)
        ref = (weights @ V_np).astype(np.float32)

        # JAX attention — needs (batch, heads, seq, d) shape
        q = jnp.array(Q_np[None, None])  # (1, 1, seq, d)
        k = jnp.array(K_np[None, None])
        v = jnp.array(V_np[None, None])
        out = np.array(attention(q, k, v), dtype=np.float32)[0, 0]

        np.testing.assert_allclose(out, ref, atol=1e-4)


class TestGradients:
    def test_gradients_exist(self, small_inputs):
        """jax.grad should work through the attention computation."""
        import jax
        import jax.numpy as jnp
        from standard_attention import attention

        q, k, v = small_inputs

        def loss(q, k, v):
            return attention(q, k, v).sum()

        grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
        dq, dk, dv = grads
        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape

    def test_gradients_finite(self, small_inputs):
        import jax
        from standard_attention import attention

        q, k, v = small_inputs

        def loss(q, k, v):
            return attention(q, k, v).sum()

        dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
        for name, g in [("dq", dq), ("dk", dk), ("dv", dv)]:
            import numpy as np
            assert np.all(np.isfinite(np.array(g))), f"{name} contains non-finite values"
