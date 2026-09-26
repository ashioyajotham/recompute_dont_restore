"""Expanded native correctness gates; these require real TPU execution.

Random cotangents exercise the complete VJP, unlike a sum loss whose incoming
gradient is uniformly one. These tests are not performance measurements.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.tpu


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize("batch,heads,seq", [(1, 1, 256), (1, 4, 512), (2, 2, 1024)])
def test_native_random_cotangent_vjp(causal, seed, batch, heads, seq):
    """Compare O and every input gradient against independent JAX attention."""
    import jax
    import jax.numpy as jnp
    from flash_bwd import flash_attention
    from standard_attention import attention
    from utils import BlockSizes

    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    shape = (batch, heads, seq, 128)
    q, k, v, cotangent = [
        jax.random.normal(key, shape, dtype=jnp.bfloat16) for key in keys
    ]
    reference = lambda q, k, v: attention(q, k, v, causal=causal)
    kernel = lambda q, k, v: flash_attention(
        q, k, v, causal, None, BlockSizes(128, 128)
    )
    expected, reference_pullback = jax.vjp(reference, q, k, v)
    actual, kernel_pullback = jax.vjp(kernel, q, k, v)
    expected_values = (expected, *reference_pullback(cotangent))
    actual_values = (actual, *kernel_pullback(cotangent))
    for name, result, oracle in zip(
        ("output", "dq", "dk", "dv"), actual_values, expected_values
    ):
        result = np.asarray(result, dtype=np.float32)
        oracle = np.asarray(oracle, dtype=np.float32)
        assert np.isfinite(result).all(), name
        assert np.isfinite(oracle).all(), name
        np.testing.assert_allclose(result, oracle, atol=0.05, rtol=0.05, err_msg=name)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_native_gqa_forward(causal, kv_heads):
    """Check MQA, grouped attention, and MHA against expanded K/V attention."""
    import jax
    import jax.numpy as jnp
    from grouped_query_attention import grouped_query_attention, reference_gqa
    from utils import BlockSizes

    keys = jax.random.split(jax.random.PRNGKey(23), 3)
    q = jax.random.normal(keys[0], (2, 4, 256, 128), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (2, kv_heads, 256, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], k.shape, dtype=jnp.bfloat16)
    result = np.asarray(grouped_query_attention(
        q, k, v, causal=causal, block_sizes=BlockSizes(128, 128)
    ), dtype=np.float32)
    oracle = np.asarray(reference_gqa(q, k, v, causal=causal), dtype=np.float32)
    assert np.isfinite(result).all()
    assert np.isfinite(oracle).all()
    np.testing.assert_allclose(result, oracle, atol=0.05, rtol=0.05)
