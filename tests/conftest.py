"""
Shared fixtures and skip guards.

Skip logic:
  - @pytest.mark.jax   : skipped if JAX is not installed
  - @pytest.mark.tpu   : skipped if no TPU device is present

All Pallas kernel tests are TPU-gated. Standard-attention and utils tests
require only JAX (CPU backend works).
"""

import pytest


def _jax_available() -> bool:
    try:
        import jax  # noqa: F401
        return True
    except ImportError:
        return False


def _tpu_available() -> bool:
    if not _jax_available():
        return False
    import jax
    return any(d.platform == "tpu" for d in jax.devices())


JAX_AVAILABLE = _jax_available()
TPU_AVAILABLE = _tpu_available()


def pytest_configure(config):
    config.addinivalue_line("markers", "jax: requires JAX installed")
    config.addinivalue_line("markers", "tpu: requires a TPU device")


def pytest_collection_modifyitems(config, items):
    skip_jax = pytest.mark.skip(reason="JAX not installed")
    skip_tpu = pytest.mark.skip(reason="no TPU device found")

    for item in items:
        if "jax" in item.keywords and not JAX_AVAILABLE:
            item.add_marker(skip_jax)
        if "tpu" in item.keywords and not TPU_AVAILABLE:
            item.add_marker(skip_tpu)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng():
    """NumPy RNG for non-JAX tests."""
    import numpy as np
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def jax_key():
    """Root JAX PRNG key."""
    import jax
    return jax.random.PRNGKey(0)


@pytest.fixture
def small_inputs(jax_key):
    """
    Small (batch=1, heads=2, seq=256, d=128) bfloat16 inputs.

    seq=256 and d=128 satisfy MIN_BLOCK_SIZE=128 constraints and are
    small enough to run fast on CPU for non-kernel tests.
    """
    import jax
    import jax.numpy as jnp
    batch, heads, seq, d = 1, 2, 256, 128
    k1, k2, k3 = jax.random.split(jax_key, 3)
    shape = (batch, heads, seq, d)
    return (
        jax.random.normal(k1, shape, dtype=jnp.bfloat16),
        jax.random.normal(k2, shape, dtype=jnp.bfloat16),
        jax.random.normal(k3, shape, dtype=jnp.bfloat16),
    )


@pytest.fixture
def gqa_inputs(jax_key):
    """
    GQA inputs: 4 query heads, 2 KV heads, seq=256.
    """
    import jax
    import jax.numpy as jnp
    batch, num_q_heads, num_kv_heads, seq, d = 1, 4, 2, 256, 128
    k1, k2, k3 = jax.random.split(jax_key, 3)
    q = jax.random.normal(k1, (batch, num_q_heads, seq, d), dtype=jnp.bfloat16)
    k = jax.random.normal(k2, (batch, num_kv_heads, seq, d), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (batch, num_kv_heads, seq, d), dtype=jnp.bfloat16)
    return q, k, v
