"""
Shared pytest fixtures and skip-guards for the recompute_dont_score test suite.

=== Skip-guard system ===

Two custom marks control whether tests run:

  @pytest.mark.jax   — skip if JAX is not importable at all.
                       Used by pure-Python tests that only need JAX arrays.
  @pytest.mark.tpu   — skip if no TPU device is present in jax.devices().
                       Used by every Pallas kernel test (flash_fwd, flash_bwd,
                       grouped_query_attention). Pallas TPU primitives
                       (PrefetchScalarGridSpec, pltpu.VMEM, CompilerParams) are
                       not available on CPU or GPU backends.

Evaluation happens once at collection time (_jax_available / _tpu_available).

=== Running Pallas tests without a real TPU ===

Pallas ships an interpreter mode that runs kernels in plain JAX on CPU.
It does NOT enforce TPU hardware constraints (e.g. MIN_BLOCK_SIZE=128) but is
useful for logic checks:

    JAX_INTERPRET_PALLAS=1 pytest tests/test_flash_attention.py -v

=== Fixture overview ===

  rng           — NumPy default_rng(42) for non-JAX tests (session-scoped).
  jax_key       — Root JAX PRNGKey(0) used to derive all other keys (session).
  small_inputs  — Standard (1, 2, 256, 128) bfloat16 Q/K/V triple (function).
  gqa_inputs    — GQA-shaped (1, 4Q/2KV, 256, 128) bfloat16 triple (function).

See individual fixture docstrings for shape rationale.
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
    """
    NumPy RNG for non-JAX tests (e.g. test_utils.py).

    Scoped to the session so the same generator is reused across all tests
    that request it.  Use NumPy here (not JAX) because some tests run before
    JAX is confirmed to be importable and because NumPy RNGs are simpler to
    use in pure-Python helper functions.
    """
    import numpy as np
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def jax_key():
    """
    Root JAX PRNG key, derived once per session.

    Session scope gives deterministic behaviour: all tests that derive sub-keys
    from this root will see the same random data across runs.  Using PRNGKey(0)
    (not a time-based seed) makes failures reproducible.

    Tests that need independent randomness should call
    ``jax.random.split(jax_key, n)`` to obtain child keys rather than
    modifying this root.
    """
    import jax
    return jax.random.PRNGKey(0)


@pytest.fixture
def small_inputs(jax_key):
    """
    Standard bfloat16 Q/K/V inputs for MHA kernel tests.

    Shape: (batch=1, heads=2, seq=256, d_k=128)

    Shape rationale
    ---------------
    * ``d_k=128``: exactly MIN_BLOCK_SIZE.  All valid block sizes on real TPU
      hardware must be multiples of 128 (the systolic-array tile dimension).
      Using 128 is the minimum that avoids a validate_shapes rejection.
    * ``seq=256``: 2 × MIN_BLOCK_SIZE, so ``get_block_sizes`` will return
      block_q = block_kv = 128 (the default for seq < 1024), and the grid has
      exactly 2 Q tiles × 2 KV tiles — small enough to run fast in interpret
      mode but exercising the multi-tile online-softmax accumulation path.
    * ``heads=2``: non-trivial (avoids single-head edge cases) while keeping
      the total tensor size < 1 MB so tests load quickly.
    * ``dtype=bfloat16``: the only dtype the kernels accept in the default
      configuration (float16 is also supported but bfloat16 is the TPU native).

    Scope is ``function`` (the default) — each test gets a fresh copy so tests
    are independent even if one mutates the returned tensors (unlikely in JAX,
    but defensive).
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
    Bfloat16 Q/K/V inputs for Grouped Query Attention tests.

    Shapes:
      Q — (batch=1, num_q_heads=4, seq=256, d_k=128)
      K — (batch=1, num_kv_heads=2, seq=256, d_k=128)
      V — (batch=1, num_kv_heads=2, seq=256, d_k=128)

    Shape rationale
    ---------------
    * ``num_q_heads=4, num_kv_heads=2`` → ``groups=2``.  This exercises the
      GQA head-mapping code (``h // groups``) with a non-trivial group size
      while remaining small.  groups=1 would degenerate to MHA; groups=4 (MQA)
      is tested separately where needed.
    * ``seq=256, d_k=128``: same constraints as ``small_inputs`` — minimum
      viable sizes for TPU block-size requirements.
    * K and V have ``num_kv_heads=2`` (not 4), which verifies that
      ``grouped_query_attention`` correctly handles the asymmetric head count
      without expanding K/V up-front.
    """
    import jax
    import jax.numpy as jnp
    batch, num_q_heads, num_kv_heads, seq, d = 1, 4, 2, 256, 128
    k1, k2, k3 = jax.random.split(jax_key, 3)
    q = jax.random.normal(k1, (batch, num_q_heads, seq, d), dtype=jnp.bfloat16)
    k = jax.random.normal(k2, (batch, num_kv_heads, seq, d), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (batch, num_kv_heads, seq, d), dtype=jnp.bfloat16)
    return q, k, v
