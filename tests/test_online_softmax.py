"""
Unit tests for the online softmax algorithm.

These tests validate the tiling math in pure NumPy — no JAX, no TPU.
They correspond directly to the algorithm in 01_flash_attention_math/ and
to the accumulation logic in 03_pallas_kernels/flash_fwd.py.

If these tests pass, the math is right. If the Pallas kernel produces wrong
results, the bug is in the Pallas-specific tiling code (indexing, VMEM
scratch lifecycle), not the algorithm.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def softmax_reference(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def attention_reference(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Standard attention without tiling."""
    scale = 1.0 / np.sqrt(Q.shape[-1])
    logits = Q @ K.T * scale
    weights = softmax_reference(logits)
    return weights @ V


def tiled_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_q: int,
    block_kv: int,
) -> np.ndarray:
    """
    Flash Attention tiling in NumPy.

    Implements the exact update rules used in flash_fwd._flash_fwd_kernel:
      m_new = max(m_prev, rowmax(logits))
      alpha = exp(m_prev - m_new)
      l_new = alpha * l_prev + rowsum(exp(logits - m_new))
      O_acc = (alpha * O_acc * l_prev + P @ V_tile) / l_new
    """
    seq_q, d_k = Q.shape
    seq_kv = K.shape[0]
    d_v = V.shape[1]
    scale = 1.0 / np.sqrt(d_k)
    O = np.zeros((seq_q, d_v), dtype=np.float64)

    for qi in range(0, seq_q, block_q):
        Q_tile = Q[qi:qi + block_q]
        m = np.full(len(Q_tile), -np.inf)
        l = np.zeros(len(Q_tile))
        O_acc = np.zeros((len(Q_tile), d_v))

        for ki in range(0, seq_kv, block_kv):
            K_tile = K[ki:ki + block_kv]
            V_tile = V[ki:ki + block_kv]

            logits = Q_tile @ K_tile.T * scale              # (bq, bkv)
            m_new = np.maximum(m, logits.max(axis=1))       # (bq,)
            P = np.exp(logits - m_new[:, None])             # (bq, bkv)
            alpha = np.exp(m - m_new)                       # (bq,)
            l_new = alpha * l + P.sum(axis=1)               # (bq,)

            O_acc = (alpha[:, None] * O_acc * l[:, None] + P @ V_tile) / l_new[:, None]
            m, l = m_new, l_new

        O[qi:qi + block_q] = O_acc

    return O


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOnlineSoftmax:
    """The tiled attention must produce the same result as reference attention."""

    @pytest.fixture(autouse=True)
    def inputs(self, rng):
        seq, d = 16, 8
        self.Q = rng.standard_normal((seq, d)).astype(np.float32)
        self.K = rng.standard_normal((seq, d)).astype(np.float32)
        self.V = rng.standard_normal((seq, d)).astype(np.float32)
        self.ref = attention_reference(self.Q, self.K, self.V)

    @pytest.mark.parametrize("block_q,block_kv", [
        (4, 4),
        (2, 8),
        (8, 2),
        (1, 1),
        (16, 16),   # single tile — must equal reference exactly
        (4, 16),    # block_kv covers full KV at once
        (16, 4),    # block_q covers full Q at once
    ])
    def test_matches_reference(self, block_q, block_kv):
        out = tiled_attention(self.Q, self.K, self.V, block_q, block_kv)
        np.testing.assert_allclose(
            out, self.ref, atol=1e-5,
            err_msg=f"block_q={block_q}, block_kv={block_kv}"
        )

    def test_single_tile_is_exact(self):
        """When tiling covers full sequence, result should be numerically identical."""
        seq = self.Q.shape[0]
        out = tiled_attention(self.Q, self.K, self.V, seq, seq)
        np.testing.assert_allclose(out, self.ref, atol=1e-6)

    def test_all_same_keys(self, rng):
        """Uniform keys -> uniform attention weights -> output is mean of V rows."""
        seq, d = 8, 4
        Q = rng.standard_normal((seq, d)).astype(np.float32)
        K = np.ones((seq, d), dtype=np.float32)
        V = rng.standard_normal((seq, d)).astype(np.float32)

        out = tiled_attention(Q, K, V, block_q=2, block_kv=4)
        ref = attention_reference(Q, K, V)
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_numerical_stability_large_logits(self):
        """Large logit values should not overflow; online max prevents this."""
        seq, d = 8, 4
        Q = np.ones((seq, d), dtype=np.float32) * 100.0
        K = np.ones((seq, d), dtype=np.float32) * 100.0
        V = np.eye(seq, dtype=np.float32)[:, :4]

        out = tiled_attention(Q, K, V, block_q=2, block_kv=4)
        assert np.all(np.isfinite(out)), "Output contains non-finite values"

    def test_single_query(self):
        """seq_q=1: edge case for the row-wise operations."""
        Q = np.ones((1, 8), dtype=np.float32)
        K = np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32)
        V = np.random.default_rng(1).standard_normal((16, 8)).astype(np.float32)
        out = tiled_attention(Q, K, V, block_q=1, block_kv=4)
        ref = attention_reference(Q, K, V)
        np.testing.assert_allclose(out, ref, atol=1e-5)


class TestCausalMaskTiling:
    """
    Causal masking: query at position i should only attend to keys j <= i.

    We test the mask logic in isolation before the Pallas kernel tests.
    """

    def _tiled_causal_attention(self, Q, K, V, block_q, block_kv):
        seq_q, d_k = Q.shape
        seq_kv = K.shape[0]
        d_v = V.shape[1]
        scale = 1.0 / np.sqrt(d_k)
        MASK_VALUE = -1e9
        O = np.zeros((seq_q, d_v), dtype=np.float64)

        for qi in range(0, seq_q, block_q):
            Q_tile = Q[qi:qi + block_q]
            bq = len(Q_tile)
            m = np.full(bq, -np.inf)
            l = np.zeros(bq)
            O_acc = np.zeros((bq, d_v))

            for ki in range(0, seq_kv, block_kv):
                K_tile = K[ki:ki + block_kv]
                V_tile = V[ki:ki + block_kv]

                logits = Q_tile @ K_tile.T * scale
                # Causal mask
                q_pos = np.arange(qi, qi + bq)[:, None]
                kv_pos = np.arange(ki, ki + len(K_tile))[None, :]
                logits = np.where(q_pos >= kv_pos, logits, MASK_VALUE)

                m_new = np.maximum(m, logits.max(axis=1))
                P = np.exp(logits - m_new[:, None])
                alpha = np.exp(m - m_new)
                l_new = alpha * l + P.sum(axis=1)
                O_acc = (alpha[:, None] * O_acc * l[:, None] + P @ V_tile) / l_new[:, None]
                m, l = m_new, l_new

            O[qi:qi + bq] = O_acc

        return O

    def _reference_causal(self, Q, K, V):
        scale = 1.0 / np.sqrt(Q.shape[-1])
        logits = Q @ K.T * scale
        seq = Q.shape[0]
        mask = np.tril(np.ones((seq, seq), dtype=bool))
        logits = np.where(mask, logits, -1e9)
        weights = softmax_reference(logits)
        return weights @ V

    @pytest.mark.parametrize("block_q,block_kv", [(4, 4), (2, 8), (8, 2)])
    def test_causal_matches_reference(self, rng, block_q, block_kv):
        seq, d = 16, 8
        Q = rng.standard_normal((seq, d)).astype(np.float32)
        K = rng.standard_normal((seq, d)).astype(np.float32)
        V = rng.standard_normal((seq, d)).astype(np.float32)

        out = self._tiled_causal_attention(Q, K, V, block_q, block_kv)
        ref = self._reference_causal(Q, K, V)
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_first_query_only_attends_to_first_key(self, rng):
        """
        With causal masking, query 0 attends only to key 0.
        Output row 0 = V[0] (since P[0, 0] = 1, P[0, j>0] = 0).
        """
        seq, d = 8, 4
        Q = rng.standard_normal((seq, d)).astype(np.float32)
        K = rng.standard_normal((seq, d)).astype(np.float32)
        V = rng.standard_normal((seq, d)).astype(np.float32)

        out = self._tiled_causal_attention(Q, K, V, block_q=4, block_kv=4)
        # Row 0: P[0, 0]=1 by construction, so O[0] = V[0]
        np.testing.assert_allclose(out[0], V[0], atol=1e-5)
