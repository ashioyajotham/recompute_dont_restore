"""
Sequence-length sweep benchmark for naive JAX attention.

Records wall-clock time and theoretical peak HBM usage across sequence lengths.
These numbers form the left axis of all comparison charts in 05_benchmarks/.

Usage:
    python 02_naive_jax_baseline/benchmark_baseline.py
    python 02_naive_jax_baseline/benchmark_baseline.py --plot
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from standard_attention import attention
from sys import path as syspath
syspath.insert(0, str(Path(__file__).parent.parent / "03_pallas_kernels"))
from utils import attention_matrix_bytes

SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]

# Default shapes matching typical LLM attention configs
DEFAULT_BATCH = 1
DEFAULT_HEADS = 8
DEFAULT_D_K = 128
DEFAULT_DTYPE = jnp.bfloat16


def make_inputs(
    seq_len: int,
    batch: int = DEFAULT_BATCH,
    heads: int = DEFAULT_HEADS,
    d_k: int = DEFAULT_D_K,
    dtype: jnp.dtype = DEFAULT_DTYPE,
    seed: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    shape = (batch, heads, seq_len, d_k)
    q = jax.random.normal(k1, shape, dtype=dtype)
    k = jax.random.normal(k2, shape, dtype=dtype)
    v = jax.random.normal(k3, shape, dtype=dtype)
    return q, k, v


def time_attention(
    fn,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    n_warmup: int = 3,
    n_trials: int = 10,
) -> float:
    """Median wall-clock time in milliseconds, after compilation warmup."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn(q, k, v))

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(q, k, v))
        times.append(time.perf_counter() - t0)

    return float(np.median(times)) * 1e3  # ms


def theoretical_peak_hbm(
    seq_len: int,
    batch: int,
    heads: int,
    d_k: int,
    dtype: jnp.dtype,
) -> dict:
    """
    Theoretical peak HBM allocation during a naive attention forward pass.

    The attention matrix (seq x seq per head) dominates at long sequences.
    Flash Attention eliminates this term — O(seq^2) becomes O(seq).
    """
    itemsize = jnp.dtype(dtype).itemsize
    qkv_bytes = 3 * batch * heads * seq_len * d_k * itemsize
    attn_matrix_bytes = attention_matrix_bytes(seq_len, heads, batch, dtype)
    output_bytes = batch * heads * seq_len * d_k * itemsize
    return {
        "qkv_bytes": qkv_bytes,
        "attn_matrix_bytes": attn_matrix_bytes,
        "output_bytes": output_bytes,
        "total_bytes": qkv_bytes + attn_matrix_bytes + output_bytes,
    }


def run_sweep(
    seq_lengths: list = SEQ_LENGTHS,
    batch: int = DEFAULT_BATCH,
    heads: int = DEFAULT_HEADS,
    d_k: int = DEFAULT_D_K,
    dtype: jnp.dtype = DEFAULT_DTYPE,
) -> list[dict]:
    records = []
    fn = lambda q, k, v: attention(q, k, v)

    print(f"{'seq_len':>8}  {'ms':>8}  {'attn_matrix_GB':>16}  {'total_hbm_GB':>14}")
    print("-" * 55)

    for seq_len in seq_lengths:
        q, k, v = make_inputs(seq_len, batch, heads, d_k, dtype)

        try:
            ms = time_attention(fn, q, k, v)
        except Exception as e:
            print(f"{seq_len:>8}  OOM or error: {e}")
            break

        mem = theoretical_peak_hbm(seq_len, batch, heads, d_k, dtype)
        attn_gb = mem["attn_matrix_bytes"] / 1e9
        total_gb = mem["total_bytes"] / 1e9

        print(f"{seq_len:>8}  {ms:>8.2f}  {attn_gb:>16.3f}  {total_gb:>14.3f}")

        records.append({
            "seq_len": seq_len,
            "batch": batch,
            "heads": heads,
            "d_k": d_k,
            "dtype": str(dtype),
            "ms": ms,
            **mem,
        })

    return records


def plot_results(records: list[dict], out_path: str = "05_benchmarks/results/baseline.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    seq_lens = [r["seq_len"] for r in records]
    times_ms = [r["ms"] for r in records]
    attn_gb = [r["attn_matrix_bytes"] / 1e9 for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(seq_lens, times_ms, "o-")
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Wall-clock time (ms)")
    ax1.set_title("Latency vs sequence length")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.plot(seq_lens, attn_gb, "o-", color="orange")
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("Attention matrix (GB)")
    ax2.set_title("Attention matrix HBM: O(n²) scaling")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Save result plots")
    parser.add_argument(
        "--out",
        default="05_benchmarks/results/baseline.json",
        help="Path for JSON results",
    )
    args = parser.parse_args()

    records = run_sweep()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nResults written to {args.out}")

    if args.plot:
        plot_results(records)
