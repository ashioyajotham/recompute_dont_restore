"""
HBM peak memory comparison: naive attention vs flash attention.

Two measurement approaches:
  1. Theoretical: analytical formula for peak HBM per method.
     Naive:  O(n^2) — attention matrix dominates.
     Flash:  O(n)   — attention matrix never materialized.
  2. Runtime: jax device memory stats before/after execution.
     This measures working set delta, not strict peak, but is a useful
     lower bound on real allocation.

Usage:
    python 05_benchmarks/memory_profile.py
    python 05_benchmarks/memory_profile.py --plot
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "02_naive_jax_baseline"))
sys.path.insert(0, str(Path(__file__).parent.parent / "03_pallas_kernels"))

from standard_attention import attention as naive_attention
from flash_fwd import flash_attention_forward
from utils import attention_matrix_bytes, get_block_sizes

SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]

DEFAULT_BATCH = 1
DEFAULT_HEADS = 8
DEFAULT_D_K = 128
DEFAULT_DTYPE = jnp.bfloat16


def theoretical_naive_peak(
    seq_len: int,
    batch: int,
    heads: int,
    d_k: int,
    dtype: jnp.dtype,
) -> int:
    """Peak HBM for naive attention (bytes). Dominated by the attention matrix."""
    itemsize = jnp.dtype(dtype).itemsize
    qkvo = 4 * batch * heads * seq_len * d_k * itemsize
    attn = attention_matrix_bytes(seq_len, heads, batch, dtype)
    return qkvo + attn


def theoretical_flash_peak(
    seq_len: int,
    batch: int,
    heads: int,
    d_k: int,
    dtype: jnp.dtype,
    block_sizes=None,
) -> int:
    """
    Peak HBM for flash attention (bytes).

    Q, K, V, O:   4 * batch * heads * seq_len * d_k * itemsize
    m, l:         2 * batch * heads * seq_len * 128 * 4  (float32)

    The attention matrix term (O(n^2)) is absent — it lives in VMEM only,
    tiled across the KV loop.
    """
    itemsize = jnp.dtype(dtype).itemsize
    qkvo = 4 * batch * heads * seq_len * d_k * itemsize
    # m and l saved in float32 with MIN_BLOCK_SIZE trailing dim
    ml = 2 * batch * heads * seq_len * 128 * 4
    return qkvo + ml


def runtime_memory_delta(fn, *args) -> int:
    """
    Working set delta in bytes using JAX device memory stats.

    Note: this is the delta between before and after the call, not the strict
    peak. For tight peak measurement, use jax.profiler.trace and read the
    profiler output.
    """
    device = jax.local_devices()[0]
    try:
        stats_before = device.memory_stats()
        jax.block_until_ready(fn(*args))
        stats_after = device.memory_stats()
        # peak_bytes_in_use is more accurate than bytes_in_use delta
        peak_key = "peak_bytes_in_use"
        if peak_key in stats_after:
            return stats_after[peak_key]
        return stats_after.get("bytes_in_use", 0) - stats_before.get("bytes_in_use", 0)
    except Exception:
        # memory_stats() may not be available on all backends
        return -1


def profile_with_xprof(fn, args: tuple, profile_dir: str) -> None:
    """
    Capture an xprof trace for manual inspection.

    Open the resulting directory in the profiler UI:
        tensorboard --logdir <profile_dir>
    or use the xprof standalone viewer.
    """
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(profile_dir):
        jax.block_until_ready(fn(*args))
    print(f"Profiler trace written to {profile_dir}")


def make_inputs(seq_len, batch, heads, d_k, dtype, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    shape = (batch, heads, seq_len, d_k)
    return (
        jax.random.normal(k1, shape, dtype=dtype),
        jax.random.normal(k2, shape, dtype=dtype),
        jax.random.normal(k3, shape, dtype=dtype),
    )


def run_memory_sweep(
    seq_lengths=SEQ_LENGTHS,
    batch=DEFAULT_BATCH,
    heads=DEFAULT_HEADS,
    d_k=DEFAULT_D_K,
    dtype=DEFAULT_DTYPE,
) -> list[dict]:
    records = []

    header = (
        f"{'seq':>6}  {'naive_theory_GB':>16}  {'flash_theory_GB':>16}  "
        f"{'ratio':>7}  {'naive_runtime_GB':>17}  {'flash_runtime_GB':>17}"
    )
    print(header)
    print("-" * len(header))

    for seq_len in seq_lengths:
        q, k, v = make_inputs(seq_len, batch, heads, d_k, dtype)

        naive_theory = theoretical_naive_peak(seq_len, batch, heads, d_k, dtype)
        flash_theory = theoretical_flash_peak(seq_len, batch, heads, d_k, dtype)
        ratio = naive_theory / flash_theory

        # Warmup for compilation
        try:
            jax.block_until_ready(naive_attention(q, k, v))
            naive_rt = runtime_memory_delta(naive_attention, q, k, v)
        except Exception as e:
            print(f"{seq_len:>6}  naive OOM/error: {e}")
            naive_rt = -1

        try:
            block_sizes = get_block_sizes(seq_len, d_k, dtype)
            flash_fn = lambda q, k, v: flash_attention_forward(q, k, v, block_sizes=block_sizes)[0]
            jax.block_until_ready(flash_fn(q, k, v))
            flash_rt = runtime_memory_delta(flash_fn, q, k, v)
        except Exception as e:
            print(f"{seq_len:>6}  flash OOM/error: {e}")
            flash_rt = -1

        print(
            f"{seq_len:>6}  {naive_theory/1e9:>16.3f}  {flash_theory/1e9:>16.3f}  "
            f"{ratio:>7.1f}x  {naive_rt/1e9:>17.3f}  {flash_rt/1e9:>17.3f}"
        )

        records.append({
            "seq_len": seq_len,
            "naive_theory_bytes": naive_theory,
            "flash_theory_bytes": flash_theory,
            "memory_ratio": ratio,
            "naive_runtime_bytes": naive_rt,
            "flash_runtime_bytes": flash_rt,
        })

    return records


def plot_results(records: list[dict], out_path="05_benchmarks/results/memory.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    seq_lens = [r["seq_len"] for r in records]
    naive_gb = [r["naive_theory_bytes"] / 1e9 for r in records]
    flash_gb = [r["flash_theory_bytes"] / 1e9 for r in records]
    ratios = [r["memory_ratio"] for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(seq_lens, naive_gb, "o-", label="Naive (theoretical)")
    ax1.plot(seq_lens, flash_gb, "s-", label="Flash (theoretical)")
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Peak HBM (GB)")
    ax1.set_title("HBM usage: naive vs flash")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(seq_lens, ratios, "o-", color="red")
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("Memory ratio (naive / flash)")
    ax2.set_title("Memory savings factor")
    ax2.set_xscale("log", base=2)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--out",
        default="05_benchmarks/results/memory.json",
    )
    args = parser.parse_args()

    records = run_memory_sweep()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nResults written to {args.out}")

    if args.plot:
        plot_results(records)
