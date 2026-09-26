"""
Throughput benchmark: TFLOP/s and Model FLOP Utilization (MFU).

FLOP count convention (standard, matching LLM literature):
  Matmuls only. Softmax and element-wise ops are ignored as they are
  memory-bound and negligible relative to matmul cost.

  Forward pass FLOPs (non-causal):
    QK^T:  2 * B * H * Sq * Skv * dk
    PV:    2 * B * H * Sq * Skv * dv
    Total: 4 * B * H * S^2 * d  (when Sq = Skv = S, dk = dv = d)

  Causal: triangular FLOPs are a useful-work estimate. This implementation
  still visits all tiles and masks future positions; its executed matmul
  count is not halved by the causal flag.

MFU = achieved TFLOP/s / peak TFLOP/s of the device.

Usage:
    python 05_benchmarks/throughput_tflops.py
    python 05_benchmarks/throughput_tflops.py --tpu v5e --plot
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "02_naive_jax_baseline"))
sys.path.insert(0, str(Path(__file__).parent.parent / "03_pallas_kernels"))

from standard_attention import attention as naive_attention
from flash_fwd import flash_attention_forward
from utils import get_block_sizes

# Peak bfloat16 TFLOP/s per chip (single-chip, not pod)
TPU_PEAK_TFLOPS: dict[str, float] = {
    "v4":  275.0,
    "v5e": 197.0,
    "v5p": 459.0,
}

SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]

DEFAULT_BATCH = 1
DEFAULT_HEADS = 8
DEFAULT_D_K = 128
DEFAULT_N_WARMUP = 3
DEFAULT_N_TRIALS = 10


def attention_flop_count(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    d_k: int,
    d_v: int,
    causal: bool = False,
    useful_causal: bool = False,
) -> int:
    """
    Matmul FLOPs for one forward pass of attention (multiply-add = 2 FLOPs).

    Two matmuls, each counted with the 2× multiply-add convention:
      QK^T :  2 · batch · heads · seq_q · seq_kv · d_k
      PV   :  2 · batch · heads · seq_q · seq_kv · d_v

    When seq_q == seq_kv == S and d_k == d_v == d this simplifies to the
    standard FA2 formula:
      total = 4 · batch · heads · S² · d    (non-causal forward)

    This is NOT the naive 2·S²·d figure (which omits the PV matmul).

    useful_causal halves the algorithmic FLOP count. The current Pallas
    kernel still executes every tile and masks future positions, so hardware
    throughput and MFU must use the full executed count.
    """
    qkt_flops = 2 * batch * heads * seq_q * seq_kv * d_k
    pv_flops  = 2 * batch * heads * seq_q * seq_kv * d_v
    total = qkt_flops + pv_flops
    if causal and useful_causal:
        total //= 2
    return total


def tflops_from_time(flop_count: int, elapsed_ms: float) -> float:
    """Convert flop count and elapsed time to TFLOP/s."""
    return flop_count / (elapsed_ms * 1e-3) / 1e12


def mfu(achieved_tflops: float, tpu_version: str) -> float:
    """Model FLOP Utilization as a fraction [0, 1]."""
    peak = TPU_PEAK_TFLOPS.get(tpu_version)
    if peak is None:
        raise ValueError(
            f"Unknown TPU version {tpu_version!r}. "
            f"Known: {list(TPU_PEAK_TFLOPS)}"
        )
    return achieved_tflops / peak


def time_fn(fn, *args, n_warmup=DEFAULT_N_WARMUP, n_trials=DEFAULT_N_TRIALS) -> float:
    """Median wall-clock time in milliseconds."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)

    return float(np.median(times)) * 1e3


def make_inputs(seq_len, batch, heads, d_k, dtype=jnp.bfloat16, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    shape = (batch, heads, seq_len, d_k)
    return (
        jax.random.normal(k1, shape, dtype=dtype),
        jax.random.normal(k2, shape, dtype=dtype),
        jax.random.normal(k3, shape, dtype=dtype),
    )


def run_throughput_sweep(
    seq_lengths=SEQ_LENGTHS,
    tpu_version="v4",
    batch=DEFAULT_BATCH,
    heads=DEFAULT_HEADS,
    d_k=DEFAULT_D_K,
    causal=False,
) -> list[dict]:
    dtype = jnp.bfloat16
    records = []

    header = (
        f"{'seq':>6}  {'method':>8}  {'ms':>8}  {'TFLOP/s':>9}  {'MFU':>6}"
    )
    print(header)
    print("-" * len(header))

    for seq_len in seq_lengths:
        q, k, v = make_inputs(seq_len, batch, heads, d_k, dtype)
        flops = attention_flop_count(batch, heads, seq_len, seq_len, d_k, d_k, causal)

        for method, fn in [
            ("naive",  lambda q, k, v: naive_attention(q, k, v, causal=causal)),
            ("flash",  lambda q, k, v: flash_attention_forward(
                q, k, v, causal=causal,
                block_sizes=get_block_sizes(seq_len, d_k, q.dtype)
            )[0]),
        ]:
            try:
                ms = time_fn(fn, q, k, v)
                tf = tflops_from_time(flops, ms)
                mfu_val = mfu(tf, tpu_version)
                print(
                    f"{seq_len:>6}  {method:>8}  {ms:>8.2f}  "
                    f"{tf:>9.2f}  {mfu_val:>6.3f}"
                )
                records.append({
                    "seq_len": seq_len,
                    "method": method,
                    "causal": causal,
                    "ms": ms,
                    "tflops": tf,
                    "mfu": mfu_val,
                    "tpu_version": tpu_version,
                    "flop_count": flops,
                })
            except Exception as e:
                print(f"{seq_len:>6}  {method:>8}  error: {e}")

    return records


def plot_results(records: list[dict], out_path="05_benchmarks/results/throughput.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    methods = list(dict.fromkeys(r["method"] for r in records))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for method in methods:
        pts = [r for r in records if r["method"] == method]
        seq = [r["seq_len"] for r in pts]
        tfs = [r["tflops"] for r in pts]
        mfus = [r["mfu"] for r in pts]
        ax1.plot(seq, tfs, "o-", label=method)
        ax2.plot(seq, mfus, "o-", label=method)

    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("TFLOP/s")
    ax1.set_title("Throughput")
    ax1.set_xscale("log", base=2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("MFU")
    ax2.set_title("Model FLOP Utilization")
    ax2.set_xscale("log", base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", default="v4", choices=list(TPU_PEAK_TFLOPS))
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--out", default="05_benchmarks/results/throughput.json"
    )
    args = parser.parse_args()

    records = run_throughput_sweep(tpu_version=args.tpu, causal=args.causal)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nResults written to {args.out}")

    if args.plot:
        plot_results(records)
