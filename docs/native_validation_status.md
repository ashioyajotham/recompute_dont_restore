# Native validation audit

Audit date: 2026-09-26. Starting source revision: `0e75416`.

## Evidence boundary and result

The earlier downstream comparison exercised imported native forward/backward
kernels on TPU. That is not validation of this repository's complete launcher,
GQA extension, notebooks, or memory benchmark claims.

On 2026-09-26, an isolated checkout of `0e75416` with the two test changes
in this audit ran on one TPU v5e device from an addressable four-device VM.
Interpreter mode and CPU fallback were disabled. The existing suite passed
**75/75** after scoping
high float32 dot precision to its NumPy-comparison test. The additional suite
passed **18/18**: 12 random-cotangent forward/VJP cases across causal masks,
two seeds, batch/head grids and sequence lengths through 1024, plus six
MHA/GQA/MQA forward cases. This validates these cases, not every model shape
or training path. GQA backward remains untested here.

## Run configuration

| Field | Recorded value |
| --- | --- |
| Cloud resource | Existing `torch-tpu-vm` in `astral-comfort-237012`, `us-south1-a`; `v5litepod-4`, 2×2 topology |
| Devices | Four addressable TPU v5e chips; one selected for every test and benchmark case; no sharding |
| Python and libraries | Python 3.12.13; JAX/JAXLIB 0.9.2; libtpu 0.0.46; NumPy 2.5.3; `absl-py` installed explicitly |
| Source | Starting commit `0e75416ccf092032fe1a528fffaa0a9c04e95d65`; scoped precision test change and added validation test copied into isolated source tree |
| Benchmark fixtures | BF16, B1/H1/D128, S=256/512/1024/2048/4096, causal and noncausal, fixed seed 0 across repetitions, `BlockSizes(128, 128)` |
| Timing | One selected chip, five warmups and 30 synchronized host wall-clock trials per method/case, three fresh Python processes; first call including compilation recorded separately |

The benchmark compares JIT-compiled naive JAX attention with the imported
Pallas forward/recomputing-backward implementation. Each process checks
output and dQ/dK/dV against naive JAX before measuring. This is an
independent *implementation* reference, not an independent framework or a
proof for arbitrary inputs. A BF16 `atol=rtol=0.05` check is deliberately
broader than a bitwise comparison. The separate expanded TPU tests use two
seeds and multi-batch/head fixtures; the timing sweep does not.

JAX's TPU v5e hardware reference lists 128 MiB VMEM per TensorCore; this
repository's 8 MiB block-selector value is only a conservative forward-tile
budget. The four-chip topology and 16 GB HBM per chip are described in the
[Cloud TPU v5e configuration guide](https://docs.cloud.google.com/tpu/docs/v5e);
VMEM capacity is from the [JAX TPU hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html).
The archived `pyproject.toml` is unmodified baseline metadata; the local
packaging fix adding `absl-py` is a separate change.

The unmodified existing suite yielded **74 passed, one failed** on this TPU:
its strict float32 NumPy comparison differed by about 0.00327 under the TPU's
default dot precision. Running the entire suite at `highest` precision was
not a valid fix: 25 tests then failed Mosaic compilation with a
`Bad lhs type` error. The final change applies `highest` only inside the
affected reference comparison; Pallas tests still run at the default
precision. Both failed control runs are retained in the private evidence.

## Timing result and interpretation

The bounded native timing sweep is separate from correctness acceptance.
Three fresh processes each ran B1/H1/D128 BF16, S=256 through 4096, causal
and noncausal, with five warmups and 30 synchronized trials for native JAX
and Pallas forward and forward-plus-backward. All 30 case/repetition records
passed the per-case `atol=rtol=0.05` output and gradient comparisons; the
largest observed absolute discrepancies across the 30 records were 0.01953
for output and 0.015625 for each of dQ, dK, and dV. The first call,
which may include compilation and cache effects, is recorded separately;
it is not an isolated compiler-time measurement.

Median of the three per-process p50s (milliseconds; lower is faster):

| S | Mask | Naive forward | Pallas forward | Naive forward+backward | Pallas forward+backward |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | Noncausal | 0.153 | 0.154 | 0.210 | 0.213 |
| 1024 | Noncausal | 0.154 | 0.185 | 0.227 | 0.293 |
| 4096 | Noncausal | 0.233 | 0.708 | 0.456 | 1.548 |
| 4096 | Causal | 0.233 | 0.705 | 0.463 | 1.614 |

These synchronized timings include host dispatch and are not device-only
kernel measurements. The small S=256 difference is near the host-timing
floor; the S=4096 gap is consistent across the three processes. There was
no TPU profiler trace, controlled block-size sweep, or isolated per-operation
HBM measurement in this run.

This pedagogical Pallas implementation was **not faster** than the naive
JAX reference in this configuration. At S=4096, median-of-repetitions
forward p50 was 0.708 ms Pallas versus 0.233 ms naive (noncausal), and
0.705 ms versus 0.233 ms (causal). Forward-plus-backward p50 was 1.548 ms
versus 0.456 ms (noncausal), and 1.614 ms versus 0.463 ms (causal).
The three per-repetition S=4096 forward ratios were 3.041–3.060x
(noncausal) and 2.994–3.049x (causal). These are wall-clock timings of
these implementations, not a claim that dense attention is generally faster
or that the Pallas algorithm lacks its expected asymptotic memory advantage.
No backward-only latency is inferred by subtracting medians, and no
unsupported runtime memory-saving or MFU claim is made. Compilation-inclusive
first calls, p95s, and raw trial samples are retained in the archive.

The [offline aggregate](native_benchmark_analysis.md) reports every case's
per-process p50 range, paired Pallas/naive ratio, p95 median, and first-call
host time. Its [JSON companion](native_benchmark_aggregate.json) contains
median/min/max summaries without raw trials or local filesystem paths. Both
are reproducible from the three archived benchmark JSON reports using the
CPU-only [summarizer](../scripts/summarize_native_benchmark.py); no TPU rerun
was needed. The fixed order of the four methods remains a design limitation.
After verifying and extracting the private archive, regenerate into an empty
output directory from the repository root:

```bash
python scripts/summarize_native_benchmark.py \
  results/benchmark-rep1.json results/benchmark-rep2.json results/benchmark-rep3.json \
  --archive-sha256 489797e795f50c8d3d1804bcb3e5c521365879a1fc068f5e3f7943470c9267a1 \
  --json-out summary/aggregate.json --markdown-out summary/analysis.md
```

The summarizer validates each report's case grid, timing samples, and
environment consistency. Its `--archive-sha256` argument records a hash;
verify the archive separately before extraction. It refuses to overwrite
existing summary outputs.

The source offers plausible explanations, not a measured bottleneck ranking:
the forward path processes every KV tile even for causal attention and masks
future positions rather than skipping whole tiles; it uses 128×128 Pallas
matmul tiles and padded float32 `m`/`l` state; and the two backward kernels
declare full-sequence input blocks and sequential inner loops. The forward
kernel also casts Q/K/V to float32 before its dot operations, while the naive
reference starts from BF16 operands. Which of these dominates latency, or how
XLA compiles the naive reference, requires a separate profiler-backed study.
The result validates a correct pedagogical reproduction, not an optimized
Flash Attention implementation or a novel algorithm. The contribution here
is a readable JAX/Pallas adaptation, corrected online-softmax recurrence,
backward recomputation wiring, and a reproducible TPU validation record.
The TorchTPU bridge comparison is a separate integration study; neither
study establishes a speedup or measured activation-memory reduction for
this native kernel.

The private archive is
`gs://torch-tpu-vm/torch-tpu-vs-jax-pallas/native-validation-0e75416-20260926T061039Z.tar.gz`
with SHA-256
`489797e795f50c8d3d1804bcb3e5c521365879a1fc068f5e3f7943470c9267a1`.
The archive and sidecar were uploaded, downloaded again, and verified locally
with `sha256sum -c`. It contains raw logs with local paths, so do not publish
it without a separate privacy review.

The essential rerun commands, after installing the pinned JAX/libtpu versions
on a TPU VM and this checkout, are:

```bash
export JAX_PLATFORMS=tpu
unset JAX_INTERPRET_PALLAS JAX_DEFAULT_MATMUL_PRECISION
python -m pytest --ignore=tests/test_native_validation.py -v
python -m pytest tests/test_native_validation.py -v
python scripts/native_benchmark.py --output benchmark-rep1.json --seed 0
python scripts/native_benchmark.py --output benchmark-rep2.json --seed 0
python scripts/native_benchmark.py --output benchmark-rep3.json --seed 0
```

Each benchmark invocation is a new process and refuses to overwrite an
existing output. The commands assume genuine TPU access; do not use Pallas
interpret mode to stand in for the hardware run.

## Other findings

- First CPU-safe run with Python 3.14.7 and JAX 0.9.2: 51 passed,
  six failed, 36 TPU cases deselected. All six failures came from a missing
  `absl` import through Pallas, not numerical disagreement. Added `absl-py`
  to package dependencies. The rerun passed: **57 passed, 36 deselected**.
  The deselected cases require TPU; this is CPU-only evidence.

- `run_tpu_vm.sh` upgrades unpinned JAX and runs only TPU-marked tests. Its
  behavior differs from the former README description; the README now states
  what it actually runs. For a reproducible run, pin versions and archive
  the result before VM shutdown.
- Existing causal backward coverage checks finiteness, not reference agreement.
  The added random-cotangent tests address this coverage gap without changing
  kernels or relaxing existing tests.
- GQA exposes a forward Pallas call, not an explicit recomputing custom VJP.
  Forward success must not be presented as GQA training validation.
- The throughput helper distinguishes executed from useful causal FLOPs. The
  current kernel masks but does not skip future KV tiles, so executed FLOPs
  must not be halved for causal throughput/MFU.
- Default benchmark sweeps reach long sequences. Begin with S <= 4096 and
  inspect allocation sizes before running larger cases.
- On the reused TPU VM, systemd's default `LimitMEMLOCK=8388608` prevented
  JAX from mapping the TPU (`Couldn't mmap: Resource temporarily unavailable`).
  An interactive shell with unlimited memlock initialized JAX successfully.
  Background test workers must set `LimitMEMLOCK=infinity`.

## Execution gates

1. Preserve the original revision and run its existing suite separately from
   new tests. Save test reports, failures, skips, versions, and device topology.
2. Reject CPU fallback and interpreter mode for hardware acceptance. Run the
   full CPU-safe and TPU suites; investigate any skip or failure.
3. Only after correctness passes, collect five warmups and 30 synchronized
   forward and forward-plus-backward trials per case, across three fresh
   processes. Record compilation separately. Do not subtract latency medians
   to claim backward-only timing.
4. Separate analytical memory from runtime counters; omit unsupported MFU
   denominators. Archive raw evidence privately with checksums and verify it
   before stopping compute.

## Cost, archival, and shutdown

The agreed first-pass compute ceiling is US$25. This run reused the existing
four-chip VM, in an isolated working directory without replacing its previous
environments. A VM-side stop timer and a local backup stop watchdog were set
before testing. The [Google Cloud TPU price list](https://cloud.google.com/tpu/pricing)
showed US$1.416 per v5e chip-hour in `us-south1` on 2026-09-26. Four allocated
chips therefore imply **US$5.664 per READY hour**, even though this experiment
used one chip. From the approximately 05:29 UTC start to 06:12 UTC stop, the
43-minute TPU-compute estimate is `43/60 × $5.664 ≈ $4.06`. This is an
estimate, **not an invoice**: exact READY duration, billing credits, storage,
and any other charges were not independently verified. The private archive
and hash above provide durable raw evidence; they are not public downloads.

After downloading and checking the archive checksum, the VM was stopped and
its state confirmed as `STOPPED` on
2026-09-26 at approximately 06:12 UTC; the scheduled timer alone was not
treated as proof of shutdown. The local backup watchdog was canceled after
confirmation. The VM-side stop timer remains scheduled for 08:20 UTC if the
same VM is restarted before then; inspect or cancel that timer before any
early restart.
