# Recompute, Don't Store

*A first-principles implementation of Flash Attention on TPU using JAX Pallas.*

> **GDE TPU Sprint — open contribution.**
> Status: research codebase, intentionally pedagogical over maximally tuned.

---

## Abstract

Standard scaled dot-product attention is a quadratic memory problem masquerading
as a mathematical one: a single `(S × S)` attention matrix at `S = 16384` and 8
heads in `bfloat16` already costs ~4.3 GB of HBM, before Q, K, V, or the output.
Flash Attention (Dao et al., 2022) resolves this on GPUs by tiling the
computation across SRAM, fusing softmax with the matmuls, and *recomputing* the
attention weights during the backward pass instead of storing them.

This repository re-derives that result on TPU silicon, using JAX Pallas as the
kernel-authoring substrate. The goal is not to ship the fastest kernel — Google
already does that — but to expose, in clean and readable form, **where GPU
intuitions about memory hierarchy and synchronization break down on TPU**, and
what has to be rewritten when you target VMEM, the systolic array, and Mosaic's
dimension semantics directly.

The codebase covers the full arc: motivation, algorithmic derivation, a naive
JAX baseline as the correctness oracle, forward and backward Pallas kernels with
the recomputation trick, a Grouped Query Attention extension, and a benchmark
suite modeling theoretical memory use and reporting throughput and Model FLOP
Utilization (MFU) against selected TPU peaks. Runtime memory counters are
backend-dependent; analytical memory alone does not demonstrate measured
savings. In the [2026-09-26 native v5e validation](docs/native_validation_status.md),
the tested kernels passed correctness checks but this pedagogical Pallas
implementation was slower than the naive JAX reference at S=4096.

---

## The problem

Naive attention materializes its attention matrix in HBM. Memory and time both
scale as `O(S²)` in the sequence dimension. Concretely, for one forward pass at
`bfloat16`, batch=1, heads=8, head_dim=128:

| `S` | Attention matrix | Total HBM (Q,K,V,O,A) |
| ---:| ----------------:| ---------------------:|
|   1K |        16.0 MB  |               18.1 MB |
|   4K |       256.0 MB  |              264.4 MB |
|  16K |         4.3 GB  |               4.4 GB |
|  32K |        17.2 GB  |              17.3 GB |

Activation memory between forward and backward is the term that breaks training
at long sequences. It is *not* the QKV tensors that hurt — it is the matrix you
never wanted to write in the first place.

Flash Attention's contribution is the observation that this matrix never needs
to exist as a tensor: softmax can be computed incrementally across tiles
provided you carry a per-row running maximum, and the backward pass can
*recompute* the weights from `Q, K, V` and a small set of saved log-sum-exp
statistics. The algorithm avoids storing the quadratic attention matrix and
retains linear-in-sequence auxiliary state; floating-point outputs still have
implementation-dependent numerical error.

---

## Why TPU is not a re-skinned GPU

Three properties of TPU silicon force a different kernel structure than GPU
Flash Attention assumes:

1. **VMEM is an explicit scratchpad.** Pallas `BlockSpec`s stage input tiles
   from HBM, and `pltpu.VMEM` declares scratch storage. This kernel carries
   three scratch tensors (`m`, `l`, `acc`) across KV tiles. GPU kernels can
   also manage on-chip shared memory explicitly; the difference here is the
   TPU/Pallas memory and compilation model, not that GPUs lack scratchpads.

2. **Tile layout constrains shapes.** This implementation chooses sequence
   tiles in multiples of 128 (`MIN_BLOCK_SIZE = 128`) and stores `m` and `l`
   with a padded trailing dimension of 128. These are implementation choices
   compatible with its TPU vector layout, not a universal minimum tile size
   on every axis. See the [Pallas TPU restrictions](https://docs.jax.dev/en/latest/pallas/tpu/details.html).

3. **The KV loop has a declared ordering dependency.** The forward grid marks
   its KV-tile axis `"arbitrary"` because successive tiles update shared
   online-softmax state. Mosaic can pipeline memory transfers around declared
   dependencies, but this run did not profile whether transfer and compute
   overlapped effectively.

A longer write-up of these tradeoffs lives in
[`docs/blog_post.md`](docs/blog_post.md).

---

## Repository layout

The directories follow the project's narrative arc. Each one is a discrete
step that you can read or run independently.

| Path | Contents | Runs on |
|---|---|---|
| `00_motivation/` | Memory complexity derivation: makes the `O(n²)` problem concrete with byte-level math | CPU |
| `01_flash_attention_math/` | Tiling algorithm and online softmax derivation; no JAX, just the math | CPU |
| `02_naive_jax_baseline/` | Pure-JAX scaled dot-product attention (correctness oracle) plus a sequence-length benchmark sweep | CPU or TPU |
| `03_pallas_kernels/` | Forward kernel (online softmax over VMEM-tiled KV), backward kernel (recomputation, no stored attention matrix), block-size selection helpers | TPU |
| `04_gqa_extension/` | Grouped Query Attention variant — same kernel body, GQA expressed entirely through `BlockSpec.index_map` | TPU |
| `05_benchmarks/` | Memory profile (naive vs flash), throughput / MFU on TPU v4 / v5e / v5p, xprof guide | TPU |
| `06_splash_attention_comparison/` | Comparison with Google's Splash Attention: where TPU-native sparse-mask kernels diverge from a port of Flash | CPU |
| `tests/` | Tiered pytest suite: pure-NumPy algorithm oracle, CPU-JAX correctness, TPU-Pallas kernel verification | CPU + TPU |
| `scripts/` | Helper scripts called by the launchers (e.g. `local_smoke_test.py`) | CPU |
| `docs/` | Long-form documents: kernel API (`kernel_api.md`), project plan (`CLAUDE.md`), TPU/GPU intuition blog post (`blog_post.md`) | — |
| `run_local.ps1`, `run_colab.sh` | One-shot launchers for the two execution paths (see Quick start) | Windows / Colab |

The split between code (`02_…` / `03_…`), measurement (`05_…`) and prose
(`00_…`, `01_…`, `06_…`, `docs/`) is deliberate: the prose is the contribution
as much as the code is.

---

## What this repository contributes

Concretely, what you get on top of "read the FlashAttention paper":

1. **A clean Pallas forward + backward** with the online-softmax update and the
   recomputation trick written in roughly 250 + 200 lines, heavily commented,
   wired together via `jax.custom_vjp`.
2. **A NumPy oracle for the tiling math** (`tests/test_online_softmax.py`) that
   you can read line-by-line against the kernel. Passing it supports the
   online-softmax math, but does not rule out untested shapes or kernel bugs.
3. **A working GQA kernel** that demonstrates the *minimal* extension: only
   the `index_map` for K/V changes (`h -> h // groups`); the kernel body is
   unchanged. This repository's GQA validation is forward-only; it does not
   establish a GQA training path.
4. **A benchmark harness** that reports an analytical tensor-footprint model,
   optional runtime memory counters where supported, TFLOP/s and MFU by TPU
   generation, and an xprof capture path. Runtime counters are not always
   true per-operation peaks and should not be equated with the analytical
   estimates.
5. **A reproducible split** between the parts of the project that can be
   reviewed and tested without TPU access (math, naive attention, online
   softmax oracle, shape/dtype validators) and the parts that genuinely need
   silicon.

---

## Quick start

The repository has CPU, Colab TPU, and native Cloud TPU VM paths. See
[Local vs Colab — what runs where](#local-vs-colab--what-runs-where) below for
the environment requirements; the same TPU-marked tests also run on a native
TPU VM.

### Local (Windows / CPU)

```powershell
./run_local.ps1                          # full setup + CPU tests + smoke + Jupyter
./run_local.ps1 -SkipInstall             # reuse an existing .venv
./run_local.ps1 -SkipNotebooks           # don't launch Jupyter at the end
./run_local.ps1 -Baseline                # also run the naive seq-length sweep
```

What runs: the `.venv` setup, `jax[cpu]` install, the CPU-safe `pytest` slice
(TPU-marked tests auto-skip), a forward + `jax.grad` smoke test, optionally the
naive-JAX baseline sweep, and finally a Jupyter server pointed at the three
story notebooks.

### Colab (TPU)

On a TPU Colab runtime (`Runtime -> Change runtime type -> TPU`):

```bash
!git clone https://github.com/ashioyajotham/recompute_dont_restore.git
%cd recompute_dont_restore
!bash run_colab.sh                       # default TPU_VERSION=v5e
!TPU_VERSION=v4 bash run_colab.sh        # use v4 peaks for MFU calculation
!SKIP_BENCH=1 bash run_colab.sh          # tests + smoke only
!SKIP_DRIVE=1 bash run_colab.sh          # don't mount Drive (results stay in VM)
```

### Native Cloud TPU VM

On a freshly provisioned TPU VM:

```bash
bash run_tpu_vm.sh
TPU_VERSION=v5e SKIP_BENCH=1 bash run_tpu_vm.sh
```

What runs: a new `.venv`, unpinned `jax[tpu]` installation, the TPU-marked test
slice, and, unless `SKIP_BENCH=1`, the memory-profile and throughput scripts.
This launcher does **not** run the full test suite, the naive baseline sweep,
or a Drive upload. It does not archive results; copy them off the VM before
deleting it. For the pinned, bounded native validation and its evidence
boundary, TPU topology, estimated cost, results, and shutdown record, see
[the native validation audit](docs/native_validation_status.md). That run
reused a four-chip v5e VM while benchmarking on one chip; the allocated
four-chip slice, not just the active chip, drives the compute estimate.
The archived timing sweep also has a [CPU-only per-case aggregate](docs/native_benchmark_analysis.md)
with an explicit variability and first-call analysis.
The [2026-09-28 TPU profiler follow-up](docs/native_profile_report.md) confirms
a device-execution gap at S=4096, but the Pallas custom call remains opaque;
no specific internal bottleneck or kernel optimization is claimed.
The [matched-stack LLO follow-up](docs/native_llo_followup.md) is prepared but
has not been run; it requires separate approval before any billable TPU work.

### Manual invocations (for the impatient)

```bash
python 02_naive_jax_baseline/benchmark_baseline.py --plot
python 05_benchmarks/memory_profile.py --plot
python 05_benchmarks/throughput_tflops.py --tpu v5e --plot
```

```bash
pytest -v                                # everything; TPU tests skip on CPU
pytest tests/test_online_softmax.py -v   # pure-NumPy algorithm oracle (no JAX needed)
pytest -m "not tpu" -v                   # CPU slice
pytest -m tpu -v                         # TPU slice
```

---

## Local vs Colab — what runs where

The cleanest way to understand the project is by what each environment can
actually exercise.

| Component | Local (CPU) | Colab (TPU) | Why |
|---|:---:|:---:|---|
| Motivation + math notebooks (`00_…`, `01_…`) | yes | yes | Pure NumPy / markdown |
| Splash comparison notebook (`06_…`) | yes | yes | Mostly conceptual prose |
| `tests/test_online_softmax.py` | yes | yes | NumPy reference; the algorithm oracle |
| `tests/test_standard_attention.py`, `test_utils.py` | yes | yes | JAX-on-CPU is fine |
| `tests/test_gqa.py::TestReferenceGQA` | yes | yes | `reference_gqa` expand-and-attend |
| Naive baseline sweep (`02_…/benchmark_baseline.py`) | slow | yes | Works on CPU; useful numbers only on accelerators |
| `tests/test_flash_attention.py` | skipped | **yes** | Requires `pltpu.VMEM`, `PrefetchScalarGridSpec` |
| `tests/test_gqa.py::TestGQAKernelVsReference` | skipped | **yes** | Same — Pallas TPU primitives |
| `flash_fwd.py` / `flash_bwd.py` (kernel dispatch) | import-only | **yes** | TPU-only Pallas calls |
| `05_benchmarks/memory_profile.py` | imports flash → fails | **yes** | Reports an analytical model and backend counters, not isolated peak HBM |
| `05_benchmarks/throughput_tflops.py` | imports flash → fails | **yes** | Reports throughput under an explicit FLOP convention |

The local script covers the CPU rows; the Colab and native TPU environments
can exercise the TPU rows. Individual tests and benchmark scripts can still
fail independently, so use the saved test and result reports to establish
which components passed.

---

## Reproducing the figures

The benchmark scripts emit JSON (raw) and PNG (plot) under
`05_benchmarks/results/`. Outputs are gitignored by default; use
`git add -f` to commit specific runs.

```bash
# 1. Baseline latency + theoretical naive HBM, S in {512, 1K, …, 32K}
python 02_naive_jax_baseline/benchmark_baseline.py --plot

# 2. Analytical memory model plus backend-dependent allocation counters
python 05_benchmarks/memory_profile.py --plot

# 3. Throughput in TFLOP/s and MFU, parameterized by TPU generation
python 05_benchmarks/throughput_tflops.py --tpu v5e --plot
python 05_benchmarks/throughput_tflops.py --tpu v5p --causal --plot
```

For the bounded capture protocol and its attribution limits, see
[`05_benchmarks/xprof_guide.md`](05_benchmarks/xprof_guide.md).

---

## Limitations and known gaps

This is research code optimized for clarity. Relative to a production kernel,
known gaps:

- **Single-device.** No multi-slice / multi-host sharding. A production
  implementation would `psum` across a model-parallel device axis.
- **Backward kernels stage full-sequence inputs in VMEM.** The dKV input
  blocks scale as `seq_q × 1792 bytes` at `d_k=d_v=128` / `bfloat16`; the dQ
  K/V blocks scale as `seq_kv × 512 bytes`. They exceed this repository's
  *conservative 8 MiB selection budget* at S=8192 for dKV; dQ reaches that
  budget at S=16384 and exceeds it at longer lengths.
  That budget is not the TPU's physical capacity: v5e has 128 MiB VMEM per
  TensorCore. Neither threshold is a measured OOM boundary. Two-level tiling
  would remove the full-sequence staging and is a candidate for future work.
  See [the JAX TPU hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html).
- **Block-size heuristic is hand-tuned.** `utils.get_block_sizes` chooses based
  on sequence length and a fixed VMEM budget; a profiler-guided autotuner would
  do better.
- **No FP8 / int8 path.** Inputs are restricted to `bfloat16` or `float16`.

JAX's `jax/experimental/pallas/ops/tpu/flash_attention.py` is a useful
implementation reference; this repository makes no production-performance
claim. The [native validation audit](docs/native_validation_status.md) records
the tested scope, observed slowdown, and estimated run cost.

---

## Background reading

For the supported downstream import surface, tensor shapes, forward/backward
residual contract, and static configuration requirements, see
[`docs/kernel_api.md`](docs/kernel_api.md).

For the proposal-style overview of the project (objectives, deliverables,
section-by-section guide), see [`docs/CLAUDE.md`](docs/CLAUDE.md).

For the deeper technical narrative on TPU vs GPU intuition divergence, see
[`docs/blog_post.md`](docs/blog_post.md).

---

## References

1. Dao, T., Fu, D., Ermon, S., Rudra, A., Ré, C. (2022).
   *FlashAttention: Fast and Memory-Efficient Exact Attention with
   IO-Awareness.* NeurIPS.
   [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
2. Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism
   and Work Partitioning.* ICLR.
   [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
3. Shazeer, N. (2019). *Fast Transformer Decoding: One Write-Head is All You
   Need* (MQA).
   [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
4. Ainslie, J., et al. (2023). *GQA: Training Generalized Multi-Query
   Transformer Models from Multi-Head Checkpoints.*
   [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
5. Milakov, M., Gimelshein, N. (2018). *Online normalizer calculation for
   softmax.* [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) — the
   incremental softmax update used inside the kernel.
6. JAX Pallas documentation.
   [jax.readthedocs.io/en/latest/pallas/](https://jax.readthedocs.io/en/latest/pallas/)
7. Google Research. *Splash Attention* — sparse, block-diagonal attention
   designed for TPUs, in
   `jax/experimental/pallas/ops/tpu/splash_attention/`.
8. JAX TPU Flash Attention reference: `jax/experimental/pallas/ops/tpu/flash_attention.py`.

---

## Acknowledgement

Google Cloud credits are provided for this project. #TPUSprint
