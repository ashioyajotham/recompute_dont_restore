# Recompute, Don't Store

Flash Attention from first principles on TPU using JAX Pallas.

This repository walks the full arc: the memory problem, the tiling / online-softmax
algorithm, a naive JAX baseline, a Pallas forward + backward (recomputation) kernel,
a GQA extension, and benchmarks vs the naive reference.

## Structure

| Path | Contents | Runs on |
|---|---|---|
| `00_motivation/` | Memory complexity derivation — makes the O(n²) problem concrete | CPU |
| `01_flash_attention_math/` | Tiling algorithm and online softmax derivation | CPU |
| `02_naive_jax_baseline/` | JAX baseline attention + sequence-length benchmark sweep | CPU or TPU |
| `03_pallas_kernels/` | Flash Attention forward + backward (recomputation) in Pallas | TPU |
| `04_gqa_extension/` | Grouped Query Attention variant (MHA / GQA / MQA) | TPU |
| `05_benchmarks/` | Memory and throughput profiling, xprof guide | TPU |
| `06_splash_attention_comparison/` | Comparison with Google's Splash Attention | CPU |
| `tests/` | Tiered pytest suite (pure-NumPy, CPU-JAX, TPU-Pallas) | CPU + TPU |
| `docs/` | Extended design doc (`CLAUDE.md`) and blog post (`blog_post.md`) | — |

## Requirements

- Python 3.10+
- For Pallas kernels: TPU access (Google Cloud TPU v4 / v5e / v5p, or Colab TPU runtime)
- For everything else: a CPU install of JAX is enough

`requirements.txt` pins `jax[tpu]` for the TPU path; on Windows / CPU-only
machines install `jax[cpu]` instead (the `run_local.ps1` script handles this).

## Quick start

### Local (Windows, CPU) — notebooks, math, naive baseline, CPU-safe tests

```powershell
./run_local.ps1              # full setup + tests + smoke test + launch Jupyter
./run_local.ps1 -SkipInstall # reuse an existing .venv
./run_local.ps1 -Baseline    # also run the naive-JAX seq-length sweep (slow on CPU)
```

The script:

1. creates `.venv/` and installs `jax[cpu]`, numpy, matplotlib, pytest, jupyter
2. runs the CPU-safe pytest slice (TPU-marked tests auto-skip)
3. smoke-tests the naive JAX attention forward + gradients
4. optionally runs `02_naive_jax_baseline/benchmark_baseline.py --plot`
5. launches Jupyter so you can open the three story notebooks:
   - `00_motivation/attention_complexity.ipynb`
   - `01_flash_attention_math/tiling_and_online_softmax.ipynb`
   - `06_splash_attention_comparison/splash_vs_flash.ipynb`

### Colab TPU — Pallas kernels and the benchmark sweeps

On a TPU Colab runtime (`Runtime -> Change runtime type -> TPU`):

```bash
!git clone https://github.com/ashioyajotham/recompute_dont_restore.git
%cd recompute_dont_restore
!bash run_colab.sh                          # default: TPU v5e
!TPU_VERSION=v4 bash run_colab.sh           # override TPU generation
!SKIP_BENCH=1 bash run_colab.sh             # tests + smoke only, no sweep
```

The script:

1. installs `jax[tpu]` against the libtpu release channel
2. verifies `jax.devices()` lists a TPU
3. runs the full `pytest -v` suite (TPU tests now execute)
4. smoke-tests `flash_fwd` against the naive reference at seq=512
5. runs the three benchmark sweeps from `05_benchmarks/`, writing JSON and PNG
   outputs into `05_benchmarks/results/`
6. mounts Google Drive and copies the results to
   `MyDrive/recompute_dont_store_results/<timestamp>/` so you can download
   them later and drop them into your local checkout under
   `05_benchmarks/results/`

Result files are ignored by default via `.gitignore`; use
`git add -f 05_benchmarks/results/<file>` to commit specific runs.

### Manual invocations

```bash
# Baseline sweep (works on CPU, faster on TPU)
python 02_naive_jax_baseline/benchmark_baseline.py --plot

# Memory profile: naive vs flash (TPU only)
python 05_benchmarks/memory_profile.py --plot

# Throughput + MFU (TPU only; --tpu selects peak TFLOP/s denominator)
python 05_benchmarks/throughput_tflops.py --tpu v5e --plot
```

### Tests

```bash
pytest -v                                   # everything; TPU tests skip on CPU
pytest tests/test_online_softmax.py -v      # pure NumPy — the algorithm oracle
pytest -m "not tpu" -v                      # CPU-only slice explicitly
pytest -m tpu -v                            # TPU-only slice (needs a TPU)
```

## Design notes

Extended write-ups live in `docs/`:

- [`docs/CLAUDE.md`](docs/CLAUDE.md) — project proposal, section-by-section guide,
  deliverables
- [`docs/blog_post.md`](docs/blog_post.md) — the TPU-vs-GPU intuition shift,
  targeted at the GDE community and *Unsupervised Insights* newsletter

## References

- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
- Dao et al. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
- Shazeer (2019). *Fast Transformer Decoding: One Write-Head is All You Need* (MQA)
- Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models*
- JAX Pallas documentation: https://jax.readthedocs.io/en/latest/pallas/
- JAX reference implementation: `jax/experimental/pallas/ops/tpu/flash_attention.py`

## Acknowledgement

Google Cloud credits are provided for this project. #TPUSprint
