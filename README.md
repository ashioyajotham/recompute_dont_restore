# Recompute, Don't Store

Flash Attention from first principles on TPU using JAX Pallas.

## Structure

| Directory | Contents |
|---|---|
| `00_motivation/` | Memory complexity derivation — makes the O(n²) problem concrete |
| `01_flash_attention_math/` | Tiling algorithm and online softmax derivation |
| `02_naive_jax_baseline/` | JAX baseline attention and benchmark sweep |
| `03_pallas_kernels/` | Flash Attention forward and backward in Pallas |
| `04_gqa_extension/` | Grouped Query Attention (GQA) variant |
| `05_benchmarks/` | Memory and throughput profiling, xprof guide |
| `06_splash_attention_comparison/` | Comparison with Google's Splash Attention |

## Requirements

Python 3.10+. TPU access required for the Pallas kernels (TPU v4 or v5 recommended).

```
pip install -r requirements.txt
```

The Pallas kernels can be run on CPU in interpret mode for debugging:

```python
pl.pallas_call(kernel, ..., debug=True)   # pure JAX execution, no TPU needed
```

## Running

```bash
# Baseline benchmark sweep
python 02_naive_jax_baseline/benchmark_baseline.py

# Memory profile: naive vs flash
python 05_benchmarks/memory_profile.py

# Throughput and MFU
python 05_benchmarks/throughput_tflops.py
```

Notebooks in each directory can be run on a TPU Colab runtime.

## References

- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
- Dao et al. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
- Shazeer (2019). *Fast Transformer Decoding: One Write-Head is All You Need* (MQA)
- Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models*
- JAX Pallas documentation: https://jax.readthedocs.io/en/latest/pallas/
- JAX reference implementation: `jax/experimental/pallas/ops/tpu/flash_attention.py`
