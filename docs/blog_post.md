# Recompute, Don't Store: Flash Attention on TPU from First Principles

Flash Attention (Dao et al., 2022) is one of the most practically impactful papers in recent LLM infrastructure. It changed how transformers are implemented everywhere — but it was designed for GPUs. This post documents what changes when you implement the same algorithm for TPU silicon, using JAX Pallas.

---

## The problem

Standard attention materializes a full $(S \times S)$ attention weight matrix in HBM. At 16K sequence length with 8 heads and bfloat16, that's 4.3 GB just for the attention matrix — before Q, K, V, or the output. Scale to multi-document contexts or high-resolution image tokens and the memory wall appears fast.

Flash Attention's solution: never write that matrix to HBM. Tile the computation across fast memory, accumulating the softmax and output simultaneously. The backward pass recomputes the attention weights from Q, K, V and a small set of saved statistics, rather than loading a stored activation.

The algorithm avoids an O(n²) stored attention matrix, though actual peak
memory depends on the implementation and compiler. Floating-point results
need not be bitwise identical to a separately compiled naive kernel.

---

## What changes on TPU

Flash Attention was originally developed for GPU memory hierarchies. GPU
shared memory is also explicitly managed; this Pallas implementation differs
in its TPU memory spaces, layout requirements, and pipeline declarations:

**1. VMEM is a scratchpad, not a cache.**

On TPU, VMEM is a vector scratchpad rather than a cache. Pallas `BlockSpec`s
describe staged input tiles, while scratch shapes reserve VMEM for state.
Actual allocation and transfer behavior remains subject to compilation.

In Pallas, this is expressed via `pltpu.VMEM(shape, dtype)` scratch shapes. The flash attention kernel allocates three VMEM tensors — the running max `m`, the running normalizer `l`, and the output accumulator `acc` — and explicitly manages their lifecycle across KV tile iterations.

**2. Tile layout constrains shapes.**

This implementation chooses sequence tiles in multiples of 128
(`MIN_BLOCK_SIZE = 128`) and pads `m` and `l` to `(block_q, 128)` for its TPU
vector layout. These choices should not be read as universal minimum tile
dimensions. See the [Pallas TPU restrictions](https://docs.jax.dev/en/latest/pallas/tpu/details.html).

**3. Mosaic uses declared grid dependencies.**

The Pallas grid declares `dimension_semantics` so Mosaic can account for
dependencies while scheduling data movement and computation.

The KV-tile axis gets `"arbitrary"` semantics because each tile consumes the
previous tile's online-softmax state. The current run did not profile whether
Mosaic actually overlapped the next DMA transfer with computation.

---

## The online softmax update

The core algorithm insight is that softmax can be computed incrementally without seeing the full row. Given a running max $m$ and normalizer $l$ after processing $j$ tiles:

When tile $j+1$ arrives with max $m' = \max_{\text{tile}} S_{ij}$:
- New global max: $m^{\text{new}} = \max(m, m')$
- Rescaling factor: $\alpha = \exp(m - m^{\text{new}})$
- New normalizer: $l^{\text{new}} = \alpha \cdot l + \sum_{\text{tile}} \exp(S_{ij} - m^{\text{new}})$
- Rescaled output: $O^{\text{new}} = (\alpha \cdot O \cdot l + P_{\text{tile}} V_{\text{tile}}) / l^{\text{new}}$

At the end of all KV tiles, $O$ is the correct normalized attention output. The attention matrix $P$ was computed tile by tile and never written to HBM.

---

## The backward pass: recomputation over storage

The forward pass saves two O(n) tensors per attention layer: the per-row running max $m$ and normalizer $l$. This implementation pads their trailing dimension to 128, so its constant factor is substantial even though their sequence-length scaling is linear.

During the backward pass, instead of loading a stored O(n²) attention matrix, we recompute:

$$P_{ij} = \frac{\exp(S_{ij} - m_i)}{l_i}$$

using Q, K, V (already needed for the backward anyway) and the saved (m, l). The backward kernel structure mirrors the forward: loop over KV tiles for each Q tile, recomputing P on the fly.

This is the intended training-memory advantage: stored attention auxiliaries
scale linearly, not quadratically, with sequence length. This project has not
measured a per-operation peak-HBM reduction or validated a 32K-token,
32-layer training workload.

---

## GQA: the minimal change

Grouped Query Attention (GQA) — used in Gemma, LLaMA-3, and Mistral — shares KV heads across groups of Q heads. In standard attention, you'd either expand K and V to full Q-head count (wasteful) or write separate code for each case.

In the Pallas implementation, GQA requires exactly one change: the `index_map` lambda for K and V in `BlockSpec`. Instead of:

```python
index_map=lambda b, h, i, j: (b, h, j, 0)
```

it becomes:

```python
index_map=lambda b, h, i, j: (b, h // groups, j, 0)
```

The grid still runs over `num_q_heads` for the head dimension. Multiple Q-head tiles map to the same KV tile via the integer division. The kernel body is unchanged. This is the cleanliness of expressing memory layout in `BlockSpec` rather than in the compute code.

---

## TPU vs GPU: the deeper intuition shift

Both GPU and TPU Flash Attention implementations reason about on-chip storage
and data movement. Here, Pallas exposes TPU VMEM and grid-dependency
declarations directly. Whether Mosaic overlaps DMA prefetch with tile compute
is a profiling question, not something the declarations alone establish.

The practical implication for this implementation is to check tile and
backward-staging footprints, validate the online-softmax dependencies, and
profile the generated TPU work before attributing latency to memory transfer.

Structured-sparsity implementations such as Splash Attention can avoid work
on fully masked tiles. This repository's causal path does not do that; it
visits every KV tile and masks future positions inside the tile.

---

## Where to go from here

The code in this repository is intentionally clear rather than maximally optimized. Here is an honest map of what remains, ordered by impact.

**Bounded backward staging.** The backward kernels currently declare
full-sequence input blocks. Their footprint grows with sequence length;
the repository's 8 MiB heuristic is not a hardware failure threshold.
Two-level tiling is a candidate design, but has not been implemented or
validated here. Compare it with JAX's Pallas attention implementations.

**Multi-device sharding.** The validated path uses one chip. Distributing
heads or sequence blocks across devices would require a separate design,
collective-communication contract, and correctness/performance evaluation.

**Profiler-guided block size tuning.** `utils.get_block_sizes` uses static
heuristics. A controlled sweep over valid `(block_q, block_kv)` pairs could
test whether another configuration improves latency; no improvement percentage
has been measured. A TPU profiler trace is needed to identify compute,
transfer, or scheduling bottlenecks.

**Quantized and paged KV cache (inference).** These are possible future
extensions, not features of this repository. Quantization and page-table
lookup introduce costs and correctness requirements that would need separate
implementation and measurement.

**Splash Attention for structured sparsity.** Google's [Splash Attention](https://github.com/google-deepmind/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/) is a separate reference for structured masks. Avoiding work for fully masked tiles would require a different scheduling and indexing design here, followed by correctness and performance checks.

---

The core insight in this codebase — *recompute cheaply, store sparingly* — is not specific to Flash Attention. It is the general principle behind activation checkpointing, rematerialization in JAX (`jax.remat`), and the broader trend of trading FLOPs (cheap, parallelizable) for HBM bandwidth (scarce, sequential). Flash Attention made this concrete at the operator level. The Pallas kernel model extends it to the tile level. Understanding both is the foundation for writing the next generation of custom TPU kernels.

---

## References

1. Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS*.
2. Dao et al. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *ICLR*.
3. Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need.
4. Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.
5. JAX Pallas documentation: https://jax.readthedocs.io/en/latest/pallas/
6. JAX reference Flash Attention (TPU): `jax/experimental/pallas/ops/tpu/flash_attention.py`
7. Google DeepMind Splash Attention: https://github.com/google-deepmind/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention

---

## Acknowledgement

Google Cloud credits are provided for this project. #TPUSprint
