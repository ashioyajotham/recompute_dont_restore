# Recompute, Don't Store: Flash Attention on TPU from First Principles

Flash Attention (Dao et al., 2022) is one of the most practically impactful papers in recent LLM infrastructure. It changed how transformers are implemented everywhere — but it was designed for GPUs. This post documents what changes when you implement the same algorithm for TPU silicon, using JAX Pallas.

---

## The problem

Standard attention materializes a full $(S \times S)$ attention weight matrix in HBM. At 16K sequence length with 8 heads and bfloat16, that's 4.3 GB just for the attention matrix — before Q, K, V, or the output. Scale to multi-document contexts or high-resolution image tokens and the memory wall appears fast.

Flash Attention's solution: never write that matrix to HBM. Tile the computation across fast memory, accumulating the softmax and output simultaneously. The backward pass recomputes the attention weights from Q, K, V and a small set of saved statistics, rather than loading a stored activation.

The result: O(n) HBM usage instead of O(n²), with the same exact output.

---

## What changes on TPU

Flash Attention was built around GPU SRAM caches and CUDA's memory hierarchy. TPUs are different in three ways that matter:

**1. VMEM is a scratchpad, not a cache.**

On a GPU, the programmer hints at cache behavior but doesn't control it. On a TPU, VMEM (the vector scratchpad) is explicitly managed. You decide what goes in and out. This is both more work and more power: you can guarantee that your KV tiles live in fast memory, not get evicted by an unrelated operation.

In Pallas, this is expressed via `pltpu.VMEM(shape, dtype)` scratch shapes. The flash attention kernel allocates three VMEM tensors — the running max `m`, the running normalizer `l`, and the output accumulator `acc` — and explicitly manages their lifecycle across KV tile iterations.

**2. The systolic array layout imposes a 128-element minimum tile dimension.**

The TPU TensorCore processes tiles in multiples of 128 elements. Smaller tiles create padding overhead or fail to compile. This is why all block sizes in the implementation are multiples of 128 (`MIN_BLOCK_SIZE = 128`), and why the `m` and `l` statistics have shape `(block_q, 128)` rather than `(block_q, 1)` — a scalar per row can't be represented efficiently in the vector tile layout.

**3. Mosaic's dimension semantics replace CUDA's explicit sync primitives.**

In CUDA Flash Attention, the programmer uses `__syncthreads()` to coordinate tile loads and compute. In Mosaic (the compiler behind Pallas on TPU), you declare *what* the data dependencies are (`dimension_semantics`), and the compiler handles pipelining.

The kv-tile axis gets `"arbitrary"` semantics — meaning tiles must be processed in order, with each tile's online softmax state depending on the previous. Setting this correctly is what enables Mosaic to pipeline the next KV tile's DMA transfer while computing on the current tile.

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

The forward pass saves two O(n) tensors per attention layer: the per-row running max $m$ and normalizer $l$. These encode the log-sum-exp without storing the attention weights.

During the backward pass, instead of loading a stored O(n²) attention matrix, we recompute:

$$P_{ij} = \frac{\exp(S_{ij} - m_i)}{l_i}$$

using Q, K, V (already needed for the backward anyway) and the saved (m, l). The backward kernel structure mirrors the forward: loop over KV tiles for each Q tile, recomputing P on the fly.

This is the memory saving that matters for training: the activation memory between forward and backward is O(n), not O(n²). For a 32K-token sequence with 32 layers, the difference is measured in hundreds of GB.

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

GPU Flash Attention optimizes *access patterns* to be SRAM-friendly. You're working with caches — you can't control them directly, but you can arrange computation to reuse data before it's evicted.

TPU Pallas kernels optimize *explicit data movement*. You decide what's in VMEM at each step. The Mosaic compiler can then pipeline the DMA prefetch for tile $j+1$ while computing on tile $j$ — but only if you've declared the dependency structure correctly via `dimension_semantics`.

The practical implication: TPU kernel debugging involves reasoning about the VMEM budget explicitly (how big can `block_q` be before VMEM pressure?) and about the pipeline structure (is the `"arbitrary"` dimension placed correctly?). On GPU, these concerns are implicit.

Google's Splash Attention takes this further: for structured sparse patterns (sliding window, local+global), it skips DMA transfers entirely for masked tiles — something the explicit VMEM model makes straightforward to express.

---

## Where to go from here

The code in this repository is intentionally clear rather than maximally optimized. Known gaps relative to a production implementation:

- The backward pass loads full-sequence Q/K/V tensors per kernel step, which creates VMEM pressure at very long sequences. A two-level tiling (block_q_major + block_q_minor) would eliminate this.
- No multi-device (multi-slice) sharding. Production implementations use `jax.lax.psum` across device axes.
- Block size selection (`utils.get_block_sizes`) uses simple heuristics. A profiler-guided tuning loop would find better values.

The JAX reference implementation at `jax/experimental/pallas/ops/tpu/flash_attention.py` handles these cases and is worth reading alongside this code.

---

## References

1. Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS*.
2. Dao et al. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *ICLR*.
3. Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need.
4. Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.
5. JAX Pallas documentation: https://jax.readthedocs.io/en/latest/pallas/
