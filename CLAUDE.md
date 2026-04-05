# Recompute, Don't Store
### A First-Principles Implementation of Flash Attention on TPU using JAX Pallas

**GDE TPU Sprint — Project Proposal**
**Category:** PyTorch on XLA/TPU — How to Write a Pallas Kernel
**Session Type:** Talk / Presentation (25–30 minutes)

---

## One-Line Description

Flash Attention was built around GPU memory constraints — but TPUs think differently. This project implements Flash Attention from first principles in JAX using Pallas, exposing exactly where TPU silicon demands you rewrite your intuitions.

---

## Background & Motivation

Standard scaled dot-product attention has a dirty secret: its memory footprint scales quadratically with sequence length. For a 16K-token sequence at bfloat16 with a single attention head, the attention matrix alone consumes ~2GB of HBM. Scale to multi-head and you hit the memory wall fast.

Flash Attention (Dao et al., 2022) solved this on GPUs by reframing the problem. Instead of materializing the full attention matrix, it tiles the computation across SRAM, fusing the softmax and matrix multiplications into a single kernel pass. The backward pass ditches stored activations entirely — it *recomputes* the attention matrix from Q, K, V on the fly, trading compute for memory. This is not a minor optimization. It is a rethinking of where computation happens.

But Flash Attention was designed around GPU HBM bandwidth bottlenecks and CUDA's memory hierarchy. TPUs are a different machine. They have:

- A systolic array architecture optimized for dense matrix multiplications
- Scratchpad memory (VMEM) rather than SRAM caches
- XLA as the compiler substrate, with aggressive operation fusion and layout constraints
- Pallas — JAX's low-level kernel authoring language — for dropping below XLA when you need explicit memory control

The interesting question is not "does Flash Attention work on TPUs?" (it does, Google ships it). The interesting question is: **what changes when you write it for TPU silicon from scratch?** Where do the GPU intuitions break down, and what do you have to rethink?

This project answers that question, hands on, in JAX.

---

## Objectives

- Derive the Flash Attention algorithm from scratch — tiling, online softmax, and the recomputation trick
- Implement a clean naive attention baseline in JAX as the benchmark
- Write a forward-pass Flash Attention kernel in Pallas with explicit VMEM tiling
- Implement the backward pass using recomputation rather than stored activations
- Extend to Grouped Query Attention (GQA) as a practically relevant variant
- Profile both implementations across sequence lengths using xprof and custom benchmarks
- Produce a reproducible codebase and Colab notebook for community use

---

## Abstract

Standard attention is a quadratic memory problem masquerading as a mathematical one. Flash Attention solved it for GPUs by rethinking *where* computation happens — tiling across SRAM, recomputing on the backward pass instead of storing activations. But TPUs are not GPUs. Their memory hierarchy, systolic array layout, and XLA compilation model demand a different set of intuitions.

This project implements Flash Attention from first principles in JAX, using Pallas — JAX's low-level kernel language — to target TPU memory directly. We profile standard attention against our implementation across sequence lengths from 512 to 32K, analyze the memory bandwidth equations on TPU v4/v5 hardware, and extend to a Grouped Query Attention (GQA) variant. The result is a reproducible, documented codebase that demystifies one of the most important efficiency innovations in modern LLM infrastructure — and shows exactly where the TPU/GPU intuitions diverge.

---

## Project Structure

```
recompute-dont-store/
│
├── README.md
├── requirements.txt
│
├── 00_motivation/
│   └── attention_complexity.ipynb
│       # Derives O(n²) memory scaling from first principles
│       # Worked numerical example: 16K tokens, bfloat16, multi-head
│       # Establishes the problem before any code is written
│
├── 01_flash_attention_math/
│   └── tiling_and_online_softmax.ipynb
│       # Algorithm derivation: why tiling works
│       # Online softmax update rule (numerically stable)
│       # The recomputation insight — intuition before implementation
│       # No JAX yet, just the math
│
├── 02_naive_jax_baseline/
│   ├── standard_attention.py
│   │   # Pure JAX scaled dot-product attention
│   │   # jit-compiled, vmapped across heads
│   │   # Fully documented, this is what we beat
│   └── benchmark_baseline.py
│       # Sequence length sweep: 512, 1K, 2K, 4K, 8K, 16K, 32K
│       # Records HBM usage and wall-clock time
│
├── 03_pallas_kernels/
│   ├── flash_fwd.py
│   │   # Forward pass in Pallas
│   │   # Explicit VMEM tiling (block_q, block_kv)
│   │   # Online softmax accumulation across KV tiles
│   │   # Inline comments explaining every non-obvious decision
│   ├── flash_bwd.py
│   │   # Backward pass using recomputation
│   │   # Recomputes attention weights from Q, K, V + log-sum-exp
│   │   # No stored activation matrix — the core of the memory saving
│   └── utils.py
│       # Block size selection heuristics for TPU VMEM
│       # Shape validation and dtype helpers
│
├── 04_gqa_extension/
│   └── grouped_query_attention.py
│       # GQA variant of the flash kernel
│       # Relevant to Gemma, Mistral, LLaMA-3 style architectures
│       # Shows how the Pallas kernel generalizes
│
├── 05_benchmarks/
│   ├── memory_profile.py
│   │   # HBM peak usage: naive vs flash, across sequence lengths
│   │   # Uses JAX device memory profiling
│   ├── throughput_tflops.py
│   │   # Compute utilization on TPU v4/v5
│   │   # MFU (Model FLOP Utilization) calculation
│   ├── xprof_guide.md
│   │   # How to capture and read xprof traces for these kernels
│   │   # Annotated screenshots: what a well-utilized TPU looks like vs not
│   └── results/
│       # Pre-generated charts and logs for reproducibility
│
├── 06_splash_attention_comparison/
│   └── splash_vs_flash.ipynb
│       # Google's Splash Attention: sparse, block-diagonal masking
│       # Designed natively for TPUs — compare design assumptions
│       # Side-by-side: where our implementation aligns and diverges
│
└── blog_post.md
    # Write-up targeting GDE community and Unsupervised Insights newsletter
    # Covers the TPU/GPU intuition divergence angle in depth
```

---

## Section-by-Section Guide

### 00 — Motivation
Before writing a line of JAX, make the problem concrete. Compute exactly how much memory a naive attention matrix consumes at various sequence lengths and head counts on bfloat16. The goal is to make the memory wall visceral — not abstract — so the solution feels necessary rather than clever.

### 01 — The Math
Derive the tiling trick and the online softmax update rule. Most tutorials skip this or bury it. The key insight is that softmax can be computed incrementally across tiles without materializing the full row — and that this is only numerically stable if you track the running maximum. Spend time here. The Pallas code only makes sense if this derivation is clear.

### 02 — Naive Baseline
A clean, well-commented JAX implementation of standard attention. `jit`-compiled, `vmap`-ed across heads, benchmarked. This is the reference point for every comparison. Keep it readable — it is teaching code as much as benchmark code.

### 03 — Pallas Kernels
The core of the project. Pallas exposes explicit VMEM tiling for TPU kernels — this is where TPU-specific intuitions live. The forward pass tiles across KV blocks, accumulating the softmax numerator and denominator online. The backward pass is the non-obvious part: instead of storing the O(n²) attention matrix, it recomputes it during the backward pass from Q, K, V and the saved log-sum-exp values. This is the memory saving in action.

### 04 — GQA Extension
Grouped Query Attention is now standard in nearly every production LLM. Extending the Pallas kernel to GQA demonstrates that the tiling approach generalizes and adds direct practical relevance to architectures like Gemma and LLaMA-3.

### 05 — Benchmarks
The payoff section. Memory usage vs sequence length plots, TFLOP/s utilization, and a comparison table. Includes an xprof guide covering how to capture traces, what good TPU utilization looks like, and how to identify memory or compute bottlenecks. All results pre-generated and committed for reproducibility.

### 06 — Splash Attention Comparison
Google's internal answer to Flash Attention for TPUs — sparse, block-diagonal masking, natively XLA-aware. Compare the design assumptions: where does Splash make different tradeoffs, and what does that tell us about what TPU silicon actually rewards?

---

## Deliverables

| Artifact | Description |
|---|---|
| GitHub repository | Full code, notebooks, benchmarks, results |
| Colab notebook | TPU runtime, runnable end-to-end by anyone |
| xprof guide | Annotated profiling walkthrough for the kernels |
| Blog post | Unsupervised Insights write-up on TPU/GPU intuition divergence |
| GDE Talk (25–30 min) | Presentation covering the full arc: problem → kernel → results |

---

## Prerequisites (for collaborators)

- Python 3.10+
- JAX with TPU backend (`jax[tpu]`)
- Pallas (`jax.experimental.pallas`)
- Familiarity with attention mechanism basics
- TPU access (Google Cloud TPU v4 or v5 recommended)

---

## References

- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
- Dao et al. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
- Jax Pallas documentation — https://jax.readthedocs.io/en/latest/pallas/
- Google Research. *Splash Attention* — https://github.com/google-deepmind/jax/tree/main/jax/experimental/splash_attention
- Shazeer (2019). *Fast Transformer Decoding: One Write-Head is All You Need* (MQA)
- Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
