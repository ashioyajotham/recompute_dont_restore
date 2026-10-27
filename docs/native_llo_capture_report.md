# Matched-stack native LLO follow-up — 2026-09-28 capture, 2026-09-29 offline review

The bounded S=4096 follow-up completed on one TPU v5e chip in a four-chip
slice. Both fresh-process method orders passed O/dQ/dK/dV correctness
(`atol=rtol=0.05`, finite; largest absolute error 0.001953125). No kernel
source was changed. The capture used commit
`03f784cbd045df1a0969a02494f881ad3ff17401` with a clean worktree,
JAX/JAXLIB 0.11.2, libtpu 0.0.48, NumPy 2.5.3, Python 3.12.13, and
`xprof-nightly` 2.24.2a20260928. The flag
`--xla_xprof_register_llo_debug_info=true` was set before JAX import;
runtime custom-call tracing was **not** enabled.

The private capture was recovered without rerunning a TPU workload. The
retrieval VM was confirmed STOPPED after upload, then the local archive was
checked against its SHA-256. Archive name:
`native-llo-debug-03f784c-20260929T051529Z.tar.gz`; SHA-256:
`e4112baaf154b238ff2d32106a7e9bf9bc0e7bd2df96da53294f655910356773`.
It is access-controlled and contains raw traces/logs and local paths; do not
publish the archive. The retrieval-only restart was bounded by a prearmed
local stop watchdog and a 600-second deadline. Compute cost is not an invoice.

## Same-stack timing evidence

XProf `get_kernel_stats --include_summary=true` reports these mean TPU device
step durations (µs). Each value includes five traced batches of repeated
calls; it is not the unprofiled host p50. The orderings were separate fresh
processes.

| Order | Naive fwd | Pallas fwd | Naive fwd+bwd | Pallas fwd+bwd |
|:--|--:|--:|--:|--:|
| Naive first | 86.8584 | 559.7494 | 249.3258 | 1336.8982 |
| Pallas first | 86.8617 | 559.7978 | 249.3233 | 1336.9378 |

Within this stack, Pallas/naive is about 6.44× for forward and 5.36× for
complete forward-plus-backward. The two orders are closely aligned. These
ratios should not be merged with the earlier JAX 0.9.2/libtpu 0.0.37 trace:
the software stack changed. The Pallas forward custom-call itself averaged
559.73–559.78 µs. The complete Pallas trace exposed three custom calls of
roughly 515.6, 487.8, and 330.4 µs plus a small fusion. This is a device
gap, not a diagnosis of what inside the kernel is slow.

## LLO attribution boundary

Using the matching local `xprof-nightly` against the eight retained XPlane
files, `get_llo_analysis` returned `LLO_DATA_ABSENT` for three of the four
Pallas traces. It returned `success: true` for the Pallas-first **forward**
trace, including a source map. Filtered to its Pallas forward module, that
result contains 1,360 **static** instructions: 740 VALU, 278 load/store,
157 SALU, 102 XLU, 32 MXU, 20 sync, 10 DMA, and 21 other. This count is a
compiled instruction inventory, not executed instruction counts, cycle
shares, utilization, or measured runtime inside the loop. The unfiltered
result includes unrelated compiled modules and must not be presented as a
Pallas-only breakdown.

The Perfetto trace shows XLA module/op events and many sync-flag events, but
no runtime LLO instruction or per-unit spans in this debug-only capture.
Neither the static VALU count nor the top-level timings establishes whether
padded statistics, BF16 conversion, DMA overlap, or backward slicing is the
dominant cause. On v5e, periodic TPU hardware counters are also unavailable
per the [OpenXLA custom-call profiling guide](https://openxla.org/xprof/custom_call_profiling).

## Decision

Preserve this partial LLO result and do **not** edit kernel logic to fit a
speculative bottleneck. If internal attribution is still worth the cost, the
smallest next test is a separately approved, fresh-process capture with both
`--xla_xprof_register_llo_debug_info=true` and
`--xla_xprof_enable_custom_call_tracing=true`, first for Pallas forward only
(which requires narrowing the current runtime-mode harness). That
instrumentation changes runtime and may drop outer HLO events, so its
durations must not be used as an uninstrumented performance comparison.
