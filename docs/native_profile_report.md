# Native v5e profiler run — 2026-09-28

The bounded follow-up captured **16 TPU traces**: naive JAX and Pallas,
forward and complete forward-plus-backward, at S=1024 and S=4096, with both
naive-first and Pallas-first fresh-process orderings. All four process
manifests finished `captured_pending_xprof_review`; O, dQ, dK, and dV passed
the finite and `atol=rtol=0.05` checks. The largest absolute error recorded
in the four manifests was 0.002930. This confirms the earlier slowdown has a
device-execution component, but **does not identify the internal Pallas
bottleneck**.

## Fixture and evidence

Single-device TPU v5e (a four-chip `v5litepod-4` slice was allocated), BF16,
B1/H1/D128, noncausal, 128×128 Pallas blocks, seed 0. Each method had five
warmups, ten synchronized unprofiled timing calls, and five traced batches.
The successful capture used JAX/JAXLIB 0.9.2, libtpu 0.0.37, NumPy 2.5.3,
and Python 3.12.13. The source snapshot is commit
`b9a8dd2d8ca608b9ad7290a460a49ac7fd74c0d1` plus the uncommitted
profiler script (SHA-256
`2ddbbc116f03a4a26b7ff83ade67bc9c0c0d2c63fe75eacf682035e263c9f4da`).
The private archive contains the complete traces, manifests, failed-attempt
diagnostics, a Git source bundle, and that exact script:

- Archive: `gs://torch-tpu-vm/torch-tpu-vs-jax-pallas/native-v5e-profiler-final-20260928.tar.gz`
- Archive SHA-256: `9f88050a89281f05a347361df92feebfa9cbc23ed6403cc31de3f034bca6dbb5`
- Hash object: same URI with `.sha256` appended.

The archive is access-controlled, not a public download. The local download
of the original VM archive passed SHA-256 verification; the final archive was
built locally to include the exact successful-run script, and its embedded
script hash was checked against the manifests. The final upload and hash
object were confirmed in the private bucket; its CRC32C (`2GF5bw==`) matched
the local archive. Raw traces and logs may contain
local paths; do not publish them without review.

## Timings

Unprofiled synchronized host p50 (ms, ten calls per method):

| S | Order | Naive fwd | Pallas fwd | Naive fwd+bwd | Pallas fwd+bwd |
|---:|:---|---:|---:|---:|---:|
| 1024 | naive first | 0.1448 | 0.1794 | 0.2089 | 0.2904 |
| 1024 | Pallas first | 0.1404 | 0.1713 | 0.2057 | 0.2816 |
| 4096 | naive first | 0.2084 | 0.7047 | 0.3878 | 1.5667 |
| 4096 | Pallas first | 0.2087 | 0.7050 | 0.3950 | 1.5717 |

Mean XProf TPU *XLA module* durations (µs per traced call, rounded; the two
orderings agreed within about 0.3 µs):

| S | Naive fwd | Pallas fwd | Naive fwd+bwd | Pallas fwd+bwd |
|---:|---:|---:|---:|---:|
| 1024 | 5.70 | 38.33 | 15.16 | 92.00 |
| 4096 | 70.21 | 560.92 | 190.62 | 1367.05 |

At S=4096 the Pallas forward module is roughly 8× the naive module on the
device; complete forward-plus-backward is roughly 7.2×. The corresponding
host p50 ratios are roughly 3.38× and 4.0×. This supports a real device-side
gap masked in part by fixed host costs. These trace averages are **not** a
replacement for the archived 30-trial, three-repetition benchmark; the
software stack also differs from that benchmark's libtpu 0.0.46. Do not
subtract medians to label a backward-only time.

## What the trace can and cannot say

The S=4096 Pallas forward appears as one `tpu_custom_call` of about 561 µs.
The complete call includes Pallas custom calls of roughly 518, 516, and
330 µs, plus a small fusion. XProf's roofline output labels the Pallas
operation `CustomCall (opaque)`; its zero-valued FLOP and bandwidth fields
are **missing attribution, not evidence of zero work**. The compute-utilization
query returned no devices, and LLO analysis returned `LLO_DATA_ABSENT`.
Consequently this run cannot rank DMA bubbles, precision/conversion cost,
padded vector work, or backward slicing. The naive trace has visible HLO
fusions and copy operations, but those do not establish whether its attention
matrix reaches HBM. [XProf's trace viewer guide](https://openxla.org/xprof/trace_viewer)
notes that some tracks are derived; we use module and op durations as scoped
timing evidence, not a complete hardware-counter breakdown.

## Profiler compatibility and cost

The earlier validated JAX 0.9.2 + libtpu 0.0.46 environment passed output
and gradient checks at S=1024 but crashed at `jax.profiler.start_trace`:
`Unexpected PLUGIN_Profiler_Api size: expected 80, got 104`, followed by
`PluginTracer::Start()` SIGSEGV. JAX 0.9.2's TPU extra declares
`libtpu==0.0.37.*`; a separate environment using that pair captured all
traces. A second setup attempt only lacked `absl-py` and did no tracing.
Neither failure should be construed as a kernel correctness result.

The VM reached READY at approximately 04:47:15 UTC and STOPPED at
approximately 05:08:10 UTC. At the [listed us-south1 on-demand v5e rate](https://cloud.google.com/tpu/pricing)
of US$1.416 per chip-hour, the four-chip READY interval implies about US$1.98
of TPU compute. This is an estimate, not an invoice or credit-balance check.
A VM-side auto-stop timer was armed before capture; STOPPED was independently
confirmed after archiving.

## Next experiment, only if internal attribution is needed

Do not change kernel logic based on these traces alone. A future small
capture should first validate a newer *matched* JAX/libtpu/XProf stack that
supports Pallas custom-call LLO tracing, then compare a smoke trace against
the present result before spending on the full matrix. The
[OpenXLA custom-call profiling guide](https://openxla.org/xprof/custom_call_profiling)
documents the newer prerequisites and pre-import LLO flags. A changed stack
must be labeled separately and rechecked for correctness and timing; an
unavailable LLO view is a valid stopping condition, not a reason to infer a
specific bottleneck.
