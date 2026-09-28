# Profiling the native JAX/Pallas attention kernels

The [offline validation](../docs/native_benchmark_analysis.md) found that the
tested Pallas kernel is slower than the naive JAX implementation at S=4096.
Those numbers are synchronized **host wall times**, not TPU device traces.
Profiling is the next diagnostic step; none of the possible bottlenecks below
has yet been established by a trace.

## Prepared bounded experiment

`scripts/profile_tpu_cases.py` is a CPU-safe planner and a TPU-only capture
harness. It does **not** start or stop the VM. First inspect its four-process
schedule without importing JAX or creating files:

```bash
python scripts/profile_tpu_cases.py --dry-run
```

Only after a separate approval to start billable compute, verify the VM is
stopped, configure an external auto-stop safeguard, and create an isolated
environment with JAX/JAXLIB 0.9.2, libtpu 0.0.37, and NumPy 2.5.3 before
running from the repository root. JAX 0.9.2 declares `libtpu==0.0.37.*` for
its TPU extra; the earlier benchmark's libtpu 0.0.46 could execute kernels
but crashed at profiler startup with a plugin API-size mismatch. Do not mix
latency numbers from those two software stacks without labeling them:

```bash
python scripts/profile_tpu_cases.py \
  --output-dir artifacts/profiles/native-v5e-<unique-run-id> \
  --max-seconds 1080
```

Choose a new output directory. `artifacts/profiles/` is gitignored; raw
traces, manifests, and logs may contain local paths and should remain private.
The controller ceiling is **not** a VM shutdown mechanism or a cost guarantee.
The four-chip slice continues billing while READY, even if the harness exits;
verify an external stop timer and confirm the VM reaches STOPPED afterward.
Before a paid run, check the current [Cloud TPU price list](https://cloud.google.com/tpu/pricing)
and available credits. The prior `us-south1` on-demand v5e rate was
US$1.416/chip-hour, or US$5.664/hour for four chips.

The harness uses BF16 B1/H1/D128, seed 0, 128×128 blocks, S=1024 and 4096,
noncausal. It profiles naive and Pallas forward and complete
forward-plus-backward loss/gradient calls. Each sequence length runs twice,
once naive-first and once Pallas-first, in fresh processes. Before a trace,
all four methods pass the output/gradient check (`atol=rtol=0.05`) and each
method completes five warmups. Ten unprofiled synchronized calls estimate
its call time; five annotated trace batches then target about 100 ms each,
capped at 500 calls per batch. Traced time is **not** a replacement for the
unprofiled latency benchmark. There is no isolated backward-only call.

JAX captures compute-and-sync TPU traces and requests performance counters.
Unsupported trace options or no identifiable TPU activity cause a visible
failure, retaining partial private artifacts. A nonempty trace and an
automated event heuristic do **not** certify useful device attribution; XProf
review is mandatory before interpreting results. The manifest records the
Git revision and dirty flag, source hashes, package versions, fixture, method
order, correctness errors, baseline samples, trace hashes, and completion status.

## Review in XProf

After capture, point a locally secured XProf or TensorBoard instance at the
private run directory. Inspect the TPU device lanes, not just host annotations;
confirm that each of the four method traces contains the expected calls and
that counters or op breakdowns are present. JAX documents both
[`start_trace`/`stop_trace`](https://docs.jax.dev/en/latest/profiling.html)
and the [XProf trace viewer](https://docs.jax.dev/en/latest/profiling.html#xprof-tensorboard-profiling).

Use the traces to test, rather than assume, these hypotheses:

| Question | Evidence to seek | Caution |
| --- | --- | --- |
| Is the Pallas forward path waiting on K/V movement? | Device DMA timeline and gaps adjacent to MXU work | `num_scalar_prefetch=0` describes SMEM scalar inputs, not K/V double-buffering. `dimension_semantics="arbitrary"` expresses a dependency, not guaranteed overlap. |
| What precision and work does the MXU actually execute? | Compiled op types, MXU utilization, and device compute timeline | Source-level float32 casts do not alone prove FP32 MXU execution or conversion cost. |
| Does padded `m/l` state dominate VPU work? | Vector/transcendental op breakdown and, if available, compiled IR | A 128-wide source expression does not prove 128× physical exponentials; compilation may simplify broadcasts. |
| Is backward staging or slicing expensive? | Compare complete forward+backward traces with forward traces, inspect backward op timeline | Do not subtract p50s to claim isolated backward latency; profile attribution may remain ambiguous. |

The similar causal and noncausal slowdown in the archived benchmark suggests
a shared problem; it does not establish that skipping fully masked causal
tiles would have no benefit. The 32 MiB BF16 matrix arithmetic size at S=4096
also does not show whether XLA materializes an attention matrix or how it
fuses the naive JAX implementation. If the trace cannot answer a question,
record it as inconclusive and plan a narrower controlled experiment before
changing kernel logic.

The [2026-09-28 capture report](../docs/native_profile_report.md) records the
completed four-process run. Its device traces confirm the slowdown, but XProf
represents the Pallas kernel as an opaque custom call: the requested counters
did not yield an internal MXU/DMA/VPU breakdown, and LLO data was absent.
