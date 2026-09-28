# Next native TPU experiment: matched-stack LLO attribution

The [2026-09-28 profile](native_profile_report.md) measured an approximately
8× S=4096 forward gap in XProf TPU XLA-module duration, but its Pallas
`CustomCall` was opaque. This follow-up is **prepared, not yet run**. It
does not alter the kernel, start/stop a VM, or authorize cloud spend.

## Interpretation to carry forward

At B1/H1/S4096/D128, the usual forward attention count `4*S*S*D` is
8,589,934,592 FLOPs, or **8.59 GFLOPs**. Dividing that count by module duration
gives approximately 122.3 TFLOP/s (naive) and 15.3 TFLOP/s (Pallas). These are
*useful-model-FLOP throughput proxies*, not XProf MXU utilization or measured
instruction throughput. A BF16-peak comparison is contextual because physical
dot operands and executed FLOPs inside the custom call were not attributed.

The two orderings agreed within about 0.3 µs for **device-module means**;
host medians varied more. The on-device durations were measured directly,
not calculated by subtracting a universal host floor. The 1024 logical
128×128 tile pairs at S4096 do not imply a 0.55 µs physical tile latency:
pipeline overlap, scheduling, and other work invalidate that division.

The source has `(128,128)` float32 m/l scratch and output shapes, but this
does not prove 128× executed exponential work. Its `.astype(float32)` dot
operands warrant inspection; `preferred_element_type=float32` concerns dot
output and accumulation, not proof of the physical MXU format. Likewise,
`num_scalar_prefetch=0` and `dimension_semantics="arbitrary"` do not prove
unoverlapped K/V DMA. [JAX's TPU pipelining guide](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
describes default two-buffer inputs/outputs and grid dimension semantics.

## Prepare without starting the VM

Use a **separate** Python 3.12 environment. Follow the
[OpenXLA custom-call profiling prerequisites](https://openxla.org/xprof/custom_call_profiling):
JAX/JAXLIB >=0.11.0, a compatible libtpu >=0.0.46, and `xprof-nightly`.
Resolve dependencies together; do not combine the old JAX 0.9.2 runtime with
libtpu 0.0.46, which previously crashed on profiler startup. Install the
repository's CPU-safe package dependencies in that environment as needed.
Record the exact installed distributions, `pip freeze`, and wheel hashes in
private run artifacts. The script's JSON lock enforces the relevant versions
at capture time; it is **not** a substitute for a reproducible package lock
or retained wheel hashes.

These commands inspect the plan without JAX import or TPU use:

```bash
python scripts/profile_tpu_llo.py --dry-run
python scripts/profile_tpu_llo.py --dry-run --mode runtime
python scripts/profile_tpu_llo.py --show-installed-stack
```

After installing the matched environment, redirect `--show-installed-stack`
to a private `stack.json` (for example under `artifacts/profiles/`) and retain
the output of `python -m pip freeze` next to it. The lock contains exact
versions of `jax`, `jaxlib`, `libtpu`, `numpy`, and `xprof-nightly`. Before any
TPU invocation, validate the environment, with the flags set **before any JAX
import**:

```bash
export JAX_PLATFORMS=tpu
export LIBTPU_INIT_ARGS='--xla_xprof_register_llo_debug_info=true'
python scripts/profile_tpu_llo.py --preflight --stack-lock artifacts/profiles/stack.json
```

`--preflight` checks versions and flags without importing JAX or accessing a
TPU. It does not prove the installed libtpu implements the flags; a failed
first TPU smoke capture is a stop condition, not a reason to change the kernel.

## Bounded capture, only after explicit cloud-run approval

Use the same four-chip v5e slice with **one selected chip**, noncausal BF16
B1/H1/S4096/D128, seed 0 and 128×128 blocks. The debug-info mode runs
naive and Pallas forward and forward-plus-backward in both fresh-process
orders. It retains the previous five warmups, ten unprofiled timing calls,
and five traced batches per method. O/dQ/dK/dV must remain finite and pass
`atol=rtol=0.05` before any trace is accepted. Use an ignored private output
directory that does not already exist:

```bash
python scripts/profile_tpu_llo.py \
  --mode debug --stack-lock artifacts/profiles/stack.json \
  --max-seconds 1200 --output-dir artifacts/profiles/llo-debug-RUN_ID
```

The 1,200-second controller deadline is **not** a VM stop timer. Before
starting the VM, separately arrange a hard stop by 30 READY minutes, a
roughly US$3 additional compute ceiling, and verify STOPPED afterward.
The estimated cumulative project spend and remaining credit must be checked
against actual billing before authorization; earlier cost numbers were
estimates, not invoices. Do not publish raw traces or logs without privacy
review.

Use `xprof-nightly` against each private trace directory and record
`get_kernel_stats`, `get_llo_analysis`, and (when needed) `get_llo_debug_string`
outputs. Require an LLO result reporting success **and** useful attribution
to Pallas instructions; a captured Perfetto file or nonzero event count alone
is insufficient. Compare naive/Pallas timings only within this matched stack.
The old 560.92 µs value is a historical reference, not a same-stack control.

Only if debug-info mode succeeds and runtime spans are needed, run the
separate opt-in capture with both LLO flags:

```bash
export LIBTPU_INIT_ARGS='--xla_xprof_register_llo_debug_info=true --xla_xprof_enable_custom_call_tracing=true'
python scripts/profile_tpu_llo.py --preflight --mode runtime \
  --stack-lock artifacts/profiles/stack.json
python scripts/profile_tpu_llo.py --mode runtime \
  --stack-lock artifacts/profiles/stack.json --max-seconds 1200 \
  --output-dir artifacts/profiles/llo-runtime-RUN_ID
```

Runtime mode captures only five calls each for Pallas forward and
forward-plus-backward. These instrumented durations are **not** performance
comparisons. [OpenXLA warns](https://openxla.org/xprof/custom_call_profiling)
that dense instrumentation can drop outer HLO events; keep the debug-only
capture intact. Periodic TPU hardware counters are not available on v5e by
the guide's hardware matrix, so LLO analysis may give instruction and cycle
estimates without a complete hardware-stall diagnosis.

If the matched stack fails correctness, Pallas compilation, profiler startup,
or LLO attribution, preserve that evidence and stop. Only after an attributed
result should one isolated kernel variant be designed and tested on the same
stack with untouched baseline control and repeated device measurements.
