"""Bounded, matched-stack LLO follow-up for the native v5e attention comparison.

This program does not provision or stop a TPU VM. Keep raw output private.
Run --show-installed-stack after installing the new stack, save that exact JSON
as a lock, and use --preflight before any capture. No JAX import occurs until
the environment and LLO flags have been checked.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import profile_tpu_cases as baseline


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("jax", "jaxlib", "libtpu", "numpy", "xprof-nightly")
DEBUG_FLAG = "--xla_xprof_register_llo_debug_info=true"
RUNTIME_FLAG = "--xla_xprof_enable_custom_call_tracing=true"
MODES = ("debug", "runtime")
DEFAULT_MAX_SECONDS = 1200  # Controller deadline, not a VM auto-stop.


def installed_stack():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def planned_runs(mode):
    if mode == "debug":
        return [{"seq": 4096, "order": order,
                 "methods": list(baseline.method_order(order))}
                for order in baseline.ORDERS]
    if mode == "runtime":
        return [{"seq": 4096, "order": "pallas-only",
                 "methods": ["pallas_forward", "pallas_forward_backward"]}]
    raise ValueError(f"Unsupported LLO mode: {mode}")


def validate_environment(lock_path, mode):
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11+ is required for this LLO experiment")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or set(lock) != set(PACKAGES) or not all(
            isinstance(value, str) and value for value in lock.values()):
        raise ValueError(f"Stack lock must contain exactly: {', '.join(PACKAGES)}")
    installed = installed_stack()
    if installed != lock:
        raise RuntimeError(f"Stack differs from lock: installed={installed}, lock={lock}")
    match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", installed["jax"])
    if match is None or tuple(map(int, match.groups())) < (0, 11):
        raise RuntimeError("LLO follow-up requires JAX >= 0.11.0")
    if installed["jaxlib"].split("+")[0] != installed["jax"].split("+")[0]:
        raise RuntimeError("JAX and jaxlib versions must match exactly")
    libtpu_match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:\.|$)", installed["libtpu"])
    if libtpu_match is None or tuple(map(int, libtpu_match.groups())) < (0, 0, 46):
        raise RuntimeError("LLO follow-up requires libtpu >= 0.0.46")
    if os.environ.get("JAX_PLATFORMS") != "tpu":
        raise RuntimeError("Set JAX_PLATFORMS=tpu; CPU fallback is not accepted")
    if os.environ.get("JAX_INTERPRET_PALLAS", "").lower() in ("1", "true", "yes", "on"):
        raise RuntimeError("Unset JAX_INTERPRET_PALLAS for TPU profiling")
    if os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"):
        raise RuntimeError("Unset JAX_DEFAULT_MATMUL_PRECISION for the matched fixture")
    flags = shlex.split(os.environ.get("LIBTPU_INIT_ARGS", ""))
    if DEBUG_FLAG not in flags:
        raise RuntimeError(f"Set LIBTPU_INIT_ARGS with {DEBUG_FLAG} before importing JAX")
    if mode == "debug" and any(flag.startswith(
            "--xla_xprof_enable_custom_call_tracing=") for flag in flags):
        raise RuntimeError("Do not enable runtime instrumentation in the debug-info capture")
    if mode == "runtime" and RUNTIME_FLAG not in flags:
        raise RuntimeError(f"Runtime mode requires {RUNTIME_FLAG}")
    return {"packages": installed, "llo_mode": mode, "libtpu_init_args": flags}


def capture_short(jax, name, workload, output_dir):
    """Limit vtrace volume; these instrumented timings are not performance data."""
    fn, fn_args = workload
    for _ in range(baseline.WARMUPS):
        jax.block_until_ready(fn(*fn_args))
    trace_dir = output_dir / name
    trace_dir.mkdir()
    jax.profiler.start_trace(str(trace_dir), create_perfetto_trace=True,
                             profiler_options=baseline.profile_options(jax))
    try:
        with jax.profiler.StepTraceAnnotation(name, step_num=0):
            for _ in range(5):
                jax.block_until_ready(fn(*fn_args))
    finally:
        jax.profiler.stop_trace()
    return {"warmups": baseline.WARMUPS, "trace_batches": 1,
            "calls_per_batch": 5, "instrumented": True,
            "trace": baseline.trace_evidence(trace_dir)}


def worker(case, output_dir, stack_lock, mode):
    environment = validate_environment(stack_lock, mode)
    import jax
    import jax.numpy as jnp

    devices = jax.devices("tpu")
    if len(devices) != 4 or "v5" not in devices[0].device_kind.lower():
        raise RuntimeError(f"Expected four addressable v5e chips; got {devices}")
    output_dir.mkdir()
    source_hashes = baseline.source_manifest()
    source_hashes["scripts/profile_tpu_llo.py"] = baseline.sha256_file(Path(__file__))
    manifest = {"schema_version": 1, "status": "incomplete", "case": case,
                "environment": environment, "python": sys.version.split()[0],
                "device_kind": devices[0].device_kind, "devices_available": len(devices),
                "devices_used": 1, "source_control": baseline.source_control(),
                "source_sha256": source_hashes, "results": {}}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        with jax.default_device(devices[0]):
            workloads, errors = baseline.build_workloads(jax, jnp, case["seq"])
            manifest["correctness"] = errors
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            for name in case["methods"]:
                capture = capture_short if mode == "runtime" else baseline.capture_one
                manifest["results"][name] = capture(jax, name, workloads[name], output_dir)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["status"] = "captured_pending_xprof_review"
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def controller(output_dir, stack_lock, mode, max_seconds):
    environment = validate_environment(stack_lock, mode)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    report = {"schema_version": 1, "status": "incomplete", "environment": environment,
              "max_seconds": max_seconds, "planned": planned_runs(mode), "runs": []}
    report_path = output_dir / "controller.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    started = time.monotonic()
    try:
        for index, case in enumerate(report["planned"], 1):
            remaining = max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("LLO controller time ceiling reached")
            name = f"s{case['seq']}-{case['order']}"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-index",
                       str(index - 1), "--mode", mode, "--stack-lock",
                       str(stack_lock.resolve()), "--output-dir", str(output_dir / name)]
            with (output_dir / f"{name}.stdout.log").open("w") as stdout, \
                 (output_dir / f"{name}.stderr.log").open("w") as stderr:
                try:
                    result = subprocess.run(command, env=os.environ.copy(), stdout=stdout,
                                            stderr=stderr, timeout=remaining, check=False)
                except subprocess.TimeoutExpired as error:
                    raise TimeoutError(f"{name} exceeded controller time ceiling") from error
            report["runs"].append({"name": name, "returncode": result.returncode})
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            if result.returncode != 0:
                raise RuntimeError(f"{name} failed; inspect private logs and manifest")
        report["status"] = "captured_pending_xprof_review"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-installed-stack", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--mode", choices=MODES, default="debug")
    parser.add_argument("--stack-lock", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.show_installed_stack:
        print(json.dumps(installed_stack(), indent=2))
        return
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "planned": planned_runs(args.mode),
                          "max_seconds": args.max_seconds}, indent=2))
        return
    if args.stack_lock is None:
        parser.error("--stack-lock is required except for --dry-run/--show-installed-stack")
    if args.preflight:
        print(json.dumps(validate_environment(args.stack_lock, args.mode), indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for capture")
    if args.worker_index is not None:
        cases = planned_runs(args.mode)
        if args.worker_index < 0 or args.worker_index >= len(cases):
            parser.error("Invalid worker index")
        worker(cases[args.worker_index], args.output_dir, args.stack_lock, args.mode)
    else:
        controller(args.output_dir, args.stack_lock, args.mode, args.max_seconds)


if __name__ == "__main__":
    main()
