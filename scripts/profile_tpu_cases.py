"""Prepare/capture bounded native JAX versus Pallas TPU profiles.

Dry-run uses only the Python standard library. A real capture requires the
previously validated JAX TPU environment; this script never starts or stops
a Cloud TPU VM. Raw traces and worker logs belong in a private, ignored path.
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = (1024, 4096)
ORDERS = ("naive-first", "pallas-first")
METHODS = ("naive_forward", "pallas_forward", "naive_forward_backward",
           "pallas_forward_backward")
EXPECTED_PACKAGES = {"jax": "0.9.2", "jaxlib": "0.9.2", "libtpu": "0.0.37",
                     "numpy": "2.5.3"}
WARMUPS = 5
BASELINE_TRIALS = 10
TRACE_BATCHES = 5
TARGET_BATCH_MS = 100.0
MAX_CALLS_PER_BATCH = 500
DEFAULT_MAX_SECONDS = 1080  # About 18 READY minutes; not a VM stop safeguard.


def method_order(order):
    if order == "naive-first":
        return METHODS
    if order == "pallas-first":
        return ("pallas_forward", "naive_forward", "pallas_forward_backward",
                "naive_forward_backward")
    raise ValueError(f"Unknown order: {order}")


def planned_runs():
    return [{"seq": seq, "order": order, "methods": list(method_order(order))}
            for seq in SEQUENCES for order in ORDERS]


def calls_per_batch(p50_ms):
    if not math.isfinite(p50_ms) or p50_ms <= 0:
        raise ValueError("Baseline p50 must be positive and finite")
    return min(MAX_CALLS_PER_BATCH,
               max(5, math.ceil(TARGET_BATCH_MS / p50_ms)))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest():
    paths = ("02_naive_jax_baseline/standard_attention.py",
             "03_pallas_kernels/flash_fwd.py", "03_pallas_kernels/flash_bwd.py",
             "03_pallas_kernels/utils.py", "scripts/profile_tpu_cases.py")
    return {path: sha256_file(ROOT / path) for path in paths}


def source_control():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def trace_evidence(trace_dir):
    """Conservative heuristic; final device attribution still needs XProf review."""
    traces = list(trace_dir.rglob("perfetto_trace.json.gz"))
    if len(traces) != 1 or traces[0].stat().st_size == 0:
        raise RuntimeError("Expected one nonempty Perfetto trace")
    with gzip.open(traces[0], "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        raise RuntimeError("Perfetto trace has no traceEvents list")
    device_pids = set()
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "M":
            continue
        args = event.get("args") or {}
        label = f"{event.get('name', '')} {args.get('name', '')}".lower()
        if "tpu" in label:
            device_pids.add(event.get("pid"))
    device_events = [event for event in events if isinstance(event, dict)
                     and event.get("ph") in ("X", "B", "E")
                     and (event.get("pid") in device_pids
                          or "tpu" in f"{event.get('cat', '')} {event.get('name', '')}".lower())]
    if not device_events:
        raise RuntimeError("No identifiable TPU activity in Perfetto trace; review in XProf")
    return {"perfetto_sha256": sha256_file(traces[0]),
            "perfetto_bytes": traces[0].stat().st_size,
            "device_events_detected": len(device_events),
            "device_event_check": "heuristic; confirm lanes and counters in XProf"}


def packages():
    found = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    if found != EXPECTED_PACKAGES:
        raise RuntimeError(f"Pinned environment required: {EXPECTED_PACKAGES}; found {found}")
    return found


def build_workloads(jax, jnp, seq):
    import numpy as np
    sys.path[:0] = [str(ROOT / "03_pallas_kernels"),
                    str(ROOT / "02_naive_jax_baseline")]
    from flash_bwd import flash_attention
    from standard_attention import attention
    from utils import BlockSizes

    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    inputs = tuple(jax.random.normal(key, (1, 1, seq, 128), dtype=jnp.bfloat16)
                   for key in keys)
    jax.block_until_ready(inputs)
    qkv = inputs[:3]
    naive = jax.jit(lambda q, k, v: attention(q, k, v, causal=False))
    pallas = jax.jit(lambda q, k, v: flash_attention(
        q, k, v, False, None, BlockSizes(128, 128)))

    def with_grad(fn):
        return jax.jit(jax.value_and_grad(
            lambda q, k, v, do: jnp.sum(fn(q, k, v).astype(jnp.float32)
                                           * do.astype(jnp.float32)),
            argnums=(0, 1, 2)))

    naive_fb, pallas_fb = with_grad(naive), with_grad(pallas)
    expected_o = jax.block_until_ready(naive(*qkv))
    actual_o = jax.block_until_ready(pallas(*qkv))
    expected_grad = jax.block_until_ready(naive_fb(*inputs))[1]
    actual_grad = jax.block_until_ready(pallas_fb(*inputs))[1]
    errors = {}
    for name, actual, expected in zip(("o", "dq", "dk", "dv"),
                                      (actual_o, *actual_grad),
                                      (expected_o, *expected_grad)):
        a = np.asarray(actual, dtype=np.float32)
        e = np.asarray(expected, dtype=np.float32)
        finite = bool(np.isfinite(a).all() and np.isfinite(e).all())
        passed = finite and bool(np.allclose(a, e, atol=0.05, rtol=0.05))
        errors[name] = {"max_abs": float(np.max(np.abs(a - e))),
                        "finite": finite, "passed": passed}
    if not all(item["passed"] for item in errors.values()):
        raise RuntimeError(f"Correctness gate failed at S={seq}: {errors}")
    return {
        "naive_forward": (naive, qkv),
        "pallas_forward": (pallas, qkv),
        "naive_forward_backward": (naive_fb, inputs),
        "pallas_forward_backward": (pallas_fb, inputs),
    }, errors


def profile_options(jax):
    options = jax.profiler.ProfileOptions()
    options.advanced_configuration = {
        "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
        "tpu_num_chips_to_profile_per_task": 1,
        "tpu_perf_counters": True,
    }
    return options


def capture_one(jax, name, workload, output_dir):
    fn, fn_args = workload
    for _ in range(WARMUPS):
        jax.block_until_ready(fn(*fn_args))
    samples = []
    for _ in range(BASELINE_TRIALS):
        start = time.perf_counter()
        jax.block_until_ready(fn(*fn_args))
        samples.append((time.perf_counter() - start) * 1000)
    p50 = statistics.median(samples)
    calls = calls_per_batch(p50)
    trace_dir = output_dir / name
    trace_dir.mkdir()
    jax.profiler.start_trace(
        str(trace_dir), create_perfetto_trace=True,
        profiler_options=profile_options(jax))
    try:
        for step in range(TRACE_BATCHES):
            with jax.profiler.StepTraceAnnotation(name, step_num=step):
                for _ in range(calls):
                    jax.block_until_ready(fn(*fn_args))
    finally:
        jax.profiler.stop_trace()
    return {"baseline_p50_ms": p50, "baseline_samples_ms": samples,
            "warmups": WARMUPS, "trace_batches": TRACE_BATCHES,
            "calls_per_batch": calls, "trace": trace_evidence(trace_dir)}


def worker(seq, order, output_dir):
    if seq not in SEQUENCES or order not in ORDERS:
        raise ValueError("Unsupported profiling case")
    if os.environ.get("JAX_PLATFORMS") != "tpu":
        raise RuntimeError("Set JAX_PLATFORMS=tpu to reject CPU fallback")
    if os.environ.get("JAX_INTERPRET_PALLAS", "").lower() in ("1", "true", "yes", "on"):
        raise RuntimeError("Pallas interpret mode is not a TPU profile")
    if os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"):
        raise RuntimeError("Unset JAX_DEFAULT_MATMUL_PRECISION for the validated fixture")
    versions = packages()
    import jax
    import jax.numpy as jnp
    devices = jax.devices("tpu")
    if len(devices) != 4:
        raise RuntimeError(f"Expected four addressable v5e chips, found {len(devices)}")
    if "v5" not in devices[0].device_kind.lower():
        raise RuntimeError(f"Unexpected device kind: {devices[0].device_kind}")
    output_dir.mkdir()
    manifest = {
        "schema_version": 1, "seq": seq, "causal": False,
        "batch": 1, "heads": 1, "head_dim": 128, "dtype": "bfloat16",
        "seed": 0, "block_q": 128, "block_kv": 128,
        "method_order": list(method_order(order)), "packages": versions,
        "python": sys.version.split()[0], "device_kind": devices[0].device_kind,
        "devices_available": len(devices), "devices_used": 1,
        "source_control": source_control(), "source_sha256": source_manifest(),
        "trace_mode": "TRACE_COMPUTE_AND_SYNC", "tpu_perf_counters": True,
        "status": "incomplete", "results": {},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        with jax.default_device(devices[0]):
            workloads, errors = build_workloads(jax, jnp, seq)
            manifest["correctness"] = errors
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            for name in method_order(order):
                manifest["results"][name] = capture_one(jax, name, workloads[name], output_dir)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["status"] = "captured_pending_xprof_review"
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def controller(output_dir, max_seconds):
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    report = {"schema_version": 1, "status": "incomplete", "max_seconds": max_seconds,
              "planned": planned_runs(), "runs": []}
    report_path = output_dir / "controller.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    started = time.monotonic()
    try:
        for case in planned_runs():
            remaining = max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Profiler controller time ceiling reached")
            name = f"s{case['seq']}-{case['order']}"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--seq", str(case["seq"]), "--order", case["order"],
                       "--output-dir", str(output_dir / name)]
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "tpu"
            with (output_dir / f"{name}.stdout.log").open("w") as stdout, \
                 (output_dir / f"{name}.stderr.log").open("w") as stderr:
                try:
                    result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr,
                                            timeout=remaining, check=False)
                    returncode = result.returncode
                except subprocess.TimeoutExpired as error:
                    raise TimeoutError(f"{name} exceeded controller time ceiling") from error
            report["runs"].append({"name": name, "returncode": returncode})
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            if returncode != 0:
                raise RuntimeError(f"{name} failed; inspect its private logs and manifest")
        report["status"] = "captured_pending_xprof_review"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seq", type=int, choices=SEQUENCES, help=argparse.SUPPRESS)
    parser.add_argument("--order", choices=ORDERS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.dry_run:
        if args.worker:
            parser.error("--dry-run cannot be used with --worker")
        print(json.dumps({"planned": planned_runs(), "trace_batches": TRACE_BATCHES,
                          "target_batch_ms": TARGET_BATCH_MS,
                          "max_calls_per_batch": MAX_CALLS_PER_BATCH,
                          "max_seconds": args.max_seconds}, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for capture")
    if args.worker:
        if args.seq is None or args.order is None:
            parser.error("--worker requires --seq and --order")
        worker(args.seq, args.order, args.output_dir)
    else:
        controller(args.output_dir, args.max_seconds)


if __name__ == "__main__":
    main()
