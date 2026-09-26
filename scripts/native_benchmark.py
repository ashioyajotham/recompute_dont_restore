"""Bounded, synchronized native-JAX validation and timing on one TPU.

Invoke once per fresh-process repetition. Compilation-inclusive first-call
latency is reported separately from warm timings; no backward-only subtraction
or allocator-based memory-saving claim is made.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "03_pallas_kernels"), str(ROOT / "02_naive_jax_baseline")]


def measure(jax, fn, args, warmups=5, trials=30):
    import numpy as np

    start = time.perf_counter()
    value = jax.block_until_ready(fn(*args))
    first_ms = (time.perf_counter() - start) * 1000
    for _ in range(warmups):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - start) * 1000)
    return value, {
        "first_call_including_compile_ms": first_ms,
        "warmups": warmups, "trials": trials, "samples_ms": samples,
        "p50_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing result")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flash_bwd import flash_attention
    from standard_attention import attention
    from utils import BlockSizes, pallas_interpret_mode

    devices = jax.devices("tpu")
    if not devices or pallas_interpret_mode():
        raise RuntimeError("Real TPU execution is required, without interpret mode")
    report = {
        "schema_version": 1,
        "packages": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu", "numpy")},
        "device_kind": devices[0].device_kind,
        "devices_available": len(devices), "devices_used": 1,
        "seed": args.seed, "atol": 0.05, "rtol": 0.05,
        "cases": [], "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with jax.default_device(devices[0]):
        for seq in (256, 512, 1024, 2048, 4096):
            for causal in (False, True):
                keys = jax.random.split(jax.random.PRNGKey(args.seed), 4)
                inputs = tuple(jax.random.normal(k, (1, 1, seq, 128), dtype=jnp.bfloat16) for k in keys)
                jax.block_until_ready(inputs)
                qkv, cotangent = inputs[:3], inputs[3]
                reference = jax.jit(lambda q, k, v: attention(q, k, v, causal=causal))
                kernel = jax.jit(lambda q, k, v: flash_attention(q, k, v, causal, None, BlockSizes(128, 128)))
                def weighted_loss(fn):
                    return jax.jit(jax.value_and_grad(
                        lambda q, k, v, do: jnp.sum(fn(q, k, v).astype(jnp.float32) * do.astype(jnp.float32)),
                        argnums=(0, 1, 2),
                    ))
                reference_fb, kernel_fb = weighted_loss(reference), weighted_loss(kernel)
                expected_o = jax.block_until_ready(reference(*qkv))
                actual_o = jax.block_until_ready(kernel(*qkv))
                expected_grads = jax.block_until_ready(reference_fb(*qkv, cotangent))[1]
                actual_grads = jax.block_until_ready(kernel_fb(*qkv, cotangent))[1]
                errors = {}
                for name, actual, expected in zip(("o", "dq", "dk", "dv"), (actual_o, *actual_grads), (expected_o, *expected_grads)):
                    a, e = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
                    errors[name] = {"max_abs": float(np.max(np.abs(a-e))), "finite": bool(np.isfinite(a).all() and np.isfinite(e).all())}
                    errors[name]["passed"] = errors[name]["finite"] and bool(np.allclose(a, e, atol=0.05, rtol=0.05))
                case = {"seq": seq, "batch": 1, "heads": 1, "head_dim": 128, "causal": causal, "errors": errors}
                report["cases"].append(case)
                if not all(e["passed"] for e in errors.values()):
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    raise AssertionError(f"Correctness failed at S={seq}, causal={causal}")
                # Clear compilation caches after correctness, so the separately
                # labeled first call is not merely the already-warm validation call.
                jax.clear_caches()
                case["timing"] = {}
                for name, fn, fn_args in (("naive_forward", reference, qkv), ("flash_forward", kernel, qkv), ("naive_forward_backward", reference_fb, inputs), ("flash_forward_backward", kernel_fb, inputs)):
                    _, case["timing"][name] = measure(jax, fn, fn_args)
                case["executed_forward_matmul_flops"] = 4 * seq * seq * 128
                case["useful_forward_matmul_flops"] = (4 * seq * (seq + 1) // 2 * 128) if causal else 4 * seq * seq * 128
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(f"S={seq} causal={causal}: correctness passed; timings saved", flush=True)
                jax.clear_caches()
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
