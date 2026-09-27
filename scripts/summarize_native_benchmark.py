"""Validate and summarize three fresh-process native benchmark JSON reports.

This is offline analysis: it imports no JAX packages and never starts a TPU.
The output includes aggregate numbers only, not raw trials or local paths.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


METHODS = (
    "naive_forward",
    "flash_forward",
    "naive_forward_backward",
    "flash_forward_backward",
)
ERRORS = ("o", "dq", "dk", "dv")
EXPECTED_CASES = {(seq, causal) for seq in (256, 512, 1024, 2048, 4096)
                  for causal in (False, True)}


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def finite_number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def validate_report(report):
    if report.get("schema_version") != 1 or report.get("passed") is not True:
        raise ValueError("Expected a complete, passing schema-v1 report")
    if report.get("devices_used") != 1 or report.get("seed") != 0:
        raise ValueError("Unexpected device count or seed")
    if report.get("atol") != 0.05 or report.get("rtol") != 0.05:
        raise ValueError("Unexpected correctness tolerance")
    cases = {}
    for case in report["cases"]:
        key = (case["seq"], case["causal"])
        if key in cases:
            raise ValueError(f"Duplicate case {key}")
        if (case["batch"], case["heads"], case["head_dim"]) != (1, 1, 128):
            raise ValueError(f"Unexpected fixture for {key}")
        for name in ERRORS:
            error = case["errors"][name]
            if error["passed"] is not True or error["finite"] is not True:
                raise ValueError(f"Incorrect or nonfinite {name} in {key}")
            finite_number(error["max_abs"], f"{name} max_abs")
        for name in METHODS:
            timing = case["timing"][name]
            samples = timing["samples_ms"]
            if timing["warmups"] != 5 or timing["trials"] != 30 or len(samples) != 30:
                raise ValueError(f"Incomplete timing in {key}: {name}")
            for sample in samples:
                finite_number(sample, f"{key} {name} sample", positive=True)
            finite_number(timing["first_call_including_compile_ms"], "first call", positive=True)
            if not math.isclose(timing["p50_ms"], statistics.median(samples), abs_tol=1e-9):
                raise ValueError(f"Bad p50 in {key}: {name}")
            if not math.isclose(timing["p95_ms"], percentile(samples, 95), abs_tol=1e-9):
                raise ValueError(f"Bad p95 in {key}: {name}")
        cases[key] = case
    if set(cases) != EXPECTED_CASES:
        raise ValueError(f"Missing or extra cases: {sorted(EXPECTED_CASES ^ set(cases))}")
    return cases


def spread(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(reports, archive_sha256):
    if len(reports) != 3:
        raise ValueError("Exactly three fresh-process reports are required")
    case_sets = [validate_report(report) for report in reports]
    packages = reports[0]["packages"]
    device_kind = reports[0]["device_kind"]
    for report in reports[1:]:
        if report["packages"] != packages or report["device_kind"] != device_kind:
            raise ValueError("Environment differs between repetitions")
    if len(archive_sha256) != 64 or any(c not in "0123456789abcdef" for c in archive_sha256):
        raise ValueError("Archive SHA-256 must be 64 lowercase hex characters")
    summaries = []
    for seq, causal in sorted(EXPECTED_CASES):
        reps = [cases[(seq, causal)] for cases in case_sets]
        timings = {}
        for method in METHODS:
            timings[method] = {
                "p50_ms": spread([case["timing"][method]["p50_ms"] for case in reps]),
                "p95_ms": spread([case["timing"][method]["p95_ms"] for case in reps]),
                "first_call_including_compile_ms": spread([
                    case["timing"][method]["first_call_including_compile_ms"] for case in reps
                ]),
            }
        paired = {}
        for label, naive, flash in (
            ("forward", "naive_forward", "flash_forward"),
            ("forward_backward", "naive_forward_backward", "flash_forward_backward"),
        ):
            paired[label] = {
                "p50_ratio": spread([case["timing"][flash]["p50_ms"] /
                                     case["timing"][naive]["p50_ms"] for case in reps]),
                "p50_gap_ms": spread([case["timing"][flash]["p50_ms"] -
                                      case["timing"][naive]["p50_ms"] for case in reps]),
            }
        summaries.append({
            "seq": seq,
            "causal": causal,
            "timing": timings,
            "paired": paired,
            "max_abs_error": {name: max(case["errors"][name]["max_abs"] for case in reps)
                              for name in ERRORS},
        })
    return {
        "schema_version": 1,
        "archive_sha256": archive_sha256,
        "repetitions": 3,
        "trials_per_method_per_repetition": 30,
        "packages": packages,
        "device_kind": device_kind,
        "cases": summaries,
    }


def markdown(summary):
    lines = [
        "# Offline native benchmark analysis",
        "",
        "Source: three fresh-process BF16 B1/H1/D128 benchmark reports from the private "
        f"archive with SHA-256 `{summary['archive_sha256']}`. All cases passed the "
        "recorded `atol=rtol=0.05` output/gradient checks. Each method has five "
        "warmups and 30 synchronized trials per process on one v5e chip.",
        "",
        "The table gives the **median of three per-process p50s** in ms "
        "(min–max across processes), followed by paired Pallas/naive p50 ratios. "
        "Forward+backward means the complete loss/gradient call, not isolated backward.",
        "",
        "| S | Mask | Naive F | Pallas F | F ratio | Naive F+B | Pallas F+B | F+B ratio |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    def fmt(stat, digits=3):
        return f"{stat['median']:.{digits}f} ({stat['min']:.{digits}f}–{stat['max']:.{digits}f})"
    for case in summary["cases"]:
        t = case["timing"]
        p = case["paired"]
        lines.append(
            f"| {case['seq']} | {'causal' if case['causal'] else 'noncausal'} | "
            f"{fmt(t['naive_forward']['p50_ms'])} | {fmt(t['flash_forward']['p50_ms'])} | "
            f"{fmt(p['forward']['p50_ratio'], 3)}× | "
            f"{fmt(t['naive_forward_backward']['p50_ms'])} | "
            f"{fmt(t['flash_forward_backward']['p50_ms'])} | "
            f"{fmt(p['forward_backward']['p50_ratio'], 3)}× |"
        )
    lines += [
        "",
        "## Tail latency and first calls",
        "",
        "The following are medians of the three per-process p95s and first-call "
        "host wall times, respectively. The first call **may include compilation "
        "and cache effects**; it is not isolated compiler time.",
        "",
        "| S | Mask | Naive F p95 | Pallas F p95 | Naive F+B p95 | Pallas F+B p95 | "
        "Naive F first | Pallas F first | Naive F+B first | Pallas F+B first |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        t = case["timing"]
        values = [f"{t[m][metric]['median']:.3f}" for metric in
                  ("p95_ms", "first_call_including_compile_ms") for m in METHODS]
        lines.append(f"| {case['seq']} | {'causal' if case['causal'] else 'noncausal'} | " +
                     " | ".join(values) + " |")
    maxima = {name: max(case["max_abs_error"][name] for case in summary["cases"])
              for name in ERRORS}
    lines += [
        "",
        "## What this establishes",
        "",
        "- Maximum recorded absolute errors across cases/repetitions: " +
        ", ".join(f"{name}={maxima[name]:.6f}" for name in ERRORS) + ".",
        "- The S=4096 slowdown persists across all three repetitions for both "
        "masks and both complete call types. At S=256, differences are small "
        "relative to the host-dispatch timing floor.",
        "- These are sequential, fixed-order host wall timings: naive forward, "
        "Pallas forward, naive forward+backward, Pallas forward+backward. "
        "They are not device-only timings; order and cache effects were not controlled.",
        "- Do not subtract medians to infer backward-only time. No per-operation "
        "peak-HBM, FLOP utilization, or causal executed-work reduction was measured.",
        "- Next paid experiment, if warranted: profile matched S=1024 and S=4096 "
        "cases with synchronized device traces and counterbalanced method order "
        "before changing kernel logic.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs=3, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    result = summarize(reports, args.archive_sha256)
    for output in (args.json_out, args.markdown_out):
        if output.exists():
            parser.error(f"Refusing to overwrite existing {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_out.write_text(markdown(result))


if __name__ == "__main__":
    main()
