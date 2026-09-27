"""CPU-only checks of the archived-result aggregator."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_native_benchmark.py"
SPEC = importlib.util.spec_from_file_location("summarize_native_benchmark", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def fake_report(multiplier=1):
    cases = []
    for seq, causal in sorted(summary.EXPECTED_CASES):
        timings = {}
        for name in summary.METHODS:
            base = 0.1 + seq / 100000
            if name.startswith("flash"):
                base *= 1.5
            if name.endswith("backward"):
                base *= 2
            samples = [base * multiplier + i / 100000 for i in range(30)]
            timings[name] = {
                "warmups": 5,
                "trials": 30,
                "samples_ms": samples,
                "p50_ms": summary.percentile(samples, 50),
                "p95_ms": summary.percentile(samples, 95),
                "first_call_including_compile_ms": base * 100,
            }
        cases.append({
            "seq": seq, "causal": causal, "batch": 1, "heads": 1, "head_dim": 128,
            "errors": {name: {"passed": True, "finite": True, "max_abs": 0.01}
                       for name in summary.ERRORS},
            "timing": timings,
        })
    return {
        "schema_version": 1, "passed": True, "devices_used": 1, "seed": 0,
        "atol": 0.05, "rtol": 0.05, "packages": {"jax": "test"},
        "device_kind": "test-tpu", "cases": cases,
    }


class TestSummary(unittest.TestCase):
    def test_three_repetitions_and_median_of_ratios(self):
        result = summary.summarize(
            [fake_report(1), fake_report(1.1), fake_report(0.9)], "a" * 64
        )
        self.assertEqual(len(result["cases"]), 10)
        case = result["cases"][0]
        self.assertGreater(case["paired"]["forward"]["p50_ratio"]["median"], 1)
        rendered = summary.markdown(result)
        self.assertIn("not isolated backward", rendered)
        self.assertNotIn("/private/tmp", rendered)

    def test_reject_bad_percentile(self):
        report = fake_report()
        report["cases"][0]["timing"]["naive_forward"]["p95_ms"] += 0.01
        with self.assertRaisesRegex(ValueError, "Bad p95"):
            summary.validate_report(report)

    def test_reject_missing_case(self):
        report = fake_report()
        report["cases"].pop()
        with self.assertRaisesRegex(ValueError, "Missing or extra"):
            summary.validate_report(report)

    def test_reject_changed_environment(self):
        reports = [fake_report() for _ in range(3)]
        reports[2]["packages"] = {"jax": "different"}
        with self.assertRaisesRegex(ValueError, "Environment differs"):
            summary.summarize(reports, "a" * 64)


if __name__ == "__main__":
    unittest.main()
