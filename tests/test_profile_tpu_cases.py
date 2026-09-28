"""Standard-library-only checks; these tests never touch a TPU."""

import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_tpu_cases.py"
SPEC = importlib.util.spec_from_file_location("profile_tpu_cases", SCRIPT)
profiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiler)


class TestProfilePlan(unittest.TestCase):
    def test_counterbalanced_four_process_schedule(self):
        plan = profiler.planned_runs()
        self.assertEqual([(case["seq"], case["order"]) for case in plan],
                         [(1024, "naive-first"), (1024, "pallas-first"),
                          (4096, "naive-first"), (4096, "pallas-first")])
        for first, second in ((plan[0], plan[1]), (plan[2], plan[3])):
            self.assertEqual(set(first["methods"]), set(second["methods"]))
            self.assertNotEqual(first["methods"], second["methods"])

    def test_batch_sizing_and_invalid_baseline(self):
        self.assertEqual(profiler.calls_per_batch(0.2), 500)
        self.assertEqual(profiler.calls_per_batch(1.0), 100)
        self.assertEqual(profiler.calls_per_batch(1000.0), 5)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                profiler.calls_per_batch(value)

    def test_dry_run_does_not_import_jax_or_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "uncreated"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--dry-run", "--output-dir", str(target)],
                check=True, capture_output=True, text=True)
            self.assertEqual(len(json.loads(completed.stdout)["planned"]), 4)
            self.assertNotIn("jax", sys.modules)
            self.assertFalse(target.exists())

    def test_existing_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                profiler.controller(Path(temp), 10)

    def test_controller_stops_after_first_failed_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "new"
            with mock.patch.object(profiler.subprocess, "run", return_value=mock.Mock(returncode=1)) as run:
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    profiler.controller(target, 10)
            self.assertEqual(run.call_count, 1)
            report = json.loads((target / "controller.json").read_text())
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(len(report["runs"]), 1)

    def test_controller_schedules_all_four_without_jax_in_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "new"
            with mock.patch.object(profiler.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
                profiler.controller(target, 10)
            self.assertEqual(run.call_count, 4)
            self.assertTrue(all(call.kwargs["env"]["JAX_PLATFORMS"] == "tpu"
                                for call in run.call_args_list))
            report = json.loads((target / "controller.json").read_text())
            self.assertEqual(report["status"], "captured_pending_xprof_review")
            self.assertEqual(len(report["runs"]), 4)


class TestTraceEvidence(unittest.TestCase):
    @staticmethod
    def write_trace(target, events):
        path = target / "perfetto_trace.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump({"traceEvents": events}, stream)

    def test_tpu_lane_and_activity_required(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            self.write_trace(target, [
                {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "TPU device"}},
                {"ph": "X", "pid": 1, "name": "compute", "dur": 100},
            ])
            result = profiler.trace_evidence(target)
            self.assertEqual(result["device_events_detected"], 1)
            self.assertEqual(len(result["perfetto_sha256"]), 64)

    def test_host_only_trace_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            self.write_trace(target, [
                {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "host"}},
                {"ph": "X", "pid": 1, "name": "python", "dur": 100},
            ])
            with self.assertRaisesRegex(RuntimeError, "No identifiable TPU"):
                profiler.trace_evidence(target)

    def test_unsupported_profiler_options_fail_without_success_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            fake_profiler = mock.Mock()
            fake_profiler.ProfileOptions.return_value = mock.Mock()
            fake_profiler.start_trace.side_effect = RuntimeError("unsupported trace mode")
            fake_jax = mock.Mock(profiler=fake_profiler)
            fake_jax.block_until_ready.side_effect = lambda value: value
            with self.assertRaisesRegex(RuntimeError, "unsupported trace mode"):
                profiler.capture_one(fake_jax, "naive_forward", (lambda: 1, ()), Path(temp))
            fake_profiler.stop_trace.assert_not_called()


if __name__ == "__main__":
    unittest.main()
