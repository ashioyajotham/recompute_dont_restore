"""Offline checks for the matched-stack LLO controller; no TPU is used."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_tpu_llo.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("profile_tpu_llo", SCRIPT)
llo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(llo)


STACK = {"jax": "0.11.0", "jaxlib": "0.11.0", "libtpu": "0.0.46",
         "numpy": "2.5.3", "xprof-nightly": "2.24.0.dev0"}


class TestLloProfile(unittest.TestCase):
    def test_small_schedule_and_separate_runtime_capture(self):
        debug = llo.planned_runs("debug")
        self.assertEqual(len(debug), 2)
        self.assertEqual([run["order"] for run in debug],
                         ["naive-first", "pallas-first"])
        self.assertEqual(set(debug[0]["methods"]), set(debug[1]["methods"]))
        runtime = llo.planned_runs("runtime")
        self.assertEqual(len(runtime), 1)
        self.assertEqual(runtime[0]["methods"],
                         ["pallas_forward", "pallas_forward_backward"])

    def test_dry_run_does_not_import_jax_or_write(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "not-created"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--dry-run", "--output-dir", str(target)],
                check=True, capture_output=True, text=True)
            self.assertEqual(len(json.loads(result.stdout)["planned"]), 2)
            self.assertFalse(target.exists())

    def test_preflight_checks_lock_and_flags_before_jax(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "stack.json"
            lock.write_text(json.dumps(STACK))
            with mock.patch.object(llo, "installed_stack", return_value=STACK), \
                 mock.patch.dict(llo.os.environ, {
                     "JAX_PLATFORMS": "tpu", "LIBTPU_INIT_ARGS": llo.DEBUG_FLAG}, clear=True):
                self.assertEqual(llo.validate_environment(lock, "debug")["packages"], STACK)
                with self.assertRaisesRegex(RuntimeError, "Runtime mode requires"):
                    llo.validate_environment(lock, "runtime")
            with mock.patch.object(llo, "installed_stack", return_value=STACK), \
                 mock.patch.dict(llo.os.environ, {
                     "JAX_PLATFORMS": "tpu", "LIBTPU_INIT_ARGS":
                     f"{llo.DEBUG_FLAG} {llo.RUNTIME_FLAG}"}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "Do not enable"):
                    llo.validate_environment(lock, "debug")
                self.assertEqual(llo.validate_environment(lock, "runtime")["llo_mode"],
                                 "runtime")

    def test_mismatch_and_old_jax_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "stack.json"
            lock.write_text(json.dumps(STACK))
            env = {"JAX_PLATFORMS": "tpu", "LIBTPU_INIT_ARGS": llo.DEBUG_FLAG}
            with mock.patch.dict(llo.os.environ, env, clear=True):
                with mock.patch.object(llo, "installed_stack", return_value={**STACK, "jax": "0.9.2"}):
                    with self.assertRaisesRegex(RuntimeError, "differs from lock"):
                        llo.validate_environment(lock, "debug")
                lock.write_text(json.dumps({**STACK, "jax": "0.9.2", "jaxlib": "0.9.2"}))
                with mock.patch.object(llo, "installed_stack", return_value={**STACK, "jax": "0.9.2", "jaxlib": "0.9.2"}):
                    with self.assertRaisesRegex(RuntimeError, "JAX >= 0.11"):
                        llo.validate_environment(lock, "debug")

    def test_controller_stops_on_failed_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "stack.json"
            lock.write_text(json.dumps(STACK))
            target = Path(temp) / "new"
            with mock.patch.object(llo, "validate_environment", return_value={"packages": STACK}), \
                 mock.patch.object(llo.subprocess, "run", return_value=mock.Mock(returncode=1)) as run:
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    llo.controller(target, lock, "debug", 10)
            self.assertEqual(run.call_count, 1)
            report = json.loads((target / "controller.json").read_text())
            self.assertEqual(report["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
