"""Exercise the real CPU measurement harness with a tiny fixture process."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any

SCRIPT = Path(__file__).resolve().parents[1] / 'research/evo-x3/measure.sh'


class CPUThreadProfileTests(unittest.TestCase):
    """Thread overrides reach the measured process and its saved profile."""

    def run_profile(
        self, threads: str | None, exit_code: int = 0,
        module: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], str]:
        """Run a copied harness, substituting only host launch dependencies."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / 'measure.sh'
            shutil.copyfile(SCRIPT, script)
            tools = root / 'bin'
            tools.mkdir()
            fixtures = {
                'systemd-run': '#!/bin/sh\n'
                'while [ "$1" != taskset ]; do shift; done\nexec "$@"\n',
                'taskset': '#!/bin/sh\nshift 2\nexec "$@"\n',
                'uv': f'#!/bin/sh\nexec "{sys.executable}" '
                '"$CMM_TEST_APP" "$@"\n',
            }
            for name, content in fixtures.items():
                path = tools / name
                path.write_text(content)
                path.chmod(0o755)
            capture = root / 'captured.json'
            app = root / 'app.py'
            app.write_text(
                'import json, os, pathlib, sys\n'
                'pathlib.Path(os.environ["CMM_TEST_CAPTURE"]).write_text('
                'json.dumps({"threads": os.environ["OMP_NUM_THREADS"], '
                '"argv": sys.argv[1:]}))\n'
                f'sys.exit({exit_code})\n',
            )
            env = os.environ.copy()
            env.pop('CMM_OMP_THREADS', None)
            env.pop('CMM_MODULE', None)
            if module is not None:
                env['CMM_MODULE'] = module
            if threads is not None:
                env['CMM_OMP_THREADS'] = threads
            # A target above any real reading skips the cooling wait.
            env['CMM_COOL_TARGET_C'] = '200'
            env.update(PATH=str(tools) + os.pathsep + env['PATH'],
                       CMM_LAB=str(root / 'lab'), CMM_TEST_APP=str(app),
                       CMM_TEST_CAPTURE=str(capture))
            result = subprocess.run(['bash', str(script), str(root), 'trial'],
                                    env=env, capture_output=True, text=True,
                                    timeout=5)
            measured = (json.loads(capture.read_text())
                        if capture.exists() else {})
            path = root / 'measurements/trial.txt'
            report = path.read_text() if path.exists() else ''
            return result, measured, report

    def test_default_profile_remains_eight_threads(self) -> None:
        """Existing calls retain the calibrated phys16-t8 contract."""
        result, process, report = self.run_profile(None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(process['threads'], '8')
        self.assertIn('profile=phys16-t8', report)
        self.assertIn('omp=8', report)

    def test_six_threads_reach_process_and_measurement(self) -> None:
        """An explicit six-thread trial must not secretly run at eight."""
        result, process, report = self.run_profile('6')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(process['threads'], '6')
        self.assertIn('profile=phys16-t6', report)
        self.assertIn('omp=6', report)
        self.assertNotIn('phys16-t8', report)

    def test_bad_thread_counts_stop_before_application(self) -> None:
        """Reject invalid or oversubscribed values for the fixed16-CPU set."""
        for value in ['0', '17', '-1', 'six', '06', '6; echo bad']:
            with self.subTest(value=value):
                result, process, report = self.run_profile(value)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(process)
                self.assertFalse(report)

    def test_failed_application_preserves_nonzero_exit_code(self) -> None:
        """A failed generation cannot be reported as a successful timing."""
        result, _, report = self.run_profile('6', 7)
        self.assertEqual(result.returncode, 7)
        self.assertIn('exit_code=7', report)

    def test_module_is_limited_to_native_or_prompt_trial(self) -> None:
        """A controlled entrypoint cannot become an arbitrary Python module."""
        result, process, _ = self.run_profile('6', module='tools.prompt_trial')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(process['threads'], '6')
        self.assertEqual(process['argv'],
                         ['run', 'python', '-m', 'tools.prompt_trial'])
        for module in ['os', 'tools.bad', 'src; echo bad']:
            with self.subTest(module=module):
                result, process, report = self.run_profile('6', module=module)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(process)
                self.assertFalse(report)
