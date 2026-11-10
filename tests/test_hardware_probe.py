"""Verify the hardware probe without model loads or downloads."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from tools import hardware_probe as probe


def report(label: str, elapsed_ns: int,
           array_mib: int = 1) -> dict[str, object]:
    """Create a hand-checked baseline with a one-second operation."""
    timing = {'iterations': 2, 'samples_ns': [elapsed_ns] * 3,
              'payload_bytes': 2_000_000_000, 'flops': 0}
    return {
        'protocol': 1, 'label': label, 'array_mib': array_mib,
        'repeats': 3, 'sample_ms': 1, 'status': 'complete', 'errors': [],
        'host': {'cpu': 'test CPU'},
        'profiles': {'default': {
            'memory_threads': 1, 'affinity': [0], 'native_threads': 1,
            'numpy_version': '2.5', 'blas': {'name': 'test', 'version': '1'},
            'cases': {name: timing for name in probe.CASE_NAMES},
        }},
    }


class ProbeContractTests(unittest.TestCase):
    """Protect units, configuration compatibility and resource limits."""

    def test_batch_timing_and_bandwidth_units(self) -> None:
        """Two operations in two seconds mean one second per operation."""
        timing = probe.Timing(iterations=2, samples_ns=[2_000_000_000] * 3,
                              payload_bytes=2_000_000_000, flops=0)
        self.assertEqual(timing.seconds, 1)
        self.assertEqual(timing.gb_per_second, 2)

    def test_preemption_cannot_create_a_zero_iteration_batch(self) -> None:
        """A clock jump before the first iteration still executes one batch."""
        calls: list[None] = []

        def operation() -> None:
            """Record actual measured work without sleeping."""
            calls.append(None)

        with patch.object(probe.time, 'perf_counter_ns',
                          side_effect=[0, 2_000_000, 3_000_000, 4_000_000]):
            timing = probe.measure_case(operation, 1, 1)
        self.assertEqual(timing.iterations, 1)
        self.assertEqual(timing.samples_ns, [1_000_000])
        self.assertEqual(len(calls), 3)

    def test_compare_keeps_each_workload_ratio(self) -> None:
        """Twice as fast is ratio 0.5, without claiming equivalence."""
        reference = probe.Report.model_validate(
            report('campus', 4_000_000_000),
        )
        candidate = probe.Report.model_validate(report('dev', 2_000_000_000))
        result = probe.compare(reference, candidate, 'default')
        self.assertEqual(result['profiles']['default']['ratios'],
                         dict.fromkeys(probe.CASE_NAMES, 0.5))
        self.assertEqual(result['profiles']['default']['worst_deviation'], .5)
        self.assertNotIn('reproduced', result)

    def test_incompatible_or_partial_reports(self) -> None:
        """Do not compare unlike buffers or incomplete measurements."""
        reference = probe.Report.model_validate(
            report('campus', 4_000_000_000),
        )
        different = probe.Report.model_validate(
            report('dev', 2_000_000_000, 2),
        )
        with self.assertRaisesRegex(ValueError, 'workload'):
            probe.compare(reference, different, 'default')
        partial = probe.Report.model_validate({
            **report('dev', 2_000_000_000), 'status': 'partial',
        })
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            probe.compare(reference, partial, 'default')

    def test_invalid_timings_and_unknown_protocol_are_rejected(self) -> None:
        """Strict report types do not accept strings, bools or zero time."""
        for samples in [[0], [-1], ['2'], [True]]:
            with self.assertRaises(ValidationError):
                probe.Timing.model_validate({
                    'iterations': 1, 'samples_ns': samples,
                    'payload_bytes': 1, 'flops': 0,
                })
        for protocol in [2, True, '1']:
            with self.assertRaises(ValidationError):
                probe.Report.model_validate({**report('x', 2),
                                             'protocol': protocol})

    def test_complete_reports_require_all_cases_and_samples(self) -> None:
        """Missing data must not become a successful comparison."""
        for profiles in [{}, {'default': {
            'memory_threads': 1, 'affinity': [0], 'native_threads': 1,
            'numpy_version': '2.5', 'blas': {}, 'cases': {},
        }}]:
            with self.assertRaises(ValidationError):
                probe.Report.model_validate({**report('x', 2),
                                             'profiles': profiles})

    def test_cache_metadata_counts_shared_l3_once(self) -> None:
        """Count both cache domains, without counting SMT views twice."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for cpu in (0, 1):
                cache = root / f'cpu{cpu}/cache/index3'
                cache.mkdir(parents=True)
                for name, value in [('level', '3'), ('size', '32768K'),
                                    ('shared_cpu_list', '0-1')]:
                    (cache / name).write_text(value)

            def path(value: str) -> Path:
                """Use fake cache files and real process facts."""
                prefix = '/sys/devices/system/cpu/'
                if str(value).startswith(prefix):
                    return root / str(value)[len(prefix):]
                return Path(value)

            with patch.object(probe, 'Path', side_effect=path), \
                    patch.object(probe.os, 'sched_getaffinity',
                                 return_value={0, 1}):
                self.assertEqual(probe.host_info()['l3_bytes_in_affinity'],
                                 32 * 1024**2)
                shared = root / 'cpu1/cache/index3/shared_cpu_list'
                shared.write_text('1')
                self.assertEqual(probe.host_info()['l3_bytes_in_affinity'],
                                 64 * 1024**2)

    def test_memory_guard_fails_before_allocating(self) -> None:
        """Keep three buffer footprints within one quarter of available RAM."""
        probe.check_memory(1, 512 * 1024**2)
        with self.assertRaisesRegex(ValueError, 'memory'):
            probe.check_memory(128, 64 * 1024**2)

    def test_memory_guard_uses_remaining_cgroup_capacity(self) -> None:
        """An almost-full cgroup matters even when host RAM is plentiful."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'memory.max').write_text(str(64 * 1024**2))
            (directory / 'memory.current').write_text(str(48 * 1024**2))
            with patch.object(probe, 'cgroup_dirs', return_value=[directory]):
                self.assertLessEqual(probe.available_memory(), 16 * 1024**2)

    def test_thread_environment_is_set_without_mutating_parent(self) -> None:
        """The worker gets consistent BLAS settings before NumPy import."""
        with patch.dict(os.environ, {'OMP_NUM_THREADS': '99'}):
            env = probe.worker_environment('4')
            for key in probe.THREAD_ENVS:
                self.assertEqual(env[key], '4')
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '99')
            native = probe.worker_environment('default')
            for key in probe.THREAD_ENVS:
                self.assertNotIn(key, native)

    def test_report_reader_rejects_duplicate_keys(self) -> None:
        """A baseline cannot silently replace protocol or measurements."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'report.json'
            path.write_text('{"protocol":1,"protocol":2}')
            with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                probe.read_report(path)

    def test_timeout_produces_partial_report(self) -> None:
        """A native worker timeout is explicit and never a complete result."""
        args = probe.parser().parse_args([
            'measure', '--label', 'test', '--threads', '1',
            '--array-mib', '1', '--sample-ms', '1', '--repeats', '1',
        ])
        with patch.object(probe.subprocess, 'run', side_effect=(
            subprocess.TimeoutExpired(['worker'], 1)
        )), contextlib.redirect_stderr(io.StringIO()):
            result = probe.collect(args)
        self.assertEqual(result.status, 'partial')
        self.assertTrue(result.errors)
        self.assertFalse(result.profiles)


class ProbeCLITests(unittest.TestCase):
    """Run the actual NumPy worker with small bounded buffers."""

    def test_real_timeout_has_partial_data_without_local_paths(self) -> None:
        """The parent kills a slow worker and does not leak its argv path."""
        result = subprocess.run([
            sys.executable, '-m', 'tools.hardware_probe', 'measure',
            '--threads', '1', '--array-mib', '1', '--sample-ms', '1000',
            '--repeats', '10', '--budget', '1',
        ], text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 1, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['status'], 'partial')
        self.assertTrue(data['errors'])
        self.assertNotIn(str(Path.cwd()), result.stdout)
        self.assertNotIn(str(Path.home()), result.stdout)

    def test_measure_then_compare_without_model_cache(self) -> None:
        """Produce a report, reuse it, and never create a model directory."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = os.environ.copy()
            environment.update(HF_HOME=str(root / 'no-model'),
                               HF_HUB_OFFLINE='1')
            measured = subprocess.run([
                sys.executable, '-m', 'tools.hardware_probe', 'measure',
                '--label', 'test', '--threads', '1', '--array-mib', '1',
                '--sample-ms', '1', '--repeats', '2', '--budget', '20',
            ], env=environment, text=True, capture_output=True, timeout=25)
            self.assertEqual(measured.returncode, 0, measured.stderr)
            path = root / 'report.json'
            path.write_text(measured.stdout)
            data = probe.read_report(path)
            self.assertEqual(data.status, 'complete')
            self.assertEqual(set(data.profiles['1'].cases),
                             set(probe.CASE_NAMES))
            for timing in data.profiles['1'].cases.values():
                self.assertEqual(len(timing.samples_ns), 2)
                self.assertTrue(all(x > 0 for x in timing.samples_ns))
            compared = subprocess.run([
                sys.executable, '-m', 'tools.hardware_probe', 'compare',
                str(path), str(path), '--reference-profile', '1',
            ], text=True, capture_output=True, timeout=5)
            self.assertEqual(compared.returncode, 0, compared.stderr)
            ratios = json.loads(compared.stdout)['profiles']['1']['ratios']
            self.assertTrue(all(value == 1 for value in ratios.values()))
            self.assertFalse((root / 'no-model').exists())
