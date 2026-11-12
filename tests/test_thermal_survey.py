"""Test heating surveys using fake sensors and real child processes."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from tools import campus_survey
from tools import thermal_survey as thermal


def args(**changes: object) -> argparse.Namespace:
    """Use a brief diagnostic with no real model or heat."""
    values = dict(lengths=[16, 64], duration=.08, budget=2, threads=None,
                  max_temp_c=None, min_free_mib=1024, hf_home=None)
    values.update(changes)
    return argparse.Namespace(**values)


class ThermalTests(unittest.TestCase):
    """Safety checks precede work and remain active during native work."""

    def setUp(self) -> None:
        """Create stable, readable CPU sensors and counters."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        hwmon = self.root / 'hwmon/hwmon0'
        hwmon.mkdir(parents=True)
        (hwmon / 'name').write_text('coretemp')
        (hwmon / 'temp1_input').write_text('50000')
        (hwmon / 'temp1_crit').write_text('100000')
        self.temp = hwmon / 'temp1_input'
        counter = self.root / 'cpu/cpu0/thermal_throttle'
        counter.mkdir(parents=True)
        self.counter = counter / 'core_throttle_count'
        self.counter.write_text('12')
        self.meminfo = self.root / 'meminfo'
        self.meminfo.write_text('MemAvailable: 8388608 kB\n')
        self.calls = self.root / 'calls'
        self.sdk = types.ModuleType('llm_sdk')
        calls = self.calls

        def model(*pos: object, **kw: object) -> object:
            with calls.open('a') as stream:
                stream.write(str(kw.get('device')) + '\n')

            def logits(ids: list[int]) -> list[float]:
                with calls.open('a') as stream:
                    stream.write(str(len(ids)) + '\n')
                time.sleep(.003)
                return [0., 1.]

            return types.SimpleNamespace(
                encode=lambda text: types.SimpleNamespace(
                    tolist=lambda: [list(range(512))]),
                get_logits_from_input_ids=logits)

        setattr(self.sdk, 'Small_LLM_Model', model)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in [('HWMON_ROOT', self.root / 'hwmon'),
                            ('CPU_ROOT', self.root / 'cpu'),
                            ('MEMINFO', self.meminfo), ('POLL_S', .015)]:
            self.stack.enter_context(patch.object(thermal, name, value))
        self.stack.enter_context(patch.object(
            campus_survey, 'cgroup_dirs', return_value=[]))
        self.stack.enter_context(patch.dict(
            sys.modules, {'llm_sdk': self.sdk}))
        self.stack.enter_context(patch.object(
            campus_survey, 'find_model',
            return_value=('explicit', self.root, self.root)))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def run_measure(self, **changes: object) -> tuple[dict, int, list[dict]]:
        """Run the production supervisor and collect its events."""
        events: list[dict] = []
        summary, code = thermal.measure(args(**changes), events.append)
        return summary, code, events

    def test_repeats_fixed_lengths_while_sampling(self) -> None:
        """Repeated identical work remains grouped by length, on CPU."""
        summary, code, events = self.run_measure()
        self.assertEqual(code, 0)
        self.assertEqual(summary['reason'], 'duration_completed')
        forwards = [e for e in events if e['event'] == 'forward']
        self.assertGreater(len(forwards), 4)
        self.assertEqual([e['input_tokens'] for e in forwards[:4]],
                         [16, 64, 16, 64])
        self.assertTrue(all(e['elapsed_ns'] > 0 for e in forwards))
        self.assertTrue(any(e['event'] == 'sample' for e in events))
        self.assertEqual(self.calls.read_text().splitlines()[0], 'cpu')

    def test_initial_unsafe_or_missing_guards_never_load_model(self) -> None:
        """No work starts without trustworthy temperature and counters."""
        for value, reason in [('100000', 'temperature_limit'),
                              ('bad', 'monitor_unavailable')]:
            self.temp.write_text(value)
            summary, code, _ = self.run_measure()
            self.assertEqual(code, 3)
            self.assertEqual(summary['reason'], reason)
            self.assertFalse(self.calls.exists())
        self.temp.write_text('50000')
        self.counter.unlink()
        summary, code, _ = self.run_measure()
        self.assertEqual((code, summary['reason']),
                         (3, 'monitor_unavailable'))
        self.assertFalse(self.calls.exists())

    def test_sensor_critical_limit_overrides_a_high_user_setting(self) -> None:
        """A user setting cannot raise the CPU's actual critical limit."""
        (self.temp.parent / 'temp1_crit').write_text('90000')
        self.temp.write_text('90000')
        summary, code, _ = self.run_measure(max_temp_c=95)
        self.assertEqual((code, summary['reason']),
                         (3, 'temperature_limit'))

    def test_throttling_sensor_loss_and_memory_stop_running_work(self) -> None:
        """Safety changes after work starts stop the real child."""
        actions = [
            (lambda: self.counter.write_text('13'), 'thermal_throttling'),
            (lambda: self.counter.write_text('11'), 'counter_reset'),
            (lambda: self.temp.unlink(), 'monitor_unavailable'),
            (lambda: self.meminfo.write_text('MemAvailable: 1024 kB\n'),
             'low_memory'),
        ]
        for action, reason in actions:
            self.counter.write_text('12')
            self.temp.write_text('50000')
            self.meminfo.write_text('MemAvailable: 8388608 kB\n')
            changed = False
            events: list[dict] = []

            def emit(event: dict) -> None:
                nonlocal changed
                events.append(event)
                if event['event'] == 'forward' and not changed:
                    changed = True
                    action()

            summary, code = thermal.measure(args(duration=.3), emit)
            self.assertEqual((code, summary['reason']), (4, reason))
            self.assertTrue(changed)
            self.assertTrue(any(e['event'] == 'forward' for e in events))

    def test_flooded_pipe_does_not_starve_throttle_checks(self) -> None:
        """Frequent results still permit periodic safety sampling."""
        changed = False

        def emit(event: dict) -> None:
            nonlocal changed
            if event['event'] == 'forward' and not changed:
                changed = True
                self.counter.write_text('13')

        summary, code = thermal.measure(args(duration=1), emit)
        self.assertEqual((code, summary['reason']),
                         (4, 'thermal_throttling'))

    def test_below_critical_temperature_continues_by_default(self) -> None:
        """No counter increase allows 99 C work below the CPU's 100 C crit."""
        self.temp.write_text('99000')
        summary, code, _ = self.run_measure()
        self.assertEqual((code, summary['reason']),
                         (0, 'duration_completed'))

    def test_throttle_reason_takes_priority_at_critical_limit(self) -> None:
        """The first counter increase remains the reported stop reason."""
        def emit(event: dict) -> None:
            if event['event'] == 'forward':
                self.temp.write_text('100000')
                self.counter.write_text('13')

        summary, code = thermal.measure(args(duration=.3), emit)
        self.assertEqual((code, summary['reason']),
                         (4, 'thermal_throttling'))

    def test_missing_critical_limit_refuses_before_model_load(self) -> None:
        """A missing emergency threshold cannot use a fake limit."""
        (self.temp.parent / 'temp1_crit').unlink()
        summary, code, _ = self.run_measure()
        self.assertEqual((code, summary['reason']),
                         (3, 'monitor_unavailable'))
        self.assertFalse(self.calls.exists())

    def test_explicit_lower_temperature_limit_is_still_enforced(self) -> None:
        """An optional lower bound can stop work but is never the default."""
        self.temp.write_text('85000')
        summary, code, _ = self.run_measure(max_temp_c=85)
        self.assertEqual((code, summary['reason']), (3, 'temperature_limit'))

    def test_blocked_child_ignoring_term_is_killed_within_budget(self) -> None:
        """A native-style hang cannot bypass the parent's deadline."""
        def stuck(*pos: object, **kw: object) -> None:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(30)

        setattr(self.sdk, 'Small_LLM_Model', stuck)
        start = time.monotonic()
        with patch.object(thermal, 'KILL_GRACE_S', .04):
            summary, code, _ = self.run_measure(budget=.08)
        self.assertEqual((code, summary['reason']), (4, 'budget_exceeded'))
        self.assertLess(time.monotonic() - start, 2)

    def test_model_failure_has_no_raw_exception_in_report(self) -> None:
        """Only a fixed failure reason crosses the process boundary."""
        def fail(*pos: object, **kw: object) -> None:
            raise OSError('/home/secret-host/private')

        setattr(self.sdk, 'Small_LLM_Model', fail)
        summary, code, events = self.run_measure()
        self.assertEqual((code, summary['reason']), (1, 'worker_failed'))
        self.assertNotIn('secret-host', json.dumps([summary, events]))

    def test_cgroup_memory_remaining_is_part_of_guard(self) -> None:
        """A nearly full cgroup stops work despite free host memory."""
        group = self.root / 'cgroup'
        group.mkdir()
        (group / 'memory.max').write_text(str(4 * 2**30))
        (group / 'memory.current').write_text(str(4 * 2**30 - 2**29))
        with patch.object(campus_survey, 'cgroup_dirs', return_value=[group]):
            summary, code, _ = self.run_measure()
        self.assertEqual((code, summary['reason']), (3, 'low_memory'))

    def test_quota_throttling_is_recorded_separately_from_heat(self) -> None:
        """An intentional CPU quota is context, not a thermal event."""
        group = self.root / 'quota'
        group.mkdir()
        (group / 'memory.max').write_text('max')
        (group / 'cpu.stat').write_text(
            'nr_throttled 7\nthrottled_usec 1200\n')
        with patch.object(campus_survey, 'cgroup_dirs', return_value=[group]):
            summary, code, events = self.run_measure()
        self.assertEqual(code, 0)
        first = next(e for e in events if e['event'] == 'sample')
        self.assertEqual(first['cgroup_cpu']['0']['nr_throttled'], 7)

    def test_report_write_failure_stops_child(self) -> None:
        """A broken recorder cannot leave unmonitored model work running."""
        def emit(event: dict) -> None:
            if event['event'] == 'forward':
                raise OSError('disk full')

        summary, code = thermal.measure(args(duration=10), emit)
        self.assertEqual((code, summary['reason']), (1, 'measurement_failed'))
        count = len(self.calls.read_text().splitlines())
        time.sleep(.04)
        self.assertEqual(len(self.calls.read_text().splitlines()), count)

    def test_sigterm_to_parent_preserves_partial_results(self) -> None:
        """The parent handler stops work and returns the signal status."""
        import multiprocessing

        ready, result = self.root / 'ready', self.root / 'result'

        def run() -> None:
            def emit(event: dict) -> None:
                if event['event'] == 'forward':
                    ready.touch()
            summary, code = thermal.measure(args(duration=10, budget=15), emit)
            result.write_text(json.dumps({'code': code, **summary}))

        parent = multiprocessing.get_context('fork').Process(target=run)
        parent.start()
        try:
            deadline = time.monotonic() + 3
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists())
            assert parent.pid is not None
            os.kill(parent.pid, signal.SIGTERM)
            parent.join(2)
            self.assertFalse(parent.is_alive())
            saved = json.loads(result.read_text())
            self.assertEqual((saved['code'], saved['reason']),
                             (143, 'interrupted'))
        finally:
            if parent.is_alive():
                parent.kill()
            parent.join()

    def test_descendant_dies_even_if_worker_has_already_exited(self) -> None:
        """Termination reaches the process group after its leader is gone."""
        original = self.sdk.Small_LLM_Model
        pid_file = self.root / 'descendant'

        def spawn(*pos: object, **kw: object) -> object:
            pid = os.fork()
            if pid == 0:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                time.sleep(30)
                os._exit(0)
            pid_file.write_text(str(pid))
            return original(*pos, **kw)

        setattr(self.sdk, 'Small_LLM_Model', spawn)
        summary, code, _ = self.run_measure()
        self.assertEqual(code, 0)
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 1
        state = ''
        while time.monotonic() < deadline:
            try:
                state = Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]
            except FileNotFoundError:
                state = 'gone'
            if state in ('Z', 'gone'):
                break
            time.sleep(.01)
        self.assertIn(state, ('Z', 'gone'))

    def test_last_sample_failure_is_a_safety_stop(self) -> None:
        """A sensor lost just before completion cannot produce success."""
        real_snapshot = thermal.snapshot
        calls = 0

        def reading(sensors: list[thermal.Sensor], counters: list[Path],
                    pid: int | None = None) -> dict:
            nonlocal calls
            calls += 1
            if calls >= 3:
                raise OSError('sensor disappeared')
            return real_snapshot(sensors, counters, pid)

        with patch.object(thermal, 'POLL_S', 5), \
                patch.object(thermal, 'snapshot', side_effect=reading):
            summary, code, _ = self.run_measure()
        self.assertEqual((code, summary['reason']),
                         (4, 'monitor_unavailable'))

    def test_cpu_topology_preserves_smt_and_unknown_core_type(self) -> None:
        """VM configuration needs sibling mappings, not only a CPU count."""
        folder = self.root / 'cpu/cpu0/topology'
        folder.mkdir()
        for name, value in [('core_id', '0'), ('physical_package_id', '0'),
                            ('thread_siblings_list', '0,16')]:
            (folder / name).write_text(value)
        with patch.object(os, 'sched_getaffinity', return_value={0}):
            topology = thermal.cpu_topology()
        self.assertEqual(topology['0']['thread_siblings_list'], '0,16')
        self.assertEqual(topology['0']['core_id'], 0)
        self.assertEqual(topology['0']['core_type'], 'unavailable (not_found)')

    def test_summary_keeps_lengths_separate(self) -> None:
        """Timing scale is late/early per input length, never cross-length."""
        summary = thermal.summarize({16: [10] * 5 + [20] * 5,
                                     64: [100] * 5 + [50] * 5,
                                     256: [1000]})
        self.assertEqual(summary['16']['late_early_ratio'], 2)
        self.assertEqual(summary['64']['late_early_ratio'], .5)
        self.assertIsNone(summary['256']['late_early_ratio'])

    def test_power_values_keep_units_and_optional_data_loss_is_nonfatal(
        self,
    ) -> None:
        """RAPL is contextual evidence; lack of it does not disable guards."""
        rapl = self.root / 'powercap/intel-rapl:0'
        rapl.mkdir(parents=True)
        (rapl / 'energy_uj').write_text('1234567')
        (rapl / 'max_energy_range_uj').write_text('4294967295')
        with patch.object(thermal, 'RAPL_ROOT', rapl.parent):
            summary, code, events = self.run_measure()
        self.assertEqual(code, 0)
        first = next(e for e in events if e['event'] == 'sample')
        self.assertEqual(first['rapl']['intel-rapl:0']['energy_uj'], 1234567)
        self.assertEqual(first['rapl']['intel-rapl:0'][
            'constraint_0_power_limit_uw'], 'unavailable (not_found)')

    def test_unlimited_cgroup_root_is_not_a_missing_sensor(self) -> None:
        """A real v2 hierarchy includes a root without memory.max."""
        root = self.root / 'hierarchy'
        leaf = root / 'scope'
        leaf.mkdir(parents=True)
        (leaf / 'memory.max').write_text('max')
        with patch.object(campus_survey, 'cgroup_dirs',
                          return_value=[leaf, root]):
            summary, code, _ = self.run_measure()
        self.assertEqual((code, summary['reason']),
                         (0, 'duration_completed'))

    def test_disappearing_cgroup_limit_still_stops_work(self) -> None:
        """A guard present at baseline cannot silently become optional."""
        group = self.root / 'limited'
        group.mkdir()
        maximum = group / 'memory.max'
        maximum.write_text(str(8 * 2**30))
        (group / 'memory.current').write_text(str(2**30))
        changed = False

        def emit(event: dict) -> None:
            nonlocal changed
            if event['event'] == 'forward' and not changed:
                changed = True
                maximum.unlink()

        with patch.object(campus_survey, 'cgroup_dirs', return_value=[group]):
            summary, code = thermal.measure(args(duration=.3), emit)
        self.assertTrue(changed)
        self.assertEqual((code, summary['reason']),
                         (4, 'monitor_unavailable'))

    def test_signal_during_child_start_does_not_orphan_work(self) -> None:
        """The parent must catch TERM before Process.start returns."""
        import multiprocessing
        from multiprocessing.process import BaseProcess

        result, pid_file = self.root / 'early-result', self.root / 'early-pid'

        def run() -> None:
            start = BaseProcess.start

            def start_then_signal(child: BaseProcess) -> None:
                start(child)
                pid_file.write_text(str(child.pid))
                os.kill(os.getpid(), signal.SIGTERM)

            with patch.object(BaseProcess, 'start', start_then_signal):
                summary, code = thermal.measure(args(duration=10),
                                                lambda e: None)
            result.write_text(json.dumps({'code': code, **summary}))

        parent = multiprocessing.get_context('fork').Process(target=run)
        parent.start()
        try:
            parent.join(3)
            self.assertFalse(parent.is_alive())
            self.assertEqual(parent.exitcode, 0)
            saved = json.loads(result.read_text())
            self.assertEqual((saved['code'], saved['reason']),
                             (143, 'interrupted'))
        finally:
            if parent.is_alive():
                parent.kill()
            parent.join()
            if pid_file.exists():
                pid = int(pid_file.read_text())
                for kill in (os.killpg, os.kill):
                    try:
                        kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


class ThermalCLITests(unittest.TestCase):
    """File ownership and partial results are part of the real CLI contract."""

    def test_default_id_reuses_hmac_without_recording_secrets(self) -> None:
        """A saved key gives a stable anonymous ID without manual input."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / 'hmac.key'
            key.write_text('0123456789abcdef')
            key.chmod(0o600)
            for number in (1, 2):
                output = root / f'run-{number}.jsonl'
                with patch('socket.gethostname',
                           return_value='CAMPUS.example'), \
                        patch.object(thermal, 'discover_sensors',
                                     return_value=[]), \
                        contextlib.redirect_stderr(io.StringIO()):
                    code = thermal.main(['--hmac-key-file', str(key),
                                         '--output', str(output)])
                self.assertEqual(code, 3)
                rows = [json.loads(line)
                        for line in output.read_text().splitlines()]
                self.assertEqual(rows[0]['pc_id'], 'h0f84f2e9e6b2')
                self.assertEqual(rows[0]['pc_id_method'], 'hmac')
                self.assertNotIn('0123456789abcdef', output.read_text())
                self.assertNotIn('CAMPUS.example', output.read_text())
            self.assertEqual(key.read_text(), '0123456789abcdef')

    def test_missing_hmac_key_does_not_create_a_report(self) -> None:
        """A missing secret must not fall back to an unhashed hostname."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'run.jsonl'
            with patch.dict(os.environ, {}, clear=True), \
                    patch('sys.stdin.isatty', return_value=False), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                thermal.main(['--hmac-key-file', str(root / 'absent'),
                              '--output', str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_existing_file_and_symlink_stay_untouched(self) -> None:
        """Refusing an output path must not overwrite its target."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'old.jsonl'
            target.write_text('keep')
            link = Path(directory) / 'link.jsonl'
            link.symlink_to(target)
            for output in [target, link]:
                with contextlib.redirect_stderr(io.StringIO()):
                    code = thermal.main(['--pc-id', 'cmm-007',
                                         '--output', str(output)])
                self.assertEqual(code, 1)
                self.assertEqual(target.read_text(), 'keep')

    def test_refused_run_leaves_private_parseable_report(self) -> None:
        """Even preflight failure has metadata and an explicit summary."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.jsonl'
            with patch.object(thermal, 'discover_sensors', return_value=[]), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = thermal.main(['--pc-id', 'cmm-007',
                                     '--output', str(output)])
            self.assertEqual(code, 3)
            events = [json.loads(line)
                      for line in output.read_text().splitlines()]
            self.assertEqual(events[0]['event'], 'metadata')
            self.assertEqual(events[-1]['event'], 'summary')
            self.assertEqual(events[-1]['reason'], 'monitor_unavailable')
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_original_cli_dispatches_thermal(self) -> None:
        """The existing entry point accepts the new mode without inference."""
        result = subprocess.run([sys.executable, '-m', 'tools.campus_survey',
                                 'thermal', '--help'],
                                text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--max-temp-c', result.stdout)
