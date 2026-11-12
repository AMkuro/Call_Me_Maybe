"""Verify the CPU-run watch (ADR-0025) with data and short real processes."""

import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from tools import cpu_watch as cw

GIB_KIB = 1024 * 1024
COOL = 40_000
PLENTY = 64 * GIB_KIB


def feed(watch: cw.Watch, temps: list[int], avail: int = PLENTY,
         start: float = 0.0) -> list[str]:
    """Give one sample per second and return the actions."""
    return [watch.step(start + i, t, avail) for i, t in enumerate(temps)]


class ReadValues(unittest.TestCase):
    """Sensor and meminfo values are parsed strictly."""

    def write(self, text: str) -> Path:
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / 'v'
        path.write_text(text)
        return path

    def test_millidegrees(self) -> None:
        self.assertEqual(cw.read_millideg(self.write('42000\n')), 42000)
        for bad in ['', '\n', 'abc', '-1', '1+1', '4.2', '999', '150001']:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cw.read_millideg(self.write(bad))

    def test_missing_sensor_file_raises(self) -> None:
        with self.assertRaises(OSError):
            cw.read_millideg(Path('/nonexistent/temp1_input'))

    def test_mem_available(self) -> None:
        good = 'MemTotal: 100 kB\nMemAvailable:   123456 kB\n'
        self.assertEqual(cw.read_mem_available_kib(self.write(good)), 123456)
        for bad in ['MemTotal: 100 kB\n', 'MemAvailable: x kB\n',
                    'MemAvailable: 5\n']:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cw.read_mem_available_kib(self.write(bad))

    def test_find_cpu_sensor_by_name(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for index, name in enumerate(['acpitz', 'amdgpu', 'k10temp']):
            (root / f'hwmon{index}').mkdir()
            (root / f'hwmon{index}' / 'name').write_text(name + '\n')
        self.assertEqual(cw.find_cpu_sensor(root),
                         root / 'hwmon2' / 'temp1_input')
        (root / 'hwmon2' / 'name').write_text('nvme\n')
        self.assertIsNone(cw.find_cpu_sensor(root))


class CoolWaitRule(unittest.TestCase):
    """Before starting: wait for 50 C; give up after 60 s with no 1 C drop."""

    def test_cool_starts_at_once(self) -> None:
        self.assertEqual(cw.CoolWait().step(0, 50_000), 'go')

    def test_falling_temperature_waits_then_goes(self) -> None:
        wait = cw.CoolWait()
        actions = [wait.step(i, t) for i, t in
                   enumerate([60_000, 58_000, 55_000, 52_000, 50_000])]
        self.assertEqual(actions, ['wait'] * 4 + ['go'])

    def test_stall_gives_up(self) -> None:
        wait = cw.CoolWait()
        actions = [wait.step(i, 70_000 - 15 * i) for i in range(62)]
        self.assertEqual(actions[59], 'wait')  # 0.9 C in 60 s is not a drop
        self.assertEqual(actions[60], 'stall')


class WatchRule(unittest.TestCase):
    """While running: pause when hot for 60 of 120 s, resume at 70 C."""

    def test_hot_59_of_120_seconds_keeps_running(self) -> None:
        actions = feed(cw.Watch(), [96_000] * 59 + [COOL] * 61)
        self.assertEqual(set(actions), {'run'})

    def test_hot_60_seconds_pauses(self) -> None:
        actions = feed(cw.Watch(), [96_000] * 61)
        self.assertEqual(actions[59], 'run')
        self.assertEqual(actions[60], 'pause')

    def test_oscillation_like_2026_11_10_pauses(self) -> None:
        # 98.7 and 94 C alternating: a reset-on-gap rule never fires here.
        actions = feed(cw.Watch(), [98_700, 94_000] * 70)
        self.assertIn('pause', actions)

    def test_paused_resumes_at_70_and_clears_window(self) -> None:
        watch = cw.Watch()
        feed(watch, [96_000] * 61)
        self.assertTrue(watch.paused)
        actions = feed(watch, [90_000, 80_000, 70_000], start=61)
        self.assertEqual(actions, ['paused', 'paused', 'resume'])
        self.assertFalse(watch.paused)
        # The window restarts: 59 more hot seconds do not pause again.
        self.assertEqual(set(feed(watch, [96_000] * 59, start=64)), {'run'})

    def test_paused_without_cooling_stops(self) -> None:
        watch = cw.Watch()
        feed(watch, [96_000] * 61)
        actions = feed(watch, [96_000] * 62, start=61)
        self.assertEqual(actions[59], 'paused')
        self.assertEqual(actions[60], 'stop:not_cooling')

    def test_low_memory_for_10_seconds_stops(self) -> None:
        low = 8 * GIB_KIB - 1
        self.assertEqual(feed(cw.Watch(), [COOL] * 10, avail=low)[-1], 'run')
        self.assertEqual(feed(cw.Watch(), [COOL] * 11, avail=low)[-1],
                         'stop:memory')

    def test_low_memory_gap_resets(self) -> None:
        watch = cw.Watch()
        low = 8 * GIB_KIB - 1
        actions = (feed(watch, [COOL] * 9, avail=low)
                   + feed(watch, [COOL], start=9)
                   + feed(watch, [COOL] * 9, avail=low, start=10))
        self.assertEqual(set(actions), {'run'})

    def test_window_counts_elapsed_time_not_samples(self) -> None:
        # Samples every 3 s: 21 hot samples cover 60 s.
        watch = cw.Watch()
        actions = [watch.step(3.0 * i, 96_000, PLENTY) for i in range(21)]
        self.assertEqual(actions[-1], 'pause')


class Processes(unittest.TestCase):
    """Pausing and stopping act on whole process groups."""

    def test_pause_resume_and_stop(self) -> None:
        proc = subprocess.Popen(['sleep', '30'], start_new_session=True)
        try:
            cw.signal_group(proc, signal.SIGSTOP)
            time.sleep(0.2)
            state = Path(f'/proc/{proc.pid}/stat').read_text().split()[2]
            self.assertEqual(state, 'T')
            cw.signal_group(proc, signal.SIGCONT)
            cw.stop_all([proc], grace=0.5)
            self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None:
                proc.kill()


class RunJobs(unittest.TestCase):
    """End to end with a cool fake sensor: jobs run pinned with 8 threads."""

    def fake(self, temp: int) -> tuple[Path, Path, Path]:
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (tmp / 'temp1_input').write_text(f'{temp}\n')
        (tmp / 'meminfo').write_text(f'MemAvailable: {PLENTY} kB\n')
        return tmp, tmp / 'temp1_input', tmp / 'meminfo'

    def test_runs_jobs_two_at_a_time_and_records(self) -> None:
        tmp, sensor, meminfo = self.fake(COOL)
        probe = ('import os; print(os.environ["OMP_NUM_THREADS"],'
                 ' len(os.sched_getaffinity(0)))')
        jobs = [[sys.executable, '-c', probe]] * 3 + [['false']]
        code = cw.run(jobs, tmp / 'out', sensor=sensor, meminfo=meminfo)
        self.assertEqual(code, 1)  # one job failed
        summary = json.loads((tmp / 'out' / 'summary.json').read_text())
        self.assertEqual([j['exit_code'] for j in summary['jobs']],
                         [0, 0, 0, 1])
        self.assertIsNone(summary['stopped'])
        self.assertEqual(summary['max_parallel'], 2)
        out = (tmp / 'out' / 'job-0.out').read_text().split()
        self.assertEqual(out, ['8', '16'])
        self.assertTrue((tmp / 'out' / 'monitor.tsv').is_file())

    def test_hot_start_does_not_cool_and_runs_nothing(self) -> None:
        tmp, sensor, meminfo = self.fake(99_000)
        marker = tmp / 'ran'
        code = cw.run([['touch', str(marker)]], tmp / 'out', sensor=sensor,
                      meminfo=meminfo, cool_stall_s=1)
        self.assertEqual(code, 3)
        self.assertFalse(marker.exists())

    def test_unreadable_sensor_stops_jobs(self) -> None:
        tmp, sensor, meminfo = self.fake(COOL)
        script = f'import os, time; os.remove({str(sensor)!r}); time.sleep(30)'
        start = time.monotonic()
        code = cw.run([[sys.executable, '-c', script]], tmp / 'out',
                      sensor=sensor, meminfo=meminfo)
        self.assertEqual(code, 4)
        self.assertLess(time.monotonic() - start, 20)
        summary = json.loads((tmp / 'out' / 'summary.json').read_text())
        self.assertEqual(summary['stopped'], 'sensor')


class CommandLine(unittest.TestCase):
    """The job list and label are checked before anything runs."""

    def test_rejects_bad_job_lists_and_labels(self) -> None:
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = tmp / 'jobs.json'
        for text in ['[]', '[[]]', '[["a", 1]]', '{"a": 1}', 'not json']:
            path.write_text(text)
            with self.subTest(text=text):
                self.assertEqual(cw.main([str(path), 'ok-label']), 2)
        path.write_text('[["true"]]')
        for label in ['', '../x', 'a b', 'x' * 61]:
            with self.subTest(label=label):
                self.assertEqual(cw.main([str(path), label]), 2)
        self.assertEqual(cw.main([str(path)]), 2)


if __name__ == '__main__':
    unittest.main()
