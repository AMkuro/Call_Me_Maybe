"""Run CPU jobs two at a time and pause them while the CPU stays hot.

ADR-0025. Before starting, wait until the CPU package temperature (k10temp
Tctl) is at most 50 C, as research/evo-x3/measure.sh does. While the jobs
run, read Tctl and MemAvailable every second:

- Tctl >= 95 C for 60 of the last 120 s: pause every job (SIGSTOP to its
  process group); resume (SIGCONT) at <= 70 C. Stop if a pause sees no 1 C
  drop for 60 s.
- MemAvailable < 8 GiB for 10 s, an unreadable sensor or meminfo, or SIGINT/
  SIGTERM to this program: stop every job (SIGTERM, then SIGKILL).

There is no time limit. Every job runs pinned to CPUs 0-15 with
OMP_NUM_THREADS=8, without a shell. Runs inside the Claude Code sandbox,
where systemd-run --user is unavailable.

Usage: python -m tools.cpu_watch JOBS.json LABEL
  JOBS.json  a JSON list of argument lists, e.g. [["python", "-m", "x"]]
  Output: research/evo-x3/watch/LABEL/ (monitor.tsv, job-N.out/.err,
  summary.json). Exit 0 all jobs succeeded, 1 a job failed, 2 bad input,
  3 the CPU did not cool, 4 stopped (memory, sensor or interrupted).
"""

from collections import deque
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import FrameType

COOL_C = 50_000
COOL_STALL_S = 60.0
HOT_C = 95_000
WINDOW_S = 120.0
HOT_LIMIT_S = 60.0
RESUME_C = 70_000
PAUSE_STALL_S = 60.0
MIN_AVAIL_KIB = 8 * 1024 * 1024
LOW_MEM_S = 10.0
MAX_JOBS = 2
THREADS = '8'
CPUS = '0-15'
DROP = 1000  # 1 C, in millidegrees
WATCH_ROOT = Path(__file__).resolve().parents[1] / 'research/evo-x3/watch'
CPU_SENSORS = {'k10temp', 'coretemp', 'zenpower'}


def read_millideg(path: Path) -> int:
    """Return a temperature in millidegrees; reject anything implausible."""
    text = path.read_text().strip()
    if not re.fullmatch(r'[0-9]{1,6}', text):
        raise ValueError(f'{path}: not a temperature: {text!r}')
    value = int(text)
    if not 1_000 <= value <= 150_000:
        raise ValueError(f'{path}: out of range: {value}')
    return value


def read_mem_available_kib(path: Path = Path('/proc/meminfo')) -> int:
    """Return MemAvailable in KiB."""
    found = re.search(r'^MemAvailable:\s+([0-9]+) kB$', path.read_text(),
                      re.MULTILINE)
    if found is None:
        raise ValueError(f'{path}: no MemAvailable line in kB')
    return int(found.group(1))


def find_cpu_sensor(root: Path = Path('/sys/class/hwmon')) -> Path | None:
    """Find the CPU package sensor by name; hwmon numbers change at boot."""
    for hwmon in sorted(root.glob('hwmon*')):
        try:
            name = (hwmon / 'name').read_text().strip()
        except OSError:
            continue
        if name in CPU_SENSORS:
            return hwmon / 'temp1_input'
    return None


class CoolWait:
    """Before starting: 'go' at the target, 'stall' without a 1 C drop."""

    def __init__(self, target: int = COOL_C,
                 stall_s: float = COOL_STALL_S) -> None:
        self.target = target
        self.stall_s = stall_s
        self.best: int | None = None
        self.last_drop = 0.0

    def step(self, now: float, temp: int) -> str:
        """Return 'go', 'wait' or 'stall' for one sample."""
        if temp <= self.target:
            return 'go'
        if self.best is None or temp <= self.best - DROP:
            self.best, self.last_drop = temp, now
        return 'stall' if now - self.last_drop >= self.stall_s else 'wait'


class Watch:
    """While running: decide run / pause / paused / resume / stop:<why>."""

    def __init__(self) -> None:
        self.samples: deque[tuple[float, bool]] = deque()
        self.low_since: float | None = None
        self.paused = False
        self.pause = CoolWait(RESUME_C, PAUSE_STALL_S)

    def hot_seconds(self, now: float) -> float:
        """Seconds of the last WINDOW_S during which the CPU was hot."""
        start = now - WINDOW_S
        total = 0.0
        times = [t for t, _ in self.samples] + [now]
        for (t, hot), end in zip(self.samples, times[1:]):
            if hot:
                total += max(0.0, end - max(t, start))
        return total

    def step(self, now: float, temp: int, avail_kib: int) -> str:
        """Take one sample and return the action."""
        if avail_kib < MIN_AVAIL_KIB:
            self.low_since = now if self.low_since is None else self.low_since
            if now - self.low_since >= LOW_MEM_S:
                return 'stop:memory'
        else:
            self.low_since = None
        if self.paused:
            state = self.pause.step(now, temp)
            if state == 'go':
                self.paused = False
                self.samples.clear()
                return 'resume'
            return 'stop:not_cooling' if state == 'stall' else 'paused'
        while self.samples and self.samples[0][0] < now - 2 * WINDOW_S:
            self.samples.popleft()
        self.samples.append((now, temp >= HOT_C))
        if self.hot_seconds(now) >= HOT_LIMIT_S:
            self.paused = True
            self.pause = CoolWait(RESUME_C, PAUSE_STALL_S)
            return 'pause'
        return 'run'


def signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    """Signal the job's whole process group (it is a session leader)."""
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def stop_all(procs: list[subprocess.Popen[bytes]],
             grace: float = 10.0) -> None:
    """SIGTERM every running job, SIGKILL what is left after the grace."""
    for proc in procs:
        if proc.poll() is None:
            signal_group(proc, signal.SIGCONT)  # a stopped job cannot exit
            signal_group(proc, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while any(p.poll() is None for p in procs) and time.monotonic() < deadline:
        time.sleep(0.1)
    for proc in procs:
        if proc.poll() is None:
            signal_group(proc, signal.SIGKILL)
            proc.wait()


class Run:
    """The records of one watched run, written to summary.json."""

    def __init__(self, jobs: list[list[str]]) -> None:
        self.exit_codes: list[int | None] = [None] * len(jobs)
        self.jobs = jobs
        self.stopped: str | None = None
        self.max_parallel = 0
        self.paused_count = 0
        self.paused_s = 0.0
        self.cool_wait_s = 0.0
        self.tctl_max_mc = 0

    def summary(self) -> dict[str, object]:
        """Return the JSON record."""
        return {
            'jobs': [{'argv': job, 'exit_code': code}
                     for job, code in zip(self.jobs, self.exit_codes)],
            'stopped': self.stopped, 'max_parallel': self.max_parallel,
            'paused_count': self.paused_count,
            'paused_s': round(self.paused_s, 1),
            'cool_wait_s': round(self.cool_wait_s, 1),
            'tctl_max_mc': self.tctl_max_mc,
            'profile': {'cpus': CPUS, 'omp': THREADS, 'max_jobs': MAX_JOBS}}

    def exit_code(self) -> int:
        """Map the result to the documented exit codes."""
        if self.stopped == 'not_cooling':
            return 3
        if self.stopped is not None:
            return 4
        return 1 if any(code != 0 for code in self.exit_codes) else 0


def cool_down(sensor: Path, stall_s: float, interrupted: list[int]) -> str:
    """Wait for the start temperature; return 'go' or why it stopped."""
    wait = CoolWait(stall_s=stall_s)
    while True:
        try:
            state = wait.step(time.monotonic(), read_millideg(sensor))
        except (OSError, ValueError) as error:
            print(f'cpu_watch: {error}', file=sys.stderr)
            return 'sensor'
        if interrupted:
            return 'interrupted'
        if state == 'go':
            return 'go'
        if state == 'stall':
            return 'not_cooling'
        time.sleep(1.0)


def run(jobs: list[list[str]], out: Path, *, sensor: Path | None = None,
        meminfo: Path = Path('/proc/meminfo'),
        cool_stall_s: float = COOL_STALL_S) -> int:
    """Run the jobs under the watch; write the records into OUT."""
    sensor = sensor or find_cpu_sensor()
    if sensor is None:
        print('cpu_watch: no CPU temperature sensor', file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    record = Run(jobs)
    interrupted: list[int] = []

    def on_signal(sig: int, frame: FrameType | None) -> None:
        interrupted.append(sig)

    old = {sig: signal.signal(sig, on_signal)
           for sig in (signal.SIGINT, signal.SIGTERM)}
    running: dict[int, subprocess.Popen[bytes]] = {}
    try:
        start = time.monotonic()
        state = cool_down(sensor, cool_stall_s, interrupted)
        record.cool_wait_s = time.monotonic() - start
        if state != 'go':
            record.stopped = state
            return record.exit_code()
        watch_jobs(jobs, out, sensor, meminfo, record, running, interrupted)
        if record.stopped is not None:
            stop_all(list(running.values()))
            for index, proc in running.items():
                record.exit_codes[index] = proc.returncode
        return record.exit_code()
    finally:
        stop_all(list(running.values()))
        for sig, handler in old.items():
            signal.signal(sig, handler)
        (out / 'summary.json').write_text(
            json.dumps(record.summary(), indent=2) + '\n')


def watch_jobs(jobs: list[list[str]], out: Path, sensor: Path,
               meminfo: Path, record: Run,
               running: dict[int, subprocess.Popen[bytes]],
               interrupted: list[int]) -> None:
    """Start jobs (MAX_JOBS at a time) and act on each one-second sample."""
    env = dict(os.environ, OMP_NUM_THREADS=THREADS)
    pending = list(range(len(jobs)))
    watch = Watch()
    start = paused_at = time.monotonic()
    with open(out / 'monitor.tsv', 'w') as monitor:
        monitor.write('elapsed_s\ttctl_mc\tavail_kib\tstate\n')
        while True:
            for index, proc in list(running.items()):
                if proc.poll() is not None:
                    record.exit_codes[index] = proc.returncode
                    del running[index]
            while pending and len(running) < MAX_JOBS and not watch.paused:
                index = pending.pop(0)
                with open(out / f'job-{index}.out', 'wb') as stdout, \
                        open(out / f'job-{index}.err', 'wb') as stderr:
                    running[index] = subprocess.Popen(
                        ['/usr/bin/taskset', '-c', CPUS, *jobs[index]],
                        stdin=subprocess.DEVNULL, stdout=stdout,
                        stderr=stderr, env=env, start_new_session=True)
                record.max_parallel = max(record.max_parallel, len(running))
            if not running and not pending:
                return
            if interrupted:
                record.stopped = 'interrupted'
                return
            now = time.monotonic()
            try:
                temp = read_millideg(sensor)
                avail = read_mem_available_kib(meminfo)
            except (OSError, ValueError) as error:
                print(f'cpu_watch: {error}', file=sys.stderr)
                record.stopped = 'sensor'
                return
            record.tctl_max_mc = max(record.tctl_max_mc, temp)
            action = watch.step(now, temp, avail)
            monitor.write(f'{now - start:.1f}\t{temp}\t{avail}\t{action}\n')
            monitor.flush()
            if action == 'pause':
                record.paused_count += 1
                paused_at = now
                for proc in running.values():
                    signal_group(proc, signal.SIGSTOP)
            elif action == 'resume':
                record.paused_s += now - paused_at
                for proc in running.values():
                    signal_group(proc, signal.SIGCONT)
            elif action.startswith('stop:'):
                record.stopped = action.removeprefix('stop:')
                return
            time.sleep(1.0)


def load_jobs(path: Path) -> list[list[str]]:
    """Read a non-empty JSON list of non-empty string lists."""
    jobs = json.loads(path.read_text())
    if (not isinstance(jobs, list) or not jobs
            or not all(isinstance(job, list) and job
                       and all(isinstance(arg, str) for arg in job)
                       for job in jobs)):
        raise ValueError('jobs must be a non-empty list of argument lists')
    return jobs


def main(argv: list[str] | None = None) -> int:
    """Check the arguments, then run the jobs."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}', args[1]):
        print('cpu_watch: LABEL must match [A-Za-z0-9_-]{1,60}',
              file=sys.stderr)
        return 2
    try:
        jobs = load_jobs(Path(args[0]))
    except (OSError, ValueError) as error:
        print(f'cpu_watch: {error}', file=sys.stderr)
        return 2
    out = WATCH_ROOT / args[1]
    if out.exists():
        print(f'cpu_watch: {out} already exists', file=sys.stderr)
        return 2
    code = run(jobs, out)
    print(out)
    return code


if __name__ == '__main__':
    sys.exit(main())
