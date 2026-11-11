"""Measure repeated CPU SDK work, stopping on heat or lost safety monitoring.

Use campus_survey thermal --help. Records are JSON Lines, not app timings.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from types import FrameType
from typing import Any, Callable

from tools import campus_survey as survey

HWMON_ROOT = Path('/sys/class/hwmon')
CPU_ROOT = Path('/sys/devices/system/cpu')
MEMINFO = Path('/proc/meminfo')
RAPL_ROOT = Path('/sys/class/powercap')
POLL_S = .5
KILL_GRACE_S = 1.
Sensor = tuple[str, Path, int]
Event = dict[str, Any]


def integer(path: Path, low: int = 0, high: int = 2**63 - 1) -> int:
    """Reject malformed, negative or implausible hardware readings."""
    value = path.read_text().strip()
    if not re.fullmatch(r'[0-9]+', value) or not low <= int(value) <= high:
        raise ValueError('invalid hardware reading')
    return int(value)


def optional_integer(path: Path) -> int | str:
    """Keep missing optional measurements distinct from measured zero."""
    try:
        return integer(path)
    except (OSError, ValueError) as error:
        return survey.unavailable(survey.reason_for(error))


def discover_sensors() -> list[Sensor]:
    """Keep CPU sensors separate, with a CPU-specific critical margin."""
    sensors = []
    for hwmon in sorted(HWMON_ROOT.glob('hwmon*')):
        name = (hwmon / 'name').read_text().strip()
        if name not in survey.CPU_SENSORS:
            continue
        for path in sorted(hwmon.glob('temp[0-9]*_input')):
            stem = path.name.removesuffix('_input')
            critical = hwmon / f'{stem}_crit'
            limit = (integer(critical, 11_000, 150_000) - 10_000
                     if critical.exists() else 150_000)
            sensors.append((f'{hwmon.name}_{name}_{stem}', path, limit))
    return sensors


def discover_counters() -> list[Path]:
    """Find Intel core/package counters without summing duplicate events."""
    return sorted(CPU_ROOT.glob(
        'cpu[0-9]*/thermal_throttle/*_throttle_count'))


def snapshot(sensors: list[Sensor], counters: list[Path],
             pid: int | None = None) -> Event:
    """Read mandatory safety data and optional context; change no settings."""
    match = re.search(r'^MemAvailable:\s+(\d+) kB$', MEMINFO.read_text(),
                      re.MULTILINE)
    if not match:
        raise ValueError('missing MemAvailable')
    available = int(match[1]) * 1024
    quota: dict[str, Any] = {}
    memory_limits = {}
    try:
        for index, directory in enumerate(survey.cgroup_dirs()):
            try:
                quota[str(index)] = {
                    key: int(value) for key, value in (
                        line.split() for line in
                        (directory / 'cpu.stat').read_text().splitlines())
                    if key in ('nr_periods', 'nr_throttled', 'throttled_usec')}
            except (OSError, ValueError) as error:
                quota[str(index)] = survey.unavailable(
                    survey.reason_for(error))
            try:
                maximum = (directory / 'memory.max').read_text().strip()
            except FileNotFoundError:
                # The v2 root and controller-disabled ancestors have no limit.
                continue
            memory_limits[str(index)] = maximum
            if maximum != 'max':
                if not maximum.isdecimal():
                    raise ValueError('invalid memory limit')
                available = min(available, max(
                    0, int(maximum) - integer(directory / 'memory.current')))
    except NotImplementedError:
        pass
    temperatures = {name: integer(path, 1_000, 150_000)
                    for name, path, _ in sensors}
    throttle = {f'{p.parents[1].name}_{p.name}': integer(p)
                for p in counters}
    energy = {}
    for path in sorted(RAPL_ROOT.glob('intel-rapl*/energy_uj')):
        if re.fullmatch(r'intel-rapl(?:-mmio)?(?::\d+)+', path.parent.name):
            energy[path.parent.name] = {
                key: optional_integer(path.parent / key) for key in (
                    'energy_uj', 'max_energy_range_uj',
                    'constraint_0_power_limit_uw',
                    'constraint_0_time_window_us',
                    'constraint_1_power_limit_uw',
                    'constraint_1_time_window_us')}
    result: Event = {
        'temperatures_mc': temperatures, 'thermal_counters': throttle,
        'available_bytes': available, 'cgroup_cpu': quota,
        'cgroup_memory_limits': memory_limits,
        'frequency_source': 'cpufreq/scaling_cur_freq',
        'frequency_is_effective_clock': False,
        'frequency_khz': {str(cpu): optional_integer(
            CPU_ROOT / f'cpu{cpu}/cpufreq/scaling_cur_freq')
            for cpu in sorted(os.sched_getaffinity(0))},
        'rapl': energy or survey.unavailable('not_found'),
        'loadavg': list(os.getloadavg()),
    }
    if pid is not None:
        try:
            result['worker_cpu_s'] = survey.process_cpu_s(pid)
        except (OSError, ValueError):
            result['worker_cpu_s'] = None
    return result


def stop_reason(sample: Event, baseline: Event, limits: dict[str, int],
                min_free_mib: int) -> str | None:
    """Apply safety bounds, without interpreting quotas as thermal events."""
    if not (baseline['cgroup_memory_limits'].keys() <=
            sample['cgroup_memory_limits'].keys()):
        return 'monitor_unavailable'
    if any(sample['temperatures_mc'][key] >= limit
           for key, limit in limits.items()):
        return 'temperature_limit'
    for key, before in baseline['thermal_counters'].items():
        after = sample['thermal_counters'][key]
        if after > before:
            return 'thermal_throttling'
        if after < before:
            return 'counter_reset'
    if sample['available_bytes'] < min_free_mib * 1024**2:
        return 'low_memory'
    return None


def worker(args: argparse.Namespace, conn: Connection,
           started_ns: int) -> None:
    """Run public SDK CPU calls in an isolated, killable process group."""
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        found = survey.find_model(args.hf_home)
        if found is None:
            raise FileNotFoundError('model not cached')
        os.environ['HF_HUB_CACHE'] = str(found[1])
        if args.threads is not None:
            for name in survey.THREAD_ENVS:
                os.environ[name] = str(args.threads)
        conn.send({'event': 'stage', 'stage': 'model_load'})
        from llm_sdk import Small_LLM_Model

        model = Small_LLM_Model(survey.MODEL_NAME, device='cpu',
                                trust_remote_code=False)
        text = survey.FILLER
        ids = model.encode(text).tolist()[0]
        while len(ids) < max(args.lengths):
            text *= 2
            ids = model.encode(text).tolist()[0]
        inputs = [ids[:length] for length in args.lengths]

        def forward(tokens: list[int], event: str) -> None:
            begin = time.perf_counter_ns()
            model.get_logits_from_input_ids(tokens)
            conn.send({'event': event, 'input_tokens': len(tokens),
                       'started_ns': begin - started_ns,
                       'elapsed_ns': time.perf_counter_ns() - begin})

        for tokens in inputs:
            forward(tokens, 'warmup')
        conn.send({'event': 'stage', 'stage': 'continuous_forward'})
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            for tokens in inputs:
                if time.monotonic() >= deadline:
                    break
                forward(tokens, 'forward')
        conn.send({'event': 'done'})
    except Exception:
        conn.send({'event': 'error'})
    finally:
        conn.close()


def stop_child(child: BaseProcess) -> None:
    """Stop the owned group, including descendants after its leader exits."""
    if child.pid is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        if child.is_alive():
            child.terminate()
    child.join(KILL_GRACE_S)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if child.is_alive():
        child.kill()
    child.join(KILL_GRACE_S)


def summarize(times: dict[int, list[int]]) -> Event:
    """Compare equal input lengths, retaining short runs without a ratio."""
    summary = {}
    for length, values in times.items():
        if values:
            summary[str(length)] = {
                'calls': len(values), 'median_ns': statistics.median(values),
                'late_early_ratio': (statistics.median(values[-5:]) /
                                     statistics.median(values[:5])
                                     if len(values) >= 10 else None)}
    return summary


def measure(args: argparse.Namespace, emit: Callable[[Event], None]
            ) -> tuple[Event, int]:
    """Monitor a child and preserve observations before stopping."""
    result: Event = {'event': 'summary', 'status': 'refused',
                     'reason': 'monitor_unavailable', 'timings': {}}
    try:
        sensors, counters = discover_sensors(), discover_counters()
        if not sensors or not counters:
            return result, 3
        baseline = snapshot(sensors, counters)
        limits = {name: min(limit, args.max_temp_c * 1000)
                  for name, _, limit in sensors}
        emit({'event': 'sample', 'elapsed_ns': 0,
              'limits_mc': limits, **baseline})
        reason = stop_reason(baseline, baseline, limits, args.min_free_mib)
        if reason:
            return {**result, 'reason': reason}, 3
    except (OSError, ValueError):
        return result, 3
    if survey.find_model(args.hf_home) is None:
        return {**result, 'reason': 'model_not_cached'}, 3
    context = multiprocessing.get_context('fork')
    receiver, sender = context.Pipe(duplex=False)
    started_ns = time.perf_counter_ns()
    child = context.Process(target=worker, args=(args, sender, started_ns),
                            daemon=True)
    interrupted: list[int] = []

    def interrupt(signum: int, frame: FrameType | None) -> None:
        """Let the supervisor stop the child and persist its partial record."""
        interrupted.append(signum)

    handlers = {sig: signal.getsignal(sig) for sig in
                (signal.SIGINT, signal.SIGTERM)}
    times: dict[int, list[int]] = {length: [] for length in args.lengths}
    code = 1
    completed = False
    try:
        deadline = time.monotonic() + args.budget
        for sig in handlers:
            signal.signal(sig, interrupt)
        if not interrupted:
            child.start()
            sender.close()
        next_sample = time.monotonic()
        result.update(status='failed', reason='worker_failed')
        while True:
            now = time.monotonic()
            if interrupted:
                result['status'] = 'stopped'
                result['reason'], code = 'interrupted', 128 + interrupted[0]
                break
            if now >= next_sample:
                try:
                    reading = snapshot(sensors, counters, child.pid)
                    reason = stop_reason(reading, baseline, limits,
                                         args.min_free_mib)
                except (OSError, ValueError):
                    reason = 'monitor_unavailable'
                else:
                    emit({'event': 'sample', 'elapsed_ns':
                          time.perf_counter_ns() - started_ns, **reading})
                if reason:
                    result['status'] = 'stopped'
                    result['reason'], code = reason, 4
                    break
                next_sample = time.monotonic() + POLL_S
            if now >= deadline:
                result['status'] = 'stopped'
                result['reason'], code = 'budget_exceeded', 4
                break
            if completed:
                result.update(status='completed', reason='duration_completed')
                code = 0
                break
            wait_s = max(0, min(POLL_S, next_sample - now, deadline - now))
            if receiver.poll(wait_s):
                try:
                    event = receiver.recv()
                except EOFError:
                    break
                if event['event'] == 'done':
                    # Sample once more so a fast last forward cannot hide heat.
                    next_sample = 0
                    completed = True
                    continue
                if event['event'] == 'error':
                    break
                emit(event)
                if event['event'] == 'forward':
                    times[event['input_tokens']].append(event['elapsed_ns'])
    except (OSError, ValueError):
        result.update(status='failed', reason='measurement_failed')
        code = 1
    finally:
        stop_child(child)
        receiver.close()
        sender.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    result['elapsed_ns'] = time.perf_counter_ns() - started_ns
    result['timings'] = summarize(times)
    return result, code


def cpu_topology() -> Event:
    """Read CPU relationships for later VM/cgroup allocation choices."""
    topology = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        directory = CPU_ROOT / f'cpu{cpu}/topology'
        siblings = survey.safe(
            lambda: survey.read(directory / 'thread_siblings_list'))
        topology[str(cpu)] = {
            key: optional_integer(directory / key)
            for key in ('physical_package_id', 'core_id', 'core_type')}
        topology[str(cpu)]['thread_siblings_list'] = (
            siblings if re.fullmatch('[0-9,-]+', siblings)
            else survey.unavailable('not_found'))
    return topology


def metadata(pc_id: str, hf_home: Path | None) -> Event:
    """Sanitize public metadata and add fixed provenance fields."""
    public = dict(line.split(' = ', 1) for line in
                  survey.render(survey.inspect(hf_home)).splitlines())
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    for name in ('uv.lock', 'tools/thermal_survey.py',
                 'tools/campus_survey.py', 'llm_sdk/llm_sdk/__init__.py'):
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                                capture_output=True, text=True, timeout=2,
                                check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = 'unavailable'
    found = survey.find_model(hf_home)
    revision = found[2].name if found is not None else ''
    if not re.fullmatch('[0-9a-f]{40}', revision):
        revision = 'unavailable'
    return {'event': 'metadata', 'protocol': 'thermal-survey-v1',
            'pc_id': pc_id, 'utc': datetime.now(timezone.utc).isoformat(),
            'commit': commit if re.fullmatch('[0-9a-f]{40}', commit)
            else 'unavailable', 'environment': public, 'sha256': hashes,
            'cpu_topology': cpu_topology(),
            'model_revision': revision,
            'scope': 'fixed SDK CPU work, not application deadline evidence'}


def main(argv: list[str] | None = None) -> int:
    """Write a new private JSONL report; never overwrite an existing path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pc-id', help='optional ID; default: hostname HMAC')
    parser.add_argument('--hmac-key-file', type=Path,
                        default=survey.HMAC_KEY_FILE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hf-home', type=Path)
    parser.add_argument('--lengths', type=survey.parse_lengths,
                        default=[16, 64, 256])
    parser.add_argument('--duration', type=int, default=300)
    parser.add_argument('--budget', type=int, default=360)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--max-temp-c', type=int, default=85)
    parser.add_argument('--min-free-mib', type=int, default=1024)
    args = parser.parse_args(argv)
    if not (1 <= args.duration < args.budget <= 900
            and 1 <= args.max_temp_c <= 95 and args.min_free_mib >= 512
            and (args.threads is None or
                 1 <= args.threads <= len(os.sched_getaffinity(0)))):
        parser.error('invalid duration/budget, temperature, memory or threads')
    try:
        if args.pc_id is None:
            pc_id = survey.hmac_pc_id(
                survey.read_hmac_key(args.hmac_key_file), socket.gethostname())
        else:
            pc_id = survey.validate_pc_id(args.pc_id)
    except (OSError, ValueError, EOFError):
        parser.error('invalid PC ID or HMAC key; initialize the key with '
                     'python -m tools.campus_survey --init-hmac-key')
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT |
                             os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            def emit(event: Event) -> None:
                """Persist individual records before waiting for more work."""
                stream.write(json.dumps(event, allow_nan=False) + '\n')
                stream.flush()

            emit({**metadata(pc_id, args.hf_home),
                  'pc_id_method': 'hmac' if args.pc_id is None else 'explicit',
                  'config': {
                key: getattr(args, key) for key in (
                    'lengths', 'duration', 'budget', 'threads',
                    'max_temp_c', 'min_free_mib')}})
            summary, code = measure(args, emit)
            emit(summary)
            print(f"thermal survey: {summary['reason']}", file=sys.stderr)
            return code
    except (OSError, ValueError):
        print('thermal survey: report or measurement failed', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
