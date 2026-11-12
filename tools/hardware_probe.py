"""Compare CPU/memory kernels without models, downloads or hardware changes."""

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.json_io import loads  # noqa: E402
from tools.campus_survey import cgroup_dirs, cpu_temp_c  # noqa: E402

CASE_NAMES = ('memory_read', 'memory_copy', 'matmul_1', 'matmul_64',
              'matmul_256')
THREAD_ENVS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
               'NUMEXPR_NUM_THREADS')
PositiveInt = Annotated[int, Field(gt=0)]
# No completed sample can outlive the maximum 600-second worker budget.
TimingInt = Annotated[int, Field(gt=0, le=600_000_000_000)]


class Timing(BaseModel):
    """Exact integer batch timings and nominal work per operation."""

    model_config = ConfigDict(extra='forbid', strict=True)
    iterations: TimingInt
    samples_ns: list[TimingInt] = Field(min_length=1)
    payload_bytes: Annotated[int, Field(ge=0)]
    flops: Annotated[int, Field(ge=0)]

    @property
    def seconds(self) -> float:
        """Return the median wall time per operation, including dispatch."""
        return statistics.median(self.samples_ns) / self.iterations / 1e9

    @property
    def gb_per_second(self) -> float:
        """Logical bytes per second, not measured physical bus traffic."""
        return self.payload_bytes / self.seconds / 1e9


class Profile(BaseModel):
    """One isolated thread configuration and its measured kernels."""

    model_config = ConfigDict(extra='forbid', strict=True)
    memory_threads: PositiveInt
    affinity: list[int]
    native_threads: PositiveInt
    numpy_version: str
    blas: dict[str, str]
    cases: dict[str, Timing]


class Report(BaseModel):
    """A comparison input; partial runs remain explicitly incomplete."""

    model_config = ConfigDict(extra='forbid', strict=True)
    protocol: Annotated[int, Field(ge=1, le=1)] = 1
    label: str
    array_mib: PositiveInt
    repeats: PositiveInt
    sample_ms: PositiveInt
    status: Literal['complete', 'partial']
    errors: list[str]
    host: dict[str, Any]
    profiles: dict[str, Profile]

    @model_validator(mode='after')
    def complete_cases(self) -> 'Report':
        """Do not treat missing cases, samples or profiles as complete."""
        if self.status == 'complete' and (not self.profiles or self.errors):
            raise ValueError('complete report needs profiles and no errors')
        for profile in self.profiles.values():
            if set(profile.cases) != set(CASE_NAMES):
                raise ValueError('workload cases are missing or different')
            if any(len(case.samples_ns) != self.repeats
                   for case in profile.cases.values()):
                raise ValueError('sample count differs from repeats')
        return self


def available_memory() -> int:
    """Respect host MemAvailable and remaining cgroup v2 memory limits."""
    info = dict(line.split(':', 1) for line in
                Path('/proc/meminfo').read_text().splitlines())
    limits = [int(info['MemAvailable'].split()[0]) * 1024]
    try:
        for directory in cgroup_dirs():
            try:
                maximum = (directory / 'memory.max').read_text().strip()
                if maximum != 'max':
                    used = int((directory / 'memory.current').read_text())
                    limits.append(max(0, int(maximum) - used))
            except OSError:
                continue
    except (OSError, NotImplementedError):
        pass
    return min(limits)


def check_memory(array_mib: int, available: int) -> None:
    """Keep estimated allocation within 25% of available memory."""
    estimate = 3 * array_mib * 1024**2 + 32 * 1024**2
    if estimate > available // 4:
        raise ValueError(
            'insufficient available memory for buffers and margin; '
            'reduce --array-mib',
        )


def temperature() -> int | None:
    """Read CPU temperature in millidegrees when the sensor is available."""
    try:
        return round(cpu_temp_c() * 1000)
    except (OSError, ValueError, NotImplementedError):
        return None


def host_info() -> dict[str, Any]:
    """Read hardware facts without hostname, username or home paths."""
    affinity = sorted(os.sched_getaffinity(0))
    text = Path('/proc/cpuinfo').read_text()
    model = next((line.split(':', 1)[1].strip() for line in text.splitlines()
                  if line.startswith('model name')), platform.machine())
    caches: dict[str, int] = {}
    for cpu in affinity:
        for entry in Path(f'/sys/devices/system/cpu/cpu{cpu}/cache').glob(
            'index*'
        ):
            if (entry / 'level').read_text().strip() == '3':
                size = (entry / 'size').read_text().strip()
                units = {'K': 1024, 'M': 1024**2, 'G': 1024**3}
                shared = (entry / 'shared_cpu_list').read_text().strip()
                caches[shared] = int(size[:-1]) * units[size[-1]]
    memory = dict(line.split(':', 1) for line in
                  Path('/proc/meminfo').read_text().splitlines())
    topology = set()
    for cpu in affinity:
        folder = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            topology.add(((folder / 'physical_package_id').read_text().strip(),
                          (folder / 'core_id').read_text().strip()))
        except OSError:
            pass
    flags = next((line.split(':', 1)[1].split() for line in text.splitlines()
                  if line.startswith('flags')), [])
    return {'cpu': model, 'architecture': platform.machine(),
            'physical_cores_in_affinity': len(topology),
            'simd_features': [name for name in
                              ('fma', 'avx2', 'avx512f', 'avx_vnni')
                              if name in flags],
            'memory_total_bytes': int(memory['MemTotal'].split()[0]) * 1024,
            'collector_sha256': hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            'python': platform.python_version(), 'affinity': affinity,
            'available_memory_bytes': available_memory(),
            'l3_bytes_in_affinity': sum(caches.values()),
            'temp_start_mc': temperature()}


def worker_environment(profile: str) -> dict[str, str]:
    """Set BLAS variables before NumPy import without changing parent state."""
    env = os.environ.copy()
    for key in THREAD_ENVS:
        env.pop(key, None)
        if profile != 'default':
            env[key] = profile
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def measure_case(operation: Callable[[], None], repeats: int, sample_ms: int,
                 payload_bytes: int = 0, flops: int = 0) -> Timing:
    """Warm up, size a batch, and retain every integer wall-clock sample."""
    operation()
    start = time.perf_counter_ns()
    iterations = 0
    while iterations == 0 or (
        time.perf_counter_ns() - start < sample_ms * 1_000_000
    ):
        operation()
        iterations += 1
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            operation()
        samples.append(time.perf_counter_ns() - start)
    return Timing(iterations=iterations, samples_ns=samples,
                  payload_bytes=payload_bytes, flops=flops)


def worker(args: argparse.Namespace) -> Profile:
    """Exercise installed NumPy with bounded arrays; never import an SDK."""
    check_memory(args.array_mib, available_memory())
    import numpy as np

    affinity = sorted(os.sched_getaffinity(0))
    threads = len(affinity) if args.threads[0] == 'default' else int(
        args.threads[0]
    )
    source = np.ones(args.array_mib * 1024**2 // 4, dtype=np.float32)
    target = np.zeros_like(source)
    chunks = list(zip(np.array_split(source, threads),
                      np.array_split(target, threads)))

    def read_chunk(pair: Any) -> None:
        """Read and reduce one disjoint float32 slice."""
        if np.sum(pair[0], dtype=np.float64) != pair[0].size:
            raise ValueError('memory read result check failed')

    def copy_chunk(pair: Any) -> None:
        """Copy one disjoint slice, without a temporary output array."""
        np.copyto(pair[1], pair[0])

    with ThreadPoolExecutor(max_workers=threads) as pool:
        def read_all() -> None:
            """Include dispatch and completion of all memory readers."""
            list(pool.map(read_chunk, chunks))

        def copy_all() -> None:
            """Include dispatch and completion of all memory copies."""
            list(pool.map(copy_chunk, chunks))

        cases = {
            'memory_read': measure_case(read_all, args.repeats, args.sample_ms,
                                        payload_bytes=source.nbytes),
            'memory_copy': measure_case(copy_all, args.repeats, args.sample_ms,
                                        payload_bytes=2 * source.nbytes),
        }
    if not np.array_equal(source, target):
        raise ValueError('memory copy result check failed')
    chunks.clear()
    del source, target
    weights = np.ones((1024, 3072), dtype=np.float32)
    for rows in (1, 64, 256):
        inputs = np.ones((rows, 1024), dtype=np.float32)
        output = np.empty((rows, 3072), dtype=np.float32)

        def multiply() -> None:
            """Multiply fixed float32 matrices into preallocated output."""
            np.matmul(inputs, weights, out=output)

        cases[f'matmul_{rows}'] = measure_case(
            multiply, args.repeats, args.sample_ms,
            flops=2 * rows * 1024 * 3072,
        )
        if not np.all(output == 1024):
            raise ValueError('matrix result check failed')
    config = np.show_config(mode='dicts') or {}
    blas = config.get('Build Dependencies', {}).get('blas', {})
    return Profile(
        memory_threads=threads, affinity=affinity,
        native_threads=len(list(Path('/proc/self/task').iterdir())),
        numpy_version=np.__version__,
        blas={key: str(blas.get(key, 'unknown'))
              for key in ('name', 'version')},
        cases=cases,
    )


def collect(args: argparse.Namespace) -> Report:
    """Run thread configurations serially with a parent-enforced deadline."""
    host = host_info()
    if args.array_mib * 1024**2 < 4 * host['l3_bytes_in_affinity']:
        host['cache_warning'] = (
            'Each array is smaller than 4x the L3 in this affinity; '
            'use a larger --array-mib before attributing speed to DRAM'
        )
    check_memory(args.array_mib, host['available_memory_bytes'])
    profiles: dict[str, Profile] = {}
    errors: list[str] = []
    deadline = time.monotonic() + args.budget
    for profile in dict.fromkeys(args.threads):
        print(f'measuring threads={profile}: memory read/copy and matmul',
              file=sys.stderr, flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), '_worker',
                   '--threads', profile, '--array-mib', str(args.array_mib),
                   '--sample-ms', str(args.sample_ms),
                   '--repeats', str(args.repeats)]
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('measurement budget exhausted')
            result = subprocess.run(
                command, env=worker_environment(profile), capture_output=True,
                text=True, timeout=remaining,
            )
            if result.returncode:
                raise ValueError(result.stderr.strip() or 'worker failed')
            measured = Profile.model_validate(loads(result.stdout))
            profiles[profile] = measured
            for name, timing in measured.cases.items():
                rate = (f'{timing.gb_per_second:.2f} logical GB/s'
                        if timing.payload_bytes else
                        f'{timing.flops / timing.seconds / 1e9:.2f} '
                        'nominal GFLOP/s')
                low = min(timing.samples_ns) / timing.iterations / 1e6
                high = max(timing.samples_ns) / timing.iterations / 1e6
                print(f'  {name}: {timing.seconds * 1000:.3f} ms; '
                      f'range {low:.3f}..{high:.3f} ms; {rate}',
                      file=sys.stderr)
        except (OSError, ValueError, TimeoutError,
                subprocess.TimeoutExpired) as error:
            reason = ('worker_timeout' if isinstance(
                error, (TimeoutError, subprocess.TimeoutExpired)
            ) else 'worker_failed')
            errors.append(f'threads={profile}: {reason}')
            print(f'worker stopped: {reason}', file=sys.stderr)
            if reason != 'worker_timeout':
                print(str(error), file=sys.stderr)
            break
    host['temp_end_mc'] = temperature()
    return Report(label=args.label, array_mib=args.array_mib,
                  repeats=args.repeats, sample_ms=args.sample_ms,
                  status='partial' if errors else 'complete', errors=errors,
                  host=host, profiles=profiles)


def read_report(path: Path) -> Report:
    """Read strict report data through the shared no-overwrite JSON parser."""
    return Report.model_validate(loads(path.read_text(encoding='utf-8')))


def compare(reference: Report, candidate: Report,
            reference_profile: str) -> dict[str, Any]:
    """Report per-kernel time ratios; never claim reproduced hardware."""
    if reference.status != 'complete' or candidate.status != 'complete':
        raise ValueError('cannot compare incomplete reports')
    if not candidate.profiles:
        raise ValueError('candidate has no measurements')
    if reference.array_mib != candidate.array_mib:
        raise ValueError('workload buffer sizes differ')
    if reference_profile not in reference.profiles:
        raise ValueError(f'reference profile {reference_profile!r} is missing')
    baseline = reference.profiles[reference_profile]
    rows: dict[str, Any] = {}
    for name, measured in candidate.profiles.items():
        if (set(baseline.cases) != set(CASE_NAMES)
                or set(measured.cases) != set(CASE_NAMES)):
            raise ValueError('workload cases are missing or different')
        ratios = {}
        for case in CASE_NAMES:
            left, right = baseline.cases[case], measured.cases[case]
            if (left.payload_bytes, left.flops) != (
                right.payload_bytes, right.flops
            ):
                raise ValueError(f'workload units differ: {case}')
            ratios[case] = right.seconds / left.seconds
        rows[name] = {
            'ratios': ratios,
            'worst_deviation': max(abs(x - 1) for x in ratios.values()),
            'software_differs': (
                baseline.numpy_version != measured.numpy_version
                or baseline.blas != measured.blas
            ),
        }
    return {'reference': reference.label, 'candidate': candidate.label,
            'reference_profile': reference_profile,
            'ratio_meaning': 'candidate/reference time; <1 means faster',
            'profiles': rows,
            'collector_differs': (reference.host.get('collector_sha256')
                                  != candidate.host.get('collector_sha256')),
            'python_differs': (reference.host.get('python')
                               != candidate.host.get('python')),
            'limit': 'Matching kernels does not prove application equivalence',
            }


def parser() -> argparse.ArgumentParser:
    """Describe collection and comparison without installing dependencies."""
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='mode', required=True)
    for mode in ('measure', '_worker'):
        command = commands.add_parser(mode)
        command.add_argument('--label', default='anonymous')
        command.add_argument('--threads', nargs='+')
        command.add_argument('--array-mib', type=int, default=256)
        command.add_argument('--repeats', type=int, default=5)
        command.add_argument('--sample-ms', type=int, default=50)
        command.add_argument('--budget', type=int, default=60)
    command = commands.add_parser('compare')
    command.add_argument('reference', type=Path)
    command.add_argument('candidate', type=Path)
    command.add_argument('--reference-profile', default='default')
    return result


def main(argv: list[str] | None = None) -> int:
    """Print a JSON report; errors and progress go to stderr."""
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.mode == 'compare':
            output = compare(read_report(args.reference),
                             read_report(args.candidate),
                             args.reference_profile)
            print(json.dumps(output, indent=2, allow_nan=False))
            return 0
        for key, low, high in [('array_mib', 1, 1024), ('repeats', 1, 10),
                               ('sample_ms', 1, 1000), ('budget', 1, 600)]:
            if not low <= getattr(args, key) <= high:
                flag = key.replace("_", "-")
                raise ValueError(f'--{flag} must be {low}..{high}')
        allowed = len(os.sched_getaffinity(0))
        if args.threads is None:
            args.threads = ['default'] + [str(n) for n in (1, 4, 8)
                                          if n <= allowed]
        for profile in args.threads:
            if profile != 'default' and (not profile.isascii()
                                         or not profile.isdecimal()
                                         or not 1 <= int(profile) <= allowed):
                raise ValueError(f'threads must be default or 1..{allowed}')
        if args.mode == '_worker':
            output = worker(args).model_dump()
            status = 0
        else:
            report = collect(args)
            output = report.model_dump()
            status = 0 if report.status == 'complete' else 1
        print(json.dumps(output, indent=2, allow_nan=False))
        return status
    except (OSError, ValueError, ImportError) as error:
        print(f'hardware probe: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
