"""校舎PCの実行環境を調査し、公開可能な匿名テキストレポートを作る。

inspect（既定）は標準ライブラリと Linux の標準情報源だけを読む軽量調査。
benchmark は明示指定時のみ、配布 llm_sdk の公開 API で CPU 推論を測る。
thermal の連続負荷と安全停止の引数は thermal --help を参照する。
モデルのダウンロードは --allow-download を付けない限り行わない。

公開レポートには許可したキーと安全な文字種の値だけを出力し、
書き込み前に hostname・username・IP/MAC 等の混入を検査する。
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import hmac
import importlib.metadata
import importlib.util
import multiprocessing
import os
import platform
import re
import resource
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from multiprocessing.connection import Connection
from typing import Callable

# Preserve direct script invocation as well as python -m tools.<name>.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.json_io import loads  # noqa: E402

SCHEMA_VERSION = '1'
BENCH_PROFILE = 'v1'
MODEL_NAME = 'Qwen/Qwen3-0.6B'
MODEL_DIR = 'models--Qwen--Qwen3-0.6B'
REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / 'research' / 'campus'
HMAC_ENV = 'CAMPUS_SURVEY_HMAC_KEY'
# 鍵はこの PC のリポジトリ内、Git 管理外（.gitignore の .cache/）に置く。
HMAC_KEY_FILE = REPO_ROOT / '.cache' / 'campus-survey' / 'hmac.key'
THREAD_ENVS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
DEPENDENCIES = ('torch', 'transformers', 'huggingface-hub', 'tokenizers',
                'safetensors', 'numpy', 'pydantic')
ISA_FLAGS = ('avx', 'avx2', 'fma', 'f16c', 'avx512f', 'avx512_vnni',
             'avx512_bf16', 'avx_vnni', 'amx_tile')
# 推定値の前提（実測ではない）。レポートにもそのまま記録する。
ASSUMED_PROMPT_TOKENS = 300
ASSUMED_GEN_TOKENS = 40
BASE_LENGTH = 64
FILLER = 'The quick brown fox jumps over the lazy dog. '

Report = dict[str, str]

PC_ID = re.compile(r'[a-z0-9][a-z0-9-]{2,31}')
META_KEY = re.compile(r'[a-z][a-z0-9_]{0,31}')
# '/' '~' '\\' 引用符などを禁止し、パスや任意文字列が出力されないようにする。
SAFE_VALUE = re.compile(r'[A-Za-z0-9 ._+\-(),=%@#]{0,120}')
ERROR_TYPE = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,60}')
FIXED_KEYS = frozenset({
    'schema_version', 'run_id', 'mode', 'pc_id', 'pc_id_method',
    'script_sha256',
    'machine.vendor', 'machine.product',
    'os.name', 'os.system', 'os.kernel', 'os.arch',
    'cpu.model', 'cpu.physical_cores', 'cpu.logical_cpus',
    'cpu.affinity_cpus', 'cpu.cgroup_quota_cpus', 'cpu.cgroup_cpuset',
    'cpu.governor', 'cpu.scaling_driver', 'cpu.energy_perf_pref',
    'cpu.turbo', 'cpu.smt_active', 'cpu.min_mhz', 'cpu.max_mhz',
    'cpu.isa_flags', 'cpu.loadavg',
    'env.omp_num_threads', 'env.mkl_num_threads',
    'env.openblas_num_threads',
    'memory.total_gib', 'memory.available_gib', 'memory.swap_total_gib',
    'memory.cgroup_max_gib', 'memory.cgroup_high_gib', 'memory.rlimit_as',
    'gpu.nvidia_driver',
    'python.version', 'python.implementation', 'python.in_venv',
    'uv.version',
    'sdk.present', 'sdk.matches_baseline', 'sdk.importable',
    'deps.transformers_supports_qwen3',
    'model.cache', 'model.cache_location', 'model.cache_complete',
    'model.cache_size_mib', 'model.locations_checked',
    'bench.status.result', 'bench.status.reason', 'bench.status.stage',
    'bench.status.error_type',
    'bench.config.profile', 'bench.config.device', 'bench.config.lengths',
    'bench.config.repeats', 'bench.config.gen_tokens',
    'bench.config.base_length', 'bench.config.budget_s',
    'bench.config.threads', 'bench.config.download_allowed',
    'bench.config.model_cached_before_run',
})
DYNAMIC_KEYS = (
    re.compile(r'meta\.[a-z][a-z0-9_]{0,31}'),
    re.compile(r'deps\.[a-z_]+'),
    re.compile(r'disk\.[a-z_]+\.(free_gib|total_gib)'),
    re.compile(r'bench\.measured\.[a-z0-9_]+'),
    re.compile(r'bench\.estimated\.[a-z0-9_]+'),
    re.compile(r'bench\.env\.[a-z0-9_]+'),
)
IPV4 = re.compile(r'(?<![\d.])\d{1,3}(\.\d{1,3}){3}(?![\d.])')
MAC = re.compile(r'(?i)\b[0-9a-f]{2}([:-][0-9a-f]{2}){5}\b')
IPV6 = re.compile(r'(?i)\b[0-9a-f]{0,4}(:[0-9a-f]{0,4}){2,7}\b')
EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
# 版番号を記録するキー。4 要素の版（例: WSL2 の kernel 6.6.87.2-...）は
# IPv4 と同じ形になるので、版番号の形式に合う値だけ IPv4 検査から外す。
VERSION_KEY = re.compile(
    r'os\.kernel|python\.version|uv\.version|deps\.[a-z_]+')
VERSION_VALUE = re.compile(r'\d+(\.\d+)+([-+_.~][A-Za-z0-9._+~-]*)?')
# 上限時間を過ぎた子プロセスに SIGTERM を送ってから SIGKILL までの猶予。
KILL_GRACE_S = 5.0
# 測定中に温度・負荷を記録する間隔。
MONITOR_INTERVAL_S = 0.5
CPU_SENSORS = ('coretemp', 'k10temp', 'zenpower', 'cpu_thermal')


class LeakError(ValueError):
    """公開レポートに出してはいけない値を検出した。"""


def unavailable(reason: str) -> str:
    """未取得値を、固定語彙の理由付きで表す。"""
    return f'unavailable ({reason})'


def reason_for(exc: BaseException) -> str:
    """例外を理由語に変換する（例外メッセージは保存しない）。"""
    if isinstance(exc, FileNotFoundError):
        return 'not_found'
    if isinstance(exc, PermissionError):
        return 'permission_denied'
    if isinstance(exc, ImportError):
        return 'not_installed'
    if isinstance(exc, NotImplementedError):
        return 'not_supported'
    if isinstance(exc, subprocess.TimeoutExpired):
        return 'timeout'
    if isinstance(exc, (ValueError, IndexError, KeyError)):
        return 'parse_error'
    return 'error'


def safe(function: Callable[[], str]) -> str:
    """1項目を取得し、失敗時は未取得として返す。"""
    try:
        return function()
    except Exception as exc:
        return unavailable(reason_for(exc))


def read(path: str | Path) -> str:
    """テキストファイルを読み、前後の空白を除く。"""
    return Path(path).read_text(encoding='utf-8', errors='replace').strip()


def gib(n_bytes: float) -> str:
    """バイト数を GiB 表記にする。"""
    return f'{n_bytes / 2**30:.1f}'


def yes_no(value: bool) -> str:
    """真偽値を yes/no にする。"""
    return 'yes' if value else 'no'


# --- 匿名ID・メタデータ ---------------------------------------------------

def sensitive_tokens() -> set[str]:
    """レポートに現れてはいけない端末・利用者固有の文字列を集める。"""
    tokens = {socket.gethostname(), platform.node(), Path.home().name}
    for getter in (getpass.getuser, lambda: os.environ.get('USER', ''),
                   lambda: os.environ.get('LOGNAME', '')):
        try:
            tokens.add(getter())
        except Exception:
            pass
    tokens |= {token.split('.')[0] for token in tokens}
    # 短すぎる文字列は誤検出が多いので対象外にする。
    return {token.lower() for token in tokens if len(token) >= 3}


def validate_pc_id(pc_id: str) -> str:
    """利用者指定の匿名IDを検証する。hostname 等そのものは拒否する。"""
    if not PC_ID.fullmatch(pc_id):
        raise ValueError('pc-id must match [a-z0-9][a-z0-9-]{2,31}')
    for token in sensitive_tokens():
        if token in pc_id or pc_id in token:
            raise ValueError('pc-id must not contain hostname or username')
    return pc_id


def hmac_pc_id(key: str, hostname: str) -> str:
    """秘密鍵付き HMAC-SHA256 で hostname から安定した匿名IDを作る。"""
    if len(key) < 16:
        raise ValueError('HMAC key must be at least 16 characters')
    normalized = hostname.strip().lower().split('.')[0]
    if not normalized:
        raise ValueError('hostname is empty')
    digest = hmac.new(key.encode(), normalized.encode(), hashlib.sha256)
    return 'h' + digest.hexdigest()[:12]


def init_hmac_key(path: Path) -> None:
    """ランダムな鍵を所有者のみ読める権限で作る。既存の鍵は上書きしない。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError('HMAC key file already exists') from None
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(secrets.token_hex(32) + '\n')


def read_hmac_key(path: Path) -> str:
    """鍵ファイルを優先し、無ければ環境変数か端末入力から読む。"""
    if path.is_file():
        if path.stat().st_mode & 0o077:
            print('warning: HMAC key file is readable by others; '
                  'run chmod 600 on it', file=sys.stderr)
        return read(path)
    key = os.environ.get(HMAC_ENV)
    if key:
        return key
    if not sys.stdin.isatty():
        raise ValueError('no HMAC key: run --init-hmac-key, '
                         f'or set {HMAC_ENV}')
    return getpass.getpass('HMAC key (not stored): ')


def parse_meta(items: list[str]) -> Report:
    """--meta key=value を検証してレポート項目にする。"""
    meta: Report = {}
    for item in items:
        key, sep, value = item.partition('=')
        if not sep or not META_KEY.fullmatch(key):
            raise ValueError('meta must be key=value, key [a-z][a-z0-9_]*')
        if not value or len(value) > 64 or not SAFE_VALUE.fullmatch(value):
            raise ValueError(f'meta {key}: value has unsafe characters')
        meta[f'meta.{key}'] = value
    return meta


# --- inspect -------------------------------------------------------------

def cgroup_dirs(root: Path = Path('/sys/fs/cgroup')) -> list[Path]:
    """自プロセスの cgroup v2 ディレクトリから root までを返す。"""
    for line in read('/proc/self/cgroup').splitlines():
        hierarchy, _, rest = line.partition(':')
        controllers, _, relative = rest.partition(':')
        if hierarchy == '0' and controllers == '':
            path = root / relative.lstrip('/')
            return [path, *[p for p in path.parents if p.is_relative_to(root)]]
    raise NotImplementedError('cgroup v2 is not in use')


def cgroup_values(filename: str) -> list[str]:
    """階層をたどり、存在する cgroup 制御ファイルの値を集める。"""
    values = []
    for directory in cgroup_dirs():
        try:
            values.append(read(directory / filename))
        except FileNotFoundError:
            continue
    if not values:
        raise FileNotFoundError(filename)
    return values


def cgroup_cpu_quota() -> str:
    """cpu.max から最も厳しい CPU 割当量を求める。"""
    limits = []
    for value in cgroup_values('cpu.max'):
        quota, period = value.split()
        if quota != 'max':
            limits.append(int(quota) / int(period))
    return f'{min(limits):.2f}' if limits else 'max'


def cgroup_memory(filename: str) -> str:
    """memory.max / memory.high の最も厳しい値を GiB で返す。"""
    limits = [int(v) for v in cgroup_values(filename) if v != 'max']
    return gib(min(limits)) if limits else 'max'


def physical_cores() -> str:
    """sysfs のトポロジから (package, core) の組を数える。"""
    cores = set()
    for topology in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/topology'):
        cores.add((read(topology / 'physical_package_id'),
                   read(topology / 'core_id')))
    if not cores:
        raise FileNotFoundError('topology')
    return str(len(cores))


def cpuinfo_field(name: str) -> str:
    """/proc/cpuinfo の最初の該当フィールドを返す。"""
    for line in read('/proc/cpuinfo').splitlines():
        key, sep, value = line.partition(':')
        if sep and key.strip() == name:
            return value.strip()
    raise KeyError(name)


def cpu_flags() -> str:
    """推論性能に関係する命令セット拡張の有無だけを返す。"""
    present = set(cpuinfo_field('flags').split())
    return ' '.join(flag for flag in ISA_FLAGS if flag in present) or 'none'


def cpu_turbo() -> str:
    """intel_pstate または cpufreq の boost 設定を返す。"""
    try:
        no_turbo = read('/sys/devices/system/cpu/intel_pstate/no_turbo')
        return 'disabled' if no_turbo == '1' else 'enabled'
    except FileNotFoundError:
        boost = read('/sys/devices/system/cpu/cpufreq/boost')
        return 'enabled' if boost == '1' else 'disabled'


def cpufreq(name: str) -> str:
    """cpu0 の cpufreq 設定を返す。"""
    return read(f'/sys/devices/system/cpu/cpu0/cpufreq/{name}')


def thread_env(name: str) -> str:
    """スレッド数系の環境変数は数値か否かだけを記録する。"""
    value = os.environ.get(name)
    if value is None:
        return 'unset'
    return value if value.isdigit() else 'set (non-numeric)'


def meminfo(name: str) -> str:
    """/proc/meminfo の kB 値を GiB にする。"""
    for line in read('/proc/meminfo').splitlines():
        key, _, value = line.partition(':')
        if key == name:
            return gib(int(value.split()[0]) * 1024)
    raise KeyError(name)


def rlimit_as() -> str:
    """アドレス空間の上限を返す。"""
    soft, _ = resource.getrlimit(resource.RLIMIT_AS)
    return 'unlimited' if soft == resource.RLIM_INFINITY else gib(soft)


def os_name() -> str:
    """/etc/os-release の PRETTY_NAME を返す。"""
    for line in read('/etc/os-release').splitlines():
        if line.startswith('PRETTY_NAME='):
            return line.partition('=')[2].strip('"')
    raise KeyError('PRETTY_NAME')


def uv_version() -> str:
    """uv --version の出力から版番号だけを取り出す。"""
    executable = shutil.which('uv')
    if executable is None:
        return unavailable('not_installed')
    output = subprocess.run(
        [executable, '--version'], capture_output=True, text=True,
        timeout=10, check=True,
    ).stdout
    match = re.search(r'\buv (\d+\.\d+\.\d+)', output)
    if match is None:
        raise ValueError('unexpected uv output')
    return match.group(1)


def sdk_matches_baseline() -> str:
    """配布 SDK が CI のレビュー済みハッシュと一致するか確かめる。"""
    baseline = loads(read(REPO_ROOT / '.github/ci/sdk-baseline.json'))
    for name, expected in baseline['files'].items():
        actual = hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            return 'no'
    return 'yes'


def sdk_importable() -> str:
    """現在のインタプリタで SDK とその依存が import 可能か（import しない）。"""
    spec = importlib.util.find_spec('llm_sdk')
    # リポジトリ直下の llm_sdk/ は名前空間パッケージとして誤検出されうる。
    if spec is None or not (spec.origin or '').endswith('__init__.py'):
        return 'no (sdk not installed)'
    missing = [name for name in ('torch', 'transformers', 'huggingface_hub')
               if importlib.util.find_spec(name) is None]
    return 'no (missing ' + ' '.join(missing) + ')' if missing else 'yes'


def dependency_version(name: str) -> str:
    """配布パッケージの版を返す（import はしない）。"""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return unavailable('not_installed')


def transformers_supports_qwen3() -> str:
    """Qwen3 には transformers 4.51 以上が必要。"""
    version = importlib.metadata.version('transformers')
    numbers = re.match(r'(\d+)\.(\d+)', version)
    if numbers is None:
        raise ValueError(version)
    major, minor = int(numbers.group(1)), int(numbers.group(2))
    return yes_no((major, minor) >= (4, 51))


def hub_candidates(hf_home: Path | None) -> list[tuple[str, Path]]:
    """モデルキャッシュの候補を (公開用ラベル, hub ディレクトリ) で返す。"""
    home = Path.home()
    candidates: list[tuple[str, Path]] = []
    if hf_home is not None:
        candidates.append(('explicit', hf_home / 'hub'))
    if os.environ.get('HF_HUB_CACHE'):
        candidates.append(('env_hf_hub_cache',
                           Path(os.environ['HF_HUB_CACHE'])))
    if os.environ.get('HF_HOME'):
        candidates.append(('env_hf_home', Path(os.environ['HF_HOME']) / 'hub'))
    candidates.append(('repo', REPO_ROOT / '.cache/huggingface/hub'))
    # 42 の goinfre / sgoinfre 配下は浅い階層だけ探す。
    for label in ('goinfre', 'sgoinfre'):
        base = home / label
        for pattern in (MODEL_DIR, f'*/{MODEL_DIR}', f'*/*/{MODEL_DIR}'):
            try:
                found = sorted(base.glob(pattern))
            except OSError:
                found = []
            candidates += [(label, path.parent) for path in found]
    candidates.append(('default', home / '.cache/huggingface/hub'))
    return candidates


def find_model(hf_home: Path | None) -> tuple[str, Path, Path] | None:
    """完全なスナップショットを持つ最初の候補を返す。"""
    for label, hub in hub_candidates(hf_home):
        for snapshot in sorted((hub / MODEL_DIR / 'snapshots').glob('*')):
            if snapshot_complete(snapshot):
                return label, hub, snapshot
    return None


def snapshot_complete(snapshot: Path) -> bool:
    """SDK のロードに必要なファイルが揃っているか。"""
    return (all((snapshot / name).is_file()
                for name in ('config.json', 'tokenizer.json'))
            and any(p.is_file() for p in snapshot.glob('*.safetensors')))


def model_cache(hf_home: Path | None) -> Report:
    """モデルキャッシュの有無と場所（ラベルのみ）を返す。"""
    candidates = hub_candidates(hf_home)
    labels = {label for label, _ in candidates} | {'goinfre', 'sgoinfre'}
    report = {'model.locations_checked': ' '.join(sorted(labels))}
    found = find_model(hf_home)
    if found is None:
        partial = any((hub / MODEL_DIR).exists() for _, hub in candidates)
        report.update({
            'model.cache': 'partial' if partial else 'not_found',
            'model.cache_location': 'none',
            'model.cache_complete': 'no',
        })
        return report
    label, _, snapshot = found
    # 実体は共有 blobs/ にあることがあるので、スナップショットの symlink を
    # たどって数える。
    size = sum(p.stat().st_size for p in snapshot.iterdir() if p.is_file())
    report.update({
        'model.cache': 'found', 'model.cache_location': label,
        'model.cache_complete': 'yes',
        'model.cache_size_mib': f'{size / 2**20:.0f}',
    })
    return report


def disk_report(hf_home: Path | None) -> Report:
    """主要な保存先の空き容量をラベル付きで返す。"""
    home = Path.home()
    locations = {
        'repo': REPO_ROOT, 'home': home, 'tmp': Path('/tmp'),
        'goinfre': home / 'goinfre', 'sgoinfre': home / 'sgoinfre',
        'uv_cache': Path(os.environ.get('UV_CACHE_DIR',
                                        home / '.cache/uv')),
    }
    if hf_home is not None:
        locations['hf_home'] = hf_home
    report: Report = {}
    for label, path in locations.items():
        # 未作成のキャッシュは、それを置くファイルシステムで測る。
        while not path.exists() and path != path.parent:
            path = path.parent
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            report[f'disk.{label}.free_gib'] = unavailable(reason_for(exc))
            continue
        report[f'disk.{label}.free_gib'] = gib(usage.free)
        report[f'disk.{label}.total_gib'] = gib(usage.total)
    return report


def inspect(hf_home: Path | None = None) -> Report:
    """軽量な環境調査。SDK・モデルは import もロードもしない。"""
    report: Report = {
        'machine.vendor': safe(lambda: read('/sys/class/dmi/id/sys_vendor')),
        'machine.product': safe(
            lambda: read('/sys/class/dmi/id/product_name')),
        'os.name': safe(os_name),
        'os.system': platform.system(),
        'os.kernel': platform.release(),
        'os.arch': platform.machine(),
        'cpu.model': safe(lambda: cpuinfo_field('model name')),
        'cpu.physical_cores': safe(physical_cores),
        'cpu.logical_cpus': str(os.cpu_count() or unavailable('error')),
        'cpu.affinity_cpus': safe(lambda: str(len(os.sched_getaffinity(0)))),
        'cpu.cgroup_quota_cpus': safe(cgroup_cpu_quota),
        'cpu.cgroup_cpuset': safe(
            lambda: cgroup_values('cpuset.cpus.effective')[0]),
        'cpu.governor': safe(lambda: cpufreq('scaling_governor')),
        'cpu.scaling_driver': safe(lambda: cpufreq('scaling_driver')),
        'cpu.energy_perf_pref': safe(
            lambda: cpufreq('energy_performance_preference')),
        'cpu.turbo': safe(cpu_turbo),
        'cpu.smt_active': safe(
            lambda: read('/sys/devices/system/cpu/smt/active')),
        'cpu.min_mhz': safe(
            lambda: str(int(cpufreq('cpuinfo_min_freq')) // 1000)),
        'cpu.max_mhz': safe(
            lambda: str(int(cpufreq('cpuinfo_max_freq')) // 1000)),
        'cpu.isa_flags': safe(cpu_flags),
        'cpu.loadavg': safe(
            lambda: ' '.join(f'{x:.2f}' for x in os.getloadavg())),
    }
    for name in THREAD_ENVS:
        report[f'env.{name.lower()}'] = thread_env(name)
    report.update({
        'memory.total_gib': safe(lambda: meminfo('MemTotal')),
        'memory.available_gib': safe(lambda: meminfo('MemAvailable')),
        'memory.swap_total_gib': safe(lambda: meminfo('SwapTotal')),
        'memory.cgroup_max_gib': safe(lambda: cgroup_memory('memory.max')),
        'memory.cgroup_high_gib': safe(lambda: cgroup_memory('memory.high')),
        'memory.rlimit_as': safe(rlimit_as),
        'gpu.nvidia_driver': yes_no(Path('/proc/driver/nvidia').exists()),
        'python.version': platform.python_version(),
        'python.implementation': platform.python_implementation(),
        'python.in_venv': yes_no(sys.prefix != sys.base_prefix),
        'uv.version': safe(uv_version),
        'sdk.present': yes_no(
            (REPO_ROOT / 'llm_sdk/llm_sdk/__init__.py').is_file()),
        'sdk.matches_baseline': safe(sdk_matches_baseline),
        'sdk.importable': safe(sdk_importable),
    })
    for name in DEPENDENCIES:
        report[f'deps.{name.replace("-", "_")}'] = dependency_version(name)
    report['deps.transformers_supports_qwen3'] = safe(
        transformers_supports_qwen3)
    report.update(safe_report(lambda: model_cache(hf_home), 'model.cache'))
    report.update(safe_report(lambda: disk_report(hf_home), 'disk.all'))
    return report


def safe_report(function: Callable[[], Report], key: str) -> Report:
    """複数項目を返す収集処理の失敗を1項目の未取得にまとめる。"""
    try:
        return function()
    except Exception as exc:
        return {key: unavailable(reason_for(exc))}


# --- benchmark -----------------------------------------------------------

def rss_mib(field: str) -> str:
    """/proc/self/status の VmRSS / VmHWM を MiB で返す。"""
    for line in read('/proc/self/status').splitlines():
        key, _, value = line.partition(':')
        if key == field:
            return f'{int(value.split()[0]) / 1024:.0f}'
    raise KeyError(field)


def seconds(value: float) -> str:
    """秒数を一定精度の文字列にする。"""
    return f'{value:.4f}'


def linear_fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    """最小二乗法で time = intercept + slope * length を求める。"""
    xs = [float(x) for x, _ in points]
    ys = [y for _, y in points]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError('need at least two distinct lengths')
    slope = sum((x - mean_x) * (y - mean_y)
                for x, y in zip(xs, ys)) / denominator
    return mean_y - slope * mean_x, slope


def estimate(points: list[tuple[int, float]], load_s: float) -> Report:
    """実測点から推定値を作る。前提はすべてレポートに記録する。"""
    intercept, slope = linear_fit(points)
    n_prompts = len(loads(read(
        REPO_ROOT / 'data/input/function_calling_tests.json')))
    # SDK は KV キャッシュを持たないため、各生成ステップで全文を再計算する。
    per_prompt = sum(intercept + slope * (ASSUMED_PROMPT_TOKENS + step)
                     for step in range(ASSUMED_GEN_TOKENS))
    return {
        'bench.estimated.method': 'linear fit of measured median logits',
        'bench.estimated.logits_fixed_s': seconds(intercept),
        'bench.estimated.logits_per_token_s': f'{slope:.6f}',
        'bench.estimated.assumed_prompts': str(n_prompts),
        'bench.estimated.assumed_prompt_tokens': str(ASSUMED_PROMPT_TOKENS),
        'bench.estimated.assumed_gen_tokens': str(ASSUMED_GEN_TOKENS),
        'bench.estimated.per_prompt_s': seconds(per_prompt),
        'bench.estimated.sample_total_s': seconds(
            load_s + n_prompts * per_prompt),
    }


def benchmark(args: argparse.Namespace) -> tuple[Report, int]:
    """CPU 推論を測る。キャッシュが無ければロード前に中止する。"""
    lengths = sorted(set(args.lengths))
    report: Report = {
        'bench.config.profile': BENCH_PROFILE,
        'bench.config.device': 'cpu',
        'bench.config.lengths': ' '.join(map(str, lengths)),
        'bench.config.base_length': str(BASE_LENGTH),
        'bench.config.repeats': str(args.repeats),
        'bench.config.gen_tokens': str(args.gen_tokens),
        'bench.config.budget_s': str(args.budget),
        'bench.config.threads': str(args.threads or 'default'),
        'bench.config.download_allowed': yes_no(args.allow_download),
    }
    found = find_model(args.hf_home)
    # 未キャッシュでダウンロードした回の model_load_s は比較に使えない。
    report['bench.config.model_cached_before_run'] = yes_no(found is not None)
    if found is None and not args.allow_download:
        print('model Qwen/Qwen3-0.6B is not cached; benchmark skipped. '
              'Use --hf-home or --allow-download explicitly.',
              file=sys.stderr)
        report.update({'bench.status.result': 'skipped',
                       'bench.status.reason': 'model_not_cached'})
        return report, 3
    if importlib.util.find_spec('torch') is None or \
            importlib.util.find_spec('transformers') is None:
        print('torch/transformers are not importable; run benchmark in '
              'the SDK environment (see research/campus/README.md).',
              file=sys.stderr)
        report.update({'bench.status.result': 'skipped',
                       'bench.status.reason': 'sdk_dependencies_missing'})
        return report, 3
    if not args.allow_download:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if found is not None:
        os.environ['HF_HUB_CACHE'] = str(found[1])
    elif args.hf_home is not None:
        os.environ['HF_HOME'] = str(args.hf_home)
    if args.threads:
        for name in THREAD_ENVS:
            os.environ[name] = str(args.threads)
    return supervise(args, lengths, report)


def cpu_temp_c() -> float:
    """CPU 温度センサー（hwmon）の最高値を返す。"""
    temps = []
    for hwmon in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            if read(hwmon / 'name') not in CPU_SENSORS:
                continue
        except OSError:
            continue
        temps += [int(read(t)) / 1000 for t in hwmon.glob('temp*_input')]
    if not temps:
        raise FileNotFoundError('cpu temperature sensor')
    return max(temps)


def throttle_count() -> int:
    """全 CPU のサーマルスロットリング発生回数の合計（Intel のみ）。"""
    paths = list(Path('/sys/devices/system/cpu').glob(
        'cpu[0-9]*/thermal_throttle/*_throttle_count'))
    if not paths:
        raise NotImplementedError('thermal_throttle')
    return sum(int(read(path)) for path in paths)


def system_busy_s() -> float:
    """システム全体の CPU 使用時間（秒）。個別のプロセスは読まない。"""
    values = [int(v) for v in read('/proc/stat').splitlines()[0].split()[1:]]
    idle = values[3] + values[4]
    return (sum(values[:8]) - idle) / os.sysconf('SC_CLK_TCK')


def process_cpu_s(pid: int) -> float:
    """自分が起動したプロセス（全スレッド）の CPU 使用時間（秒）。"""
    fields = read(f'/proc/{pid}/stat').rsplit(')', 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')


class RunMonitor:
    """benchmark 実行中の温度・スロットリング・他プロセス負荷を記録する。

    公平な比較のため、測定条件を乱した要因を後から見分けられるようにする。
    他プロセス負荷は「システム全体の使用量 - benchmark 子プロセスの使用量」。
    """

    def __init__(self, pid: int) -> None:
        """開始時点の値を記録する。取得できない値は None にする。"""
        self.pid = pid
        self.throttle_start = self._try(throttle_count)
        self.temp_start = self._try(cpu_temp_c)
        self.temp_max = self.temp_start
        self.first = self.last = self._usage()
        self.other_max: float | None = None

    @staticmethod
    def _try(function: Callable[[], float]) -> float | None:
        """取得に失敗したら None を返す。"""
        try:
            return float(function())
        except Exception:
            return None

    def _usage(self) -> tuple[float, float, float] | None:
        """(時刻, システム全体, benchmark) の CPU 使用時間。"""
        try:
            return (time.monotonic(), system_busy_s(),
                    process_cpu_s(self.pid))
        except Exception:
            return None

    def sample(self) -> None:
        """現在の温度と、前回からの他プロセス負荷を記録する。"""
        temp = self._try(cpu_temp_c)
        if temp is not None:
            self.temp_max = max(temp, self.temp_max or temp)
        now = self._usage()
        if now is None or self.last is None:
            return
        elapsed = now[0] - self.last[0]
        if elapsed < MONITOR_INTERVAL_S / 2:
            return
        other = max(0.0, (now[1] - self.last[1]) - (now[2] - self.last[2]))
        self.other_max = max(other / elapsed, self.other_max or 0.0)
        self.last = now

    def summary(self) -> Report:
        """記録した値を bench.env.* にまとめる。"""
        def fmt(value: float | None, reason: str) -> str:
            return unavailable(reason) if value is None else f'{value:.1f}'

        throttle_end = self._try(throttle_count)
        events = (None if throttle_end is None or self.throttle_start is None
                  else throttle_end - self.throttle_start)
        report = {
            'bench.env.temp_start_c': fmt(self.temp_start, 'not_found'),
            'bench.env.temp_max_c': fmt(self.temp_max, 'not_found'),
            'bench.env.throttle_events': (
                unavailable('not_supported') if events is None
                else str(int(events))),
            'bench.env.other_load_max_cores': fmt(self.other_max, 'error'),
        }
        if self.first is not None and self.last is not None and \
                self.last[0] > self.first[0]:
            elapsed = self.last[0] - self.first[0]
            bench = (self.last[2] - self.first[2]) / elapsed
            other = max(0.0, (self.last[1] - self.first[1]) / elapsed - bench)
            report['bench.env.other_load_avg_cores'] = f'{other:.1f}'
            report['bench.env.bench_cpu_avg_cores'] = f'{bench:.1f}'
        else:
            report['bench.env.other_load_avg_cores'] = unavailable('error')
            report['bench.env.bench_cpu_avg_cores'] = unavailable('error')
        return report


def supervise(
    args: argparse.Namespace, lengths: list[int], report: Report,
) -> tuple[Report, int]:
    """推論を子プロセスで実行し、上限時間を親プロセスから強制する。

    PyTorch のネイティブ処理中は Python のシグナルハンドラが動かないため、
    同じプロセス内のタイマーでは上限を守れない。親はパイプで実測値を
    逐次受け取り、締め切りを過ぎたら子を SIGTERM、猶予後に SIGKILL する。
    """
    # 親は torch を import していないので fork しても安全（Linux 前提）。
    context = multiprocessing.get_context('fork')
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_measure_in_child,
                            args=(args, lengths, sender), daemon=True)
    deadline = time.monotonic() + args.budget
    child.start()
    sender.close()
    monitor = RunMonitor(child.pid or 0)
    stage = 'sdk import'
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                report.update({'bench.status.result': 'partial',
                               'bench.status.reason': 'budget_exceeded',
                               'bench.status.stage': stage})
                return report, 1
            if not receiver.poll(min(remaining, MONITOR_INTERVAL_S)):
                monitor.sample()
                continue
            try:
                kind, *payload = receiver.recv()
            except EOFError:
                kind, payload = 'error', ['ChildExited']
            if kind == 'stage':
                stage = payload[0]
            elif kind == 'value':
                report[payload[0]] = payload[1]
            elif kind == 'done':
                report['bench.status.result'] = 'completed'
                report.update(payload[0])
                return report, 0
            else:
                name = payload[0]
                report.update({
                    'bench.status.result': 'failed',
                    'bench.status.stage': stage,
                    'bench.status.error_type': (
                        name if ERROR_TYPE.fullmatch(name) else 'error'),
                })
                return report, 1
    finally:
        # 子を止める前に最後の値を記録する（return 済みの report にも反映）。
        monitor.sample()
        report.update(monitor.summary())
        receiver.close()
        if child.is_alive():
            child.terminate()
            child.join(KILL_GRACE_S)
        if child.is_alive():
            child.kill()
        child.join()


def _measure_in_child(
    args: argparse.Namespace, lengths: list[int], conn: Connection,
) -> None:
    """子プロセス側。段階名と実測値を親へ逐次送る。"""
    try:
        estimated = run_measurements(
            args, lengths,
            lambda key, value: conn.send(('value', key, value)),
            lambda name: conn.send(('stage', name)),
        )
        conn.send(('done', estimated))
    except Exception as exc:
        if args.debug:
            traceback.print_exc()  # 端末表示のみ。レポートには保存しない。
        conn.send(('error', type(exc).__name__))
    finally:
        conn.close()


def run_measurements(
    args: argparse.Namespace, lengths: list[int],
    record: Callable[[str, str], None], step: Callable[[str], None],
) -> Report:
    """各測定を実行し、実測値は record で随時渡し、推定値を返す。

    step には実行中の段階名を渡し、失敗・時間切れ時の報告に使う。
    """
    started = time.perf_counter()
    from llm_sdk import Small_LLM_Model
    record('bench.measured.sdk_import_s',
           seconds(time.perf_counter() - started))
    record('bench.measured.rss_after_import_mib', safe(
        lambda: rss_mib('VmRSS')))

    step('model load')
    started = time.perf_counter()
    model = Small_LLM_Model(MODEL_NAME, device='cpu', trust_remote_code=False)
    load_s = time.perf_counter() - started
    record('bench.measured.model_load_s', seconds(load_s))
    record('bench.measured.rss_after_load_mib', safe(lambda: rss_mib('VmRSS')))

    step('encode')
    needed = max([*lengths, BASE_LENGTH]) + args.gen_tokens
    text = FILLER
    ids: list[int] = model.encode(text).tolist()[0]
    while len(ids) < needed:
        text *= 2
        ids = model.encode(text).tolist()[0]

    def timed_logits(input_ids: list[int]) -> tuple[float, list[float]]:
        begin = time.perf_counter()
        logits = model.get_logits_from_input_ids(input_ids)
        return time.perf_counter() - begin, logits

    step('first logits')
    first, logits = timed_logits(ids[:BASE_LENGTH])
    record('bench.measured.vocab_size', str(len(logits)))
    record(f'bench.measured.first_logits_len_{BASE_LENGTH}_s', seconds(first))

    step('repeated logits')
    repeated = [timed_logits(ids[:BASE_LENGTH])[0]
                for _ in range(args.repeats)]
    prefix = f'bench.measured.repeat_logits_len_{BASE_LENGTH}'
    record(f'{prefix}_min_s', seconds(min(repeated)))
    record(f'{prefix}_median_s', seconds(statistics.median(repeated)))
    record(f'{prefix}_max_s', seconds(max(repeated)))

    points: list[tuple[int, float]] = []
    for length in lengths:
        step(f'logits length {length}')
        times = [timed_logits(ids[:length])[0] for _ in range(args.repeats)]
        median = statistics.median(times)
        points.append((length, median))
        record(f'bench.measured.logits_len_{length}_median_s', seconds(median))

    step('generation')
    sequence = ids[:BASE_LENGTH]
    started = time.perf_counter()
    for _ in range(args.gen_tokens):
        logits = model.get_logits_from_input_ids(sequence)
        sequence.append(max(range(len(logits)), key=logits.__getitem__))
    total = time.perf_counter() - started
    record(f'bench.measured.generate_{args.gen_tokens}_tokens_s',
           seconds(total))
    record('bench.measured.generate_per_token_s', seconds(
        total / args.gen_tokens))
    record('bench.measured.peak_rss_mib', safe(lambda: rss_mib('VmHWM')))

    step('estimate')
    return safe_report(lambda: estimate(points, load_s),
                       'bench.estimated.status')


# --- 出力 ----------------------------------------------------------------

def key_allowed(key: str) -> bool:
    """許可リストにあるキーか。"""
    return key in FIXED_KEYS or any(p.fullmatch(key) for p in DYNAMIC_KEYS)


def render(report: Report) -> str:
    """許可キー・安全な値だけを key = value 形式で出力する。"""
    lines = []
    tokens = sensitive_tokens()
    for key, value in report.items():
        if not key_allowed(key):
            raise LeakError(f'key is not allowed: {key}')
        if not SAFE_VALUE.fullmatch(value):
            value = unavailable('unsafe_value')
        check_leaks(key, value, tokens)
        lines.append(f'{key} = {value}')
    return '\n'.join(lines) + '\n'


def check_leaks(key: str, value: str, tokens: set[str]) -> None:
    """端末固有情報やアドレス形式の文字列が混入していないか項目ごとに検査する。"""
    line = f'{key} = {value}'
    lowered = line.lower()
    for token in tokens:
        if re.search(rf'(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])',
                     lowered):
            raise LeakError('report contains hostname or username')
    is_version = (VERSION_KEY.fullmatch(key) is not None
                  and VERSION_VALUE.fullmatch(value) is not None)
    for name, pattern in (('IPv4', IPV4), ('MAC', MAC), ('IPv6', IPV6),
                          ('email', EMAIL)):
        if name == 'IPv4' and is_version:
            continue
        if pattern.search(line):
            raise LeakError(f'report contains an {name}-like value')


def write_report(directory: Path, pc_id: str, run_id: str, text: str) -> Path:
    """同じ端末の再測定は、同じファイルへ run 単位で追記する。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'pc-{pc_id}.txt'
    new = not path.exists()
    with path.open('a', encoding='utf-8') as stream:
        if new:
            stream.write('# Call Me Maybe campus PC survey '
                         '(public, anonymized)\n')
        stream.write(f'\n=== run {run_id} ===\n{text}')
    return path


def parse_lengths(value: str) -> list[int]:
    """'16,64,256' を正の整数リストにする。"""
    try:
        lengths = [int(part) for part in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError('comma-separated integers') from None
    if len(set(lengths)) < 2 or min(lengths) < 1 or max(lengths) > 2048:
        raise argparse.ArgumentTypeError('need >=2 distinct values in 1-2048')
    return lengths


def build_parser() -> argparse.ArgumentParser:
    """CLI 引数を定義する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='inspect',
                        choices=('inspect', 'benchmark'))
    ident = parser.add_mutually_exclusive_group(required=True)
    ident.add_argument('--pc-id', help='anonymous ID chosen by you')
    ident.add_argument('--pc-id-from-hostname', action='store_true',
                       help='HMAC of hostname with the local key file')
    ident.add_argument('--init-hmac-key', action='store_true',
                       help='create the local key file, print the ID, exit')
    parser.add_argument('--hmac-key-file', type=Path, default=HMAC_KEY_FILE,
                        help='default: .cache/campus-survey/hmac.key')
    parser.add_argument('--meta', action='append', default=[],
                        help='public metadata key=value (e.g. floor=2f)')
    parser.add_argument('--write', action='store_true',
                        help='append to research/campus/pc-<id>.txt')
    parser.add_argument('--output-dir', type=Path, default=REPORT_DIR)
    parser.add_argument('--hf-home', type=Path,
                        help='HF_HOME holding the cached model')
    bench = parser.add_argument_group('benchmark only')
    bench.add_argument('--allow-download', action='store_true')
    bench.add_argument('--budget', type=int, default=180,
                       help='hard time limit in seconds (default 180)')
    bench.add_argument('--lengths', type=parse_lengths,
                       default=[16, 64, 256])
    bench.add_argument('--repeats', type=int, default=3)
    bench.add_argument('--gen-tokens', type=int, default=8)
    bench.add_argument('--threads', type=int)
    bench.add_argument('--debug', action='store_true',
                       help='print tracebacks to stderr (never saved)')
    return parser


def main(argv: list[str] | None = None) -> int:
    """調査を実行してレポートを表示し、--write 時のみ保存する。"""
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == 'thermal':
        if not __package__:
            sys.path.insert(0, str(REPO_ROOT))
        from tools.thermal_survey import main as thermal_main
        return thermal_main(arguments[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 10 <= args.budget <= 900 or not 1 <= args.repeats <= 10 or \
            not 1 <= args.gen_tokens <= 64 or \
            (args.threads is not None and args.threads < 1):
        parser.error('budget 10-900, repeats 1-10, gen-tokens 1-64, '
                     'threads >= 1')
    try:
        if args.init_hmac_key:
            init_hmac_key(args.hmac_key_file)
            pc_id = hmac_pc_id(read(args.hmac_key_file),
                               socket.gethostname())
            print(f'created HMAC key; this PC is --pc-id-from-hostname '
                  f'-> {pc_id}')
            return 0
        if args.pc_id_from_hostname:
            pc_id, method = hmac_pc_id(read_hmac_key(args.hmac_key_file),
                                       socket.gethostname()), 'hmac'
        else:
            pc_id, method = validate_pc_id(args.pc_id), 'explicit'
        meta = parse_meta(args.meta)
    except ValueError as exc:
        parser.error(str(exc))
    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%SZ') + '-' + args.mode
    script = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report: Report = {
        'schema_version': SCHEMA_VERSION, 'run_id': run_id,
        'mode': args.mode, 'pc_id': pc_id, 'pc_id_method': method,
        'script_sha256': script[:12], **meta,
    }
    report.update(inspect(args.hf_home))
    status = 0
    if args.mode == 'benchmark':
        bench, status = benchmark(args)
        report.update(bench)
    try:
        text = render(report)
    except LeakError as exc:
        print(f'refused to output report: {exc}', file=sys.stderr)
        return 2
    print(text, end='')
    if args.write:
        path = write_report(args.output_dir, pc_id, run_id, text)
        shown = (path.relative_to(REPO_ROOT)
                 if path.is_relative_to(REPO_ROOT) else path.name)
        print(f'appended: {shown}', file=sys.stderr)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
