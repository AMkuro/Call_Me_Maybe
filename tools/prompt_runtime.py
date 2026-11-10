"""Evidence identity, pinned offline model caches and owned process cleanup."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import subprocess
import threading
from typing import Any, TypeVar

from src.json_io import loads

_T = TypeVar('_T')
_R = TypeVar('_R')
_CHILDREN: set[subprocess.Popen[Any]] = set()
_CHILD_LOCK = threading.Lock()
_STOP = threading.Event()


def has_oom(result: dict[str, Any]) -> bool:
    """Recognize startup and per-case GPU memory failures."""
    if (result.get('failure_kind') == 'gpu_oom'
            or 'out of memory' in str(result.get('error', '')).lower()):
        return True
    for name in ('trial', 'trial_attempt'):
        nested = result.get(name)
        if isinstance(nested, dict) and has_oom(nested):
            return True
    return any(has_oom(row) for row in result.get('records', []))


def accept_editor_result(
    path: Path, stamp: str, code: int,
) -> dict[str, Any]:
    """Reject stale, incomplete or failed editor output."""
    data: dict[str, Any] = loads(path.read_text()) if path.exists() else {}
    valid = (code == 0 and data.get('fingerprint') == stamp
             and data.get('finished') is True
             and data.get('trial', {}).get('finished') is True
             and re.fullmatch(r'[a-f0-9]{64}', data.get('candidate_id', '')))
    if valid:
        status = 'evaluation_failed' if has_oom(data) else 'evaluated'
        return dict(data, status=status)
    failed: dict[str, Any] = {
        'status': 'editor_failed', 'exit_code': code,
        'error': data.get('error', 'missing, stale or incomplete result'),
    }
    candidate_id = data.get('candidate_id', '')
    if (isinstance(candidate_id, str)
            and re.fullmatch(r'[a-f0-9]{64}', candidate_id)):
        failed['candidate_id'] = candidate_id
    return failed


def evidence_fingerprint(
    code_sha: str, candidate: Path, fixture: Path, runtime: dict[str, Any],
) -> str:
    """Bind evidence to actual weights, environment, data and instructions."""
    value = {'code_sha': code_sha, 'runtime': runtime,
             'candidate': hashlib.sha256(candidate.read_bytes()).hexdigest(),
             'fixture': hashlib.sha256(fixture.read_bytes()).hexdigest()}
    serialized = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()


def pin_model_cache(
    source: Path, view: Path, models: dict[str, str],
) -> dict[str, Any]:
    """Give the public SDK private refs/main pointing to immutable snapshots.

    This uses the documented Hugging Face cache layout, not SDK internals.
    Weights stay in the existing shared cache; no copy or download occurs.
    """
    pinned: dict[str, Any] = {}
    for model, revision in models.items():
        if (not re.fullmatch(r'[\w.-]+/[\w.-]+', model)
                or not re.fullmatch(r'[a-f0-9]{40}', revision)):
            raise ValueError('model ID and full revision SHA required')
        name = 'models--' + model.replace('/', '--')
        snapshot = source / 'hub' / name / 'snapshots' / revision
        required = {'config.json', 'tokenizer.json', 'tokenizer_config.json',
                    'vocab.json'}
        index = snapshot / 'model.safetensors.index.json'
        if index.is_file():
            required.add(index.name)
            required.update(loads(index.read_text())['weight_map'].values())
        else:
            required.add('model.safetensors')
        files = []
        for filename in sorted(required):
            if Path(filename).name != filename:
                raise ValueError('unexpected checkpoint file path')
            path = snapshot / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'incomplete {model}@{revision}: {filename}')
            files.append({'name': filename, 'bytes': path.stat().st_size,
                          'blob': path.resolve().name})
        repo = view / 'hub' / name
        (repo / 'refs').mkdir(parents=True, exist_ok=True)
        (repo / 'snapshots').mkdir(exist_ok=True)
        link = repo / 'snapshots' / revision
        if not link.exists():
            link.symlink_to(snapshot.resolve(), target_is_directory=True)
        if link.resolve() != snapshot.resolve():
            raise ValueError('pinned cache snapshot target changed')
        main = repo / 'refs/main'
        if main.exists() and main.read_text() != revision:
            raise ValueError('cannot change a pinned cache revision in place')
        main.write_text(revision)
        pinned[model] = {'revision': revision, 'files': files}
    return pinned


def environment_metadata() -> dict[str, Any]:
    """Record installed code versions, including the SDK actually imported."""
    packages = {}
    for name in ('torch', 'transformers', 'tokenizers', 'huggingface-hub',
                 'numpy', 'pydantic', 'llm-sdk', 'accelerate'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = 'not-installed'
    spec = util.find_spec('llm_sdk')
    sdk_hash = None
    if spec is not None and spec.origin is not None:
        sdk_hash = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    return {'python': platform.python_version(), 'executable': sys.executable,
            'packages': packages, 'sdk_source_sha256': sdk_hash}


def cancel_signal(signum: int, frame: Any) -> None:
    """Turn SIGTERM into the same cleanup route as a user interrupt."""
    _STOP.set()
    raise KeyboardInterrupt


def _terminate(process: subprocess.Popen[Any]) -> None:
    """Stop and reap an owned process group, including nested trials."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def stop_children() -> None:
    """Stop all owned groups after interruption."""
    _STOP.set()
    with _CHILD_LOCK:
        children = list(_CHILDREN)
    for child in children:
        _terminate(child)


def run_logged(
    command: list[str], log: Path, env: dict[str, str], timeout: int = 1200,
) -> int:
    """Run a fixed command, retaining logs and tracking its process group."""
    with log.open('w', encoding='utf-8') as stream:
        with _CHILD_LOCK:
            if _STOP.is_set():
                raise InterruptedError('experiment cancelled')
            process = subprocess.Popen(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
            _CHILDREN.add(process)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            _terminate(process)
            raise
        finally:
            with _CHILD_LOCK:
                _CHILDREN.discard(process)


def parallel(function: Callable[[_T], _R], items: list[_T], workers: int
             ) -> list[_R]:
    """Wait for workers while letting main stop their process groups."""
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        return list(pool.map(function, items))
    except BaseException:
        stop_children()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
