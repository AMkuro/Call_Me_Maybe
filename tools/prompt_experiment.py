"""Bounded GPU prompt search with controlled Git, CPU and held-out gates."""

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import tempfile
import subprocess
import sys
import time
from typing import Any
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from src import prompt
from src.json_io import loads
from src.pipeline import MAX_VALUE_STEPS
from src.schema import FunctionCall, FunctionDef, validate_number
from tools.prompt_candidates import (
    Candidate, save_candidate, score_records, select_candidate, write_json,
    eligible,
)
from tools.prompt_runtime import (
    accept_editor_result, cancel_signal, environment_metadata,
    evidence_fingerprint, has_oom, parallel, pin_model_cache, run_logged,
    stop_children,
)

DATA_REF = "8d63548988a37bd1862b7192076fea3b1e946154"
# Held-out regex set (ADR-0020), untracked and written by an isolated agent.
HELDOUT_REGEX = Path(".cache/heldout-regex/regex_heldout.json")
TARGET_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
OPTIMIZER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


class SearchState(BaseModel):
    """Durable phase boundary; selection never returns to an opened holdout."""

    model_config = ConfigDict(strict=True, extra='forbid')
    identity: str
    phase: Literal['search', 'selected', 'heldout', 'done'] = 'search'
    candidate_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    next_round: int = Field(default=0, ge=0, le=4)
    trials: list[dict[str, str]] = []


def read_state(path: Path, identity: str) -> dict[str, Any]:
    """Resume a fixed selection or restart an unused search."""
    if not path.exists():
        return SearchState(identity=identity).model_dump()
    state = SearchState.model_validate(loads(path.read_text())).model_dump()
    if state['identity'] != identity:
        if state['phase'] in {'heldout', 'done'}:
            raise ValueError('selection frozen; conditions changed')
        if state['phase'] == 'selected':
            return dict(state, identity=identity)
        return SearchState(
            identity=identity, trials=state['trials'],
            next_round=state['next_round']).model_dump()
    return state


def search_status(edits: list[dict[str, Any]]) -> str:
    """Separate failed proposals from valid candidates with no improvement."""
    evaluated = any(e.get('status') == 'evaluated' and not has_oom(e)
                    for e in edits)
    return 'no_improvement' if evaluated else 'search_failed'


def choose_update(
    local: str, remote: str, ancestor: Callable[[tuple[str, str]], bool],
) -> str:
    """Use the newer comparable ref; never guess between divergent sources."""
    if local == remote or ancestor((remote, local)):
        return local
    if ancestor((local, remote)):
        return remote
    raise ValueError("local/remote feat/implementation diverged; sync stopped")


def load_cached(path: Path, fingerprint: str) -> dict[str, Any] | None:
    """Reuse only an artifact for precisely the requested source/settings."""
    if not path.exists():
        return None
    result: dict[str, Any] = loads(path.read_text(encoding="utf-8"))
    return result if result.get("fingerprint") == fingerprint else None


def estimate_cpu(seconds: float) -> dict[str, float]:
    """Keep actual elapsed time and apply the owner's one-per-run offset."""
    return {"raw_seconds": seconds, "estimated_seconds": float(
        Decimal(str(seconds)) + Decimal(8))}


def git(*args: str) -> str:
    """Read or update the authorized experimental worktree."""
    return subprocess.check_output(["git", *args], text=True).strip()


def synchronize(log: Path) -> bool:
    """Merge upstream only at idle boundaries; fail closed on dirty state."""
    if git("branch", "--show-current") != "exp/local-prompt-improvement":
        raise ValueError("use the exp/local-prompt-improvement worktree")
    if git("status", "--porcelain"):
        raise ValueError("uncommitted changes prevent synchronization")
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run(["git", "fetch", "origin"], check=True,
                       stdout=stream, stderr=stream, timeout=120)
        target = choose_update(
            git("rev-parse", "feat/implementation"),
            git("rev-parse", "origin/feat/implementation"),
            lambda pair: subprocess.run(
                ["git", "merge-base", "--is-ancestor", *pair],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0,
        )
        if subprocess.run(["git", "merge-base", "--is-ancestor", target,
                           "HEAD"]).returncode == 0:
            return False
        merged = subprocess.run(["git", "merge", "--no-edit", target],
                                stdout=stream, stderr=stream)
        if merged.returncode:
            subprocess.run(["git", "merge", "--abort"], check=False,
                           stdout=stream, stderr=stream)
            raise ValueError("merge conflicted; original state restored")
        env = os.environ.copy()
        env.pop("UV_PROJECT_ENVIRONMENT", None)
        env.update(UV_OFFLINE="1")
        subprocess.run(["make", "install", "lint", "test"],
                       env=env, check=True,
                       stdout=stream, stderr=stream, timeout=600)
    return True


def fingerprint(candidate: Path, fixture: Path, env: dict[str, str]) -> str:
    """Use the actual immutable runtime record passed to every child."""
    provenance = loads(Path(env['CMM_PROMPT_PROVENANCE']).read_text())
    return evidence_fingerprint(provenance['code_sha'], candidate, fixture,
                                provenance['runtime'])


def editor_session(
    folder: Path, context: Path, store: Path, fixture: Path,
    optimizer: str, env: dict[str, str], serial_retry: bool = False,
    guard: list[Path] | None = None,
) -> dict[str, Any]:
    """Keep immutable attempts; reject stale or failed results."""
    context_data = loads(context.read_text())
    parent = store / f"{context_data['parent_id']}.json"
    stamp = hashlib.sha256((fingerprint(parent, fixture, env)
                            + context.read_text()).encode()).hexdigest()
    if not serial_retry:
        old_paths = sorted(folder.glob('attempt-*/result.json'),
                           key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for old_path in old_paths:
            old = load_cached(old_path, stamp)
            if old is not None:
                return old
    attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=folder))
    shutil.copyfile(context, attempt / 'context.json')
    output = attempt / 'editor.json'
    workspace = Path(tempfile.mkdtemp(prefix='cmm-prompt-'))
    templates = Path('research/prompt-improvement')
    tool_dir = workspace / '.opencode/tools'
    tool_dir.mkdir(parents=True)
    for template in templates.joinpath('tools').glob('*.ts'):
        shutil.copyfile(template, tool_dir / template.name)
    write_json(workspace / '.opencode/package.json', {
        'private': True, 'dependencies': {'@opencode-ai/plugin': '1.15.10'}})
    config = loads((templates / 'opencode.json').read_text())
    if config['model'] != 'local/' + optimizer:
        raise ValueError('OSS configuration and optimizer model differ')
    provenance = loads(Path(env['CMM_PROMPT_PROVENANCE']).read_text())
    served_id = optimizer + '@' + provenance['runtime']['models'][
        optimizer]['revision']
    models = config['provider']['local']['models']
    models[served_id] = models.pop(optimizer)
    config['model'] = config['small_model'] = 'local/' + served_id
    config['provider']['local']['options']['baseURL'] = env.get(
        'CMM_OPTIMIZER_URL', 'http://127.0.0.1:8011/v1')
    write_json(workspace / 'opencode.json', config)
    shutil.copyfile(templates / 'improver.txt', workspace / 'improver.txt')
    write_json(workspace / 'tool-config.json', {
        'store': str(store), 'fixture': str(fixture), 'output': str(output),
        'provenance': env['CMM_PROMPT_PROVENANCE'], 'fingerprint': stamp,
        'code_sha': provenance['code_sha'],
        'parent_id': context_data['parent_id'],
        'guard': [str(p) for p in [fixture, *(guard or [])]],
    })
    cache_home = str(Path('.cache/opencode/cache').resolve())
    child_env = dict(env, CMM_PYTHON=sys.executable, CMM_REPO=str(Path.cwd()),
                     OPENCODE_CONFIG=str(workspace / 'opencode.json'),
                     OPENCODE_CONFIG_DIR=str(workspace / '.opencode'),
                     OPENCODE_DISABLE_MODELS_FETCH='true',
                     XDG_CONFIG_HOME=str(workspace / '.state/config'),
                     XDG_DATA_HOME=str(workspace / '.state/data'),
                     XDG_STATE_HOME=str(workspace / '.state/state'),
                     XDG_CACHE_HOME=cache_home)
    for inherited in ('OPENCODE_CONFIG_CONTENT', 'OPENCODE_PERMISSION'):
        child_env.pop(inherited, None)
    write_json(attempt / 'workspace.json', {'workspace': str(workspace)})
    message = ('Improve one prompt candidate using this development context:\n'
               + json.dumps(context_data, default=validate_number))
    try:
        code = run_logged([
            'opencode', '--pure', 'run', '--dir', str(workspace), '--agent',
            'prompt-improver', '--format', 'json', message,
        ], attempt / 'editor.log', child_env)
        result = accept_editor_result(output, stamp, code)
    except subprocess.TimeoutExpired:
        result = {'status': 'editor_failed',
                  'error': 'editor deadline reached'}
    result['fingerprint'] = stamp
    result['artifact'] = str(attempt)
    if serial_retry:
        result['serial_retry'] = True
    if result['status'] == 'evaluated':
        candidate = store / f"{result['candidate_id']}.json"
        expected = fingerprint(candidate, fixture, env)
        if result['trial'].get('fingerprint') != expected:
            result = dict(result, status='editor_failed',
                          error='trial identity mismatch')
            result.pop('trial', None)
    write_json(attempt / 'result.json', result)
    return result


def run_trial(
    candidate: Path, fixture: Path, report: Path, env: dict[str, str],
) -> dict[str, Any]:
    """Run the fixed pipeline; reject incomplete cached reports."""
    stamp = fingerprint(candidate, fixture, env)
    cached = load_cached(report, stamp)
    if cached is not None and cached.get("finished"):
        return cached
    report.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "tools.prompt_trial", "--candidate",
               str(candidate), "--fixture", str(fixture), "--report",
               str(report), "--fingerprint", stamp]
    if report.exists() or report.with_suffix('.log').exists():
        suffix = f'-previous-{time.time_ns()}'
        for source in (report, report.with_suffix('.log')):
            if source.exists():
                source.rename(source.with_stem(source.stem + suffix))
    code = run_logged(command, report.with_suffix(".log"), env, timeout=900)
    data = loads(fixture.read_text())
    result = load_cached(report, stamp)
    if code or result is None or not result.get("finished"):
        result = {"fingerprint": stamp, "finished": False,
                  "error": f"worker failed with exit {code}",
                  "records": [] if result is None else result["records"]}
        result["score"] = score_records(data["cases"], result["records"])
        write_json(report, result)
    return result


def copy_calibrated(root: Path) -> list[Path]:
    """Read fixed fixture blobs without merging their source branch."""
    paths = []
    for suite in ("original", "transfer", "boolean", "difference",
                  "magnitude", "word_pairs", "timer_flags", "feature_flags"):
        path = root / f"calibrated-{suite}.json"
        path.write_bytes(subprocess.check_output([
            "git", "show", f"{DATA_REF}:tests/fixtures/calibrated/{suite}/"
            "measurement_fixture.json",
        ]))
        paths.append(path)
    return paths


def cpu_check(
    root: Path, candidate: Path, fixture: Path, env: dict[str, str],
    gpu_report: dict[str, Any], cache: Path,
) -> dict[str, Any]:
    """Measure the native CLI under the prescribed CPU scope."""
    data = loads(fixture.read_text())
    functions = root / "cpu-functions.json"
    inputs = root / "cpu-inputs.json"
    output = root / "cpu-calls.json"
    write_json(functions, data["functions"])
    write_json(inputs, [{"prompt": c["prompt"]} for c in data["cases"]])
    lab = root / "lab"
    lab.mkdir(exist_ok=True)
    if not (lab / "hf").exists():
        (lab / "hf").symlink_to(cache, target_is_directory=True)
    script = root / "cpu-measure.sh"
    shutil.copyfile("research/evo-x3/measure.sh", script)
    cpu_env = dict(env, CMM_LAB=str(lab), CMM_OMP_THREADS="6",
                   CMM_MODULE="tools.prompt_trial",
                   CMM_CPU_TELEMETRY=str(root / "cpu-runtime.json"),
                   CUDA_VISIBLE_DEVICES="",
                   HIP_VISIBLE_DEVICES="")
    cpu_env.pop("UV_PROJECT_ENVIRONMENT", None)
    code = run_logged([
        "bash", str(script), str(Path.cwd()), "selected-cpu", "--candidate",
        str(candidate), "--cli", "--functions_definition", str(functions),
        "--input", str(inputs), "--output", str(output),
    ], root / "cpu.log", cpu_env, timeout=600)
    raw = (root / "measurements/selected-cpu.txt").read_text()
    match = re.search(r"^wall_s=([0-9.]+)", raw, re.MULTILINE)
    if match is None:
        raise ValueError("CPU harness did not produce wall_s")
    records = []
    calls = loads(output.read_text()) if output.exists() else []
    for case in data["cases"]:
        matching = [call for call in calls if call["prompt"] == case["prompt"]]
        if len(matching) == 1:
            records.append({"id": case["id"], "call": matching[0]})
    definitions = {f["name"]: FunctionDef.model_validate(f)
                   for f in data["functions"]}
    score = score_records(data["cases"], records, definitions)
    normalized = [FunctionCall.model_validate(r["call"]).model_dump()
                  for r in records]
    gpu_calls = [r["call"] for r in gpu_report["records"] if "call" in r]
    identical = (json.dumps(normalized, sort_keys=True)
                 == json.dumps([FunctionCall.model_validate(c).model_dump()
                                for c in gpu_calls], sort_keys=True))
    result: dict[str, Any] = {
              **estimate_cpu(float(match[1])), "exit_code": code,
              "score": score, "gpu_calls_identical": identical,
              "profile": raw, "code_sha": git("rev-parse", "HEAD"),
              'runtime': loads((root / 'cpu-runtime.json').read_text())}
    result["passed"] = (
        code == 0 and identical
        and score["correct_ids"] == gpu_report["score"]["correct_ids"]
        and score["valid"] == score["total"]
        and result["estimated_seconds"] < 300
    )
    write_json(root / "cpu-result.json", result)
    return result


def editor_context(
    base: dict[str, Any], fixture: Path, candidate: Path, role: int,
    previous: list[dict[str, Any]], direction: str | None = None,
) -> dict[str, Any]:
    """Expose failed requests and observed outputs, never gold values."""
    data = loads(fixture.read_text())
    observed = {r["id"]: r for r in base["records"]}
    return {
        "current_prompt": loads(candidate.read_text()),
        "previous_attempts": previous,
        "development_functions": data["functions"],
        "failures": [{"id": case["id"], "prompt": case["prompt"],
                      "observed": {k: v for k, v in observed.get(
                          case["id"], {}).items() if k in {"call", "error"}}}
                     for case in data["cases"]
                     if case["id"] not in base["score"]["correct_ids"]],
        "parent_id": candidate.stem,
        "goal": direction or (
            "Clarify input argument roles; do not execute functions or return "
            "their results. Keep instructions concise and general."
            if role == 0 else
            "Improve exact extraction of signs, literal replacement symbols, "
            "and meaningful whitespace. Generalize beyond these examples."
        ),
    }


def main() -> int:
    """Run the finite search; resume a frozen candidate without retuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--optimizer', required=True)
    parser.add_argument('--optimizer-revision', default=OPTIMIZER_REVISION)
    parser.add_argument('--rounds', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--direction', type=Path,
                        help='relay memo from Claude Code; replaces the goal')
    args = parser.parse_args()
    direction = (args.direction.read_text(encoding='utf-8')
                 if args.direction else None)
    root, cache = args.output.resolve(), args.cache.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / 'state.json'
    summary_path = root / 'summary.json'
    signal.signal(signal.SIGTERM, cancel_signal)
    try:
        # Finished runs display their report without restarting a search.
        if state_path.exists():
            saved = loads(state_path.read_text())
            if saved.get('phase') == 'done':
                print(summary_path.read_text(), flush=True)
                return 0 if loads(summary_path.read_text())['status'] in {
                    'completed', 'no_improvement'} else 1
        if synchronize(root / 'sync.log'):
            os.execv(sys.executable, [
                sys.executable, '-m', 'tools.prompt_experiment', *sys.argv[1:],
            ])
        code_sha = git('rev-parse', 'HEAD')
        view = root / 'hf' / args.optimizer_revision
        models = pin_model_cache(cache, view, {
            'Qwen/Qwen3-0.6B': TARGET_REVISION,
            args.optimizer: args.optimizer_revision,
        })
        env = dict(os.environ, HF_HOME=str(view),
                   HF_HUB_CACHE=str(view / 'hub'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   OMP_NUM_THREADS='4')
        for variable in ('UV_PROJECT_ENVIRONMENT', 'TRANSFORMERS_CACHE',
                         'HUGGINGFACE_HUB_CACHE'):
            env.pop(variable, None)
        preflight = loads(subprocess.check_output([
            sys.executable, '-c', "import json,torch; "
            "assert torch.cuda.is_available(), 'GPU required'; "
            "print(json.dumps({'torch':torch.__version__,"
            "'gpu':torch.cuda.get_device_name(0),'hip':torch.version.hip}))",
        ], env=env, text=True))
        version_env = dict(env,
                           XDG_DATA_HOME=str(root / 'opencode/data'),
                           XDG_CONFIG_HOME=str(root / 'opencode/config'),
                           XDG_CACHE_HOME=str(root / 'opencode/cache'),
                           XDG_STATE_HOME=str(root / 'opencode/state'))
        harness_version = subprocess.check_output(
            ['opencode', '--version'], env=version_env, text=True).strip()
        runtime = {'models': models, 'environment': environment_metadata(),
                   'harness_version': harness_version,
                   'gpu': preflight, 'optimizer': args.optimizer,
                   'optimizer_dtype': 'bfloat16', 'optimizer_max_tokens': 512,
                   'optimizer_backend': 'opencode+transformers-serve',
                   'optimizer_use_cache': True,
                   'target_dtype': 'float32', 'target_decoding': 'greedy',
                   'value_token_limit': MAX_VALUE_STEPS}
        fixture = Path('tests/fixtures/prompt_development.json').resolve()
        identity = hashlib.sha256(json.dumps({
            'code_sha': code_sha, 'runtime': runtime, 'data_ref': DATA_REF,
            'dev_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
            'workers': args.workers, 'rounds': args.rounds,
            'direction': direction,
        }, sort_keys=True).encode()).hexdigest()
        state = read_state(state_path, identity)
        epoch = root / identity
        epoch.mkdir(exist_ok=True)
        provenance = epoch / 'provenance.json'
        write_json(provenance, {'code_sha': code_sha, 'runtime': runtime})
        env['CMM_PROMPT_PROVENANCE'] = str(provenance)
        manifest = {'code_sha': code_sha, 'runtime': runtime,
                    'data_ref': DATA_REF,
                    'workers': args.workers, 'rounds': args.rounds,
                    'direction': direction, 'identity': identity}
        write_json(epoch / 'manifest.json', manifest)
        # Candidates may not copy any evaluation set, including held-out ones.
        guard = copy_calibrated(epoch) + [
            p.resolve() for p in [HELDOUT_REGEX] if p.exists()]
        store = root / 'candidates'
        baseline_id = save_candidate(store, Candidate(
            name_instructions=prompt.NAME_INSTRUCTIONS,
            parameter_instructions=prompt.PARAMETER_INSTRUCTIONS))
        baseline_path = store / f'{baseline_id}.json'
        base = run_trial(baseline_path, fixture, epoch / 'baseline.json', env)
        if not base.get('finished') or has_oom(base):
            raise ValueError('baseline failed; see baseline artifact/log')
        best_path, best = baseline_path, base
        trials: list[tuple[str, dict[str, Any]]] = []
        for ref in state['trials']:
            trial = loads(Path(ref['report']).read_text())
            candidate = store / f"{ref['candidate_id']}.json"
            if trial.get('fingerprint') != fingerprint(
                    candidate, fixture, env):
                carried = epoch / f"carry-{ref['candidate_id']}.json"
                trial = run_trial(candidate, fixture, carried, env)
                ref['report'] = str(carried)
            if trial.get('finished'):
                trials.append((ref['candidate_id'], trial['score']))
                chosen = select_candidate(base['score'], trials)
                if chosen == ref['candidate_id']:
                    best_path, best = candidate, trial
        all_edits: list[dict[str, Any]] = []
        if state['phase'] == 'search':
            for round_index in range(state['next_round'], args.rounds):
                if len(best['score']['correct_ids']) == best['score']['total']:
                    break
                if synchronize(root / 'sync.log'):
                    os.execv(sys.executable, [
                        sys.executable, '-m', 'tools.prompt_experiment',
                        *sys.argv[1:],
                    ])
                tasks = []
                for worker in range(args.workers):
                    folder = epoch / f'round-{round_index}-worker-{worker}'
                    folder.mkdir(exist_ok=True)
                    context = folder / 'context.json'
                    write_json(context, editor_context(
                        best, fixture, best_path, worker, [
                            {'prompt': loads((store / (cid + '.json'))
                                             .read_text()), 'score': score}
                            for cid, score in dict(trials).items()],
                        direction))
                    tasks.append((folder, context))

                def edit(task: tuple[Path, Path]) -> dict[str, Any]:
                    """Keep one immutable editor attempt per proposal slot."""
                    folder, context = task
                    return editor_session(
                        folder, context, store, fixture, args.optimizer,
                        dict(env, CMM_OPTIMIZER_URL=(
                            'http://127.0.0.1:' + str(8011 + int(
                                folder.name.rsplit('-', 1)[1])) + '/v1')),
                        guard=guard)

                edits = parallel(edit, tasks, args.workers)
                for worker_index, edited in enumerate(edits):
                    # Only repeat a saved candidate's evaluation after OOM;
                    # no extra model proposal or hidden candidate budget.
                    artifact = Path(edited['artifact'])
                    log_oom = 'out of memory' in (
                        artifact / 'editor.log').read_text().lower()
                    if (not edited.get('serial_retry')
                            and (has_oom(edited) or log_oom)):
                        edited['serial_retry'] = 'started'
                        write_json(artifact / 'result.json', edited)
                        if 'candidate_id' in edited:
                            retry = run_trial(store / (
                                edited['candidate_id'] + '.json'), fixture,
                                artifact / 'serial-oom-retry.json', env)
                            edited['serial_retry'] = retry
                            edited['trial'] = retry
                            edited['status'] = (
                                'evaluated' if retry.get('finished') and
                                not has_oom(retry) else 'evaluation_failed')
                            write_json(artifact / 'result.json', edited)
                        else:
                            # No candidate existed: retry this same proposal
                            # slot once after all other editors have exited.
                            folder, context = tasks[worker_index]
                            prior = str(artifact)
                            edited = editor_session(
                                folder, context, store, fixture,
                                args.optimizer, env, serial_retry=True,
                                guard=guard)
                            edited['retry_of'] = prior
                            artifact = Path(edited['artifact'])
                            write_json(artifact / 'result.json', edited)
                    all_edits.append(edited)
                    if edited['status'] != 'evaluated':
                        continue
                    candidate_id = edited['candidate_id']
                    trial_path = artifact / 'accepted-trial.json'
                    write_json(trial_path, edited['trial'])
                    trials.append((candidate_id, edited['trial']['score']))
                    state['trials'].append({
                        'candidate_id': candidate_id,
                        'report': str(trial_path)})
                    chosen = select_candidate(base['score'], trials)
                    if chosen == candidate_id:
                        best_path = store / f'{chosen}.json'
                        best = edited['trial']
                    print(json.dumps({'round': round_index,
                                      'candidate_id': candidate_id,
                                      'correct': len(edited['trial']['score'][
                                          'correct_ids'])}), flush=True)
                state['next_round'] = round_index + 1
                write_json(state_path, state)
            if synchronize(root / 'sync.log'):
                os.execv(sys.executable, [
                    sys.executable, '-m', 'tools.prompt_experiment',
                    *sys.argv[1:],
                ])
            chosen = select_candidate(base['score'], trials)
            if chosen is None:
                all_edits = [loads(path.read_text()) for path in
                             epoch.glob('round-*/attempt-*/result.json')]
                status = ('no_improvement' if trials or len(
                    base['score']['correct_ids']) ==
                    base['score']['total'] else
                    search_status(all_edits))
                summary = {'status': status, 'manifest': manifest,
                           'baseline': base['score'],
                           'selected_candidate': None,
                           'editor_outcomes': [{k: e.get(k) for k in (
                               'status', 'error', 'exit_code', 'artifact')}
                               for e in all_edits]}
                write_json(summary_path, summary)
                state['phase'] = 'done'
                write_json(state_path, state)
                print(json.dumps(summary, default=validate_number), flush=True)
                return 0 if status == 'no_improvement' else 1
            state.update(phase='selected', candidate_id=chosen)
            write_json(state_path, state)
        selected = store / f"{state['candidate_id']}.json"
        best = run_trial(selected, fixture, epoch / 'selected.json', env)
        if (not best.get('finished')
                or not eligible(base['score'], best['score'])):
            raise ValueError('fixed candidate no longer improves baseline')
        summary = {'manifest': manifest, 'baseline': base['score'],
                   'selected_candidate': state['candidate_id'],
                   'status': 'cpu_failed'}
        units = os.environ.get('CMM_OPTIMIZER_UNITS', '').split()
        if units:
            subprocess.run(['systemctl', '--user', 'stop', *units], check=True)
        elif not os.environ.get('CMM_SERVERS_ALREADY_STOPPED'):
            raise ValueError('stop owned GPU model servers before CPU timing')
        cpu_path = epoch / 'cpu-result.json'
        if cpu_path.exists():
            cpu = loads(cpu_path.read_text())
        else:
            cpu_folder = Path(tempfile.mkdtemp(
                prefix='cpu-attempt-', dir=epoch))
            cpu = cpu_check(cpu_folder, selected, fixture, env, best, view)
        write_json(cpu_path, cpu)
        summary['cpu'] = cpu
        if cpu['passed']:
            state['phase'] = 'heldout'
            write_json(state_path, state)
            summary['calibrated'] = {}
            for heldout in copy_calibrated(epoch):
                comparison = {}
                for label, candidate in [('baseline', baseline_path),
                                         ('candidate', selected)]:
                    result = run_trial(candidate, heldout,
                                       epoch / f'{heldout.stem}-{label}.json',
                                       env)
                    if not result.get('finished'):
                        raise ValueError('heldout failed; resume fixed pair')
                    comparison[label] = result['score']
                summary['calibrated'][heldout.stem] = comparison
            summary['status'] = 'completed'
        write_json(summary_path, summary)
        state['phase'] = 'done'
        write_json(state_path, state)
        print(json.dumps(summary, default=validate_number), flush=True)
        return 0 if summary['status'] == 'completed' else 1
    except KeyboardInterrupt:
        stop_children()
        write_json(root / 'interrupted.json', {'time_ns': time.time_ns()})
        print('prompt experiment interrupted; owned workers stopped',
              file=sys.stderr)
        return 130
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        stop_children()
        write_json(root / 'failure.json', {
            'error': str(error), 'time_ns': time.time_ns()})
        print(f'prompt experiment stopped: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
