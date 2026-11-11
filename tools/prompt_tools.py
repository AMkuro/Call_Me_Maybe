"""The two data-only tools exposed to the existing OpenCode agent harness."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.errors import StartupError
from src.json_io import loads
from src.schema import validate_number
from tools.prompt_candidates import (
    Candidate, copies_evaluation_data, save_candidate, write_json,
)
from tools.prompt_runtime import evidence_fingerprint
from tools.prompt_trial import apply_candidate, evaluate


class ToolConfig(BaseModel):
    """Trusted workspace settings, never arguments chosen by the model."""

    model_config = ConfigDict(extra='forbid', strict=True)
    store: str
    fixture: str
    output: str
    provenance: str
    fingerprint: str
    code_sha: str
    parent_id: str = ''
    guard: list[str] = []


def call_tool(workspace: Path, action: str, payload: dict[str, Any]
              ) -> dict[str, Any]:
    """Save one candidate or evaluate it against the fixed development data."""
    config = ToolConfig.model_validate(loads(
        (workspace / 'tool-config.json').read_text()))
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      text=True).strip()
    if current != config.code_sha:
        raise ValueError('code changed; recreate this workspace')
    output = Path(config.output)
    record: dict[str, Any] = {
        'fingerprint': config.fingerprint, 'finished': False, 'actions': [],
        'started_monotonic_ns': time.monotonic_ns(), 'harness': 'opencode'}
    if output.exists():
        record = loads(output.read_text())
    store = Path(config.store)
    if action == 'write_prompt':
        candidate = Candidate.model_validate(payload)
        # Rejected before saving, so the workspace still accepts a candidate.
        if config.parent_id and candidate == Candidate.model_validate(loads(
                (store / (config.parent_id + '.json')).read_text())):
            raise ValueError('candidate equals current_prompt; revise it')
        if copies_evaluation_data(candidate, [Path(p) for p in config.guard]):
            raise ValueError('candidate copies evaluation data; '
                             'write general instructions and new examples')
        if 'candidate_id' in record:
            existing = Candidate.model_validate(loads((store / (
                record['candidate_id'] + '.json')).read_text()))
            if existing != candidate:
                raise ValueError('one candidate per workspace; call run now')
        candidate_id = save_candidate(store, candidate)
        selected = workspace / 'candidate.json'
        try:
            # shortcut: same filesystem only, use atomic copy across mounts.
            os.link(store / (candidate_id + '.json'), selected)
        except FileExistsError:
            previous = Candidate.model_validate(loads(selected.read_text()))
            if previous != candidate:
                raise ValueError('this workspace already has a candidate')
        record['candidate_id'] = candidate_id
        record['actions'].append({'tool': action,
                                  'candidate_id': candidate_id})
        write_json(output, record)
        return {'candidate_id': candidate_id, 'next': 'call run without args'}
    if action != 'run' or payload:
        raise ValueError('run accepts no arguments')
    if 'candidate_id' not in record:
        raise ValueError('call write_prompt before run')
    path = store / (record['candidate_id'] + '.json')
    provenance = loads(Path(config.provenance).read_text())
    stamp = evidence_fingerprint(config.code_sha, path, Path(config.fixture),
                                 provenance['runtime'])
    if not record.get('finished'):
        apply_candidate(path)
        try:
            trial = evaluate(Path(config.fixture), workspace / 'trial.json',
                             stamp)
        except (StartupError, ValueError, OSError) as error:
            record['error'] = str(error)
            write_json(output, record)
            raise
        record.update(trial=trial, finished=True,
                      ended_monotonic_ns=time.monotonic_ns())
        record['actions'].append({'tool': action})
        write_json(output, record)
    trial = record['trial']
    if trial['fingerprint'] != stamp:
        raise ValueError('cached evaluation identity mismatch')
    return {'candidate_id': record['candidate_id'], 'score': trial['score']}


def main() -> int:
    """Bridge native OSS tools to fixed Python operations, without a shell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('write_prompt', 'run'))
    parser.add_argument('workspace', type=Path)
    args = parser.parse_args()
    try:
        payload = loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError('tool arguments must be an object')
        result = call_tool(args.workspace.resolve(), args.action, payload)
        print(json.dumps(result, default=validate_number, allow_nan=False))
        return 0
    except (ValueError, OSError, StartupError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
