"""Evaluate instruction data using the unchanged application pipeline/CLI."""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any

from src import prompt
from src.errors import PromptError, StartupError
from src.json_io import loads
from src.loader import index_functions
from src.model_port import Model
from src.name_trie import build_name_trie
from src.pipeline import process_prompt
from src.schema import FunctionDef
from src.vocab import load_vocab
from tools.prompt_candidates import Candidate, score_records, write_json
from tools.prompt_runtime import environment_metadata, has_oom


def apply_candidate(path: Path) -> Candidate:
    """Change only the two instruction globals for this worker process."""
    candidate = Candidate.model_validate(loads(path.read_text()))
    prompt.NAME_INSTRUCTIONS = candidate.name_instructions
    prompt.PARAMETER_INSTRUCTIONS = candidate.parameter_instructions
    return candidate


def evaluate(fixture: Path, output: Path, fingerprint: str) -> dict[str, Any]:
    """Run all fixed cases, retaining failures and incremental raw evidence."""
    data = loads(fixture.read_text(encoding="utf-8"))
    functions = index_functions([FunctionDef.model_validate(f)
                                for f in data["functions"]])
    started = time.monotonic_ns()
    import torch
    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    model = Model()
    vocab = load_vocab(Path(model.vocab_path()))
    trie = build_name_trie(list(functions), model.encode)
    records: list[dict[str, Any]] = []
    result = {"fingerprint": fingerprint, "records": records,
              "pid": os.getpid(), 'started_monotonic_ns': started,
              'device': 'cuda' if gpu else 'cpu',
              'threads': torch.get_num_threads()}
    for case in data["cases"]:
        row: dict[str, Any] = {"id": case["id"], "choices": []}

        def observe(key: str, value: str) -> None:
            """Keep existing token-level diagnostics without extra forwards."""
            if key in {"stage", "choice", "metrics", "warning", "function"}:
                row["choices"].append({"event": key, "value": value})

        try:
            row["call"] = process_prompt(
                case["prompt"], functions, model, trie, vocab,
                observe,
            ).model_dump()
        except PromptError as error:
            row["error"] = str(error)
        records.append(row)
        write_json(output, result)
    if has_oom(result):
        result['failure_kind'] = 'gpu_oom'
    result["score"] = score_records(data["cases"], records, functions)
    result['ended_monotonic_ns'] = time.monotonic_ns()
    if gpu:
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    result["finished"] = True
    write_json(output, result)
    return result


def main() -> int:
    """Run a fixed fixture or delegate to the entire native application CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--fingerprint", default="")
    parser.add_argument("--cli", action="store_true")
    args, remaining = parser.parse_known_args()
    try:
        apply_candidate(args.candidate)
        if args.cli:
            telemetry = os.environ.get('CMM_CPU_TELEMETRY')
            if telemetry:
                import torch
                runtime = environment_metadata()
                runtime.update(cpus=sorted(os.sched_getaffinity(0)),
                               threads=torch.get_num_threads(),
                               gpu_available=torch.cuda.is_available())
                group = Path('/proc/self/cgroup').read_text().strip()
                group = group.split('::')[-1]
                for name in ('memory.max', 'memory.swap.max'):
                    path = Path('/sys/fs/cgroup') / group.lstrip('/') / name
                    runtime[name] = path.read_text().strip()
                write_json(Path(telemetry), runtime)
                if (runtime['cpus'] != list(range(16))
                        or runtime['threads'] != 6 or runtime['gpu_available']
                        or runtime['memory.max'] != str(16 * 1024 ** 3)
                        or runtime['memory.swap.max'] != '0'):
                    raise ValueError('CPU execution profile was not applied')
            from src.__main__ import main as application
            return application(remaining)
        if remaining or args.fixture is None or args.report is None:
            raise ValueError("evaluation needs --fixture and --report")
        evaluate(args.fixture, args.report, args.fingerprint)
        return 0
    except (OSError, ValueError, PromptError, StartupError) as error:
        if args.report is not None:
            write_json(args.report, {
                'fingerprint': args.fingerprint, 'finished': False,
                'records': [], 'error': str(error),
                'failure_kind': ('gpu_oom' if 'out of memory' in
                                 str(error).lower() else 'execution_failed'),
            })
        print(f"prompt trial failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
