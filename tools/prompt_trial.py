"""Evaluate instruction data using the unchanged application pipeline/CLI."""

import argparse
import os
from pathlib import Path
import sys
from typing import Any

from src import prompt
from src.errors import PromptError
from src.json_io import loads
from src.loader import index_functions
from src.model_port import Model
from src.name_trie import build_name_trie
from src.pipeline import process_prompt
from src.schema import FunctionDef
from src.vocab import load_vocab
from tools.prompt_candidates import Candidate, score_records, write_json


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
    model = Model()
    vocab = load_vocab(Path(model.vocab_path()))
    trie = build_name_trie(list(functions), model.encode)
    records: list[dict[str, Any]] = []
    result = {"fingerprint": fingerprint, "records": records,
              "pid": os.getpid()}
    for case in data["cases"]:
        row: dict[str, Any] = {"id": case["id"], "choices": []}

        def observe(key: str, value: str) -> None:
            """Keep existing token-level diagnostics without extra forwards."""
            if key in {"stage", "choice", "metrics", "warning"}:
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
    result["score"] = score_records(data["cases"], records, functions)
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
            from src.__main__ import main as application
            return application(remaining)
        if remaining or args.fixture is None or args.report is None:
            raise ValueError("evaluation needs --fixture and --report")
        evaluate(args.fixture, args.report, args.fingerprint)
        return 0
    except (OSError, ValueError, PromptError) as error:
        print(f"prompt trial failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
