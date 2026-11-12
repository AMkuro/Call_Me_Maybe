"""Entry point: ``uv run python -m src``."""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .cli import parse_args
from .display import terminal_progress
from .errors import PromptError, StartupError
from .loader import load_functions, load_prompts
from .model_port import Model
from .name_trie import build_name_trie
from .pipeline import process_prompt
from .schema import FunctionCall
from .vocab import load_vocab
from .writer import prepare_output, save_results


def main(argv: Sequence[str] | None = None) -> int:
    """Validate inputs and output readiness before loading the model.

    Non-zero when a startup step fails or any prompt was skipped.

    Effect: reads and writes files, loads the model, prints to stderr.
    """
    try:
        args = parse_args(argv)
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        prepare_output(args.output)
        model = Model()
        vocab = load_vocab(Path(model.vocab_path()))
        trie = build_name_trie(list(functions), model.encode)
    except StartupError as e:
        print(e, file=sys.stderr)
        return 1

    calls: list[FunctionCall] = []
    skipped = False
    try:
        with terminal_progress(len(prompts)) as report:
            for index, user_prompt in enumerate(prompts, 1):
                report("input", f"{index}/{len(prompts)}")
                report("prompt", user_prompt)
                try:
                    call = process_prompt(
                        user_prompt, functions, model, trie, vocab, report,
                        report.top_k,
                    )
                except PromptError as e:
                    report("error", f"skipped: {e}")
                    skipped = True
                    continue
                calls.append(call)
                save_results(args.output, calls)
                report("result", f"{call.name} " + json.dumps(call.parameters))
            if not calls:
                save_results(args.output, calls)
    except OSError as e:
        print(f"cannot save output to {args.output}: {e}", file=sys.stderr)
        return 1
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
