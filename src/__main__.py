"""Entry point: ``uv run python -m src``."""

import sys
from collections.abc import Sequence
from pathlib import Path

from .cli import parse_args
from .errors import PromptError, StartupError
from .loader import load_functions, load_prompts
from .model_port import Model
from .name_trie import build_name_trie
from .pipeline import process_prompt
from .schema import FunctionCall
from .vocab import load_vocab
from .writer import save_results


def main(argv: Sequence[str] | None = None) -> int:
    """Run the flow of docs/architecture.md section 4; return the exit code.

    Non-zero when a startup step fails or any prompt was skipped.

    Effect: reads and writes files, loads the model, prints to stderr.
    """
    try:
        args = parse_args(argv)
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        model = Model()
        vocab = load_vocab(Path(model.vocab_path()))
        trie = build_name_trie([fn.name for fn in functions], model.encode)
    except StartupError as e:
        print(e, file=sys.stderr)
        return 1

    calls: list[FunctionCall] = []
    skipped = False
    try:
        for user_prompt in prompts:
            try:
                call = process_prompt(
                    user_prompt, functions, model, trie, vocab,
                )
            except PromptError as e:
                print(f"skipped: {e}", file=sys.stderr)
                skipped = True
                continue
            calls.append(call)
            save_results(args.output, calls)
        if not calls:
            save_results(args.output, calls)
    except OSError as e:
        print(f"cannot save output to {args.output}: {e}", file=sys.stderr)
        return 1
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
