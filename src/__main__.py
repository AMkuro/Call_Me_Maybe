"""Entry point: ``uv run python -m src``."""

import sys
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

from pydantic import ValidationError

from .cli import Args, parse_args
from .display import Dashboard
from .errors import PromptError, StartupError
from .loader import load_functions, load_prompts
from .model_port import Model
from .name_trie import build_name_trie
from .pipeline import GenerationResources, process_prompt
from .schema import FunctionCall
from .vocab import load_vocab
from .writer import prepare_output, save_results


def main(argv: Sequence[str] | None = None) -> int:
    """Start, process every prompt, and return the exit status.

    Non-zero when a startup step fails or any prompt was skipped.

    Effect: reads and writes files, loads the model, prints to stderr.
    """
    try:
        args, prompts, resources = _start(argv)
    except StartupError as e:
        print(e, file=sys.stderr)
        return 1
    try:
        with Dashboard(len(prompts), resources.model,
                       MappingProxyType(resources.vocab)) as dashboard:
            all_done = _process_all(prompts, resources, args.output, dashboard)
    except OSError as e:
        print(f"cannot save output to {args.output}: {e}", file=sys.stderr)
        return 1
    return 0 if all_done else 1


def _start(
    argv: Sequence[str] | None,
) -> tuple[Args, list[str], GenerationResources]:
    """Check the inputs and the output path, then load the model.

    The model is loaded last, so a bad file fails fast. Raises
    StartupError.

    Effect: reads files, checks the output path, loads the model.
    """
    args = parse_args(argv)
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)
    prepare_output(args.output)
    model = Model()
    vocab = load_vocab(Path(model.vocab_path()))
    trie = build_name_trie(list(functions), model.encode)
    try:
        resources = GenerationResources(model=model, vocab=vocab,
                                        functions=functions, trie=trie)
    except ValidationError as e:
        raise StartupError(f"invalid generation resources: {e}") from e
    return args, prompts, resources


def _process_all(
    prompts: list[str], resources: GenerationResources, output: Path,
    dashboard: Dashboard,
) -> bool:
    """Process every prompt; True when none failed.

    A failed prompt is reported and skipped. The results so far are saved
    after each success, so an interruption keeps them. Raises OSError when
    saving fails.

    Effect: uses the model and writes ``output``.
    """
    calls: list[FunctionCall] = []
    for index, user_prompt in enumerate(prompts, 1):
        dashboard.report("input", f"{index}/{len(prompts)}")
        dashboard.report("prompt", user_prompt)
        try:
            calls.append(process_prompt(user_prompt, resources,
                                        progress=dashboard))
        except PromptError as e:
            dashboard.report("error", str(e))
            continue
        save_results(output, calls)
        dashboard.report("result", calls[-1].name)
    if not calls:
        save_results(output, calls)  # all failed: still write "[]"
    return len(calls) == len(prompts)


if __name__ == "__main__":
    raise SystemExit(main())
