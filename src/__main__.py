"""Entry point: ``uv run python -m src``."""

import sys
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

from .cli import Args, parse_args
from .display import Dashboard
from .errors import PromptError, StartupError
from .loader import load_functions, load_prompts
from .pipeline import (
    GenerationResources, build_generation_resources, process_prompt,
)
from .schema import FunctionCall
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
    resources = build_generation_resources(functions)
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
        dashboard.prompt_started(index, user_prompt)
        try:
            call = process_prompt(user_prompt, resources, progress=dashboard)
        except PromptError as e:
            dashboard.prompt_failed(str(e))
            continue
        calls.append(call)
        save_results(output, calls)
        # Only after the save: if it fails, the summary shows this prompt
        # as not finished rather than done.
        dashboard.prompt_done(call.name)
    if not calls:
        save_results(output, calls)  # all failed: still write "[]"
    return len(calls) == len(prompts)


if __name__ == "__main__":
    raise SystemExit(main())
