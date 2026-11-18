"""Entry point: ``uv run python -m src``."""

import sys
from collections.abc import Sequence

from .cli import Args, parse_args
from .display import Dashboard
from .errors import StartupError
from .pipeline import process_all
from .runtime import GenerationAssets


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
    return _end(args, prompts, resources)


def _start(
    argv: Sequence[str] | None,
) -> tuple[Args, tuple[str, ...], GenerationAssets]:
    """Check the inputs and the output path, then load the model.

    The model is loaded last, so a bad file fails fast. Raises
    StartupError.

    Effect: reads files, checks the output path, loads the model.
    """
    args = parse_args(argv)
    resources = GenerationAssets(args.functions)
    return args, args.prompts, resources


def _end(
    args: Args,
    prompts: Sequence[str],
    resources: GenerationAssets,
) -> int:
    """Run the batch, close the terminal, and report save failures.

    Dashboard owns terminal cleanup, including on interruption. Report
    OSError only after its context has closed; other exceptions propagate.
    """
    try:
        with Dashboard(
            len(prompts), resources.model, resources.vocab
        ) as dashboard:
            all_done = process_all(prompts, resources, args.output, dashboard)
    # Outside the with, so the terminal is restored before the message.
    # Only save_results raises OSError here: Dashboard keeps its own
    # stream errors, and Model turns its errors into PromptError.
    except OSError as e:
        print(f"cannot save output to {args.output}: {e}", file=sys.stderr)
        return 1
    return 0 if all_done else 1


if __name__ == "__main__":
    raise SystemExit(main())
