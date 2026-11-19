"""Entry point: ``uv run python -m src``."""

import sys
from collections.abc import Sequence

from .cli import Args, parse_args
from .display import Dashboard
from .errors import StartupError
from .pipeline import process_all
from .runtime import GenerationAssets


def main(argv: Sequence[str] | None = None) -> int:
    """Run the program and return its exit status.

    Reads the two input files, loads the model, writes the output file and
    reports progress and errors on stderr.

    Args:
        argv: The command-line arguments without the program name, or None
            to read sys.argv.

    Returns:
        0 when every prompt was saved; 1 when startup fails, a prompt is
        skipped or saving fails.

    Raises:
        SystemExit: From argparse, for ``--help`` (status 0) or an invalid
            option (status 2).
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
    """Read the arguments and inputs, prepare the output, load the model.

    The model is loaded last, so an invalid file or output path is reported
    without waiting for it.

    Args:
        argv: As for main.

    Returns:
        The arguments, their validated prompts and the generation
        resources.

    Raises:
        StartupError: An input, the output path or the model cannot be
            used.
    """
    args = parse_args(argv)
    resources = GenerationAssets(args.functions)
    return args, args.prompts, resources


def _end(
    args: Args,
    prompts: Sequence[str],
    resources: GenerationAssets,
) -> int:
    """Process the prompts inside the dashboard and return the exit status.

    The dashboard restores the terminal on leaving, also after an exception
    or Ctrl-C. A save failure is reported after that, so the message is not
    hidden by the dashboard.

    Args:
        args: The arguments; ``args.output`` is the output file.
        prompts: The prompts to process, in order.
        resources: The model and the data shared by every prompt.

    Returns:
        0 when every prompt was saved, otherwise 1.
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
