"""Save the results."""

from pathlib import Path

from .schema import FunctionCall


def save_results(path: Path, calls: list[FunctionCall]) -> None:
    """Write ``calls`` as a JSON array, replacing the file atomically.

    Called after every prompt with all results so far.

    Effect: writes a file.

    Judgment: rewrite the whole file after every prompt (temporary file,
    then replace), so a crash keeps the results so far. Slow when there
    are many prompts (not measured).
    """
    raise NotImplementedError
