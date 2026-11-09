"""Save the results."""

from pathlib import Path

from .schema import FunctionCall


def save_results(path: Path, calls: list[FunctionCall]) -> None:
    """Write ``calls`` as a JSON array, replacing the file atomically."""
    raise NotImplementedError
