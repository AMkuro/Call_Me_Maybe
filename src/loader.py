"""Read and validate the two input files."""

from pathlib import Path

from .schema import FunctionDef


def load_functions(path: Path) -> list[FunctionDef]:
    """Read function definitions.

    Raises StartupError on a missing or invalid file, an empty list,
    duplicate names, or an unsupported parameter type.
    """
    raise NotImplementedError


def load_prompts(path: Path) -> list[str]:
    """Read the prompts (the ``prompt`` key of each input entry)."""
    raise NotImplementedError
