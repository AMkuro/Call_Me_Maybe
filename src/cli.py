"""Command-line arguments."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel


class Args(BaseModel):
    """Validated command-line arguments."""

    functions_definition: Path = Path("data/input/functions_definition.json")
    input: Path = Path("data/input/function_calling_tests.json")
    output: Path = Path("data/output/function_calls.json")


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """Parse ``--functions_definition``, ``--input`` and ``--output``."""
    raise NotImplementedError
