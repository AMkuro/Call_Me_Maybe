"""Command-line arguments."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Args(BaseModel):
    """Validated command-line arguments.

    Judgment: the defaults follow the Subject (default input and output
    live under data/). The default output path is ADR-0011.
    """

    model_config = ConfigDict(frozen=True)

    functions_definition: Path = Path("data/input/functions_definition.json")
    input: Path = Path("data/input/function_calling_tests.json")
    output: Path = Path("data/output/function_calls.json")


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """Parse ``--functions_definition``, ``--input`` and ``--output``.

    Effect: reads sys.argv when ``argv`` is None.
    """
    raise NotImplementedError
