"""Command-line arguments."""

import argparse
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
    parser = argparse.ArgumentParser(
        description="Generate JSON function calls with constrained decoding.",
        allow_abbrev=False,
    )
    defaults = Args()
    parser.add_argument("--functions_definition", type=Path,
                        default=defaults.functions_definition)
    parser.add_argument("--input", type=Path, default=defaults.input)
    parser.add_argument("--output", type=Path, default=defaults.output)
    return Args.model_validate(vars(parser.parse_args(argv)))
