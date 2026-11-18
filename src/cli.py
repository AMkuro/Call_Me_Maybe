"""Command-line arguments."""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .loader import load_functions, load_prompts
from .schema import FunctionDef
from .writer import prepare_output


class Args(BaseModel):
    """Validated command-line arguments.

    Judgment: the defaults follow the Subject (default input and output
    live under data/). The Subject gives the output file as both
    data/output/function_calls.json and output/function_calls.json; the
    first matches the data/ default.
    Initialization loads validated input snapshots and probes the output
    destination before a model can be loaded. Existing output is untouched.
    """

    model_config = ConfigDict(frozen=True)

    functions_definition: Path = Path("data/input/functions_definition.json")
    input: Path = Path("data/input/function_calling_tests.json")
    output: Path = Path("data/output/function_calls.json")

    functions: Mapping[str, FunctionDef] = Field(
        default_factory=dict, exclude=True, repr=False,
    )
    prompts: tuple[str, ...] = Field(default=(), exclude=True, repr=False)

    def __init__(self, **data: Any) -> None:
        """Validate inputs and output, then install immutable snapshots."""
        super().__init__(**data)
        functions = load_functions(self.functions_definition)
        prompts = load_prompts(self.input)
        prepare_output(self.output)
        # Only initialization installs these already validated, frozen fields.
        object.__setattr__(self, "functions", MappingProxyType(functions))
        object.__setattr__(self, "prompts", tuple(prompts))


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """Parse ``--functions_definition``, ``--input`` and ``--output``.

    Effect: reads inputs and prepares output after successful parsing.
    Reads sys.argv when ``argv`` is None. Help and parse errors skip
    input loading and output preparation.
    """
    parser = argparse.ArgumentParser(
        description="Generate JSON function calls with constrained decoding.",
        allow_abbrev=False,
    )
    defaults = Args.model_fields
    parser.add_argument("--functions_definition", type=Path,
                        default=defaults["functions_definition"].default)
    parser.add_argument("--input", type=Path,
                        default=defaults["input"].default)
    parser.add_argument("--output", type=Path,
                        default=defaults["output"].default)
    return Args.model_validate(vars(parser.parse_args(argv)))
