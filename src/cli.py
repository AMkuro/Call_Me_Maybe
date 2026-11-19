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
    """The three paths and the inputs read from them, validated together.

    Creating an Args reads and validates both input files and prepares the
    output: missing parent directories are created and a temporary file is
    written and removed next to it. An existing output file is left
    unchanged. The defaults are the Subject's paths under data/.

    Attributes:
        functions_definition: The function definitions file.
        input: The prompts file.
        output: The output file.
        functions: The definitions, read-only, keyed by name in input
            order. Left out of model_dump.
        prompts: The prompts in input order. Left out of model_dump.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    functions_definition: Path = Path("data/input/functions_definition.json")
    input: Path = Path("data/input/function_calling_tests.json")
    output: Path = Path("data/output/function_calls.json")

    functions: Mapping[str, FunctionDef] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
    )
    prompts: tuple[str, ...] = Field(default=(), exclude=True, repr=False)

    def __init__(self, **data: Any) -> None:
        """Validate the paths, read the inputs, then prepare the output.

        Args:
            **data: The path fields functions_definition, input and output.

        Raises:
            StartupError: An input file is unreadable or invalid, or the
                output cannot be prepared.
            ValidationError: A value is not a path, or a field is unknown.
        """
        super().__init__(**data)
        functions = load_functions(self.functions_definition)
        prompts = load_prompts(self.input)
        prepare_output(self.output)
        # Only initialization installs these already validated, frozen fields.
        object.__setattr__(self, "functions", MappingProxyType(functions))
        object.__setattr__(self, "prompts", tuple(prompts))


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """Parse ``--functions_definition``, ``--input`` and ``--output``.

    Args:
        argv: The arguments without the program name, or None to read
            sys.argv.

    Returns:
        The resulting Args: the inputs have been read and the output
        prepared.

    Raises:
        StartupError: As for Args.
        SystemExit: From argparse, for ``--help`` or an invalid option,
            before any file is touched.
    """
    parser = argparse.ArgumentParser(
        description="Generate JSON function calls with constrained decoding.",
        allow_abbrev=False,
    )
    defaults = Args.model_fields
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=defaults["functions_definition"].default,
    )
    parser.add_argument(
        "--input", type=Path, default=defaults["input"].default
    )
    parser.add_argument(
        "--output", type=Path, default=defaults["output"].default
    )
    return Args.model_validate(vars(parser.parse_args(argv)))
