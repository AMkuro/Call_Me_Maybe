"""Run the shared sequential CLI independently of any generation algorithm."""

import argparse
from pathlib import Path

from src.workflow import GenerateCall, run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Subject CLI paths and report usage errors."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--functions_definition", type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )
    parser.add_argument(
        "--output", "-output", type=Path,
        default=Path("data/output/function_calls.json"),
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None, *, generate: GenerateCall | None = None,
) -> int:
    """Parse CLI arguments and delegate execution to the workflow."""
    args = parse_args(argv)
    return run(
        args.functions_definition, args.input, args.output, generate=generate,
    )
