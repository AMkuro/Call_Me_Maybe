"""Run the shared sequential CLI independently of any generation algorithm."""

import argparse
from pathlib import Path

from src.inputs import InputFiles
from src.processing import GenerateCall, generate_call
from src.storage import ResultStore
from src.workflow import Application


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
    """Parse arguments, compose the application, and call its entry point."""
    args = parse_args(argv)
    application = Application(
        inputs=InputFiles(
            functions_path=args.functions_definition, input_path=args.input,
        ),
        output=ResultStore(path=args.output),
        generate=generate_call if generate is None else generate,
    )
    return application.run()
