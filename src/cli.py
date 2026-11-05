"""Run the shared sequential CLI independently of any generation algorithm."""

import argparse
import sys
from pathlib import Path
from typing import Callable

from pydantic import JsonValue

from src.storage import save_results
from src.validation import (
    FunctionDefinition,
    Prompt,
    load_definitions,
    load_prompts,
    validate_result,
)

GenerateCall = Callable[[str, list[FunctionDefinition]], str]


def generate_call(prompt: str, definitions: list[FunctionDefinition]) -> str:
    """Fail explicitly until a real constrained LLM generator is integrated.

    Future adapters must return only a successfully completed final JSON text.
    SDK failures, early EOS, impossible constraints, or exceeded limits must
    raise exceptions, even if partial text happens to parse as valid JSON.
    The adapter owns model loading/reuse and internal generation metrics.
    """
    raise NotImplementedError(
        "LLM generation is not implemented; integrate a real constrained "
        "generation method before running nonempty batches"
    )


def _check_output_path(output: Path, inputs: tuple[Path, Path]) -> None:
    """Reject input aliases before initializing output could destroy data."""
    for source in inputs:
        if output.resolve() == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise ValueError(f"output {output} aliases input {source}")


def main(
    argv: list[str] | None = None, *, generate: GenerateCall | None = None,
) -> int:
    """Process one input at a time and publish only validated successes.

    The injected callable is a seam for integration and model-free tests.
    Codes: 0 success, 1 failure, 2 argparse usage error, 130 Ctrl-C.
    """
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
    args = parser.parse_args(argv)
    functions_path: Path = args.functions_definition
    input_path: Path = args.input
    output_path: Path = args.output
    generator = generate_call if generate is None else generate
    stage = "output path validation"
    try:
        _check_output_path(output_path, (functions_path, input_path))
        stage = f"initialize output {output_path}"
        save_results(output_path, [])
        stage = f"load function definitions {functions_path}"
        definitions = load_definitions(functions_path)
        stage = f"load prompts {input_path}"
        records = load_prompts(input_path)
        results: list[dict[str, JsonValue]] = []
        for index, record in enumerate(records, start=1):
            stage = f"input {index} input validation ({input_path})"
            prompt = Prompt.model_validate(record).prompt
            stage = f"input {index} generation"
            text = generator(
                prompt, [item.model_copy(deep=True) for item in definitions],
            )
            stage = f"input {index} result validation"
            result = validate_result(text, prompt, definitions)
            stage = f"input {index} save output {output_path}"
            results.append(result.model_dump())
            save_results(output_path, results)
    except KeyboardInterrupt:
        print(f"interrupted: {stage}", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {stage}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0
