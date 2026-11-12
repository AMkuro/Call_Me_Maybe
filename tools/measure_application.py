"""Repeat real-model accuracy/stress cases, without making timing claims."""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.decoder import SelectionMode
from src.progress import Stage
from src.errors import PromptError, StartupError
from src.json_io import loads
from src.loader import index_functions
from src.model_port import Model
from src.name_trie import build_name_trie
from src.pipeline import MAX_VALUE_STEPS, GenerationResources, process_prompt
from src.schema import FunctionCall, FunctionDef, ParamType
from src.vocab import load_vocab


class MeasurementProgress(BaseModel):
    """Collect per-parameter token counts for one accuracy case."""

    counts: dict[str, int] = Field(default_factory=dict)
    parameter: str = ""

    @property
    def available(self) -> bool:
        """This in-memory diagnostic sink remains available."""
        return True

    def stage(self, stage: Stage) -> None:
        """Stage changes need no measurement."""

    def function_selected(self, name: str) -> None:
        """Function correctness is assessed from the completed call."""

    def parameter_started(self, name: str, param_type: ParamType) -> None:
        """Start the counter under the existing report's parameter key."""
        self.parameter = f"{name} ({param_type})"
        self.counts[self.parameter] = 0

    def name_token(
        self, tokens: Sequence[int], *, mode: SelectionMode,
        raw_top: int | None,
    ) -> None:
        """Name observations are not retained by this measurement."""

    def value_token(
        self, written: bytes, step: int, *, token: int, mode: SelectionMode,
    ) -> None:
        """Record the actual count without retaining token histories."""
        self.counts[self.parameter] = step


def assess_call(case: dict[str, Any], call: FunctionCall) -> bool:
    """Compare held-out expectations, including normalized numeric values."""
    if call.name != case["expected_name"]:
        return False
    if "expected_parameters" in case:
        expected = case["expected_parameters"]
        if call.parameters.keys() != expected.keys():
            return False
        for name, value in call.parameters.items():
            wanted = expected[name]
            if isinstance(wanted, (int, float, Decimal)) and not isinstance(
                wanted, bool
            ):
                if (not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or Decimal(str(value)) != Decimal(str(wanted))):
                    return False
            elif type(value) is not type(wanted) or value != wanted:
                return False
        return True
    requested = Decimal(case["requested_number_literal"])
    numeric_value = call.parameters.get("number")
    if (not isinstance(numeric_value, (int, float))
            or isinstance(numeric_value, bool)
            or (isinstance(numeric_value, float)
                and not math.isfinite(numeric_value))):
        return False
    # Compare the number that is actually serialized, independently of
    # any internal numeric conversion used by production validation.
    return Decimal(str(numeric_value)) == requested


def main() -> int:
    """Load once, repeat cases and save every completed observation.

    Effect: loads the real model, reads fixtures, writes the report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    try:
        fixture = loads(args.cases.read_text(encoding="utf-8"))
        functions = index_functions(FunctionDef.model_validate(fn)
                                    for fn in fixture["functions"])
        model = Model()
        vocab = load_vocab(Path(model.vocab_path()))
        trie = build_name_trie(list(functions), model.encode)
        resources = GenerationResources(model=model, vocab=vocab,
                                        functions=functions, trie=trie)
        results: list[dict[str, Any]] = []
        report: dict[str, Any] = {
            "measurement": "accuracy/stress; no timing measurements",
            "model": "Qwen/Qwen3-0.6B", "value_token_limit": MAX_VALUE_STEPS,
            "repetitions": args.repeat, "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for repeat in range(1, args.repeat + 1):
            for case in fixture["cases"]:
                progress = MeasurementProgress()

                result: dict[str, Any] = {"id": case["id"], "repeat": repeat}
                if "requested_number_literal" in case:
                    result["requested_number_literal"] = (
                        case["requested_number_literal"]
                    )
                try:
                    call = process_prompt(case["prompt"], resources,
                                          progress=progress)
                    result.update(status="completed", call=call.model_dump(),
                                  correct=assess_call(case, call))
                    if "requested_number_literal" in case:
                        result["exact_numeric_agreement"] = result["correct"]
                except PromptError as e:
                    result.update(status="error", error=str(e), correct=False)
                result["generated_value_tokens"] = progress.counts
                results.append(result)
                args.output.write_text(json.dumps(report, indent=2,
                                                  allow_nan=False) + "\n",
                                       encoding="utf-8")
                print(json.dumps({key: value for key, value in result.items()
                                  if key != "call"}), flush=True)
    except (StartupError, OSError, ValueError, KeyError) as e:
        print(f"measurement failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
