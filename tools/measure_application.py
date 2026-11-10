"""Repeat real-model accuracy/stress cases, without making timing claims."""

import argparse
import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.errors import PromptError, StartupError
from src.loader import index_functions
from src.model_port import Model
from src.name_trie import build_name_trie
from src.pipeline import MAX_VALUE_STEPS, process_prompt
from src.schema import FunctionCall, FunctionDef
from src.vocab import load_vocab


def assess_call(case: dict[str, Any], call: FunctionCall) -> bool:
    """Compare held-out expectations, including normalized numeric values."""
    if call.name != case["expected_name"]:
        return False
    if "expected_parameters" in case:
        return bool(call.parameters == case["expected_parameters"])
    requested = Decimal(case["requested_number_literal"])
    value = call.parameters.get("number")
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value)):
        return False
    # Compare the number that is actually serialized, independently of
    # the float conversion used by production validation.
    return Decimal(str(value)) == requested


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
        fixture = json.loads(args.cases.read_text(encoding="utf-8"))
        functions = index_functions(FunctionDef.model_validate(fn)
                                    for fn in fixture["functions"])
        model = Model()
        vocab = load_vocab(Path(model.vocab_path()))
        trie = build_name_trie(list(functions), model.encode)
        results: list[dict[str, Any]] = []
        report: dict[str, Any] = {
            "measurement": "accuracy/stress; no timing measurements",
            "model": "Qwen/Qwen3-0.6B", "value_token_limit": MAX_VALUE_STEPS,
            "repetitions": args.repeat, "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for repeat in range(1, args.repeat + 1):
            for case in fixture["cases"]:
                counts: dict[str, int] = {}
                parameter = ""

                def observe(label: str, text: str) -> None:
                    """Collect actual generated counts from pipeline events."""
                    nonlocal parameter
                    if label == "parameter":
                        parameter = text
                    elif label == "value_tokens":
                        counts[parameter] = int(text)

                result: dict[str, Any] = {"id": case["id"], "repeat": repeat}
                if "requested_number_literal" in case:
                    result["requested_number_literal"] = (
                        case["requested_number_literal"]
                    )
                try:
                    call = process_prompt(case["prompt"], functions, model,
                                          trie, vocab, observe)
                    result.update(status="completed", call=call.model_dump(),
                                  correct=assess_call(case, call))
                    if "requested_number_literal" in case:
                        literal = case["requested_number_literal"]
                        result["requested_number_literal"] = literal
                        result["normalized_agreement"] = (
                            call.name == case["expected_name"]
                            and call.parameters == {"number": float(literal)}
                        )
                        result["exact_numeric_agreement"] = result["correct"]
                except PromptError as e:
                    result.update(status="error", error=str(e), correct=False)
                result["generated_value_tokens"] = counts
                results.append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n",
                                       encoding="utf-8")
                print(json.dumps({key: value for key, value in result.items()
                                  if key != "call"}), flush=True)
    except (StartupError, OSError, ValueError, KeyError) as e:
        print(f"measurement failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
