"""Instruction-only tool protocol and independent, typed experiment scoring."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from src.schema import (
    FunctionCall, FunctionDef, build_parameters_model, validate_number,
)
from tools.measure_application import assess_call


class Candidate(BaseModel):
    """The entire editable surface; no Python, paths or decoding settings."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    name_instructions: str
    parameter_instructions: str

    @field_validator("name_instructions", "parameter_instructions")
    @classmethod
    def require_plain_instruction(cls, value: str) -> str:
        """Preserve text but forbid empty instructions and role delimiters."""
        if (not value.strip() or "<|" in value
                or re.search(r"</?think>", value)):
            raise ValueError("instructions must be nonblank plain text")
        return value


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a development artifact with standard JSON."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                      default=validate_number)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text + '\n')
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_candidate(root: Path, candidate: Candidate) -> str:
    """Save immutable content-addressed data, deduplicating parallel writes."""
    root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(candidate.model_dump(), sort_keys=True,
                      ensure_ascii=False, separators=(",", ":")) + "\n"
    candidate_id = hashlib.sha256(text.encode()).hexdigest()
    path = root / f"{candidate_id}.json"
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_text(encoding='utf-8') != text:
                raise ValueError('candidate storage integrity mismatch')
    finally:
        temporary.unlink(missing_ok=True)
    return candidate_id


def regex_agrees(case: dict[str, Any], call: FunctionCall) -> bool:
    """Check finite examples in a timeout-bounded child, never in the loop."""
    expected = case["expected_parameters"]
    actual = call.parameters
    if (call.name != case["expected_name"] or actual.keys() != expected.keys()
            or any(actual[k] != expected[k] or type(actual[k]) is not type(
                expected[k]) for k in expected if k != "regex")
            or not isinstance(actual.get("regex"), str)):
        return False
    program = (
        "import json,re,sys\n"
        "d=json.load(sys.stdin)\n"
        "ok=all(re.sub(d['regex'],d['replacement'],e['input'])==e['output'] "
        "for e in d['checks'])\n"
        "sys.exit(0 if ok else 1)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", program], input=json.dumps({
                "regex": actual["regex"], "replacement": actual["replacement"],
                "checks": case["regex_checks"],
            }), text=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=1,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def score_records(
    cases: list[dict[str, Any]], records: list[dict[str, Any]],
    functions: dict[str, FunctionDef] | None = None,
) -> dict[str, Any]:
    """Score all inputs, distinguishing exact gold and finite regex checks."""
    indexed = {row["id"]: row for row in records}
    if len(indexed) != len(records):
        raise ValueError("duplicate result case IDs")
    result: dict[str, Any] = {
        "total": len(cases), "completed": 0, "valid": 0, "name_correct": 0,
        "correct_ids": [], "regex_semantic_ids": [], "errors": [],
        "by_function": {}, "by_type": {},
    }
    for case in cases:
        name = case["expected_name"]
        group = result["by_function"].setdefault(
            name, {"total": 0, "correct": 0})
        group["total"] += 1
        for value in case['expected_parameters'].values():
            kind = ('boolean' if isinstance(value, bool) else
                    'string' if isinstance(value, str) else 'number')
            result['by_type'].setdefault(kind, {'total': 0,
                                                'correct': 0})['total'] += 1
        row = indexed.get(case["id"], {})
        if "call" not in row:
            result["errors"].append(case["id"])
            continue
        result["completed"] += 1
        try:
            call = FunctionCall.model_validate(row["call"])
            if call.prompt != case["prompt"]:
                raise ValueError("prompt copy mismatch")
            if functions is not None:
                build_parameters_model(functions[call.name]).model_validate(
                    call.parameters)
        except (ValueError, KeyError):
            result["errors"].append(case["id"])
            continue
        result["valid"] += 1
        result["name_correct"] += call.name == name
        correct = assess_call(case, call)
        if correct:
            result["correct_ids"].append(case["id"])
            group["correct"] += 1
        if "regex_checks" in case and regex_agrees(case, call):
            result["regex_semantic_ids"].append(case["id"])
        for key, value in case["expected_parameters"].items():
            kind = ("boolean" if isinstance(value, bool) else
                    "string" if isinstance(value, str) else "number")
            counts = result["by_type"].setdefault(
                kind, {"total": 0, "correct": 0})
            counts["correct"] += assess_call(
                dict(case, expected_parameters={key: value}),
                FunctionCall(prompt=call.prompt, name=call.name,
                             parameters={key: call.parameters[key]})
            ) if key in call.parameters else 0
    groups = result["by_function"].values()
    result["macro_accuracy"] = (sum(g["correct"] / g["total"] for g in groups)
                                / len(result["by_function"]))
    return result


def eligible(base: dict[str, Any], trial: dict[str, Any]) -> bool:
    """Require more exact correct cases without regressions or lost records."""
    return (trial["total"] == base["total"]
            and trial["valid"] == trial["total"]
            and set(base["correct_ids"]) < set(trial["correct_ids"]))


def select_candidate(
    base: dict[str, Any], trials: list[tuple[str, dict[str, Any]]],
) -> str | None:
    """Pick one by development correctness, keeping first proposal on ties."""
    accepted = [(name, row) for name, row in trials if eligible(base, row)]
    if not accepted:
        return None
    return max(accepted, key=lambda item: len(item[1]["correct_ids"]))[0]
