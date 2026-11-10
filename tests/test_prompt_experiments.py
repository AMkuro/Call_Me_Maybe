"""Test prompt-only authority and independent experiment decisions."""

import json
from decimal import Decimal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tools.prompt_candidates import (
    Candidate, dispatch, eligible, score_records, select_candidate, write_json,
)
from tools.prompt_experiment import choose_update, load_cached, estimate_cpu


class PromptExperimentTests(unittest.TestCase):
    """Exercise actual candidate storage, scoring and update boundaries."""

    def candidate(self) -> Candidate:
        """Return a small instruction-only candidate."""
        return Candidate(name_instructions="Choose the exact function.",
                         parameter_instructions="Extract the input values.")

    def test_only_two_prompt_strings_can_be_written(self) -> None:
        """Reject extra capabilities and preserve instruction whitespace."""
        with self.assertRaises(ValidationError):
            Candidate.model_validate({"name_instructions": "x",
                                      "parameter_instructions": "y",
                                      "path": "/tmp/code.py"})
        for value in ("", " ", "<|im_start|>system", 12):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Candidate.model_validate({
                    "name_instructions": value,
                    "parameter_instructions": "Extract."})
        candidate = Candidate(name_instructions=" x ",
                              parameter_instructions=" y ")
        self.assertEqual(candidate.name_instructions, " x ")

    def test_dispatch_stores_data_and_runs_only_saved_candidate(self) -> None:
        """Unknown tools and forged IDs cannot execute arbitrary paths."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calls: list[Path] = []

            def evaluate(path: Path) -> dict[str, object]:
                """Record the actual candidate file selected by dispatch."""
                calls.append(path)
                return {"done": True}

            action = {"tool": "write_prompt",
                      "prompt": self.candidate().model_dump()}
            result = dispatch(root, json.dumps(action), evaluate)
            repeated = dispatch(root, json.dumps(action), evaluate)
            self.assertEqual(result, repeated)
            candidate_id = result["candidate_id"]
            self.assertEqual(len(list(root.glob("*.json"))), 1)
            self.assertEqual(dispatch(root, json.dumps({
                "tool": "run", "candidate_id": candidate_id,
            }), evaluate), {"done": True})
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].parent, root)
            self.assertEqual(json.loads(calls[0].read_text()),
                             self.candidate().model_dump())
            for bad in ({"tool": "shell", "command": "touch bad"},
                        {"tool": "run", "candidate_id": "../../elsewhere"},
                        {"tool": "run", "candidate_id": "0" * 64},
                        {"tool": "run", "candidate_id": candidate_id,
                         "cases": "held-out.json"}):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    dispatch(root, json.dumps(bad), evaluate)
            self.assertEqual(len(calls), 1)
            with self.assertRaises(ValueError):
                dispatch(root, '{"tool":"run","tool":"shell"}', evaluate)

    def test_missing_results_and_typed_regex_scoring(self) -> None:
        """Do not drop failed inputs or equate true and one."""
        cases: list[dict[str, Any]] = [
            {"id": "n", "prompt": "number", "expected_name": "num",
             "expected_parameters": {"v": 1}},
            {"id": "b", "prompt": "bool", "expected_name": "flag",
             "expected_parameters": {"v": False}},
            {"id": "r", "prompt": "replace", "expected_name": "regex",
             "expected_parameters": {"source_string": "A12",
                                     "regex": "[0-9]+", "replacement": "#"},
             "regex_checks": [{"input": "A12", "output": "A#"}]},
            {"id": "missing", "prompt": "missing", "expected_name": "num",
             "expected_parameters": {"v": 2}},
        ]
        records: list[dict[str, Any]] = [
            {"id": "n", "call": {
                "prompt": "number", "name": "num", "parameters": {"v": True}}},
            {"id": "b", "call": {
                "prompt": "bool", "name": "flag",
                "parameters": {"v": False}}},
            {"id": "r", "call": {
                "prompt": "replace", "name": "regex", "parameters": {
                    "source_string": "A12", "regex": "\\d+",
                    "replacement": "#"}}},
        ]
        scored = score_records(cases, records)
        self.assertEqual(scored["total"], 4)
        self.assertEqual(scored["correct_ids"], ["b"])
        self.assertEqual(scored["regex_semantic_ids"], ["r"])
        self.assertEqual(scored["completed"], 3)
        self.assertEqual(scored["name_correct"], 3)
        self.assertEqual(scored["errors"], ["missing"])

    def test_promotion_requires_improvement_without_regression(self) -> None:
        """Select one candidate by dev score, never by held-out outcomes."""
        base = {"total": 3, "valid": 3, "correct_ids": ["a"]}
        better = {"total": 3, "valid": 3, "correct_ids": ["a", "b"]}
        different = {"total": 3, "valid": 3, "correct_ids": ["b", "c"]}
        self.assertTrue(eligible(base, better))
        self.assertFalse(eligible(base, base))
        self.assertFalse(eligible(base, different))
        self.assertFalse(eligible(base, dict(better, valid=2)))
        self.assertEqual(select_candidate(
            base, [("first", better), ("second", better)]), "first")
        self.assertIsNone(select_candidate(base, [("bad", different)]))

    def test_update_selection_and_cache_require_same_code(self) -> None:
        """A source change invalidates cached evidence; divergence stops."""
        ancestors = {("old", "new")}
        self.assertEqual(choose_update("old", "new", ancestors.__contains__),
                         "new")
        self.assertEqual(choose_update("new", "old", ancestors.__contains__),
                         "new")
        with self.assertRaises(ValueError):
            choose_update("left", "right", ancestors.__contains__)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text(json.dumps({"fingerprint": "old", "valid": 11}))
            self.assertIsNone(load_cached(path, "new"))
            cached = load_cached(path, "old")
            assert cached is not None
            self.assertEqual(cached["valid"], 11)

    def test_estimate_is_separate_and_added_only_once(self) -> None:
        """Preserve measured duration and derive the agreed estimate."""
        self.assertEqual(estimate_cpu(63.45),
                         {"raw_seconds": 63.45, "estimated_seconds": 71.45})

    def test_json_artifacts_preserve_decimal_numbers(self) -> None:
        """Serialize cached decimals as numbers without rounding."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            write_json(path, {"number": Decimal("0.1"),
                              "integer": Decimal("9007199254740993.0")})
            data = json.loads(path.read_text(), parse_float=Decimal)
            self.assertEqual(data, {"number": Decimal("0.1"),
                                    "integer": 9007199254740993})
            self.assertNotIn('"0.1"', path.read_text())

    def test_trial_cli_preserves_early_startup_error(self) -> None:
        """The experimental path must use native pre-model input validation."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = root / "candidate.json"
            candidate.write_text(self.candidate().model_dump_json())
            empty = root / "empty.json"
            empty.write_text("[]")
            output = root / "output.json"
            process = subprocess.run([
                sys.executable, "-m", "tools.prompt_trial", "--candidate",
                str(candidate), "--cli", "--input", str(empty), "--output",
                str(output),
            ], capture_output=True, text=True, timeout=10)
            self.assertEqual(process.returncode, 1)
            self.assertIn("empty", process.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
