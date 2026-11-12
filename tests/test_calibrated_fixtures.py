"""Validate the calibrated fixtures without loading the LLM or SDK."""

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "fixtures" / "calibrated"
FREQUENCIES = {
    "original": (4, 1, 1, 2, 3),
    "transfer": (1, 3, 2, 2, 3),
    "boolean": (2, 1, 1, 4, 3),
    "difference": (3, 1, 2, 2, 3),
    "magnitude": (1, 2, 3, 2, 3),
    "word_pairs": (2, 3, 1, 2, 3),
    "timer_flags": (2, 1, 2, 3, 3),
    "feature_flags": (3, 2, 1, 2, 3),
}
BOOLEAN_SUITES = {"boolean", "timer_flags", "feature_flags"}


class CalibratedFixtureTests(unittest.TestCase):
    """Check schema shape, reference labels and test-set coverage."""

    def test_fixture_consistency(self) -> None:
        """Verify inputs against function definitions and gold labels."""
        for suite, distribution in FREQUENCIES.items():
            with self.subTest(suite=suite):
                root = ROOT / suite
                functions = json.loads(
                    (root / "functions_definition.json").read_text(
                        encoding="utf-8"))
                prompts = json.loads(
                    (root / "function_calling_tests.json").read_text(
                        encoding="utf-8"))
                fixture = json.loads(
                    (root / "measurement_fixture.json").read_text(
                        encoding="utf-8"))
                self.assertEqual(functions, fixture["functions"])
                self.assertEqual(prompts, [
                    {"prompt": case["prompt"]} for case in fixture["cases"]
                ])
                self.assertEqual(len(functions), 5)
                self.assertEqual(len(prompts), 11)
                names = [function["name"] for function in functions]
                self.assertEqual(len(set(names)), 5)
                self.assertEqual([
                    len(function["parameters"]) for function in functions
                ], [2, 1, 1, 1, 3])
                indexed = {fn["name"]: fn for fn in functions}
                counts = {name: 0 for name in indexed}
                seen: set[str] = set()
                regex_count = 0
                bool_values: set[bool] = set()
                mixed_types = False
                for case in fixture["cases"]:
                    self.assertNotIn(case["prompt"], seen)
                    seen.add(case["prompt"])
                    name = case["expected_name"]
                    self.assertIn(name, indexed)
                    counts[name] += 1
                    parameters = indexed[name]["parameters"]
                    self.assertEqual(
                        set(parameters),
                        set(case["expected_parameters"]),
                    )
                    types = {
                        spec["type"] for spec in parameters.values()
                    }
                    mixed_types |= types == {"number", "boolean"}
                    for key, spec in parameters.items():
                        value = case["expected_parameters"][key]
                        kind = spec["type"]
                        expected_types: dict[str, type | tuple[type, ...]] = {
                            "string": str,
                            "number": (int, float),
                            "boolean": bool,
                        }
                        self.assertIsInstance(value, expected_types[kind])
                        if kind == "number":
                            self.assertNotIsInstance(value, bool)
                        if kind == "boolean":
                            bool_values.add(value)
                    if "regex_checks" in case:
                        regex_count += 1
                        args = case["expected_parameters"]
                        for check in case["regex_checks"]:
                            self.assertEqual(
                                re.sub(
                                    args["regex"], args["replacement"],
                                    check["input"],
                                ),
                                check["output"],
                            )
                self.assertEqual(
                    tuple(counts.values()), distribution)
                self.assertEqual(regex_count, 3)
                if suite in BOOLEAN_SUITES:
                    self.assertEqual(bool_values, {True, False})
                    self.assertTrue(mixed_types)
                else:
                    self.assertEqual(bool_values, set())


if __name__ == "__main__":
    unittest.main()
