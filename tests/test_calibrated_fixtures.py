"""Validate the held-out JSON fixtures without model inference."""

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "fixtures" / "calibrated"
SUITES = ("original", "transfer", "boolean")


class CalibratedFixtureTests(unittest.TestCase):
    """Check the native inputs against the separately labeled cases."""

    def test_fixture_consistency(self) -> None:
        """Verify shape, distribution, argument values and regex examples."""
        for suite in SUITES:
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
                indexed = {fn["name"]: fn for fn in functions}
                self.assertEqual(len(indexed), 5)
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
                    params = indexed[name]["parameters"]
                    self.assertEqual(
                        set(params), set(case["expected_parameters"]))
                    types = {spec["type"] for spec in params.values()}
                    mixed_types |= types == {"number", "boolean"}
                    for key, spec in params.items():
                        value = case["expected_parameters"][key]
                        kind = spec["type"]
                        expected_type = {
                            "string": str,
                            "number": (int, float),
                            "boolean": bool,
                        }[kind]
                        self.assertIsInstance(value, expected_type)
                        if kind == "number":
                            self.assertNotIsInstance(value, bool)
                        if kind == "boolean":
                            bool_values.add(value)
                    if "regex_checks" in case:
                        regex_count += 1
                        args = case["expected_parameters"]
                        for example in case["regex_checks"]:
                            self.assertEqual(re.sub(
                                args["regex"], args["replacement"],
                                example["input"]), example["output"])
                self.assertEqual(sum(counts.values()), 11)
                self.assertGreaterEqual(len(set(counts.values())), 3)
                self.assertEqual(regex_count, 3)
                if suite == "boolean":
                    self.assertEqual(bool_values, {True, False})
                    self.assertTrue(mixed_types)
                else:
                    self.assertEqual(bool_values, set())


if __name__ == "__main__":
    unittest.main()
