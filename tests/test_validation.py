"""Exercise dynamic Subject data contracts without the SDK or a model."""

import json
import tempfile
import unittest
from pathlib import Path

from src.validation import (
    FunctionDefinition,
    Prompt,
    load_definitions,
    load_prompts,
    validate_result,
)


class ValidationTests(unittest.TestCase):
    """Reject invalid structure and types, while leaving semantics to A06."""

    def setUp(self) -> None:
        """Use names and arguments unrelated to the repository samples."""
        self.definition = {
            "name": "measure_温度",
            "description": "Record a measurement.",
            "parameters": {
                "reading": {"type": "number"},
                "unit": {"type": "string"},
                "enabled": {"type": "boolean"},
            },
            "returns": {"type": "boolean"},
        }
        self.definitions = [FunctionDefinition.model_validate(self.definition)]
        self.prompt = '記録: "temperature"\n'
        self.result = {
            "prompt": self.prompt,
            "name": "measure_温度",
            "parameters": {"reading": -12.5, "unit": "", "enabled": True},
        }

    def validate(self, result: object) -> dict[str, object]:
        """Pass real serialized JSON through the public validation boundary."""
        return validate_result(
            json.dumps(result), self.prompt, self.definitions
        ).model_dump()

    def test_dynamic_result_preserves_prompt_and_values(self) -> None:
        """A semantically wrong reading is still a valid saved result."""
        result = self.validate(self.result)
        self.assertEqual(result, {
            "prompt": '記録: "temperature"\n',
            "name": "measure_温度",
            "parameters": {"reading": -12.5, "unit": "", "enabled": True},
        })

    def test_invalid_result_shapes_and_types(self) -> None:
        """Missing, additional, coercible, and wrong fields must all fail."""
        cases: list[object] = [
            [], None, "prose", 42,
            {**self.result, "prompt": self.prompt.strip()},
            {**self.result, "prompt": 1},
            {**self.result, "name": "not_a_candidate"},
            {**self.result, "name": ["measure_温度"]},
            {**self.result, "parameters": []},
            {**self.result, "extra": "metadata"},
            {k: v for k, v in self.result.items() if k != "name"},
        ]
        parameters = self.result["parameters"]
        assert isinstance(parameters, dict)
        for key in parameters:
            cases.append({
                **self.result,
                "parameters": {
                    k: v for k, v in parameters.items() if k != key
                },
            })
        for key, value in (
            ("reading", "-12.5"), ("reading", True), ("reading", None),
            ("unit", 10), ("enabled", 1), ("enabled", "true"),
            ("surplus", "extra"),
        ):
            cases.append({**self.result, "parameters": {
                **parameters, key: value,
            }})
        for result in cases:
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.validate(result)

    def test_json_syntax_and_nonstandard_numbers_are_rejected(self) -> None:
        """No fences, trailing prose, duplicate keys, or NaN/Infinity."""
        valid = json.dumps(self.result)
        for text in (
            "", "{", f"```json\n{valid}\n```", valid + " comment",
            valid + valid, valid.replace("-12.5", "NaN"),
            valid.replace("-12.5", "Infinity"),
            valid.replace("-12.5", "-Infinity"),
            valid.replace("-12.5", "1e999"),
            valid.replace('"unit": ""', '"unit": "", "unit": "C"'),
            '{"prompt":"wrong",' + valid[1:],
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_result(text, self.prompt, self.definitions)

    def test_number_preserves_large_integers(self) -> None:
        """JSON integers must not undergo float rounding during validation."""
        for value in (0, -9, 10 ** 100, 0.25, -2.5e20):
            with self.subTest(value=value):
                result = self.validate({**self.result, "parameters": {
                    "reading": value, "unit": "C", "enabled": False,
                }})
                parameters = result["parameters"]
                assert isinstance(parameters, dict)
                self.assertEqual(parameters["reading"], value)
                self.assertIs(type(parameters["reading"]), type(value))

    def test_all_flat_json_types_and_reserved_parameter_names(self) -> None:
        """Aliases allow arbitrary JSON keys, including Pydantic methods."""
        definition = FunctionDefinition.model_validate({
            "name": "dynamic",
            "description": "Flat JSON types.",
            "parameters": {
                "model_dump": {"type": "integer"},
                "__private": {"type": "null"},
                "not-an-identifier": {"type": "array"},
                "argument_0": {"type": "object"},
            },
            "returns": {"type": "null"},
        })
        result = {
            "prompt": "", "name": "dynamic",
            "parameters": {
                "model_dump": 7, "__private": None,
                "not-an-identifier": [True, {"x": "y"}, 3],
                "argument_0": {"nested": [None]},
            },
        }
        validated = validate_result(json.dumps(result), "", [definition])
        self.assertEqual(validated.model_dump(), result)
        for key, value in (
            ("model_dump", 7.0), ("model_dump", True),
            ("__private", "null"), ("not-an-identifier", {}),
            ("argument_0", []),
        ):
            parameters = result["parameters"]
            assert isinstance(parameters, dict)
            invalid = {**result, "parameters": {**parameters, key: value}}
            with self.subTest(key=key, value=value), \
                    self.assertRaises(ValueError):
                validate_result(json.dumps(invalid), "", [definition])

    def test_zero_argument_function(self) -> None:
        """No listed parameters means exactly an empty object is valid."""
        definition = FunctionDefinition.model_validate({
            **self.definition, "parameters": {},
        })
        valid = {**self.result, "parameters": {}}
        self.assertEqual(
            validate_result(json.dumps(valid), self.prompt, [definition])
            .model_dump(), valid,
        )
        with self.assertRaises(ValueError):
            validate_result(json.dumps(self.result), self.prompt, [definition])

    def test_input_files_and_definition_errors(self) -> None:
        """Malformed and ambiguous definitions fail before generation."""
        invalid_definitions: list[object] = [
            {}, [], [self.definition, self.definition], [None],
            [{**self.definition, "name": ""}],
            [{**self.definition, "description": 4}],
            [{**self.definition, "parameters": []}],
            [{**self.definition, "parameters": {"x": {"type": "float"}}}],
            [{**self.definition, "parameters": {"x": {}}}],
            [{**self.definition, "parameters": {
                "x": {"type": "number", "minimum": 0},
            }}],
            [{**self.definition, "returns": {"type": "unknown"}}],
            [{k: v for k, v in self.definition.items() if k != "returns"}],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            with self.assertRaises(OSError):
                load_definitions(path)
            for value in invalid_definitions:
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_definitions(path)
            for text in ("{", "null", '[{"prompt":"a","prompt":"b"}]'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_prompts(path)
            path.write_bytes(b"\xff")
            with self.assertRaises(UnicodeError):
                load_prompts(path)
            path.write_text(json.dumps([self.definition]), encoding="utf-8")
            self.assertEqual(load_definitions(path)[0].name, "measure_温度")

    def test_prompt_records_are_validated_individually(self) -> None:
        """File parsing keeps malformed records for the fail-fast loop."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text('[{"prompt":""},{"prompt":4}]', encoding="utf-8")
            records = load_prompts(path)
            self.assertEqual(Prompt.model_validate(records[0]).prompt, "")
            with self.assertRaises(ValueError):
                Prompt.model_validate(records[1])
        for record in ("hi", {}, {"prompt": None}, {"prompt": "a", "x": 1}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                Prompt.model_validate(record)
