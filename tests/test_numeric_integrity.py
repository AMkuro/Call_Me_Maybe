"""Check exact JSON numeric values independently of binary float storage."""

import json
import math
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from test_application import definition

from src.errors import PromptError
from src.pipeline import validate_call
from src.schema import (
    FunctionCallRequest,
    build_argument_validation_model,
    validate_number,
)
from src.writer import save_results
from tools.measure_application import assess_call


def record(literal: str) -> str:
    """Embed a hand-written numeric literal without rounding through float."""
    return (
        '{"prompt":"x","name":"dynamic","parameters":{"value":'
        + literal
        + "}}"
    )


class NumericIntegrityTests(unittest.TestCase):
    """Number validation must preserve the eventual JSON number or fail."""

    def test_direct_numeric_inputs_become_checked_floats(self) -> None:
        """Direct construction and declared numbers use the same checks."""
        model = build_argument_validation_model(definition(value="number"))
        values: list[Any] = [10.0, Decimal("10.0"), Decimal("0.1"), 2.75]
        for value in values:
            with self.subTest(value=value):
                self.assertIs(type(validate_number(value)), float)
                checked = model.model_validate({"value": value})
                result = checked.model_dump(by_alias=True)["value"]
                call = FunctionCallRequest(
                    prompt="x", name="dynamic", parameters={"value": value}
                )
                self.assertIs(type(result), float)
                self.assertTrue(math.isfinite(result))
                self.assertIs(type(call.parameters["value"]), float)
                self.assertEqual(Decimal(json.dumps(result)), Decimal(value))
        invalid: list[tuple[Any, str]] = [
            (10, "integer"),
            (0, "integer"),
            (-3, "integer"),
            (9007199254740993, "integer"),
            (Decimal(9007199254740993), "precision"),
            (10**400, "integer"),
            (Decimal("1e400"), "overflow"),
            (Decimal("1e-400"), "underflow"),
        ]
        for value, reason in invalid:
            with self.subTest(reason=reason, type=type(value)):
                with self.assertRaisesRegex(ValueError, reason):
                    validate_number(value)
                with self.assertRaisesRegex(ValidationError, reason):
                    model.model_validate({"value": value})
                with self.assertRaisesRegex(ValidationError, reason):
                    FunctionCallRequest(
                        prompt="x", name="dynamic", parameters={"value": value}
                    )

    def test_regular_decimals_and_integral_decimal_are_preserved(self) -> None:
        """Allow 0.1 and preserve integer-valued decimals without rounding."""
        fn = definition(value="number")
        for literal in [
            "0.1",
            "2.5",
            "-12.75",
            "1.00",
            "-0.0",
            "1e-300",
            "9007199254740992.0",
        ]:
            with self.subTest(literal=literal):
                call = validate_call(record(literal), fn, "x")
                encoded = json.dumps(call.parameters, allow_nan=False)
                parsed = json.loads(encoded, parse_float=Decimal)
                self.assertNotIsInstance(parsed["value"], str)
                self.assertEqual(
                    Decimal(str(parsed["value"])), Decimal(literal)
                )

    def test_precision_loss_and_out_of_range_decimals_are_errors(self) -> None:
        """Reject rounding, underflow and overflow rather than saving them."""
        fn = definition(value="number")
        for literal in [
            "0.1234567890123456789",
            "1.0000000000000000001",
            "0.100000000000000005",
            "1e309",
            "-1e309",
            "1e-999",
            "-1e-999",
            "NaN",
            "Infinity",
            "-Infinity",
        ]:
            with self.subTest(literal=literal):
                with self.assertRaises(PromptError):
                    validate_call(record(literal), fn, "x")

    def test_saved_json_keeps_numeric_type_and_decimal_value(self) -> None:
        """Use an independent decimal parser on the actual saved file."""
        fn = definition(value="number")
        literals = [
            "9007199254740992.0",
            "1" + "0" * 300 + ".0",
            "-12.75",
            "0.1",
            "-0.0",
        ]
        calls = [
            validate_call(record(literal), fn, "x") for literal in literals
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calls.json"
            save_results(path, calls)
            saved = json.loads(path.read_text(), parse_float=Decimal)
            standard = json.loads(path.read_text())
        for literal, call, standard_call in zip(literals, saved, standard):
            value = call["parameters"]["value"]
            self.assertIs(type(value), Decimal)
            self.assertIs(type(standard_call["parameters"]["value"]), float)
            self.assertEqual(Decimal(str(value)), Decimal(literal))

    def test_generated_integers_require_float_spelling_and_range(self) -> None:
        """Generated records reject integer spelling and float overflow."""
        for literal in ["2", "9007199254740993", "9" * 400, "9" * 400 + ".0"]:
            with self.subTest(literal=literal[:35]):
                with self.assertRaises(PromptError):
                    validate_call(
                        record(literal), definition(value="number"), "x"
                    )

    def test_accepted_records_pass_the_removed_float_recheck(self) -> None:
        """Whatever validate_call accepts loads as a finite float in json."""
        fn = definition(value="number")
        accepted = 0
        for literal in [
            "0.1",
            "-0.0",
            "2.50",
            "1e5",
            "1e-300",
            "2",
            "-7",
            "9007199254740992.0",
            "9007199254740993",
            "1e309",
            "1e-999",
            "0.1234567890123456789",
        ]:
            with self.subTest(literal=literal):
                value = json.loads(record(literal))["parameters"]["value"]
                old_check = isinstance(value, float) and math.isfinite(value)
                try:
                    validate_call(record(literal), fn, "x")
                except PromptError:
                    continue
                accepted += 1
                self.assertTrue(old_check, literal)
        self.assertEqual(accepted, 6)

    def test_strict_numeric_and_boolean_types(self) -> None:
        """Do not interpret numeric strings, bools or boolean-like strings."""
        number = build_argument_validation_model(definition(value="number"))
        boolean = build_argument_validation_model(definition(value="boolean"))
        for value in [10, 0, -3, "2", "２", True, False, None, [], {}]:
            with self.subTest(number=value):
                with self.assertRaises(ValueError):
                    validate_number(value)
                with self.assertRaises(ValidationError):
                    number.model_validate({"value": value})
        for value in [1, 0, 1.0, "false", "true", "1", None]:
            with self.subTest(boolean=value):
                with self.assertRaises(ValidationError):
                    boolean.model_validate({"value": value})
        for value in [2.0, 2.5]:
            self.assertEqual(
                number.model_validate({"value": value}).model_dump()[
                    "parameter_0"
                ],
                value,
            )
        for value in [True, False]:
            self.assertIs(
                boolean.model_validate({"value": value}).model_dump()[
                    "parameter_0"
                ],
                value,
            )

    def test_nonfinite_numbers_and_unknown_scalars(self) -> None:
        """Pydantic also protects non-JSON construction before saving."""
        for value in [
            math.inf,
            -math.inf,
            math.nan,
            Decimal("NaN"),
            Decimal("Infinity"),
            None,
            [],
            {},
        ]:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    FunctionCallRequest.model_validate(
                        {
                            "prompt": "x",
                            "name": "dynamic",
                            "parameters": {"value": value},
                        }
                    )

    def test_runtime_integer_conversion_limit_is_a_clear_error(self) -> None:
        """Use the runtime's actual technical limit, not a project constant."""
        get_limit = getattr(sys, "get_int_max_str_digits", lambda: 0)
        limit = get_limit()
        if not limit:
            self.skipTest("this interpreter has no integer-string limit")
        with self.assertRaisesRegex(PromptError, "limit|digits"):
            validate_call(
                record("9" * (limit + 1)), definition(value="number"), "x"
            )

    def test_measurement_checks_types_and_exact_decimals(self) -> None:
        """Python True == 1 must not make a wrong typed argument correct."""
        self.assertFalse(
            assess_call(
                {
                    "expected_name": "dynamic",
                    "expected_parameters": {"value": 1},
                },
                FunctionCallRequest(
                    prompt="x", name="dynamic", parameters={"value": True}
                ),
            )
        )
        self.assertTrue(
            assess_call(
                {
                    "expected_name": "dynamic",
                    "expected_parameters": {"value": Decimal("0.1")},
                },
                FunctionCallRequest(
                    prompt="x", name="dynamic", parameters={"value": 0.1}
                ),
            )
        )
