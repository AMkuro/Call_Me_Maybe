"""Check exact JSON numeric values independently of binary float storage."""

import json
import math
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from src.errors import PromptError
from src.pipeline import validate_call
from src.schema import FunctionCall, build_parameters_model
from src.writer import save_results
from tools.measure_application import assess_call
from test_application import definition


def record(literal: str) -> str:
    """Embed a hand-written numeric literal without rounding through float."""
    return ('{"prompt":"x","name":"dynamic","parameters":{"value":'
            + literal + '}}')


class NumericIntegrityTests(unittest.TestCase):
    """Number validation must preserve the eventual JSON number or fail."""

    def test_integer_boundaries_and_large_integers_remain_exact(self) -> None:
        """Integers beyond binary64 precision or range remain Python ints."""
        fn = definition(value='number')
        for literal in ['0', '-2', '2', '9007199254740991',
                        '9007199254740992', '9007199254740993',
                        '-' + '9' * 400]:
            with self.subTest(literal=literal[:35]):
                value = validate_call(record(literal), fn).parameters['value']
                self.assertIs(type(value), int)
                self.assertEqual(value, int(literal))

    def test_regular_decimals_and_integral_decimal_are_preserved(self) -> None:
        """Allow 0.1 and preserve integer-valued decimals without rounding."""
        fn = definition(value='number')
        for literal in ['0.1', '2.5', '-12.75', '1.00', '-0.0',
                        '1e-300', '9007199254740993.0']:
            with self.subTest(literal=literal):
                call = validate_call(record(literal), fn)
                encoded = json.dumps(call.parameters, allow_nan=False)
                parsed = json.loads(encoded, parse_float=Decimal)
                self.assertNotIsInstance(parsed['value'], str)
                self.assertEqual(Decimal(str(parsed['value'])),
                                 Decimal(literal))

    def test_precision_loss_and_out_of_range_decimals_are_errors(self) -> None:
        """Reject rounding, underflow and overflow rather than saving them."""
        fn = definition(value='number')
        for literal in ['0.1234567890123456789', '1.0000000000000000001',
                        '0.100000000000000005', '1e309', '-1e309',
                        '1e-999', '-1e-999', 'NaN', 'Infinity', '-Infinity']:
            with self.subTest(literal=literal):
                with self.assertRaises(PromptError):
                    validate_call(record(literal), fn)

    def test_saved_json_keeps_numeric_type_and_decimal_value(self) -> None:
        """Use an independent decimal parser on the actual saved file."""
        fn = definition(value='number')
        literals = ['9007199254740993', '9' * 400, '-12.75', '0.1',
                    '9007199254740993.0']
        calls = [validate_call(record(literal), fn) for literal in literals]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'calls.json'
            save_results(path, calls)
            saved = json.loads(path.read_text(), parse_float=Decimal)
        for literal, call in zip(literals, saved):
            value = call['parameters']['value']
            self.assertIn(type(value), (int, Decimal))
            self.assertEqual(Decimal(str(value)), Decimal(literal))

    def test_strict_numeric_and_boolean_types(self) -> None:
        """Do not interpret numeric strings, bools or boolean-like strings."""
        number = build_parameters_model(definition(value='number'))
        boolean = build_parameters_model(definition(value='boolean'))
        for value in ['2', '２', True, False, None, [], {}]:
            with self.subTest(number=value):
                with self.assertRaises(ValidationError):
                    number.model_validate({'value': value})
        for value in [1, 0, 1.0, 'false', 'true', '1', None]:
            with self.subTest(boolean=value):
                with self.assertRaises(ValidationError):
                    boolean.model_validate({'value': value})
        for value in [2, 2.5]:
            self.assertEqual(number.model_validate({'value': value})
                             .model_dump()['parameter_0'], value)
        for value in [True, False]:
            self.assertIs(boolean.model_validate({'value': value})
                          .model_dump()['parameter_0'], value)

    def test_nonfinite_numbers_and_unknown_scalars(self) -> None:
        """Pydantic also protects non-JSON construction before saving."""
        for value in [math.inf, -math.inf, math.nan, Decimal('NaN'),
                      Decimal('Infinity'), None, [], {}]:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    FunctionCall.model_validate({
                        'prompt': 'x', 'name': 'dynamic',
                        'parameters': {'value': value},
                    })

    def test_runtime_integer_conversion_limit_is_a_clear_error(self) -> None:
        """Use the runtime's actual technical limit, not a project constant."""
        get_limit = getattr(sys, 'get_int_max_str_digits', lambda: 0)
        limit = get_limit()
        if not limit:
            self.skipTest('this interpreter has no integer-string limit')
        with self.assertRaisesRegex(PromptError, 'limit|digits'):
            validate_call(record('9' * (limit + 1)),
                          definition(value='number'))

    def test_measurement_does_not_cast_huge_integers_to_float(self) -> None:
        """Scoring an exact integer outside float range must not overflow."""
        literal = '9' * 400
        call = FunctionCall(prompt='x', name='dynamic',
                            parameters={'number': int(literal)})
        self.assertTrue(assess_call({
            'expected_name': 'dynamic', 'requested_number_literal': literal,
        }, call))

    def test_measurement_checks_types_and_exact_decimals(self) -> None:
        """Python True == 1 must not make a wrong typed argument correct."""
        self.assertFalse(assess_call({
            'expected_name': 'dynamic', 'expected_parameters': {'value': 1},
        }, FunctionCall(prompt='x', name='dynamic',
                        parameters={'value': True})))
        self.assertTrue(assess_call({
            'expected_name': 'dynamic',
            'expected_parameters': {'value': Decimal('0.1')},
        }, FunctionCall(prompt='x', name='dynamic',
                        parameters={'value': 0.1})))
