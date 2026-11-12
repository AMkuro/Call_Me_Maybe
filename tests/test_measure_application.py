"""Check that measurement distinguishes normalization from precision loss."""

import unittest

from src.schema import FunctionCall
from tools.measure_application import assess_call


class NumericOracleTests(unittest.TestCase):
    """Requested literals are an independent mathematical oracle."""

    def test_precision_loss_is_not_an_exact_correct_answer(self) -> None:
        """Large integers and long decimals cannot pass by rounding both."""
        for literal in ['9007199254740993', '0.12345678901234567']:
            case = {'expected_name': 'store_number',
                    'requested_number_literal': literal}
            call = FunctionCall(prompt='test', name='store_number',
                                parameters={'number': float(literal)})
            self.assertFalse(assess_call(case, call))

    def test_exact_numbers_and_changed_names(self) -> None:
        """Ordinary float normalization and notation alone remain valid."""
        for literal in ['2', '-0.25', '100000000000000000000']:
            case = {'expected_name': 'store_number',
                    'requested_number_literal': literal}
            call = FunctionCall(prompt='test', name='store_number',
                                parameters={'number': float(literal)})
            self.assertTrue(assess_call(case, call))
        self.assertFalse(assess_call(
            {'expected_name': 'other', 'expected_parameters': {}}, call,
        ))


class ProgressMeasurementTests(unittest.TestCase):
    """The new synchronous sink preserves the existing report's count keys."""

    def test_counts_reset_per_parameter_and_keep_no_name_history(self) -> None:
        """One case owns its counts; name notifications do not retain IDs."""
        from tools.measure_application import MeasurementProgress

        progress = MeasurementProgress()
        ids = [1]
        progress.stage('select_name')
        progress.name_token(ids, mode='forced', raw_top=None)
        progress.function_selected('f')
        progress.parameter_started('a', 'number')
        progress.value_token(b'12,', 2, token=3, mode='greedy')
        progress.parameter_started('b', 'boolean')
        self.assertEqual(progress.counts,
                         {'a (number)': 2, 'b (boolean)': 0})
        self.assertTrue(progress.available)
        self.assertFalse(any(value is ids
                             for value in vars(progress).values()))
        self.assertEqual(MeasurementProgress().counts, {})
