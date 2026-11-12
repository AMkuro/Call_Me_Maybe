"""Few-shot parameter examples in the parameter prompt (ADR-0020)."""

import json
from unittest.mock import patch
import unittest

from src import prompt
from src.schema import FunctionDef, Scalar

FN = FunctionDef.model_validate({
    'name': 'fn_scale', 'description': 'Scale a value.',
    'parameters': {'x': {'type': 'number'}, 'unit': {'type': 'string'},
                   'round': {'type': 'boolean'}},
    'returns': {'type': 'number'}})
REQUEST = 'Scale -0.25 in "m/s", rounded'
VALUES: dict[str, Scalar] = {'x': -0.25, 'unit': 'm/s', 'round': True}


class PromptExampleTests(unittest.TestCase):
    """Examples are earlier chat turns in exactly the generated format."""

    def test_no_examples_show_the_user_request_first(self) -> None:
        """The request is the user's; the function is the call for it."""
        definition = json.dumps(FN.model_dump(), separators=(',', ':'))
        expected = (
            f'<|im_start|>system\n{prompt.PARAMETER_INSTRUCTIONS}<|im_end|>\n'
            f'<|im_start|>user\nUser request: "Go"\nFunction to call: '
            f'{definition}'
            '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
        with patch.object(prompt, 'PARAMETER_EXAMPLES', []):
            self.assertEqual(prompt.build_parameters_prompt('Go', FN),
                             expected)

    def test_example_turn_precedes_request_without_think_block(self) -> None:
        """Matches the Qwen3 template for a past assistant turn."""
        with patch.object(prompt, 'PARAMETER_EXAMPLES',
                          [(FN, REQUEST, VALUES)]):
            text = prompt.build_parameters_prompt('Go', FN)
        answer = prompt.example_answer(REQUEST, FN, VALUES)
        self.assertEqual(text.count('<think>'), 1)
        self.assertLess(text.index(answer + '<|im_end|>'),
                        text.index('User request: "Go"'))
        self.assertEqual(answer, '{"prompt": "Scale -0.25 in \\"m/s\\", '
                         'rounded", "name": "fn_scale", "parameters": '
                         '{"x": -0.25, "unit": "m/s", "round": true}}')


if __name__ == '__main__':
    unittest.main()
