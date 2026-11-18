"""Few-shot parameter examples in the parameter prompt (ADR-0020)."""

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

    def test_instructions_are_the_four_needed_sentences(self) -> None:
        """Experiment 3 (2026-11-18): five other sentences changed nothing."""
        self.assertEqual(
            prompt.PARAMETER_INSTRUCTIONS,
            'Extract the arguments of that call from the user request: copy '
            'each value as it appears in the request; do not answer the '
            'request or compute a result. Return only a JSON object with '
            'prompt, name, parameters. Use JSON numbers, strings, true/false. '
            'Convert descriptions of patterns into actual regular '
            'expressions; do not list the matched substrings.')

    def test_no_examples_show_the_user_request_first(self) -> None:
        """The request is the user's; the definition omits "returns"."""
        definition = ('{"name":"fn_scale","description":"Scale a value.",'
                      '"parameters":{"x":{"type":"number"},'
                      '"unit":{"type":"string"},"round":{"type":"boolean"}}}')
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
        answer = prompt.build_example_call_json(REQUEST, FN, VALUES)
        self.assertEqual(text.count('<think>'), 1)
        self.assertLess(text.index(answer + '<|im_end|>'),
                        text.index('User request: "Go"'))
        # The user's text keeps its own ", " inside the JSON string.
        self.assertEqual(answer, '{"prompt":"Scale -0.25 in \\"m/s\\", '
                         'rounded","name":"fn_scale","parameters":'
                         '{"x": -0.25,"unit": "m/s","round": true}}')


if __name__ == '__main__':
    unittest.main()
