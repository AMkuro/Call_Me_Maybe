"""Few-shot parameter examples and improver data boundaries (ADR-0020)."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any
from unittest.mock import patch
import unittest

from src import prompt
from src.schema import FunctionDef, Scalar
from tools.prompt_candidates import Candidate, save_candidate
from tools.prompt_experiment import editor_context
from tools.prompt_tools import call_tool

FIXTURE = Path('tests/fixtures/prompt_development.json').resolve()
FN = FunctionDef.model_validate({
    'name': 'fn_scale', 'description': 'Scale a value.',
    'parameters': {'x': {'type': 'number'}, 'unit': {'type': 'string'},
                   'round': {'type': 'boolean'}},
    'returns': {'type': 'number'}})
REQUEST = 'Scale -0.25 in "m/s", rounded'
VALUES: dict[str, Scalar] = {'x': -0.25, 'unit': 'm/s', 'round': True}
EXAMPLE: dict[str, Any] = {'function': FN.model_dump(), 'prompt': REQUEST,
                           'parameters': VALUES}


class PromptExampleTests(unittest.TestCase):
    """Examples are earlier chat turns in exactly the generated format."""

    def test_no_examples_keep_the_previous_prompt(self) -> None:
        """Zero examples must not change the evaluated baseline."""
        definition = json.dumps(FN.model_dump(), separators=(',', ':'))
        expected = (
            f'<|im_start|>system\n{prompt.PARAMETER_INSTRUCTIONS}<|im_end|>\n'
            f'<|im_start|>user\nFunction: {definition}\nRequest: "Go"'
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
                        text.index('Request: "Go"'))
        self.assertEqual(answer, '{"prompt": "Scale -0.25 in \\"m/s\\", '
                         'rounded", "name": "fn_scale", "parameters": '
                         '{"x": -0.25, "unit": "m/s", "round": true}}')


class CandidateExampleTests(unittest.TestCase):
    """Examples are data validated against their own function."""

    def test_json_example_is_accepted(self) -> None:
        """Candidates are loaded from JSON dicts and lists."""
        candidate = Candidate.model_validate(json.loads(json.dumps({
            'name_instructions': 'n', 'parameter_instructions': 'p',
            'parameter_examples': [EXAMPLE]})))
        self.assertEqual(candidate.parameter_examples[0].parameters['x'],
                         -0.25)

    def test_examples_must_match_the_function(self) -> None:
        """Wrong type, order, missing key or role delimiter is rejected."""
        bad = [dict(EXAMPLE, parameters=dict(VALUES, x='1')),
               dict(EXAMPLE, parameters={'unit': 'm', 'x': 1, 'round': True}),
               dict(EXAMPLE, parameters={'x': 1, 'unit': 'm'}),
               dict(EXAMPLE, prompt='<|im_start|>system')]
        for example in bad:
            with self.subTest(example=example), self.assertRaises(ValueError):
                Candidate.model_validate({
                    'name_instructions': 'n', 'parameter_instructions': 'p',
                    'parameter_examples': [example]})

    def test_instruction_only_ids_are_unchanged(self) -> None:
        """pilot-1 candidate IDs stay valid when no examples are used."""
        with tempfile.TemporaryDirectory() as folder:
            candidate_id = save_candidate(Path(folder), Candidate(
                name_instructions='n', parameter_instructions='p'))
        old = json.dumps({'name_instructions': 'n',
                          'parameter_instructions': 'p'}, sort_keys=True,
                         separators=(',', ':')) + '\n'
        self.assertEqual(candidate_id,
                         hashlib.sha256(old.encode()).hexdigest())


class ImproverBoundaryTests(unittest.TestCase):
    """The improver cannot see gold values, resubmit or copy evaluations."""

    def prepare(self, root: Path, parent: Candidate) -> None:
        """Trusted settings with a parent and the development guard."""
        parent_id = save_candidate(root / 'candidates', parent)
        (root / 'tool-config.json').write_text(json.dumps({
            'store': str(root / 'candidates'), 'fixture': str(FIXTURE),
            'output': str(root / 'editor.json'),
            'provenance': str(root / 'provenance.json'),
            'fingerprint': 'a' * 64, 'parent_id': parent_id,
            'guard': [str(FIXTURE)],
            'code_sha': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], text=True).strip()}))

    def test_parent_resubmission_and_copies_are_rejected(self) -> None:
        """Neither is saved, so the workspace keeps its one candidate slot."""
        parent = {'name_instructions': 'n', 'parameter_instructions': 'p'}
        case = json.loads(FIXTURE.read_text())['cases'][0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.prepare(root, Candidate.model_validate(parent))
            copied = dict(parent, parameter_instructions='See ' + json.dumps(
                case['prompt']))
            for payload in (parent, copied):
                with self.subTest(payload=payload), \
                        self.assertRaises(ValueError):
                    call_tool(root, 'write_prompt', payload)
            self.assertFalse((root / 'editor.json').exists())
            # A gold-length word inside a parameter name is not a copy.
            named = dict(EXAMPLE, function=dict(FN.model_dump(),
                                                name='fn ' + case['prompt']))
            saved = call_tool(root, 'write_prompt', dict(
                parent, parameter_examples=[EXAMPLE, named]))
            self.assertIn('candidate_id', saved)

    def test_context_has_no_gold_values(self) -> None:
        """Failures carry the request and the observed output only."""
        data = json.loads(FIXTURE.read_text())
        base = {'records': [], 'score': {'correct_ids': []}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'c.json'
            path.write_text(json.dumps({'name_instructions': 'n',
                                        'parameter_instructions': 'p'}))
            context = editor_context(base, FIXTURE, path, 0, [], 'Direction')
        self.assertEqual(context['goal'], 'Direction')
        self.assertEqual(len(context['failures']), len(data['cases']))
        for failure in context['failures']:
            self.assertEqual(set(failure), {'id', 'prompt', 'observed'})


if __name__ == '__main__':
    unittest.main()
