"""Check skeleton orchestration with fixed data and replaced leaf stubs."""

import contextlib
from collections.abc import Sequence
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import BaseModel

from src import __main__ as entry
from src import pipeline
from src.cli import Args
from src.errors import PromptError
from src.model_port import Model
from src.name_trie import NameTrie
from src.schema import FunctionCall, FunctionDef
from src.skeleton import Forced, ValueSlot


def encode(text: str) -> list[int]:
    """Represent fixture characters as IDs without loading a tokenizer."""
    return [ord(char) for char in text]


def decode(ids: list[int]) -> str:
    """Decode the fixture IDs without loading a model."""
    return ''.join(chr(token) for token in ids)


def definition(name: str = 'f') -> FunctionDef:
    """Provide a function whose parameters exercise all supported types."""
    return FunctionDef.model_validate({
        'name': name, 'description': 'fixture',
        'parameters': {
            'number': {'type': 'number'}, 'text': {'type': 'string'},
            'flag': {'type': 'boolean'},
        },
        'returns': {'type': 'number'},
    })


class Parameters(BaseModel):
    """Stand in for the unavailable dynamic model with real validation."""

    number: float
    text: str
    flag: bool


class PipelineTests(unittest.TestCase):
    """Test real pipeline control flow, not the unfinished leaf algorithms."""

    def test_numbers_continue_until_the_chosen_delimiter(self) -> None:
        """A valid numeric prefix must not truncate an integer or decimal."""
        for target in ('12,', '1.5}', '-0.25,', '0}'):
            with self.subTest(target=target):
                model = object.__new__(Model)
                slot = ValueSlot.model_validate({
                    'name': 'number', 'type': 'number',
                    'terminator': target[-1],
                })

                def complete(
                    written: str, kind: str, terminator: str = '',
                ) -> bool:
                    """Implement the documented completion contract as data."""
                    if not written or not written.endswith(terminator):
                        return False
                    value = written[:-1] if terminator else written
                    try:
                        return type(json.loads(value)) in (int, float)
                    except ValueError:
                        return False

                def allowed(
                    vocab: dict[int, str], kind: str, written: str,
                    terminator: str = '',
                ) -> list[int]:
                    """Offer the fixture continuation and another digit."""
                    token = ord(target[len(written)])
                    return [token, ord('0') if token != ord('0') else ord('1')]

                def logits(ids: list[int]) -> list[float]:
                    """Return fixed scores for each fixture position."""
                    scores = [0.0] * 128
                    scores[ord(target[len(ids) - 1])] = 1.0
                    return scores

                with patch.object(Model, 'decode', side_effect=decode), \
                        patch.object(Model, 'logits', side_effect=logits), \
                        patch.object(pipeline, 'is_value_complete',
                                     side_effect=complete), \
                        patch.object(pipeline, 'allowed_value_tokens',
                                     side_effect=allowed), \
                        patch('src.decoder.choose_token', side_effect=(
                            lambda scores, permitted:
                            max(permitted, key=scores.__getitem__)
                        )):
                    result = pipeline.generate_value(slot, [127], model, {})
                self.assertEqual(decode(result), target)

    def test_names_and_delimiters_round_trip_through_json(self) -> None:
        """Escaped names and slot-owned delimiters form one valid record."""
        for name in ('normal', 'f"x', 'f\\path', 'f\nname'):
            with self.subTest(name=name):
                model = object.__new__(Model)
                fn = FunctionDef.model_validate({
                    'name': name, 'description': 'fixture',
                    'parameters': {
                        'a': {'type': 'number'}, 'b': {'type': 'boolean'},
                    },
                    'returns': {'type': 'number'},
                })
                parts = [
                    Forced(text=', "parameters": {"a": '),
                    ValueSlot.model_validate({
                        'name': 'a', 'type': 'number', 'terminator': ',',
                    }),
                    Forced(text=' "b": '),
                    ValueSlot.model_validate({
                        'name': 'b', 'type': 'boolean', 'terminator': '}',
                    }),
                    Forced(text='}'),
                ]
                with patch.object(Model, 'encode', side_effect=encode), \
                        patch.object(Model, 'decode', side_effect=decode), \
                        patch.object(pipeline, 'build_parameters_prompt',
                                     return_value=''), \
                        patch.object(pipeline, 'build_skeleton',
                                     return_value=parts), \
                        patch.object(pipeline, 'generate_value', side_effect=[
                            encode('12,'), encode('true}'),
                        ]):
                    text = pipeline.generate_parameters(
                        'a "quoted" prompt', fn, model, {},
                    )
                try:
                    record = json.loads(text)
                except ValueError:
                    self.fail(f'invalid assembled JSON: {text!r}')
                self.assertEqual(record, {
                    'prompt': 'a "quoted" prompt', 'name': name,
                    'parameters': {'a': 12, 'b': True},
                })

    def test_validated_parameters_reach_the_output(self) -> None:
        """The output uses normalized numbers and preserves other types."""
        text = json.dumps({
            'prompt': 'x', 'name': 'f',
            'parameters': {'number': 12, 'text': '12', 'flag': True},
        })
        with patch.object(pipeline, 'build_parameters_model',
                          return_value=Parameters):
            result = pipeline.validate_call(text, definition())
        self.assertIs(type(result.parameters['number']), float)
        self.assertEqual(result.model_dump(), {
            'prompt': 'x', 'name': 'f',
            'parameters': {'number': 12.0, 'text': '12', 'flag': True},
        })

    def test_invalid_parameters_still_raise_prompt_error(self) -> None:
        """Normalization must not swallow validation failures."""
        with patch.object(pipeline, 'build_parameters_model',
                          return_value=Parameters):
            with self.assertRaises(PromptError):
                pipeline.validate_call(
                    '{"prompt":"x","name":"f","parameters":{}}',
                    definition(),
                )


class MainTests(unittest.TestCase):
    """Test persistence decisions while model and file loaders are stubs."""

    def setUp(self) -> None:
        """Replace unavailable leaves; keep actual output files observable."""
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.output = Path(directory) / 'calls.json'
        self.stderr = io.StringIO()
        self.saved: list[list[dict[str, object]]] = []
        model = object.__new__(Model)
        fixtures = {
            'parse_args': Args(output=self.output),
            'load_functions': [definition()], 'Model': model,
            'load_vocab': {}, 'build_name_trie': NameTrie(),
        }
        for name, result in fixtures.items():
            self.stack.enter_context(patch.object(
                entry, name, return_value=result,
            ))
        self.stack.enter_context(patch.object(
            Model, 'vocab_path', return_value='unused',
        ))
        self.stack.enter_context(patch.object(
            entry, 'save_results', side_effect=self.save,
        ))

    def save(self, path: Path, calls: list[FunctionCall]) -> None:
        """Write fixture results in place of the unfinished atomic writer."""
        records = [call.model_dump() for call in calls]
        self.saved.append(records)
        path.write_text(json.dumps(records), encoding='utf-8')

    def run_main(
        self, prompts: list[str], results: Sequence[FunctionCall | Exception],
    ) -> int:
        """Run actual CLI orchestration with fixed per-prompt outcomes."""
        with patch.object(entry, 'load_prompts', return_value=prompts), \
                patch.object(entry, 'process_prompt', side_effect=results), \
                contextlib.redirect_stderr(self.stderr):
            return entry.main([])

    def test_save_error_is_reported_and_stops_processing(self) -> None:
        """A failed write exits nonzero before another prompt is processed."""
        call = FunctionCall(prompt='x', name='f', parameters={})
        with patch.object(entry, 'save_results',
                          side_effect=OSError('disk full')):
            try:
                result = self.run_main(['x', 'y'], [call])
            except OSError:
                self.fail('save failure escaped main')
        self.assertEqual(result, 1)
        self.assertIn('disk full', self.stderr.getvalue())

    def test_zero_results_replace_the_previous_run(self) -> None:
        """Empty input and all-failed input save an empty result array."""
        for prompts, results, status in (
            ([], [], 0), (['x'], [PromptError('failed')], 1),
        ):
            with self.subTest(prompts=prompts):
                self.output.write_text('[{"prompt":"previous run"}]')
                self.assertEqual(self.run_main(prompts, results), status)
                self.assertEqual(json.loads(self.output.read_text()), [])

    def test_empty_result_save_error_is_reported(self) -> None:
        """Saving an empty array uses the same error handling."""
        with patch.object(entry, 'save_results',
                          side_effect=OSError('permission denied')):
            self.assertEqual(self.run_main([], []), 1)
        self.assertIn('permission denied', self.stderr.getvalue())

    def test_partial_success_keeps_each_successful_result(self) -> None:
        """A skipped prompt does not discard either successful result."""
        first = FunctionCall(prompt='a', name='f', parameters={})
        last = FunctionCall(prompt='c', name='f', parameters={})
        self.assertEqual(self.run_main(
            ['a', 'b', 'c'], [first, PromptError('failed'), last],
        ), 1)
        self.assertEqual(json.loads(self.output.read_text()), [
            {'prompt': 'a', 'name': 'f', 'parameters': {}},
            {'prompt': 'c', 'name': 'f', 'parameters': {}},
        ])
        self.assertEqual([len(records) for records in self.saved], [1, 2])


if __name__ == '__main__':
    unittest.main()
