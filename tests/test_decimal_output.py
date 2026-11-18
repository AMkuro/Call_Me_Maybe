"""Independent REQ-031 grammar, generation and exact-save regression checks."""

import json
import re
import tempfile
import unittest
from collections.abc import Callable
from typing import Any
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src import __main__ as entry, pipeline
from src.errors import PromptError
from src.model_port import Model
from src.name_trie import NameTrie
from src.schema import FunctionCall, FunctionDef
from src.skeleton import ValueSlot
from src.values import allowed_value_tokens, is_value_complete
from src.writer import save_results

VALID = ('0.0', '-0.0', '2.0', '2.75', '-42.125', '9007199254740993.0',
         '0.00000001', '123456789012345678901234567890.0', '2.00')
INVALID = ('2', '-2', '2.', '.5', '+2.0', '01.0', '1e10', '1.0e10',
           '1.0E-5', 'NaN', 'Infinity', '1.0, extra')


def definition() -> FunctionDef:
    """Combine a number, numeric-looking string and boolean."""
    return FunctionDef.model_validate({
        'name': 'record', 'description': 'Record values unchanged.',
        'parameters': {'x': {'type': 'number'}, 's': {'type': 'string'},
                       'b': {'type': 'boolean'}},
        'returns': {'type': 'number'},
    })


def generated(literal: str, prompt: str = '依頼') -> FunctionCall:
    """Keep real validation and the production raw-text setter."""
    fn = definition()
    model = object.__new__(Model)
    resources = pipeline.GenerationResources(
        model=model, vocab={}, functions={'record': fn},
        trie=NameTrie(paths={'record': (1,)}),
    )
    text = ('{"prompt":' + json.dumps(prompt) + ',"name":"record",'
            '"parameters":{"x":' + literal + ',"s":'
            + json.dumps('日本語 " \\ \n 1e10 2.0 <|im_end|>') + ',"b":false}}')
    with patch.object(pipeline, 'select_name', return_value=fn), \
            patch.object(pipeline, 'generate_parameters', return_value=text):
        return pipeline.process_prompt(prompt, resources)


class DecimalGrammarTests(unittest.TestCase):
    """Compare masks and completion against independently supplied literals."""

    def test_every_valid_prefix_reaches_both_delimiters(self) -> None:
        """No legal next byte is masked before the complete decimal value."""
        vocab = {token: bytes([token]) for token in range(128)}
        for literal in VALID:
            for delimiter in (',', '}'):
                with self.subTest(literal=literal, delimiter=delimiter):
                    written = b''
                    for token in (literal + delimiter).encode():
                        self.assertFalse(is_value_complete(
                            written, 'number', delimiter))
                        self.assertIn(token, allowed_value_tokens(
                            vocab, 'number', written, delimiter))
                        written += bytes([token])
                    self.assertTrue(is_value_complete(
                        written, 'number', delimiter))
                    self.assertEqual(allowed_value_tokens(
                        vocab, 'number', written, delimiter), [])

    def test_invalid_literals_cannot_be_committed_whole(self) -> None:
        """Incomplete prefixes can continue, but cannot accept a delimiter."""
        for literal in INVALID:
            for delimiter in (',', '}'):
                with self.subTest(literal=literal, delimiter=delimiter):
                    payload = (literal + delimiter).encode()
                    self.assertFalse(is_value_complete(
                        payload, 'number', delimiter))
                    self.assertEqual(allowed_value_tokens(
                        {9: payload}, 'number', b'', delimiter), [])
        for prefix in (b'2', b'-2', b'2.'):
            for delimiter in (',', '}'):
                self.assertEqual(allowed_value_tokens(
                    {9: delimiter.encode()}, 'number', prefix, delimiter), [])

    def test_merged_token_candidates_obey_the_whole_token(self) -> None:
        """A token cannot hide an exponent or trailing material."""
        vocab = {1: b'2.0', 2: b'2.0,', 3: b'.0}', 4: b'1e10',
                 5: b'1.0e10', 6: b'0,', 7: b'2.0, extra'}
        self.assertEqual(allowed_value_tokens(vocab, 'number', b'', ','),
                         [1, 2])
        self.assertEqual(allowed_value_tokens(vocab, 'number', b'2', '}'),
                         [1, 3])
        self.assertEqual(allowed_value_tokens(vocab, 'number', b'2.', ','),
                         [6])
        self.assertTrue(is_value_complete(b'2.0}', 'number', '}'))
        self.assertFalse(is_value_complete(b'2.0', 'number', '}'))

    def test_fraction_selects_from_forced_context(self) -> None:
        """The model chooses 5, with the forced decimal prefix in context."""
        model = object.__new__(Model)
        vocab = {1: b'2.', 2: b'0}', 3: b'5}', 4: b'1e10'}
        scores = [0.0, 0.0, 1.0, 2.0, 1000.0]
        slot = ValueSlot(name='x', type='number', terminator='}')
        with patch.object(Model, 'logits', return_value=scores) as forward, \
                patch.object(Model, 'encode') as encode:
            ids = pipeline.generate_value(slot, [90, 91], model=model,
                                          vocab=vocab)
        self.assertEqual(ids, [1, 3])
        self.assertEqual(b''.join(vocab[t] for t in ids), b'2.5}')
        forward.assert_called_once_with([90, 91, 1])
        encode.assert_not_called()

    def test_real_number_grammar_completes_at_256_not_257(self) -> None:
        """Use the actual number mask and termination at the existing cap."""
        model = object.__new__(Model)
        vocab = {1: b'1', 2: b'.', 3: b'0', 4: b'}'}
        for length in (256, 257):
            target = b'1' * (length - 3) + b'.0}'

            def logits(ids: list[int]) -> list[float]:
                """Choose the fixture continuation among legal tokens."""
                scores = [0.0] * 5
                next_byte = target[len(ids)]
                scores[{49: 1, 46: 2, 48: 3, 125: 4}[next_byte]] = 1.0
                return scores

            with self.subTest(length=length), \
                    patch.object(Model, 'logits', side_effect=logits):
                slot = ValueSlot(name='x', type='number', terminator='}')
                if length == 256:
                    ids = pipeline.generate_value(slot, [], model=model,
                                                  vocab=vocab)
                    self.assertEqual(b''.join(vocab[t] for t in ids), target)
                else:
                    with self.assertRaisesRegex(
                            PromptError, 'token limit 256'):
                        pipeline.generate_value(slot, [], model=model,
                                                vocab=vocab)


class DecimalSaveTests(unittest.TestCase):
    """Inspect the saved file using standard JSON and Decimal hooks."""

    def test_generated_lexemes_survive_cumulative_save(self) -> None:
        """Preserve digits, sign and trailing zeros for every value."""
        calls = [generated(literal, str(index))
                 for index, literal in enumerate(VALID)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calls.json'
            for count in range(1, len(calls) + 1):
                save_results(path, calls[:count])
                lexemes: list[str] = []

                def decimal(literal: str) -> Decimal:
                    """Observe only numeric literals, never string content."""
                    lexemes.append(literal)
                    return Decimal(literal)

                rows = json.loads(path.read_text(), parse_float=decimal)
                self.assertEqual(lexemes, list(VALID[:count]))
                self.assertEqual([r['prompt'] for r in rows],
                                 [str(i) for i in range(count)])
                for literal, row in zip(VALID, rows):
                    self.assertEqual(row['parameters']['x'].as_tuple(),
                                     Decimal(literal).as_tuple())
                    self.assertIs(row['parameters']['b'], False)
                    self.assertEqual(row['parameters']['s'],
                                     '日本語 " \\ \n 1e10 2.0 <|im_end|>')
        for call in calls:
            self.assertEqual(set(call.model_dump()),
                             {'prompt', 'name', 'parameters'})
        self.assertEqual(set(FunctionCall.model_json_schema()['properties']),
                         {'prompt', 'name', 'parameters'})

    def test_precision_rejections_keep_existing_reasons(self) -> None:
        """Keep precision and finite-range checks."""
        for literal, reason in (
            ('0.1234567890123456789', 'precision'),
            ('1.' + '0' * 18 + '1', 'precision'),
            ('0.' + '0' * 400 + '1', 'underflow'),
            ('9' * 309 + '.0', 'overflow'),
        ):
            with self.subTest(literal=literal[:30]):
                self.assertTrue(re.fullmatch(r'-?(0|[1-9][0-9]*)\.[0-9]+',
                                             literal))
                with self.assertRaisesRegex(PromptError, reason):
                    generated(literal)

    def test_generated_text_must_pass_validation(self) -> None:
        """An unvalidated body cannot enter the normal trusted-text path."""
        for literal in ('true', '"2.0"', 'NaN', '2.0,"x":3.0'):
            with self.subTest(literal=literal), self.assertRaises(PromptError):
                generated(literal)
        with self.assertRaises(ValidationError):
            FunctionCall.model_validate({
                'prompt': 'x', 'name': 'record', 'parameters': {'x': 2.0},
                '_generated_json': '{"unvalidated":true}',
            })

    def test_manual_call_keeps_the_documented_fallback(self) -> None:
        """Manually built calls still use ordinary JSON serialization."""
        call = FunctionCall(prompt='手動', name='record',
                            parameters={'x': 2, 's': '1e10', 'b': True})
        self.assertIsNone(call._generated_json)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calls.json'
            save_results(path, [call])
            self.assertEqual(json.loads(path.read_text()), [call.model_dump()])
            save_results(path, [])
            self.assertEqual(path.read_text(), '[]\n')

    def test_replace_failure_preserves_previous_file(self) -> None:
        """Fail replacement after writing raw generated text."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calls.json'
            path.write_text('["previous"]\n')
            with patch('src.writer.os.replace',
                       side_effect=OSError('replace')):
                with self.assertRaisesRegex(OSError, 'replace'):
                    save_results(path, [generated('2.00')])
            self.assertEqual(path.read_text(), '["previous"]\n')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_raw_write_failure_preserves_previous_file(self) -> None:
        """A write error during generated text never replaces prior output."""
        real_temporary = tempfile.NamedTemporaryFile

        def temporary(**kwargs: Any) -> Any:
            """Keep real temporary-file lifecycle and fail only its write."""
            output = real_temporary(**kwargs)
            original: Callable[[str], int] = output.write

            def write(text: str) -> int:
                """Raise while writing the raw record, after the array head."""
                if text.startswith('{'):
                    raise OSError('write failed')
                return original(text)

            writer = patch.object(output, 'write', side_effect=write)
            writer.start()
            self.addCleanup(writer.stop)
            return output

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calls.json'
            path.write_text('["previous"]\n')
            with patch('src.writer.tempfile.NamedTemporaryFile',
                       side_effect=temporary):
                with self.assertRaisesRegex(OSError, 'write failed'):
                    save_results(path, [generated('2.00')])
            self.assertEqual(path.read_text(), '["previous"]\n')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_success_order_after_prompt_failure(self) -> None:
        """Run the real loop and writer, skipping one failed prompt."""
        import contextlib
        import io
        from src.display import Dashboard

        model = object.__new__(Model)
        fn = definition()
        resources = pipeline.GenerationResources(
            model=model, vocab={}, functions={'record': fn},
            trie=NameTrie(paths={'record': (1,)}),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calls.json'
            with contextlib.redirect_stderr(io.StringIO()):
                dashboard = Dashboard(3, model, {})
            outcomes = [generated('2.00', 'first'), PromptError('skip'),
                        generated('0.00000001', 'last')]
            with patch.object(entry, 'process_prompt', side_effect=outcomes):
                self.assertFalse(entry._process_all(
                    ['first', 'bad', 'last'], resources, path, dashboard))
            prompts = [r['prompt'] for r in json.loads(path.read_text())]
            self.assertEqual(prompts, ['first', 'last'])
            with patch.object(entry, 'process_prompt',
                              side_effect=PromptError('skip')):
                self.assertFalse(entry._process_all(
                    ['bad'], resources, path, dashboard))
            self.assertEqual(path.read_text(), '[]\n')
