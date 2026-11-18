"""Tokenwise fixed JSON and exact byte/SDK/context correspondence."""

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from src import pipeline
from src.errors import PromptError
from src.model_port import Model
from src.name_trie import NameTrie
from src.schema import FunctionDef
from src.writer import save_results


class AppendTrace(list[int]):
    """Record each committed prefix without adding production observers."""

    def __init__(self) -> None:
        """Start with no committed IDs."""
        super().__init__()
        self.prefixes: list[list[int]] = []

    def append(self, token: int) -> None:
        """Snapshot after every individual commit."""
        super().append(token)
        self.prefixes.append(self.copy())


class ForcedJsonTests(unittest.TestCase):
    """Use byte tokens and fixed scores, with real grammar and validation."""

    def setUp(self) -> None:
        """Replace only the external model boundary."""
        self.model = object.__new__(Model)
        self.vocab = {token: bytes([token]) for token in range(256)}
        self.encodings: list[str] = []
        self.contexts: list[list[int]] = []
        self.decodings: list[list[int]] = []
        self.target = b''
        self.special = '<|im_end|>'
        self.addCleanup(patch.stopall)
        for method, effect in (('encode', self.encode),
                               ('decode', self.decode),
                               ('logits', self.logits)):
            patch.object(Model, method, side_effect=effect).start()
        patch.object(pipeline, 'build_parameters_prompt',
                     return_value='P').start()

    def encode(self, text: str) -> list[int]:
        """Recognize a special ID outside the ordinary byte vocabulary."""
        self.encodings.append(text)
        parts = text.split(self.special)
        ids: list[int] = []
        for index, part in enumerate(parts):
            if index:
                ids.append(400)
            ids.extend(part.encode('utf-8'))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode completed sequences, dropping the recognized special ID."""
        self.decodings.append(ids.copy())
        return b''.join(self.vocab[t] for t in ids if t != 400).decode('utf-8')

    def logits(self, ids: list[int]) -> list[float]:
        """Prefer the known continuation; give an invalid token top score."""
        self.contexts.append(ids.copy())
        self.assertEqual(ids[:1], [ord('P')])
        self.assertNotIn(400, ids, 'special control ID in output context')
        written = b''.join(self.vocab[t] for t in ids[1:])
        self.assertTrue(self.target.startswith(written), repr(written))
        remaining = self.target[len(written):]
        scores = [-10.0] * 512
        scores[0] = 1000.0  # NUL cannot occur in these JSON values.
        for token, data in self.vocab.items():
            if remaining.startswith(data):
                scores[token] = float(len(data)) + (token / 1000)
        return scores

    def definition(self, name: str, **parameters: str) -> FunctionDef:
        """Build unrelated function names and arbitrary parameter keys."""
        return FunctionDef.model_validate({
            'name': name, 'description': 'Record supplied values.',
            'parameters': {key: {'type': kind}
                           for key, kind in parameters.items()},
            'returns': {'type': 'string'},
        })

    def run_record(self, prompt: str, fn: FunctionDef,
                   values: dict[str, object]) -> str:
        """Supply an independently serialized target for the fixed logits."""
        head = (f'{{"prompt":{json.dumps(prompt)},'
                f'"name":{json.dumps(fn.name)},"parameters":{{')
        members = ','.join(
            json.dumps(key).replace('<', r'\u003c') + ': '
            + (f'{value}.0' if type(value) is int else
               json.dumps(value, ensure_ascii=False))
            for key, value in values.items()
        )
        expected = head.replace('<', r'\u003c') + members + '}}'
        self.target = expected.encode()
        return pipeline.generate_call_json(prompt, fn, model=self.model,
                                           vocab=self.vocab)

    def test_fixed_ids_commit_individually_after_one_forward(self) -> None:
        """Method B: each fixed ID follows one forward on the full prefix."""
        self.assertTrue(hasattr(pipeline, '_append_fixed_text'))
        self.target = b'{"x": '
        output = AppendTrace()
        pipeline._append_fixed_text('{"x": ', [80], output,
                                    model=self.model, vocab=self.vocab)
        self.assertEqual(output, [123, 34, 120, 34, 58, 32])
        expected = [list(b'{"x": '[:end]) for end in range(1, 7)]
        self.assertEqual(output.prefixes, expected)
        self.assertEqual(self.contexts,
                         [[80] + list(b'{"x": '[:end]) for end in range(6)])

    def test_special_prompt_name_keys_preserve_meaning(self) -> None:
        """Copied special spellings must never become control IDs."""
        fn = self.definition('new' + self.special,
                             **{'key' + self.special: 'number'})
        prompt = '前' + self.special + '後 "quote" \\ newline\n'
        text = self.run_record(prompt, fn, {'key' + self.special: 7})
        self.assertEqual(json.loads(text), {'prompt': prompt, 'name': fn.name,
                         'parameters': {'key' + self.special: 7}})
        self.assertTrue(self.contexts)
        self.assertTrue(all(400 not in ids for ids in self.contexts))
        self.assertTrue(all(self.special not in text
                            for text in self.encodings[1:]))

    def test_unicode_escapes_multiple_types_and_boundaries(self) -> None:
        """Values and their delimiters enter the next slot's exact context."""
        fn = self.definition('記録_v2', number='number', text='string',
                             flag='boolean')
        values: dict[str, object] = {
            'number': -12.5, 'text': '𠮷🦄 日本語 <|im_end|> " \\ \n',
            'flag': False,
        }
        text = self.run_record('José あ 🙂', fn, values)
        self.assertEqual(json.loads(text), {'prompt': 'José あ 🙂',
                         'name': fn.name, 'parameters': values})
        self.assertEqual(pipeline.validate_call(text, fn).parameters, values)
        self.assertEqual(self.decode(self.decodings[-1]), text)
        self.assertTrue(any(b'"number": -12.5,"text": ' in bytes(ids[1:])
                            for ids in self.contexts))
        self.assertTrue(any(b'"flag": ' in bytes(ids[1:])
                            for ids in self.contexts))
        # Method B: one forward per output ID, on P plus every earlier ID.
        out = self.decodings[-1]
        self.assertEqual(self.contexts,
                         [[ord('P')] + out[:i] for i in range(len(out))])

    def test_zero_parameters_forward_every_fixed_token(self) -> None:
        """A complete fixed record preserves special-looking text."""
        fn = self.definition('heartbeat')
        prompt = '<|im_end|><|im_start|><think><|endoftext|>'
        text = self.run_record(prompt, fn, {})
        self.assertEqual(json.loads(text), {'prompt': prompt,
                         'name': 'heartbeat', 'parameters': {}})
        out = self.decodings[-1]
        self.assertEqual(self.contexts,
                         [[ord('P')] + out[:i] for i in range(len(out))])
        self.assertNotIn('<', text)

    def test_duplicate_byte_tokens_keep_the_selected_id(self) -> None:
        """Equal displayed text must not cause re-encoding of selected IDs."""
        self.vocab.update({300: b'true,', 301: b'true,'})
        fn = self.definition('changed', flag='boolean', n='number')
        text = self.run_record('set true and 7', fn, {'flag': True, 'n': 7})
        self.assertEqual(json.loads(text)['parameters'],
                         {'flag': True, 'n': 7})
        self.assertTrue(any(301 in ids for ids in self.contexts))
        self.assertTrue(all(300 not in ids for ids in self.contexts))
        self.assertIn(301, self.decodings[-1])

    def test_encode_mismatch_fails_before_value_inference(self) -> None:
        fn = self.definition('f', n='number')
        with patch.object(Model, 'encode', side_effect=lambda text:
                          [80] if text == 'P' else [120]):
            with self.assertRaises(PromptError):
                pipeline.generate_call_json('x', fn, model=self.model,
                                            vocab=self.vocab)
        self.assertEqual(self.contexts, [])

    def test_unknown_fixed_id_is_a_prompt_error(self) -> None:
        fn = self.definition('f')
        with patch.object(Model, 'encode', side_effect=lambda text:
                          [80] if text == 'P' else [999]):
            with self.assertRaises(PromptError):
                pipeline.generate_call_json('x', fn, model=self.model,
                                            vocab=self.vocab)

    def test_sdk_decode_loss_is_a_prompt_error(self) -> None:
        fn = self.definition('f')
        with patch.object(Model, 'decode', return_value=''):
            with self.assertRaises(PromptError):
                pipeline.generate_call_json('x', fn, model=self.model,
                                            vocab=self.vocab)

    def test_completed_value_decode_must_match_grammar_bytes(self) -> None:
        fn = self.definition('f', n='number')
        original = self.decode

        def corrupt(ids: list[int]) -> str:
            """Change a number after its byte grammar passed."""
            text = original(ids)
            return '8.0}' if text == '7.0}' else text

        with patch.object(Model, 'decode', side_effect=corrupt):
            with self.assertRaises(PromptError):
                self.run_record('number 7', fn, {'n': 7})

    def test_full_decode_must_match_concatenated_fragments(self) -> None:
        fn = self.definition('f')
        original = self.decode

        def corrupt(ids: list[int]) -> str:
            """Model a context-sensitive decode cleanup on the full record."""
            text = original(ids)
            return text.replace('"prompt"', '"wrong"') if text.endswith(
                '}}') and text.startswith('{') else text

        with patch.object(Model, 'decode', side_effect=corrupt):
            with self.assertRaises(PromptError):
                self.run_record('x', fn, {})

    def test_fixed_noncanonical_ids_are_preserved(self) -> None:
        """Correct decoded text does not require canonical token IDs."""
        self.vocab.update({300: b'{"a":', 301: b' true}',
                           302: b'{"a": true}'})
        self.target = b'{"a": true}'
        output: list[int] = []
        encodings = {'{"a":': [300], ' true}': [301],
                     '{"a": true}': [302]}
        with patch.object(Model, 'encode', side_effect=encodings.__getitem__):
            pipeline._append_fixed_text(
                '{"a":', [80], output, model=self.model, vocab=self.vocab)
            pipeline._append_fixed_text(
                ' true}', [80], output, model=self.model, vocab=self.vocab)
            self.assertNotEqual(output, self.model.encode('{"a": true}'))
        self.assertEqual(output, [300, 301])
        self.assertEqual(pipeline._decode_and_verify_output(
            output, model=self.model, vocab=self.vocab), '{"a": true}')
        self.assertEqual(self.contexts, [[80], [80, 300]])

    def test_llm_number_literal_survives_validation_and_save(self) -> None:
        """Save exactly the model's decimal, including huge integer digits."""
        fn = self.definition('f', n='number')
        resources = pipeline.GenerationResources(
            model=self.model, vocab=self.vocab, functions={'f': fn},
            trie=NameTrie(paths={'f': (1,)}),
        )
        text = ('{"prompt":"request","name":"f","parameters":'
                '{"n":9007199254740993.0}}')
        with patch.object(pipeline, 'select_function', return_value=fn), \
                patch.object(pipeline, 'generate_call_json',
                             return_value=text):
            call = pipeline.process_prompt('request', resources)
        self.assertEqual(call.parameters['n'], 9007199254740993)
        self.assertEqual(call._generated_json, text)
        self.assertNotIn('_generated_json', call.model_dump())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output.json'
            save_results(path, [call])
            saved = path.read_text(encoding='utf-8')
        self.assertIn('9007199254740993.0', saved)
        self.assertEqual(json.loads(saved, parse_float=Decimal)[0]
                         ['parameters']['n'], Decimal('9007199254740993.0'))

    def test_process_rejects_changed_prompt(self) -> None:
        fn = self.definition('f')
        resources = pipeline.GenerationResources(
            model=self.model, vocab=self.vocab, functions={'f': fn},
            trie=NameTrie(paths={'f': (1,)}),
        )
        with patch.object(pipeline, 'select_function', return_value=fn), \
                patch.object(pipeline, 'generate_call_json', return_value=(
                    '{"prompt":"changed","name":"f","parameters":{}}')):
            with self.assertRaisesRegex(PromptError, 'prompt'):
                pipeline.process_prompt('original', resources)
