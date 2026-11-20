"""Check actual trie, byte grammar and prompt implementations directly."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel, ValidationError

from src.errors import PromptError, StartupError
from src.model_port import EOS, Model
from src.name_trie import NameTrie
from trie_fixtures import trie_from_paths
from src.prompt import build_name_prompt, build_arguments_prompt
from src.schema import ParamType
from src.values import (
    _COMPLETE, _INCOMPLETE, _INVALID, _value_syntax_state,
    allowed_tokens_for_arguments, is_argument_generation_complete,
)
from src.vocab import load_vocab
from test_application import definition


def encode_name(text: str) -> list[int]:
    """Encode characters, and each EOS as the one token 0 like the SDK."""
    ids: list[int] = []
    for index, part in enumerate(text.split(EOS)):
        ids += ([0] if index else []) + [ord(char) for char in part]
    return ids


class TrieTests(unittest.TestCase):
    """EOS separates valid complete names from longer candidates."""

    def setUp(self) -> None:
        """Replace SDK encoding while retaining real trie initialization."""
        self.model = object.__new__(Model)
        encode = patch.object(Model, 'encode', side_effect=encode_name)
        encode.start()
        self.addCleanup(encode.stop)

    def test_prefix_names_choose_end_or_continue(self) -> None:
        """A shorter name becomes a leaf only after its EOS."""
        trie = NameTrie(['fn_add', 'fn_add_numbers'], self.model)
        prefix = encode_name('fn_add')
        self.assertEqual(set(trie.next_tokens(prefix)), {0, ord('_')})
        self.assertIsNone(trie.leaf_name(prefix))
        self.assertEqual(trie.leaf_name(prefix + [0]), 'fn_add')
        longer = encode_name('fn_add_numbers<|im_end|>')
        self.assertEqual(trie.leaf_name(longer), 'fn_add_numbers')
        self.assertEqual(trie.next_tokens(longer), [])
        self.assertNotIn(0, trie.next_tokens(encode_name('fn_ad')))

    def test_arbitrary_names_and_invalid_paths(self) -> None:
        """Names are leaf data, including quotes, Unicode and long names."""
        names = ['say"hello', 'path\\name', '日本語', 'a' * 100]
        trie = NameTrie(names, self.model)
        for name in names:
            self.assertEqual(
                trie.leaf_name(encode_name(name + '<|im_end|>')), name)
        for lookup in (trie.leaf_name, trie.next_tokens):
            with self.subTest(lookup=lookup.__name__):
                with self.assertRaises(PromptError):
                    lookup([999999])

    def test_leaf_name_separates_unfinished_end_and_invalid(self) -> None:
        """An unfinished valid path is None; an unknown path is an error."""
        trie = NameTrie(['ab', 'abc'], self.model)
        self.assertIsNone(trie.leaf_name([]))
        self.assertIsNone(trie.leaf_name(encode_name('ab')))
        self.assertEqual(trie.leaf_name(encode_name('ab') + [0]), 'ab')
        for path in (encode_name('ax'), encode_name('ab') + [0, 0],
                     encode_name('abc') + [0, 1]):
            with self.subTest(path=path), self.assertRaises(PromptError):
                trie.leaf_name(path)

    def test_encoding_failures_are_startup_errors(self) -> None:
        """Tokenizer failures cannot escape the startup boundary."""
        def fail(text: str) -> list[int]:
            """Represent a failure from the actual wrapper boundary."""
            raise PromptError('tokenizer failed')

        with patch.object(Model, 'encode', side_effect=fail):
            with self.assertRaisesRegex(StartupError, 'tokenizer failed'):
                NameTrie(['f'], self.model)

    def test_eos_text_comes_from_model_port(self) -> None:
        """Encode the whole name with the existing model's EOS constant."""
        paths = {'ab' + EOS: [97, 98, 7], 'a' + EOS: [97, 7]}
        with patch.object(Model, 'encode', side_effect=paths.__getitem__):
            trie = NameTrie(['ab', 'a'], self.model)
        self.assertEqual(dict(trie.paths), {'ab': (97, 98, 7), 'a': (97, 7)})
        self.assertEqual(trie.next_tokens([97]), [98, 7])
        self.assertEqual(trie.leaf_name([97, 7]), 'a')

    def test_next_tokens_are_unique_in_first_seen_order(self) -> None:
        """Shared prefixes list each next token once, in input order."""
        trie = trie_from_paths({'c': (1, 3, 0), 'a': (1, 2, 0),
                               'b': (1, 3, 9, 0), 'd': (4, 0)})
        self.assertEqual(trie.next_tokens([]), [1, 4])  # 1 seen three times
        self.assertEqual(trie.next_tokens([1]), [3, 2])  # not sorted
        self.assertEqual(trie.next_tokens([1, 3]), [0, 9])
        self.assertEqual(trie.next_tokens([1, 3, 0]), [])  # full path
        self.assertEqual(trie.leaf_name([1, 3, 9, 0]), 'b')

    def test_multi_token_eos_is_followed_to_the_full_path(self) -> None:
        """A terminator of several tokens is just more steps on the path."""
        def encode(text: str) -> list[int]:
            """Encode characters; the EOS terminator becomes 1, 2, 3."""
            ids: list[int] = []
            for index, part in enumerate(text.split(EOS)):
                ids += ([1, 2, 3] if index else []) + [ord(c) for c in part]
            return ids

        with patch.object(Model, 'encode', side_effect=encode):
            trie = NameTrie(['fn_add', 'fn_add_numbers'], self.model)
        short = encode('fn_add')
        self.assertEqual(set(trie.next_tokens(short)), {1, ord('_')})
        for step in (short + [1], short + [1, 2]):
            self.assertIsNone(trie.leaf_name(step))  # unfinished terminator
            self.assertEqual(len(trie.next_tokens(step)), 1)
        self.assertEqual(trie.leaf_name(short + [1, 2, 3]), 'fn_add')
        self.assertEqual(trie.next_tokens(short + [1, 2, 3]), [])
        self.assertEqual(trie.leaf_name(encode('fn_add_numbers' + EOS)),
                         'fn_add_numbers')

    def test_terminator_may_merge_with_the_name(self) -> None:
        """encode(name + eos) need not be encode(name) + encode(eos)."""
        def encode(text: str) -> list[int]:
            """Encode characters, merging b + EOS into token 50 like BPE."""
            merged = text.replace('b' + EOS, '\0').replace(EOS, '!')
            ids = [ord(c) for c in merged]
            return [50 if token == 0 else token for token in ids]

        self.assertEqual(encode('ab' + EOS), [97, 50])
        self.assertNotEqual(encode('ab' + EOS), encode('ab') + encode(EOS))
        with patch.object(Model, 'encode', side_effect=encode):
            trie = NameTrie(['ab', 'a'], self.model)
        self.assertEqual(dict(trie.paths), {'ab': (97, 50), 'a': (97, 33)})
        self.assertEqual(trie.leaf_name([97, 50]), 'ab')
        self.assertEqual(trie.leaf_name([97, 33]), 'a')

    def test_bad_full_paths_are_rejected_at_build(self) -> None:
        """Empty, shared and prefix-colliding full paths are ambiguous."""
        cases = {
            'empty full path': (['f'], lambda text: []),
            'same path for two names': (['a', 'b'], lambda text: [1, 0]),
            'one path is a prefix of another': (
                ['a', 'a' + EOS + 'b'], encode_name),
            'duplicate names': (['a', 'a'], encode_name),
            'no names': ([], encode_name),
        }
        for case, (names, encode) in cases.items():
            with self.subTest(case), \
                    patch.object(Model, 'encode', side_effect=encode), \
                    self.assertRaises(StartupError):
                NameTrie(names, self.model)
        # An EOS token inside one name is only text; no structure is broken.
        trie = NameTrie(['a' + EOS + 'b'], self.model)
        self.assertEqual(trie.leaf_name([97, 0, 98, 0]), 'a' + EOS + 'b')

    def test_pydantic_validators_check_the_whole_trie(self) -> None:
        """Exercise strict path validation independently of encoding."""
        cases: dict[str, dict[str, object]] = {
            'missing fields': {},
            'no names (root is a dead end)': {'paths': {}},
            'empty full path': {'paths': {'a': ()}},
            'two names on one path': {'paths': {'a': (97, 0), 'b': (97, 0)}},
            'full path is a proper prefix': {
                'paths': {'a': (97,), 'ab': (97, 98)}},
            'prefix listed after the longer path': {
                'paths': {'ab': (97, 98), 'a': (97,)}},
            'list instead of tuple': {'paths': {'a': [97, 0]}},
            'unknown field': {'eos': 0, 'paths': {'a': (97, 0)}},
        }
        for case, data in cases.items():
            with self.subTest(case), self.assertRaises(ValidationError):
                trie = object.__new__(NameTrie)
                BaseModel.__init__(trie, **data)
        trie = trie_from_paths({'a': (97, 0), 'ab': (97, 98, 0)})
        self.assertEqual(trie.next_tokens([97]), [0, 98])
        self.assertEqual(trie.next_tokens([97, 0]), [])  # leaf: no children
        self.assertIsNone(trie.leaf_name([97]))  # inner node: no name

    def test_unvalidated_trie_cannot_be_used(self) -> None:
        """model_construct skips validation, so no index is ever built."""
        trie = NameTrie.model_construct(paths={'a': (97, 0)})
        for lookup in (trie.leaf_name, trie.next_tokens):
            with self.subTest(lookup.__name__):
                with self.assertRaises(AttributeError):
                    lookup([97])

    def test_finished_trie_is_immutable(self) -> None:
        """Neither the trie nor its source dictionary can change its paths."""
        source = {'a': (97, 0), 'b': (98, 0)}
        trie = trie_from_paths(source)
        source['c'] = (99, 0)
        del source['a']
        self.assertEqual(dict(trie.paths), {'a': (97, 0), 'b': (98, 0)})
        self.assertEqual(trie.next_tokens([]), [97, 98])
        with self.assertRaises(TypeError):
            trie.paths['c'] = (99, 0)  # type: ignore[index]
        with self.assertRaises(ValidationError):
            trie.paths = {}
        allowed = trie.next_tokens([])
        allowed.append(99)
        self.assertEqual(trie.next_tokens([]), [97, 98])
        with self.assertRaises(PromptError):
            trie.next_tokens([99])

    def test_serialization_keeps_paths_as_plain_data(self) -> None:
        """Read-only paths remain serializable without private indexes."""
        trie = NameTrie(['ab', 'a'], self.model)
        self.assertEqual(trie.model_dump(),
                         {'paths': {'ab': (97, 98, 0), 'a': (97, 0)}})
        self.assertEqual(json.loads(trie.model_dump_json()),
                         {'paths': {'ab': [97, 98, 0], 'a': [97, 0]}})


class VocabularyTests(unittest.TestCase):
    """Read Qwen's byte-level alphabet without decoding partial UTF-8."""

    def test_byte_alphabet_preserves_split_unicode(self) -> None:
        """Visible byte symbols map to exact bytes, not replacement text."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vocab.json'
            path.write_text(json.dumps({
                'a': 0, 'Ġ': 1, 'Ċ': 2, 'ãģ': 3, 'Ĥ': 4,
                'Ã': 5, '©': 6,
            }), encoding='utf-8')
            vocab = load_vocab(path)
        self.assertEqual(vocab[1], b' ')
        self.assertEqual(vocab[2], b'\n')
        self.assertEqual(vocab[3] + vocab[4], 'あ'.encode())
        self.assertEqual(vocab[5] + vocab[6], 'é'.encode())

    def test_loaded_vocabulary_is_read_only(self) -> None:
        """Consumers cannot add, replace or delete validated vocabulary IDs."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vocab.json'
            path.write_text('{"b":2,"a":1}')
            vocab = load_vocab(path)
        self.assertEqual(list(vocab), [2, 1])
        self.assertEqual(dict(vocab), {2: b'b', 1: b'a'})
        for token in [1, 3]:
            with self.subTest(token=token), self.assertRaises(TypeError):
                vocab[token] = b'x'  # type: ignore[index]
        with self.assertRaises(TypeError):
            del vocab[1]  # type: ignore[attr-defined]

    def test_bad_vocab_is_a_startup_error(self) -> None:
        """Invalid files, duplicate IDs and unsupported symbols fail early."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vocab.json'
            for text in ['{}', '[]', '{', '{"a": -1}',
                         '{"a": 1, "b": 1}', '{"あ": 0}']:
                path.write_text(text, encoding='utf-8')
                with self.subTest(text=text):
                    with self.assertRaises(StartupError):
                        load_vocab(path)

    def test_bad_vocab_messages_name_the_problem(self) -> None:
        """An empty file is not reported as a negative ID."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vocab.json'
            for text, reason in [('{}', 'empty'), ('{"a": -1}', 'negative'),
                                 ('{"a": 1, "b": 1}', 'duplicate')]:
                path.write_text(text, encoding='utf-8')
                with self.subTest(text=text):
                    with self.assertRaisesRegex(StartupError, reason):
                        load_vocab(path)


class GrammarTests(unittest.TestCase):
    """Allow every valid prefix and reject bytes past slot terminators."""

    def test_number_prefixes_and_completion(self) -> None:
        """Integers and decimals continue until the actual terminator."""
        for value in ['0.0', '-0.0', '12.0', '1.5', '-0.25', '9' * 400 + '.0',
                      '0.12345678901234567', '0', '-0', '12', '-7']:
            for terminator in [',', '}']:
                data = value.encode() + terminator.encode()
                for index in range(len(data)):
                    written = data[:index]
                    token = data[index:index + 1]
                    self.assertFalse(is_argument_generation_complete(
                        written, 'number', terminator,
                    ))
                    self.assertIn(1, allowed_tokens_for_arguments(
                        {1: token}, 'number', written, terminator,
                    ))
                self.assertTrue(is_argument_generation_complete(data, 'number', terminator))
        self.assertTrue(is_argument_generation_complete(b' \n-12.5 \t}', 'number', '}'))

    def test_invalid_numbers_and_crossing_tokens_are_masked(self) -> None:
        """A candidate cannot add bad syntax or text after a terminator."""
        vocab = {0: b'1', 1: b'.', 2: b',', 3: b', "next":',
                 4: b'e2', 5: b'}', 6: b'', 7: b'0'}
        self.assertEqual(set(allowed_tokens_for_arguments(
            vocab, 'number', b'12', ',',
        )), {0, 1, 2, 7})
        for value in [b'01,', b'+1,', b'.1,', b'1.,', b'NaN,', b'-,',
                      b'1e3,', b'1.0e3,', b'1, ', b'1,true', b'1}']:
            self.assertFalse(is_argument_generation_complete(value, 'number', ','))

    def test_boolean_grammar(self) -> None:
        """Only lowercase JSON booleans can reach their terminator."""
        self.assertEqual(allowed_tokens_for_arguments(
            {1: b'ue}', 2: b'ue}junk', 3: b'ue,', 4: b'ue'},
            'boolean', b'tr', '}',
        ), [1, 4])
        for value in [b'true}', b'false}', b' true \n}']:
            self.assertTrue(is_argument_generation_complete(value, 'boolean', '}'))
        for value in [b'True}', b'1}', b'false', b'true},', b'true} ']:
            self.assertFalse(is_argument_generation_complete(value, 'boolean', '}'))

    def test_strings_accept_every_byte_prefix_and_json_escape(self) -> None:
        """Split UTF-8 and split escapes stay valid until completion."""
        for text in ['', ' ', 'the "best" day', '\\', '\n\t',
                     '日本語é🙂', 'commas, and braces }']:
            value = json.dumps(text, ensure_ascii=False).encode() + b'}'
            for index in range(len(value)):
                self.assertIn(1, allowed_tokens_for_arguments(
                    {1: value[index:index + 1]}, 'string', value[:index], '}',
                ), msg=f'{text!r}, byte {index}')
            self.assertTrue(is_argument_generation_complete(value, 'string', '}'))
        escaped = b'"\\u3042"}'
        for index in range(len(escaped)):
            self.assertIn(1, allowed_tokens_for_arguments(
                {1: escaped[index:index + 1]}, 'string', escaped[:index], '}',
            ))

    def test_invalid_string_bytes_are_masked(self) -> None:
        """Valid terminators inside strings never terminate a value early."""
        for token in [b'"a\n"}', b'"\\q"}', b'"\\u00xz"}', b'"a"}x',
                      b'"a"} ', b'"\xff"}', b'"\xe3"}', b'"a",']:
            self.assertEqual(allowed_tokens_for_arguments(
                {1: token}, 'string', b'', '}',
            ), [], msg=repr(token))
        self.assertFalse(is_argument_generation_complete(b'"a}', 'string', '}'))
        self.assertEqual(allowed_tokens_for_arguments(
            {1: b'\x82"}', 2: b'"}', 3: b'\xff'},
            'string', b'"\xe3\x81', '}',
        ), [1])

    def test_slot_states_have_independent_grammar_expectations(self) -> None:
        """Classify syntax, UTF-8 and completion without the candidate path."""
        cases = [
            (b'', _INCOMPLETE), (b' \n', _INCOMPLETE),
            (b'"', _INCOMPLETE), (b'"text', _INCOMPLETE),
            (b'"text"', _INCOMPLETE), (b'"text" \t', _INCOMPLETE),
            (b'"text"}', _COMPLETE), (b'""}', _COMPLETE),
            (b'"\\', _INCOMPLETE), (b'"\\"', _INCOMPLETE),
            (b'"\\u', _INCOMPLETE), (b'"\\u0', _INCOMPLETE),
            (b'"\\u00', _INCOMPLETE), (b'"\\u00a', _INCOMPLETE),
            (b'"\\u00aF', _INCOMPLETE), (b'"\\u00aF"}', _COMPLETE),
            (b'"\\u"}', _INVALID), (b'"\\u0"}', _INVALID),
            (b'"\\u00"}', _INVALID), (b'"\\u00a"}', _INVALID),
            (b'"\xe3', _INCOMPLETE), (b'"\xe3\x81', _INCOMPLETE),
            ('"あ"}'.encode(), _COMPLETE),
            (b'"\xe3"}', _INVALID), (b'"\xff', _INVALID),
            (b'"\xe0\x80', _INVALID), (b'"\\u00x', _INVALID),
            (b'"\\q', _INVALID), (b'"a\n', _INVALID),
            (b'""} ', _INVALID), (b'""}x', _INVALID),
            (b'"",', _INVALID), (b'x', _INVALID),
        ]
        for data, expected in cases:
            with self.subTest(data=data):
                self.assertEqual(_value_syntax_state(data, 'string', b'}'), expected)

    def test_candidate_order_and_ids_match_slot_state(self) -> None:
        """Incremental candidates obey the reference contract at boundaries."""
        rng = random.Random(1942)
        fragments = [b'', b'"', b'\\', b'\\"', b'\\u3042', b'}', b',',
                     b'"}', b'"} ', b'"}x', b'\x81\x82"}', b'\xe3',
                     b'\xe3\x81', b'\xe3\x81\x82', b'\xff', b'\xe0\x80']
        fragments += [bytes([byte]) for byte in range(256)]
        fragments += [rng.randbytes(rng.randrange(9)) for _ in range(256)]
        # Descending, nonconsecutive IDs expose sorting or ID substitution.
        vocab = {10000 - index * 3: fragment
                 for index, fragment in enumerate(fragments)}
        kinds: tuple[ParamType, ...] = ('string', 'number', 'boolean')
        for kind in kinds:
            for terminator in (',', '}'):
                end = terminator.encode()
                if kind == 'string':
                    texts = ['', 'normal', '日本語🙂', '"\\\n', ',}']
                    values = [
                        json.dumps(text, ensure_ascii=ascii_only).encode()
                        for text in texts for ascii_only in (True, False)
                    ]
                    for _ in range(30):
                        text = ''.join(rng.choices('abcあ🙂"\\\n', k=12))
                        values.append(json.dumps(
                            text, ensure_ascii=rng.choice((True, False)),
                        ).encode())
                elif kind == 'number':
                    values = [b'0', b'-12.34', b'123', b'0.01']
                else:
                    values = [b'true', b'false']
                prefixes = set[bytes]()
                if kind == 'string':
                    prefixes.update((b'"\xff', b'"\xe0\x80'))
                for value in values:
                    data = b' \t' + value + b' \n' + end
                    prefixes.update(data[:index]
                                    for index in range(len(data) + 1))
                for written in sorted(prefixes):
                    expected = [token for token, value in vocab.items()
                                if value and _value_syntax_state(
                                    written + value, kind, end,
                                ) != _INVALID]
                    with self.subTest(kind=kind, end=end, written=written):
                        self.assertEqual(allowed_tokens_for_arguments(
                            vocab, kind, written, terminator,
                        ), expected)


class PromptTests(unittest.TestCase):
    """Prompt templates use definitions, not example-specific answers."""

    def test_name_prompt_lists_functions_and_requests_only_name(self) -> None:
        """The name stage is a non-thinking assistant completion."""
        fn = definition(a='number')
        prompt = build_name_prompt('Add 2 and 3', {fn.name: fn})
        self.assertIn(fn.description, prompt)
        self.assertIn(fn.name, prompt)
        self.assertIn('Add 2 and 3', prompt)
        self.assertIn('<think>\n\n</think>\n\n', prompt)
        self.assertTrue(prompt.endswith('Function name:\n'))
        self.assertIn('name only', prompt)

    def test_parameter_prompt_has_selected_definition(self) -> None:
        """Parameter extraction must not execute the requested function."""
        fn = definition(text='string')
        prompt = build_arguments_prompt('Keep the text', fn)
        self.assertIn(fn.description, prompt)
        self.assertIn('Keep the text', prompt)
        self.assertIn('Extract the arguments', prompt)
        self.assertIn('from the user request', prompt)
        self.assertIn('do not answer the request or compute a result', prompt)
