"""Check actual trie, byte grammar and prompt implementations directly."""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.errors import PromptError, StartupError
from src.model_port import EOS
from src.name_trie import (
    NameTrie, build_name_trie, leaf_name, max_name_steps, next_tokens,
)
from src.prompt import build_name_prompt, build_parameters_prompt
from src.values import allowed_slot_tokens, is_slot_complete
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

    def test_prefix_names_choose_end_or_continue(self) -> None:
        """A shorter name becomes a leaf only after its EOS."""
        trie = build_name_trie(['fn_add', 'fn_add_numbers'], encode_name, EOS)
        prefix = encode_name('fn_add')
        self.assertEqual(set(next_tokens(trie, prefix)), {0, ord('_')})
        self.assertIsNone(leaf_name(trie, prefix))
        self.assertEqual(leaf_name(trie, prefix + [0]), 'fn_add')
        longer = encode_name('fn_add_numbers<|im_end|>')
        self.assertEqual(leaf_name(trie, longer), 'fn_add_numbers')
        self.assertEqual(next_tokens(trie, longer), [])
        self.assertNotIn(0, next_tokens(trie, encode_name('fn_ad')))

    def test_arbitrary_names_and_invalid_paths(self) -> None:
        """Names are leaf data, including quotes, Unicode and long names."""
        names = ['say"hello', 'path\\name', '日本語', 'a' * 100]
        trie = build_name_trie(names, encode_name, EOS)
        for name in names:
            self.assertEqual(leaf_name(
                trie, encode_name(name + '<|im_end|>'),
            ), name)
        for lookup in (leaf_name, next_tokens):
            with self.subTest(lookup=lookup.__name__):
                with self.assertRaises(PromptError):
                    lookup(trie, [999999])

    def test_leaf_name_separates_unfinished_end_and_invalid(self) -> None:
        """An unfinished valid path is None; an unknown path is an error."""
        trie = build_name_trie(['ab', 'abc'], encode_name, EOS)
        self.assertIsNone(leaf_name(trie, []))
        self.assertIsNone(leaf_name(trie, encode_name('ab')))
        self.assertEqual(leaf_name(trie, encode_name('ab') + [0]), 'ab')
        for path in (encode_name('ax'), encode_name('ab') + [0, 0],
                     encode_name('abc') + [0, 1]):
            with self.subTest(path=path), self.assertRaises(PromptError):
                leaf_name(trie, path)

    def test_encoding_failures_are_startup_errors(self) -> None:
        """Tokenizer failures cannot escape the startup boundary."""
        def fail(text: str) -> list[int]:
            """Represent a failure from the actual wrapper boundary."""
            raise PromptError('tokenizer failed')

        with self.assertRaisesRegex(StartupError, 'tokenizer failed'):
            build_name_trie(['f'], fail, EOS)

    def test_eos_text_comes_from_the_caller(self) -> None:
        """Each full path is encode(name + eos), whatever eos encodes to."""
        self.assertEqual(EOS, '<|im_end|>')  # Qwen's response end
        trie = build_name_trie(['ab', 'a'], lambda text: (
            [ord(c) for c in text.removesuffix('<END>')]
            + ([7] if text.endswith('<END>') else [])), '<END>')
        self.assertEqual(dict(trie.paths), {'ab': (97, 98, 7), 'a': (97, 7)})
        self.assertEqual(next_tokens(trie, [97]), [98, 7])  # input order
        self.assertEqual(leaf_name(trie, [97, 7]), 'a')
        self.assertEqual(max_name_steps(trie), 3)

    def test_next_tokens_are_unique_in_first_seen_order(self) -> None:
        """Shared prefixes list each next token once, in input order."""
        trie = NameTrie(paths={'c': (1, 3, 0), 'a': (1, 2, 0),
                               'b': (1, 3, 9, 0), 'd': (4, 0)})
        self.assertEqual(next_tokens(trie, []), [1, 4])  # 1 seen three times
        self.assertEqual(next_tokens(trie, [1]), [3, 2])  # not sorted
        self.assertEqual(next_tokens(trie, [1, 3]), [0, 9])
        self.assertEqual(next_tokens(trie, [1, 3, 0]), [])  # full path
        self.assertEqual(leaf_name(trie, [1, 3, 9, 0]), 'b')
        self.assertEqual(max_name_steps(trie), 4)

    def test_multi_token_eos_is_followed_to_the_full_path(self) -> None:
        """A terminator of several tokens is just more steps on the path."""
        def encode(text: str) -> list[int]:
            """Encode characters; the terminator '$$' becomes 1, 2, 3."""
            ids: list[int] = []
            for index, part in enumerate(text.split('$$')):
                ids += ([1, 2, 3] if index else []) + [ord(c) for c in part]
            return ids

        trie = build_name_trie(['fn_add', 'fn_add_numbers'], encode, '$$')
        short = encode('fn_add')
        self.assertEqual(set(next_tokens(trie, short)), {1, ord('_')})
        for step in (short + [1], short + [1, 2]):
            self.assertIsNone(leaf_name(trie, step))  # unfinished terminator
            self.assertEqual(len(next_tokens(trie, step)), 1)
        self.assertEqual(leaf_name(trie, short + [1, 2, 3]), 'fn_add')
        self.assertEqual(next_tokens(trie, short + [1, 2, 3]), [])
        self.assertEqual(leaf_name(trie, encode('fn_add_numbers$$')),
                         'fn_add_numbers')

    def test_terminator_may_merge_with_the_name(self) -> None:
        """encode(name + eos) need not be encode(name) + encode(eos)."""
        def encode(text: str) -> list[int]:
            """Encode characters, merging 'b!' into token 50 like BPE."""
            ids = [ord(c) for c in text.replace('b!', '\0')]
            return [50 if token == 0 else token for token in ids]

        self.assertEqual(encode('ab!'), [97, 50])
        self.assertNotEqual(encode('ab!'), encode('ab') + encode('!'))
        trie = build_name_trie(['ab', 'a'], encode, '!')
        self.assertEqual(dict(trie.paths), {'ab': (97, 50), 'a': (97, 33)})
        self.assertEqual(leaf_name(trie, [97, 50]), 'ab')
        self.assertEqual(leaf_name(trie, [97, 33]), 'a')

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
            with self.subTest(case), self.assertRaises(StartupError):
                build_name_trie(names, encode, EOS)
        # An EOS token inside one name is only text; no structure is broken.
        trie = build_name_trie(['a' + EOS + 'b'], encode_name, EOS)
        self.assertEqual(leaf_name(trie, [97, 0, 98, 0]), 'a' + EOS + 'b')

    def test_direct_construction_validates_the_whole_trie(self) -> None:
        """No constructor path yields an unchecked or partial trie."""
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
                NameTrie.model_validate(data)
        trie = NameTrie(paths={'a': (97, 0), 'ab': (97, 98, 0)})
        self.assertEqual(next_tokens(trie, [97]), [0, 98])
        self.assertEqual(next_tokens(trie, [97, 0]), [])  # leaf: no children
        self.assertIsNone(leaf_name(trie, [97]))  # inner node: no name

    def test_unvalidated_trie_cannot_be_used(self) -> None:
        """model_construct skips validation, so no index is ever built."""
        trie = NameTrie.model_construct(paths={'a': (97, 0)})
        for lookup in (leaf_name, next_tokens):
            with self.subTest(lookup.__name__):
                with self.assertRaises(AttributeError):
                    lookup(trie, [97])

    def test_finished_trie_is_immutable(self) -> None:
        """Neither the trie nor its source dictionary can change its paths."""
        source = {'a': (97, 0), 'b': (98, 0)}
        trie = NameTrie(paths=source)
        source['c'] = (99, 0)
        del source['a']
        self.assertEqual(dict(trie.paths), {'a': (97, 0), 'b': (98, 0)})
        self.assertEqual(next_tokens(trie, []), [97, 98])
        with self.assertRaises(TypeError):
            trie.paths['c'] = (99, 0)  # type: ignore[index]
        with self.assertRaises(ValidationError):
            trie.paths = {}
        allowed = next_tokens(trie, [])
        allowed.append(99)
        self.assertEqual(next_tokens(trie, []), [97, 98])
        with self.assertRaises(PromptError):
            next_tokens(trie, [99])

    def test_serialization_round_trip(self) -> None:
        """Dumped data validates back to an equal, usable trie."""
        trie = build_name_trie(['ab', 'a'], encode_name, EOS)
        for again in (NameTrie.model_validate(trie.model_dump()),
                      NameTrie.model_validate_json(trie.model_dump_json())):
            self.assertEqual(again, trie)
            self.assertEqual(leaf_name(again, [97, 0]), 'a')


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


class GrammarTests(unittest.TestCase):
    """Allow every valid prefix and reject bytes past slot delimiters."""

    def test_number_prefixes_and_completion(self) -> None:
        """Decimal-only numbers continue until the actual delimiter."""
        for value in ['0.0', '-0.0', '12.0', '1.5', '-0.25', '9' * 400 + '.0',
                      '0.12345678901234567']:
            for terminator in [',', '}']:
                data = value.encode() + terminator.encode()
                for index in range(len(data)):
                    written = data[:index]
                    token = data[index:index + 1]
                    self.assertFalse(is_slot_complete(
                        written, 'number', terminator,
                    ))
                    self.assertIn(1, allowed_slot_tokens(
                        {1: token}, 'number', written, terminator,
                    ))
                self.assertTrue(is_slot_complete(data, 'number', terminator))
        self.assertTrue(is_slot_complete(b' \n-12.5 \t}', 'number', '}'))

    def test_invalid_numbers_and_crossing_tokens_are_masked(self) -> None:
        """A candidate cannot add bad syntax or text after a delimiter."""
        vocab = {0: b'1', 1: b'.', 2: b',', 3: b', "next":',
                 4: b'e2', 5: b'}', 6: b'', 7: b'0'}
        self.assertEqual(set(allowed_slot_tokens(
            vocab, 'number', b'12', ',',
        )), {0, 1, 7})
        for value in [b'01,', b'+1,', b'.1,', b'1.,', b'NaN,',
                      b'1e3,', b'1.0e3,', b'1,', b'1, ', b'1,true',
                      b'1}']:
            self.assertFalse(is_slot_complete(value, 'number', ','))

    def test_boolean_grammar(self) -> None:
        """Only lowercase JSON booleans can reach their delimiter."""
        self.assertEqual(allowed_slot_tokens(
            {1: b'ue}', 2: b'ue}junk', 3: b'ue,', 4: b'ue'},
            'boolean', b'tr', '}',
        ), [1, 4])
        for value in [b'true}', b'false}', b' true \n}']:
            self.assertTrue(is_slot_complete(value, 'boolean', '}'))
        for value in [b'True}', b'1}', b'false', b'true},', b'true} ']:
            self.assertFalse(is_slot_complete(value, 'boolean', '}'))

    def test_strings_accept_every_byte_prefix_and_json_escape(self) -> None:
        """Split UTF-8 and split escapes stay valid until completion."""
        for text in ['', ' ', 'the "best" day', '\\', '\n\t',
                     '日本語é🙂', 'commas, and braces }']:
            value = json.dumps(text, ensure_ascii=False).encode() + b'}'
            for index in range(len(value)):
                self.assertIn(1, allowed_slot_tokens(
                    {1: value[index:index + 1]}, 'string', value[:index], '}',
                ), msg=f'{text!r}, byte {index}')
            self.assertTrue(is_slot_complete(value, 'string', '}'))
        escaped = b'"\\u3042"}'
        for index in range(len(escaped)):
            self.assertIn(1, allowed_slot_tokens(
                {1: escaped[index:index + 1]}, 'string', escaped[:index], '}',
            ))

    def test_invalid_string_bytes_are_masked(self) -> None:
        """Valid delimiters inside strings never terminate a value early."""
        for token in [b'"a\n"}', b'"\\q"}', b'"\\u00xz"}', b'"a"}x',
                      b'"a"} ', b'"\xff"}', b'"\xe3"}', b'"a",']:
            self.assertEqual(allowed_slot_tokens(
                {1: token}, 'string', b'', '}',
            ), [], msg=repr(token))
        self.assertFalse(is_slot_complete(b'"a}', 'string', '}'))
        self.assertEqual(allowed_slot_tokens(
            {1: b'\x82"}', 2: b'"}', 3: b'\xff'},
            'string', b'"\xe3\x81', '}',
        ), [1])


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
        prompt = build_parameters_prompt('Keep the text', fn)
        self.assertIn(fn.description, prompt)
        self.assertIn('Keep the text', prompt)
        self.assertIn('Extract the arguments', prompt)
        self.assertIn('from the user request', prompt)
        self.assertIn('do not answer the request or compute a result', prompt)
