"""Check actual trie, byte grammar and prompt implementations directly."""

import json
import tempfile
import unittest
from pathlib import Path

from src.errors import PromptError, StartupError
from src.name_trie import build_name_trie, leaf_name, next_tokens
from src.prompt import build_name_prompt, build_parameters_prompt
from src.values import allowed_value_tokens, is_value_complete
from src.vocab import load_vocab
from test_application import definition


def encode_name(text: str) -> list[int]:
    """Encode characters plus the EOS marker for direct trie tests."""
    suffix = '<|im_end|>'
    if text.endswith(suffix):
        return [ord(char) for char in text[:-len(suffix)]] + [0]
    return [ord(char) for char in text]


class TrieTests(unittest.TestCase):
    """EOS separates valid complete names from longer candidates."""

    def test_prefix_names_choose_end_or_continue(self) -> None:
        """A shorter name becomes a leaf only after its EOS."""
        trie = build_name_trie(['fn_add', 'fn_add_numbers'], encode_name)
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
        trie = build_name_trie(names, encode_name)
        for name in names:
            self.assertEqual(leaf_name(
                trie, encode_name(name + '<|im_end|>'),
            ), name)
        self.assertIsNone(leaf_name(trie, [999999]))
        with self.assertRaises(PromptError):
            next_tokens(trie, [999999])

    def test_encoding_failures_are_startup_errors(self) -> None:
        """Tokenizer failures cannot escape the startup boundary."""
        def fail(text: str) -> list[int]:
            """Represent a failure from the actual wrapper boundary."""
            raise PromptError('tokenizer failed')

        with self.assertRaisesRegex(StartupError, 'tokenizer failed'):
            build_name_trie(['f'], fail)
        with self.assertRaises(StartupError):
            build_name_trie(['a', 'b'], lambda text: [1])


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
        """Integers and decimals continue until the actual delimiter."""
        for value in ['0', '-0', '12', '1.5', '-0.25', '9' * 400,
                      '0.12345678901234567']:
            for terminator in [',', '}']:
                data = value.encode() + terminator.encode()
                for index in range(len(data)):
                    written = data[:index]
                    token = data[index:index + 1]
                    self.assertFalse(is_value_complete(
                        written, 'number', terminator,
                    ))
                    self.assertIn(1, allowed_value_tokens(
                        {1: token}, 'number', written, terminator,
                    ))
                self.assertTrue(is_value_complete(data, 'number', terminator))
        self.assertTrue(is_value_complete(b' \n-12.5 \t}', 'number', '}'))

    def test_invalid_numbers_and_crossing_tokens_are_masked(self) -> None:
        """A candidate cannot add bad syntax or text after a delimiter."""
        vocab = {0: b'1', 1: b'.', 2: b',', 3: b', "next":',
                 4: b'e2', 5: b'}', 6: b'', 7: b'0'}
        self.assertEqual(set(allowed_value_tokens(
            vocab, 'number', b'12', ',',
        )), {0, 1, 2, 7})
        for value in [b'01,', b'+1,', b'.1,', b'1.,', b'NaN,',
                      b'1e3,', b'1, ', b'1,true', b'1}']:
            self.assertFalse(is_value_complete(value, 'number', ','))

    def test_boolean_grammar(self) -> None:
        """Only lowercase JSON booleans can reach their delimiter."""
        self.assertEqual(allowed_value_tokens(
            {1: b'ue}', 2: b'ue}junk', 3: b'ue,', 4: b'ue'},
            'boolean', b'tr', '}',
        ), [1, 4])
        for value in [b'true}', b'false}', b' true \n}']:
            self.assertTrue(is_value_complete(value, 'boolean', '}'))
        for value in [b'True}', b'1}', b'false', b'true},']:
            self.assertFalse(is_value_complete(value, 'boolean', '}'))

    def test_strings_accept_every_byte_prefix_and_json_escape(self) -> None:
        """Split UTF-8 and split escapes stay valid until completion."""
        for text in ['', ' ', 'the "best" day', '\\', '\n\t',
                     '日本語é🙂', 'commas, and braces }']:
            value = json.dumps(text, ensure_ascii=False).encode() + b'}'
            for index in range(len(value)):
                self.assertIn(1, allowed_value_tokens(
                    {1: value[index:index + 1]}, 'string', value[:index], '}',
                ), msg=f'{text!r}, byte {index}')
            self.assertTrue(is_value_complete(value, 'string', '}'))
        escaped = b'"\\u3042"}'
        for index in range(len(escaped)):
            self.assertIn(1, allowed_value_tokens(
                {1: escaped[index:index + 1]}, 'string', escaped[:index], '}',
            ))

    def test_invalid_string_bytes_are_masked(self) -> None:
        """Valid delimiters inside strings never terminate a value early."""
        for token in [b'"a\n"}', b'"\\q"}', b'"\\u00xz"}', b'"a"}x',
                      b'"a"} ', b'"\xff"}', b'"\xe3"}', b'"a",']:
            self.assertEqual(allowed_value_tokens(
                {1: token}, 'string', b'', '}',
            ), [], msg=repr(token))
        self.assertFalse(is_value_complete(b'"a}', 'string', '}'))
        self.assertEqual(allowed_value_tokens(
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
        self.assertIn('do not execute', prompt)
