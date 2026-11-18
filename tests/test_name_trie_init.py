"""Exercise name encoding through the public trie constructor."""

import unittest
from unittest.mock import patch

from src.errors import PromptError, StartupError
from src.model_port import EOS, Model
from src.name_trie import NameTrie


class TrieConstructorTests(unittest.TestCase):
    """Use a loaded Model instance without loading the SDK in tests."""

    def test_mapping_keys_and_combined_eos_keep_input_order(self) -> None:
        """Encode complete names once and retain first-seen branch order."""
        names = {'ab': object(), 'a': object()}
        model = object.__new__(Model)
        paths = {'ab' + EOS: [97, 50], 'a' + EOS: [97, 33]}
        with patch.object(Model, 'encode', side_effect=paths.__getitem__), \
                patch.object(Model, '__init__', side_effect=AssertionError):
            trie = NameTrie(names, model)
        self.assertEqual(dict(trie.paths), {'ab': (97, 50), 'a': (97, 33)})
        self.assertEqual(trie.next_tokens([97]), [50, 33])
        self.assertEqual(trie.leaf_name([97, 33]), 'a')
        names.clear()
        paths['ab' + EOS].clear()
        self.assertEqual(trie.leaf_name([97, 50]), 'ab')

    def test_encoding_error_preserves_startup_context_and_cause(self) -> None:
        """Constructor keeps the old factory's error boundary."""
        error = PromptError('tokenizer failed')
        with patch.object(Model, 'encode', side_effect=error):
            with self.assertRaisesRegex(
                StartupError, '^cannot build name tree: tokenizer failed$'
            ) as raised:
                NameTrie(['f'], object.__new__(Model))
        self.assertIs(raised.exception.__cause__, error)
