"""Build generation assets from definitions validated by the input loader."""

import unittest
from unittest.mock import patch

from pydantic import ValidationError

from src.errors import StartupError
from src.loader import index_functions
from src.model_port import Model
from src.runtime import GenerationAssets
from src.schema import FunctionDef, ParamSpec
from test_constraints import encode_name


def function(name: str) -> FunctionDef:
    """Return a minimal definition named ``name``."""
    return FunctionDef(name=name, description='test', parameters={},
                       returns=ParamSpec(type='string'))


class ResourceTests(unittest.TestCase):
    """Keep real trie construction; replace only model and vocabulary IO."""

    def setUp(self) -> None:
        self.functions = index_functions([function('g'), function('f')])
        self.vocab = {1: b'a'}
        for mock in (
            patch.object(Model, '__init__', return_value=None),
            patch.object(Model, 'encode', side_effect=encode_name),
            patch.object(Model, 'vocab_path', return_value='unused.json'),
            patch('src.runtime.load_vocab', return_value=self.vocab),
        ):
            mock.start()
            self.addCleanup(mock.stop)

    def test_builds_matching_trie_and_read_only_functions(self) -> None:
        """Input keys agree with names and select the same definitions."""
        resource = GenerationAssets(self.functions)
        self.assertEqual(list(resource.functions), ['g', 'f'])
        self.assertEqual(resource.trie.names, {'g', 'f'})
        for key, fn in self.functions.items():
            self.assertEqual(key, fn.name)
            self.assertIs(resource.functions[key], fn)
            self.assertEqual(resource.trie.leaf_name(
                encode_name(key + '<|im_end|>')), key)
        self.assertIs(resource.vocab, self.vocab)
        self.functions.clear()
        self.assertEqual(list(resource.functions), ['g', 'f'])
        with self.assertRaises(TypeError):
            resource.functions['h'] = function('h')  # type: ignore[index]
        with self.assertRaises(ValidationError):
            resource.functions = {}

    def test_ambiguous_token_paths_are_startup_errors(self) -> None:
        """NameTrie rejects collisions during asset construction."""
        with patch.object(Model, 'encode', return_value=[1]):
            with self.assertRaises(StartupError):
                GenerationAssets(self.functions)
