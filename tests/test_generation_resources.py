"""Shared resources keep references and reject a mismatched function set."""

import unittest
from typing import Any
from unittest.mock import patch

from pydantic import ValidationError

from src.model_port import Model
from src.name_trie import EOS, NameTrie, build_name_trie
from src.pipeline import GenerationResources
from src.schema import FunctionDef, ParamSpec
from test_constraints import encode_name


def function(name: str) -> FunctionDef:
    """Return a minimal definition named ``name``."""
    return FunctionDef(name=name, description='test', parameters={},
                       returns=ParamSpec(type='string'))


class ResourceTests(unittest.TestCase):
    """Assembly checks the envelope and the functions/trie correspondence."""

    def setUp(self) -> None:
        """Use an unloaded Model whose encode is the test tokenizer."""
        encode = patch.object(Model, 'encode', side_effect=encode_name)
        encode.start()
        self.addCleanup(encode.stop)
        self.inputs: dict[str, Any] = {
            'model': object.__new__(Model), 'vocab': {1: b'a'},
            'functions': {'f': function('f')},
            'trie': build_name_trie(['f'], encode_name),
        }

    def test_all_references_and_nested_models_are_retained(self) -> None:
        """No dictionary, function or tree is reconstructed at assembly."""
        resource = GenerationResources(**self.inputs)
        for key, value in self.inputs.items():
            self.assertIs(getattr(resource, key), value)
        self.assertIs(resource.functions['f'], self.inputs['functions']['f'])
        self.assertIs(resource.functions['f'].parameters,
                      self.inputs['functions']['f'].parameters)
        self.assertIs(resource.functions['f'].returns,
                      self.inputs['functions']['f'].returns)

    def test_wrong_roots_missing_and_extra_fields_are_rejected(self) -> None:
        """Root checks do not pretend to validate SDK initialization."""
        for field, value in (('model', object()), ('vocab', []),
                             ('functions', []), ('trie', {})):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                GenerationResources(**(self.inputs | {field: value}))
        with self.assertRaises(ValidationError):
            GenerationResources(**(self.inputs | {'report': None}))
        for key in self.inputs:
            with self.subTest(missing=key), self.assertRaises(ValidationError):
                GenerationResources(**{k: v for k, v in self.inputs.items()
                                       if k != key})

    def test_functions_and_trie_must_correspond(self) -> None:
        """Name sets, keys and token paths must all match one function set."""
        f, g = function('f'), function('g')
        cases: dict[str, dict[str, Any]] = {
            'trie has a missing function': {
                'trie': build_name_trie(['f', 'g'], encode_name)},
            'functions have a name not in the trie': {
                'functions': {'f': f, 'g': g}},
            'trie built from other names': {
                'trie': build_name_trie(['g'], encode_name)},
            'key differs from FunctionDef.name': {
                'functions': {'f': g}},
            'value is not a FunctionDef': {'functions': {'f': object()}},
            # Same name set, but the path does not spell "f" + EOS.
            'path does not spell the name': {'trie': NameTrie(children={
                ord('g'): NameTrie(children={0: NameTrie(name='f')})})},
            'trie from another tokenizer': {'trie': build_name_trie(
                ['f'], lambda text: [5] if text == EOS else [6, 5])},
        }
        for case, change in cases.items():
            with self.subTest(case), self.assertRaises(ValidationError):
                GenerationResources(**(self.inputs | change))

    def test_vocab_is_not_rescanned_and_frozen_is_shallow(self) -> None:
        """Vocab contents stay unchecked; aliases remain mutable."""
        bad_vocab = {'bad-id': 'not-bytes'}
        resource = GenerationResources(**(self.inputs | {'vocab': bad_vocab}))
        self.assertIs(resource.vocab, bad_vocab)
        with self.assertRaises(ValidationError):
            resource.vocab = {}
        with self.assertRaises(ValidationError):
            del resource.trie
        bad_vocab['later'] = 'still mutable'
        self.assertIn('later', resource.vocab)
        resource.functions['later'] = function('later')
        self.assertIn('later', resource.functions)  # not rechecked later
