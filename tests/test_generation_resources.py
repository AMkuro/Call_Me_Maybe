"""Shared resources combine validated parts and keep them consistent."""

import unittest
from typing import Any
from unittest.mock import patch

from pydantic import ValidationError

from src.model_port import EOS, Model
from src.name_trie import NameTrie, build_name_trie
from src.pipeline import GenerationResources
from src.schema import FunctionDef, ParamSpec
from test_constraints import encode_name


def function(name: str) -> FunctionDef:
    """Return a minimal definition named ``name``."""
    return FunctionDef(name=name, description='test', parameters={},
                       returns=ParamSpec(type='string'))


def trie_of(*names: str) -> NameTrie:
    """Build a trie with the test tokenizer, outside the resource check."""
    return build_name_trie(list(names), encode_name, EOS)


class ResourceTests(unittest.TestCase):
    """Assembly checks names only; the trie owns its own paths."""

    def setUp(self) -> None:
        """Fail any test that tokenizes at assembly (a rebuild needs it)."""
        encode = patch.object(Model, 'encode', side_effect=AssertionError(
            'encode called during assembly'))
        encode.start()
        self.addCleanup(encode.stop)
        self.inputs: dict[str, Any] = {
            'model': object.__new__(Model), 'vocab': {1: b'a'},
            'functions': {'g': function('g'), 'f': function('f')},
            'trie': trie_of('g', 'f'),
        }

    def test_references_and_read_only_functions(self) -> None:
        """Model, vocab and trie are shared; functions is a frozen copy."""
        resource = GenerationResources(**self.inputs)
        for key in ('model', 'vocab', 'trie'):
            self.assertIs(getattr(resource, key), self.inputs[key])
        source = self.inputs['functions']
        self.assertIsNot(resource.functions, source)
        self.assertEqual(list(resource.functions), ['g', 'f'])  # input order
        for key, fn in source.items():
            self.assertIs(resource.functions[key], fn)  # no deep copy
            self.assertIs(resource.functions[key].parameters, fn.parameters)

    def test_functions_cannot_change_after_assembly(self) -> None:
        """Neither the source dict nor the resource view breaks the match."""
        resource = GenerationResources(**self.inputs)
        source = self.inputs['functions']
        source['h'] = function('h')
        del source['f']
        self.assertEqual(list(resource.functions), ['g', 'f'])
        with self.assertRaises(TypeError):
            resource.functions['h'] = function('h')  # type: ignore[index]
        with self.assertRaises(TypeError):
            del resource.functions['f']  # type: ignore[attr-defined]
        with self.assertRaises(ValidationError):
            resource.functions = {}

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

    def test_functions_and_trie_names_must_match(self) -> None:
        """Key sets and FunctionDef.name must agree with the trie's names."""
        f, g, h = function('f'), function('g'), function('h')
        cases: dict[str, dict[str, Any]] = {
            'trie has a missing function': {'trie': trie_of('g', 'f', 'h')},
            'functions have a name not in the trie': {
                'functions': {'g': g, 'f': f, 'h': h}},
            'trie built from other names': {'trie': trie_of('h')},
            'key differs from FunctionDef.name': {
                'functions': {'g': g, 'f': h}},
            'value is not a FunctionDef': {
                'functions': {'g': g, 'f': object()}},
        }
        for case, change in cases.items():
            with self.subTest(case), self.assertRaises(ValidationError):
                GenerationResources(**(self.inputs | change))

    def test_unvalidated_trie_is_rejected(self) -> None:
        """A trie that skipped its own validation has no names to match."""
        trie = NameTrie.model_construct(eos=0, paths={'g': (103, 0),
                                                      'f': (102, 0)})
        with self.assertRaises(AttributeError):
            GenerationResources(**(self.inputs | {'trie': trie}))

    def test_vocab_is_not_rescanned_and_stays_shared(self) -> None:
        """Vocab contents stay unchecked; the shared dict is not copied."""
        bad_vocab = {'bad-id': 'not-bytes'}
        resource = GenerationResources(**(self.inputs | {'vocab': bad_vocab}))
        self.assertIs(resource.vocab, bad_vocab)
        with self.assertRaises(ValidationError):
            del resource.trie
        bad_vocab['later'] = 'still mutable'
        self.assertIn('later', resource.vocab)
