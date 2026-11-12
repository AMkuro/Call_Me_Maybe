"""Shared resources validate reference types without rebuilding their data."""

import unittest
from typing import Any

from pydantic import ValidationError

from src.model_port import Model
from src.name_trie import NameTrie
from src.pipeline import GenerationResources
from src.schema import FunctionDef, ParamSpec


class ResourceTests(unittest.TestCase):
    """Assembly validates the envelope; loaders own content validation."""

    def setUp(self) -> None:
        """Use already constructed resources without loading the SDK."""
        fn = FunctionDef(name='f', description='test', parameters={},
                         returns=ParamSpec(type='string'))
        self.inputs: dict[str, Any] = {
            'model': object.__new__(Model), 'vocab': {1: b'a'},
            'functions': {'f': fn}, 'trie': NameTrie(),
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

    def test_contents_are_not_revalidated_and_frozen_is_shallow(self) -> None:
        """The wrapper checks no contents and leaves aliases mutable."""
        bad_vocab = {'bad-id': 'not-bytes'}
        bad_functions = {'bad': object()}
        resource = GenerationResources(**(self.inputs | {
            'vocab': bad_vocab, 'functions': bad_functions,
        }))
        self.assertIs(resource.vocab, bad_vocab)
        self.assertIs(resource.functions, bad_functions)
        with self.assertRaises(ValidationError):
            resource.vocab = {}
        with self.assertRaises(ValidationError):
            del resource.trie
        bad_vocab['later'] = 'still mutable'
        self.assertIn('later', resource.vocab)
