"""Keep schema errors strict and incomplete metadata warning-only."""

import contextlib
import io
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src import __main__ as entry, pipeline
from src.loader import load_functions
from src.model_port import Model
from src.schema import FunctionDef, ParamSpec
from test_application import definition
from test_constraints import encode_name


class DefinitionValidationTests(unittest.TestCase):
    """Use Pydantic for schema checks without changing valid field values."""

    def test_missing_extra_wrong_type_and_unsupported_fields(self) -> None:
        """Do not accept malformed definitions or implicit text conversions."""
        valid = definition(value='number').model_dump()
        invalid = [{k: v for k, v in valid.items() if k != missing}
                   for missing in valid]
        invalid += [
            {**valid, 'extra': 1}, {**valid, 'name': 1},
            {**valid, 'description': False}, {**valid, 'parameters': []},
            {**valid, 'returns': 'number'}, {**valid, 'name': b'dynamic'},
            {**valid, 'description': b'description'},
            {**valid, 'parameters': {'a': {'type': 'integer'}}},
            {**valid, 'parameters': {'a': {'type': 'number', 'extra': 1}}},
        ]
        for data in invalid:
            with self.subTest(data=data):
                with self.assertRaises(ValidationError):
                    FunctionDef.model_validate(data)
        for spec in [{}, {'type': True}, {'type': 'unknown'},
                     {'type': 'number', 'extra': 1}]:
            with self.assertRaises(ValidationError):
                ParamSpec.model_validate(spec)

    def test_blank_function_names_are_pydantic_errors(self) -> None:
        """Blank identifiers are rejected at the model-validation boundary."""
        valid = definition().model_dump()
        for name in ['', ' \t\n']:
            with self.assertRaises(ValidationError):
                FunctionDef.model_validate({**valid, 'name': name})

    def test_blank_metadata_warns_and_is_preserved(self) -> None:
        """Warn for description and parameter names without repairing them."""
        raw = {
            'name': ' original name ', 'description': ' \t',
            'parameters': {'': {'type': 'string'}, '  ': {'type': 'number'}},
            'returns': {'type': 'boolean'},
        }
        with self.assertLogs(level='WARNING') as logs:
            fn = FunctionDef.model_validate(raw)
        self.assertEqual(fn.model_dump(), raw)
        self.assertEqual(len(logs.records), 3)
        messages = '\n'.join(item.getMessage() for item in logs.records)
        self.assertIn(repr(raw['name']), messages)
        self.assertIn('description', messages)
        self.assertIn('parameter', messages)
        self.assertIn("''", messages)
        self.assertIn("'  '", messages)

    def test_zero_parameters_and_nonblank_metadata_do_not_warn(self) -> None:
        """Zero-argument functions remain valid and unchanged."""
        raw = definition().model_dump()
        with self.assertNoLogs(level='WARNING'):
            fn = FunctionDef.model_validate(raw)
        self.assertEqual(fn.model_dump(), raw)

    def test_loader_reports_warnings_without_altering_input(self) -> None:
        """The file boundary keeps metadata bytes meaningful and warns."""
        raw = {**definition().model_dump(), 'description': '',
               'parameters': {' ': {'type': 'string'}}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'functions.json'
            path.write_text(json.dumps([raw]))
            with self.assertLogs(level='WARNING') as logs:
                functions = load_functions(path)
        self.assertEqual(functions['dynamic'].model_dump(), raw)
        self.assertEqual(len(logs.records), 2)


class ValidationFlowTests(unittest.TestCase):
    """Real orchestration must preserve errors, warnings and saved values."""

    def test_prompt_errors_and_stderr_warnings(self) -> None:
        """Keep exact successes around a failing value, without JSON logs."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = {**definition().model_dump(), 'description': ' ',
                   'parameters': {'': {'type': 'number'}}}
            functions = root / 'functions.json'
            functions.write_text(json.dumps([raw]))
            prompts = root / 'prompts.json'
            prompts.write_text('[{"prompt":"x"},{"prompt":"x"},'
                               '{"prompt":"x"}]')
            vocab = root / 'vocab.json'
            vocab.write_text('{"a":0}')
            output = root / 'output.json'
            model = object.__new__(Model)
            bodies = [
                '{"prompt":"x","name":"dynamic","parameters":{"":'
                + literal + '}}'
                for literal in ['9007199254740993',
                                '0.1234567890123456789', '0.1']
            ]
            stderr, stdout = io.StringIO(), io.StringIO()
            with patch.object(entry, 'Model', return_value=model), \
                    patch.object(Model, 'encode', side_effect=encode_name), \
                    patch.object(Model, 'decode', side_effect=(
                        lambda ids: ''.join(chr(x) for x in ids if x)
                    )), patch.object(Model, 'vocab_path',
                                     return_value=str(vocab)), \
                    patch.object(Model, 'logits', side_effect=AssertionError(
                        'one-name paths are forced'
                    )), patch.object(pipeline, 'generate_parameters',
                                     side_effect=bodies), \
                    contextlib.redirect_stderr(stderr), \
                    contextlib.redirect_stdout(stdout):
                status = entry.main([
                    '--functions_definition', str(functions),
                    '--input', str(prompts), '--output', str(output),
                ])
            self.assertEqual(status, 1)
            saved = json.loads(output.read_text(), parse_float=Decimal)
            self.assertEqual([row['parameters'][''] for row in saved],
                             [9007199254740993, Decimal('0.1')])
            self.assertEqual(stdout.getvalue(), '')
            self.assertIn('description', stderr.getvalue())
            self.assertIn('parameter', stderr.getvalue())
            self.assertIn('failed', stderr.getvalue())
            self.assertNotIn('warning', output.read_text().lower())
