"""Reject JSON data loss before Pydantic shape/type validation."""

import contextlib
import io
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from src.errors import PromptError, StartupError
from src.loader import load_functions, load_prompts
from src.pipeline import validate_call
from src.vocab import load_vocab
from tools import campus_survey, measure_application
from tools.check_subject import check_inputs
from test_application import definition


class JSONParserTests(unittest.TestCase):
    """The shared parser owns duplicates, JSON syntax and source numerals."""

    def test_normal_json_preserves_decimal_source_values(self) -> None:
        """Parse decimal digits exactly before any float conversion."""
        from src.json_io import loads

        data = loads('{"a":1,"nested":{"a":0.1234567890123456789},'
                     '"text":"NaN", "bool":true, "null":null}')
        self.assertEqual(data, {
            'a': 1, 'nested': {'a': Decimal('0.1234567890123456789')},
            'text': 'NaN', 'bool': True, 'null': None,
        })
        self.assertIs(type(data['a']), int)

    def test_duplicate_keys_are_rejected_in_every_object(self) -> None:
        """Escaped duplicates and nested objects cannot overwrite values."""
        from src.json_io import loads

        for text in ['{"a":1,"a":2}', '{"x":{"a":1,"a":1}}',
                     '[{"a":1,"a":2}]', '{"a":1,"\\u0061":2}']:
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                    loads(text)
        self.assertEqual(loads('{"a":{"a":1},"b":{"a":2}}'),
                         {'a': {'a': 1}, 'b': {'a': 2}})

    def test_bad_syntax_and_nonstandard_numbers_are_rejected(self) -> None:
        """Do not accept extensions, partial JSON or parser-limit crashes."""
        from src.json_io import loads

        for text in ['NaN', 'Infinity', '-Infinity', '{"a":NaN}',
                     '{', '{"a":1,}', '[1] trailing', '01', '0x10',
                     '1e999999999999999999999999999999999']:
            with self.subTest(text=text[:60]):
                with self.assertRaises(ValueError):
                    loads(text)
        with patch("src.json_io.json.loads", side_effect=RecursionError):
            with self.assertRaisesRegex(ValueError, "recursion limit"):
                loads("[]")


class JSONBoundaryTests(unittest.TestCase):
    """All external JSON paths use the same rejection policy."""

    def test_function_and_prompt_files_reject_duplicate_keys(self) -> None:
        """Detect metadata/type replacement before building definitions."""
        good = json.dumps([definition(value='number').model_dump()])
        bad = [good.replace('"dynamic"', '"dynamic", "name":"other"'),
               good.replace('"value": {"type": "number"}',
                            '"value":{"type":"number"},'
                            '"value":{"type":"boolean"}'),
               good.replace('"type": "number"',
                            '"type":"number","type":"boolean"')]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'input.json'
            for text in bad:
                path.write_text(text, encoding='utf-8')
                with self.assertRaisesRegex(StartupError,
                                            'duplicate JSON key'):
                    load_functions(path)
            path.write_text('[{"prompt":"first","prompt":"second"}]')
            with self.assertRaisesRegex(StartupError, 'duplicate JSON key'):
                load_prompts(path)

    def test_generated_json_rejects_duplicate_keys(self) -> None:
        """Neither record keys nor argument keys may be overwritten."""
        fn = definition(value='number')
        for text in [
            '{"prompt":"x","name":"wrong","name":"dynamic",'
            '"parameters":{"value":2}}',
            '{"prompt":"x","name":"dynamic",'
            '"parameters":{"value":1,"value":2}}',
        ]:
            with self.assertRaisesRegex(PromptError, 'duplicate JSON key'):
                validate_call(text, fn)

    def test_vocabulary_rejects_bad_json(self) -> None:
        """A repeated byte-token spelling must not silently change its ID."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'vocab.json'
            for text in ['{"a":0,"a":1}', '{"a":NaN}',
                         '{"a":Infinity}', '{"a":true}', '{"a":"1"}']:
                path.write_text(text)
                with self.assertRaises(StartupError):
                    load_vocab(path)

    def test_input_checker_rejects_duplicate_keys(self) -> None:
        """Development preflight must not disagree with application input."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'data/input'
            folder.mkdir(parents=True)
            (folder / 'functions_definition.json').write_text(json.dumps([
                definition().model_dump(),
            ]))
            (folder / 'function_calling_tests.json').write_text(
                '[{"prompt":"first","prompt":"second"}]',
            )
            self.assertIn('duplicate JSON key', ' '.join(check_inputs(root)))

    def test_measurement_rejects_duplicates_before_model(self) -> None:
        """Do not run an evaluation on silently replaced fixture data."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'fixture.json'
            path.write_text('{"functions":[],"functions":[],"cases":[]}')
            stderr = io.StringIO()
            with patch('sys.argv', ['measure', '--cases', str(path),
                                    '--output', str(root / 'out.json')]), \
                    patch.object(measure_application, 'Model',
                                 side_effect=StartupError('model sentinel')) \
                    as model, contextlib.redirect_stderr(stderr):
                self.assertEqual(measure_application.main(), 1)
            model.assert_not_called()
            self.assertIn('duplicate JSON key', stderr.getvalue())

    def test_campus_readers_reject_duplicate_keys(self) -> None:
        """Both baseline verification and prompt counting reject overwrite."""
        for operation in [campus_survey.sdk_matches_baseline,
                          lambda: campus_survey.estimate([(1, .1), (2, .2)],
                                                         .1)]:
            with patch.object(campus_survey, 'read',
                              return_value='{"x":1,"x":2}'):
                with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                    operation()
