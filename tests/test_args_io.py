"""Validate input snapshots and output preparation before expensive startup."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import __main__ as entry
from src.cli import Args, parse_args
from src.errors import StartupError
from src.loader import load_prompts
from src.writer import save_results


class ArgsIOTests(unittest.TestCase):
    """Use actual files and keep the model boundary outside these tests."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.functions = self.root / 'functions.json'
        self.functions.write_text(json.dumps([{
            'name': 'ping', 'description': 'Check health.',
            'parameters': {}, 'returns': {'type': 'boolean'},
        }]))
        self.prompts = self.root / 'prompts.json'
        self.prompts.write_text('[{"prompt":"ping"}]')
        self.output = self.root / 'results' / 'nested' / 'calls.json'

    def arguments(self) -> list[str]:
        """Use the unchanged public CLI options with controlled paths."""
        return ['--functions_definition', str(self.functions),
                '--input', str(self.prompts), '--output', str(self.output)]

    def test_initialization_loads_inputs_and_prepares_output(self) -> None:
        """Completed Args holds validated input and a prepared destination."""
        args = Args(functions_definition=self.functions, input=self.prompts,
                    output=self.output)
        self.assertTrue(self.output.parent.is_dir())
        self.assertEqual(list(self.output.parent.iterdir()), [])
        self.assertEqual(list(args.functions), ['ping'])
        self.assertEqual(args.prompts, ['ping'])
        self.assertEqual(set(args.model_dump()),
                         {'functions_definition', 'input', 'output'})

    def test_bad_inputs_do_not_prepare_output(self) -> None:
        """Unreadable, malformed and insufficient inputs fail first."""
        cases = ['missing', 'directory', '{', '[]']
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as d:
                path = Path(d) / 'prompts.json'
                if case == 'directory':
                    path.mkdir()
                elif case != 'missing':
                    path.write_text(case)
                with self.assertRaises(StartupError):
                    Args(functions_definition=self.functions, input=path,
                         output=self.output)
                self.assertFalse(self.output.parent.exists())

    def test_start_reads_each_input_once_and_keeps_its_snapshot(self) -> None:
        """Keep startup snapshots; loaders read current files."""
        original = Path.read_text
        reads: list[Path] = []

        def read(path: Path, *args: object, **kwargs: object) -> str:
            reads.append(path)
            return original(path, encoding='utf-8')

        with patch.object(Path, 'read_text', read), \
                patch.object(entry, 'GenerationAssets'):
            args, prompts, _ = entry._start(self.arguments())
        self.assertEqual(reads, [self.functions, self.prompts])
        self.prompts.write_text('[{"prompt":"changed"}]')
        self.assertEqual(prompts, ['ping'])
        self.assertEqual(args.prompts, ['ping'])
        self.assertEqual(load_prompts(self.prompts), ['changed'])
        self.prompts.unlink()
        self.assertEqual(args.prompts, ['ping'])
        with self.assertRaises(StartupError):
            load_prompts(self.prompts)

    def test_help_and_bad_options_do_no_file_io(self) -> None:
        """Argument parsing exits before validation, including default Args."""
        for option, status in [('--help', 0), ('--unknown', 2)]:
            with self.subTest(option=option), \
                    patch.object(Path, 'read_text',
                                 side_effect=AssertionError), \
                    patch.object(Path, 'mkdir', side_effect=AssertionError), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parse_args(self.arguments() + [option])
                self.assertEqual(raised.exception.code, status)
        self.assertFalse(self.output.parent.exists())

    def test_save_rechecks_a_changed_destination(self) -> None:
        """Successful preflight does not certify a later filesystem state."""
        parse_args(self.arguments())
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_text('previous')
        with self.assertRaisesRegex(OSError, 'not a regular file'):
            save_results(self.output, [])
        self.assertEqual(marker.read_text(), 'previous')
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])
