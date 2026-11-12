"""Reject unusable atomic-output destinations before loading the model."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src import __main__ as entry
from src.errors import StartupError
from src.writer import save_results


class OutputPreflightTests(unittest.TestCase):
    """Use real input loading and destination paths with no actual model."""

    def setUp(self) -> None:
        """Build valid input files independent of the output destination."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.functions = self.root / 'functions.json'
        self.functions.write_text(json.dumps([{
            'name': 'ping', 'description': 'Check health.', 'parameters': {},
            'returns': {'type': 'boolean'},
        }]), encoding='utf-8')
        self.prompts = self.root / 'prompts.json'
        self.prompts.write_text('[{"prompt":"Check health."}]',
                                encoding='utf-8')
        self.stderr = io.StringIO()

    def arguments(self, output: Path) -> list[str]:
        """Pass actual paths through the actual argument parser."""
        return ['--functions_definition', str(self.functions),
                '--input', str(self.prompts), '--output', str(output)]

    def assert_rejected_before_model(self, output: Path) -> None:
        """A path failure must be reported instead of attempting startup."""
        with patch.object(entry, 'Model', side_effect=StartupError(
            'model startup sentinel'
        )) as model, contextlib.redirect_stderr(self.stderr):
            status = entry.main(self.arguments(output))
        self.assertEqual(status, 1)
        model.assert_not_called()
        self.assertIn(str(output), self.stderr.getvalue())
        self.assertIn('cannot prepare output', self.stderr.getvalue())

    def test_directory_destination_fails_before_model(self) -> None:
        """An existing directory cannot be an atomic JSON destination."""
        output = self.root / 'destination'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('keep', encoding='utf-8')
        self.assert_rejected_before_model(output)
        self.assertEqual(marker.read_text(), 'keep')

    def test_parent_that_is_a_file_fails_before_model(self) -> None:
        """A non-directory path component fails without expensive startup."""
        parent = self.root / 'blocked'
        parent.write_text('keep', encoding='utf-8')
        self.assert_rejected_before_model(parent / 'calls.json')
        self.assertEqual(parent.read_text(), 'keep')

    def test_unopenable_probe_preserves_existing_output(self) -> None:
        """An injected file-open failure must not truncate the last output."""
        output = self.root / 'calls.json'
        previous = b'["previous run"]\n'
        output.write_bytes(previous)
        with patch('src.writer.tempfile.NamedTemporaryFile',
                   side_effect=PermissionError('probe open denied')):
            self.assert_rejected_before_model(output)
        self.assertEqual(output.read_bytes(), previous)
        self.assertIn('probe open denied', self.stderr.getvalue())

    def test_invalid_output_path_is_a_startup_error(self) -> None:
        """A path containing a NUL fails through the CLI error boundary."""
        self.assert_rejected_before_model(self.root / 'bad\x00name.json')

    def test_probe_flush_failure_is_early_and_removes_probe(self) -> None:
        """Opening alone is insufficient when writing cannot be flushed."""
        output = self.root / 'result' / 'calls.json'
        original = tempfile.NamedTemporaryFile

        @contextlib.contextmanager
        def full_probe(**kwargs: Any) -> Iterator[Any]:
            """Keep real file creation/removal and fail at the I/O boundary."""
            with original(**kwargs) as probe:
                with patch.object(probe, 'flush',
                                  side_effect=OSError('probe disk full')):
                    yield probe

        with patch('src.writer.tempfile.NamedTemporaryFile', full_probe):
            self.assert_rejected_before_model(output)
        self.assertEqual(list(output.parent.iterdir()), [])
        self.assertIn('probe disk full', self.stderr.getvalue())

    def test_missing_parents_are_prepared_before_model(self) -> None:
        """The probe is removed and no final output is created at startup."""
        output = self.root / 'new' / 'nested' / 'calls.json'

        def startup() -> None:
            """Inspect the filesystem at the exact Model boundary."""
            self.assertTrue(output.parent.is_dir())
            self.assertEqual(list(output.parent.iterdir()), [])
            raise StartupError('model startup sentinel')

        with patch.object(entry, 'Model', side_effect=startup), \
                contextlib.redirect_stderr(self.stderr):
            self.assertEqual(entry.main(self.arguments(output)), 1)
        self.assertFalse(output.exists())
        self.assertIn('model startup sentinel', self.stderr.getvalue())

    def test_existing_output_survives_successful_preparation(self) -> None:
        """Preparation must leave existing bytes and no sibling probe."""
        output = self.root / 'result' / 'calls.json'
        output.parent.mkdir()
        previous = b'["previous run"]\n'
        output.write_bytes(previous)

        def startup() -> None:
            """Inspect preservation after the actual preflight completes."""
            self.assertEqual(output.read_bytes(), previous)
            self.assertEqual(list(output.parent.iterdir()), [output])
            raise StartupError('model startup sentinel')

        with patch.object(entry, 'Model', side_effect=startup), \
                contextlib.redirect_stderr(self.stderr):
            self.assertEqual(entry.main(self.arguments(output)), 1)
        self.assertEqual(output.read_bytes(), previous)

    def test_readonly_file_can_still_be_atomically_replaced(self) -> None:
        """Atomic saving needs parent access, not direct target-file writes."""
        output = self.root / 'calls.json'
        output.write_text('previous', encoding='utf-8')
        output.chmod(0o400)
        try:
            with patch.object(entry, 'Model', side_effect=StartupError(
                'model startup sentinel'
            )) as model, contextlib.redirect_stderr(self.stderr):
                self.assertEqual(entry.main(self.arguments(output)), 1)
            model.assert_called_once_with()
            self.assertIn('model startup sentinel', self.stderr.getvalue())
            self.assertEqual(output.read_text(), 'previous')
        finally:
            output.chmod(0o600)

    def test_atomic_save_filename_limit_passes_preflight(self) -> None:
        """Probing must not impose a stricter filename limit than saving."""
        limit = os.pathconf(self.root, 'PC_NAME_MAX')
        output = self.root / ('x' * (limit - 14))
        save_results(output, [])
        with patch.object(entry, 'Model', side_effect=StartupError(
            'model startup sentinel'
        )) as model, contextlib.redirect_stderr(self.stderr):
            self.assertEqual(entry.main(self.arguments(output)), 1)
        model.assert_called_once_with()
        self.assertEqual(output.read_text(), '[]\n')

    @unittest.skipIf(os.geteuid() == 0, 'root bypasses directory permissions')
    def test_readonly_directory_fails_before_model(self) -> None:
        """Real filesystem permissions are checked by opening a probe."""
        parent = self.root / 'readonly'
        parent.mkdir()
        parent.chmod(0o500)
        try:
            self.assert_rejected_before_model(parent / 'calls.json')
        finally:
            parent.chmod(0o700)
