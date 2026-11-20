"""Preserve batch persistence and terminal cleanup across module boundaries."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src import __main__ as entry, pipeline
from src.cli import Args
from src.errors import PromptError
from src.model_port import Model
from src.schema import FunctionCallRequest


class BatchLifecycleTests(unittest.TestCase):
    """Exercise actual batch decisions and the CLI's cleanup boundary."""

    def test_batch_saves_successes_and_reports_failure(self) -> None:
        """A failed prompt preserves both successes and exact saved JSON."""
        prompts = ['first', 'bad', 'last']
        first = FunctionCallRequest(prompt='first', name='f', parameters={})
        last = FunctionCallRequest(prompt='last', name='f', parameters={})
        resources = MagicMock()
        dashboard = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'calls.json'
            with patch.object(pipeline, 'process_prompt', side_effect=[
                first, PromptError('skip'), last,
            ]):
                self.assertFalse(pipeline.process_all(
                    prompts, resources, output, dashboard))
            self.assertEqual(json.loads(output.read_text()), [
                {'prompt': 'first', 'name': 'f', 'parameters': {}},
                {'prompt': 'last', 'name': 'f', 'parameters': {}},
            ])
        self.assertEqual(dashboard.prompt_started.call_count, 3)
        dashboard.prompt_failed.assert_called_once_with('skip')
        self.assertEqual(dashboard.prompt_done.call_count, 2)

    def test_end_closes_terminal_before_reporting_save_error(self) -> None:
        """A save failure leaves the context before the error is printed."""
        stderr = io.StringIO()
        resources = MagicMock(model=object.__new__(Model), vocab={})
        args = Args.model_construct(
            output=Path('/tmp/cmm-lifecycle-output.json'))
        with patch.object(entry, 'Dashboard') as dashboard, \
                patch.object(entry, 'process_all',
                             side_effect=OSError('disk full')), \
                contextlib.redirect_stderr(stderr):
            def closed(*exc_info: object) -> None:
                self.assertEqual(stderr.getvalue(), '')
            dashboard.return_value.__exit__.side_effect = closed
            self.assertEqual(entry._end(args, ['x'], resources), 1)
            dashboard.return_value.__exit__.assert_called_once()
        self.assertIn('cannot save output', stderr.getvalue())
        self.assertIn('disk full', stderr.getvalue())

    def test_end_keeps_keyboard_interrupt_after_cleanup(self) -> None:
        """User interruption restores the context and propagates unchanged."""
        resources = MagicMock(model=object.__new__(Model), vocab={})
        with patch.object(entry, 'Dashboard') as dashboard, \
                patch.object(entry, 'process_all',
                             side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                entry._end(Args.model_construct(), ['x'], resources)
            dashboard.return_value.__exit__.assert_called_once()
