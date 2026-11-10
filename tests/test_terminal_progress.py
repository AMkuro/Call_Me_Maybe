"""Verify token observations and terminal output without model inference."""

import contextlib
import io
import math
import os
import unittest
from unittest.mock import patch

from src import decoder
from src import pipeline
from src.errors import PromptError
from src.model_port import Model
from src.name_trie import build_name_trie
from test_application import definition


class ObservationTests(unittest.TestCase):
    """Observing scores must never change greedy selection or model calls."""

    def test_token_diagnostics_preserve_decoding(self) -> None:
        """Show raw preference, allowed probability mass and forced steps."""
        observed: list[decoder.TokenChoice] = []
        calls: list[list[int]] = []

        def scores(ids: list[int]) -> list[float]:
            """Return one vocabulary distribution, including an invalid max."""
            calls.append(ids)
            return [3.0, 1.0, 2.0]

        def allowed(ids: list[int]) -> list[int] | None:
            """Use an actual branch followed by a forced token."""
            return [[1, 2], [1], None][len(ids)]

        result = decoder.generate(scores, [9], allowed, 2,
                                  on_token=observed.append)
        self.assertEqual(result, [2, 1])
        self.assertEqual(calls, [[9]])

        first, second = observed
        self.assertTrue(first.constrained)
        self.assertEqual(first.raw_top, 0)
        self.assertEqual(first.token, 2)
        expected = (math.exp(1) + math.exp(2)) / sum(
            math.exp(score) for score in [3, 1, 2]
        )
        assert first.allowed_mass is not None
        assert first.margin is not None
        self.assertAlmostEqual(first.allowed_mass, expected)
        self.assertAlmostEqual(first.margin, 1.0)
        self.assertTrue(second.forced)
        self.assertIsNone(second.allowed_mass)
        self.assertIsNone(second.margin)
        self.assertEqual(first.tokens, [2])

        calls.clear()
        self.assertEqual(decoder.generate(scores, [9], allowed, 2), result)
        self.assertEqual(calls, [[9]])

    def test_diagnostic_decode_failure_preserves_name_selection(self) -> None:
        """A display-text failure cannot change a valid selected function."""
        fn = definition()
        trie = build_name_trie([fn.name], lambda text: [1, 2])
        model = object.__new__(Model)
        with patch.object(Model, 'encode', return_value=[9]), \
                patch.object(Model, 'decode',
                             side_effect=PromptError('diagnostic decode')):
            plain = pipeline.select_name('request', [fn], model, trie)
            events: list[tuple[str, str]] = []
            observed = pipeline.select_name(
                'request', [fn], model, trie,
                lambda label, text: events.append((label, text)),
            )
        self.assertEqual(observed, plain)
        self.assertTrue(any('unavailable' in text for _, text in events))


class TerminalTests(unittest.TestCase):
    """Check real stderr streams and an actual pseudo-terminal."""

    def test_redirected_output_has_no_control_sequences(self) -> None:
        """Files receive readable result and warning lines, not ANSI frames."""
        from src.display import terminal_progress

        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            with terminal_progress(2) as report:
                report('input', '1/2')
                report('prompt', 'request\x1b[2J')
                report('function', 'store_text')
                report('choice', 'step=1 id=42')
                report('warning', 'allowed_mass=0.04')
                report('result', 'store_text {"text":"hello"}')
        text = output.getvalue()
        self.assertNotIn('\x1b', text)
        self.assertIn('store_text', text)
        self.assertIn('allowed_mass=0.04', text)
        self.assertIn('[1/2]', text)

    def test_pseudo_terminal_restores_cursor_after_exception(self) -> None:
        """The live view is cleaned up even when a prompt fails."""
        from src.display import terminal_progress

        master, slave = os.openpty()
        try:
            with os.fdopen(slave, 'w', encoding='utf-8') as output:
                with contextlib.redirect_stderr(output):
                    with patch.dict(os.environ, {'TERM': 'xterm'}):
                        with self.assertRaisesRegex(RuntimeError, 'test'):
                            with terminal_progress(1) as report:
                                report('input', '1/1')
                                report('function', 'store_number')
                                raise RuntimeError('test')
            chunks = []
            while True:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                chunks.append(chunk)
            text = b''.join(chunks).decode()
            self.assertIn('\x1b[?25l', text)
            self.assertIn('\x1b[?25h', text)
            self.assertIn('store_number', text)
        finally:
            os.close(master)

    def test_frame_escapes_input_controls_and_fits_small_width(self) -> None:
        """Prompt text cannot inject terminal commands or break the layout."""
        from src.display import render_progress

        frame = render_progress({'input': '1/1', 'prompt': 'a\x1b[2J\nb'},
                                width=35, height=20)
        self.assertNotIn('\x1b', frame)
        self.assertIn('\\u001b', frame)
        self.assertTrue(all(len(line) <= 35 for line in frame.splitlines()))
