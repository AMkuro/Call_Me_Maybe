"""Verify token observations and terminal output without model inference."""

import contextlib
import io
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
        """Raw preference and forced steps do not change selection."""
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
        self.assertEqual(first.raw_top, 0)
        self.assertEqual(first.token, 2)
        self.assertTrue(second.forced)
        self.assertIsNone(second.raw_top)
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
            run = pipeline.Run(model, {}, {fn.name: fn}, trie)
            plain = pipeline.select_name('request', run)
            events: list[tuple[str, str]] = []
            observed = pipeline.select_name('request', pipeline.Run(
                model, {}, {fn.name: fn}, trie,
                lambda label, text: events.append((label, text)),
            ))
        self.assertEqual(observed, plain)
        self.assertTrue(any('unavailable' in text for _, text in events))

    def test_name_warning_and_progress_without_ranking(self) -> None:
        """Warn only for a masked raw preference; keep token progress."""
        first = definition().model_copy(update={'name': 'fn_a'})
        second = definition().model_copy(update={'name': 'fn_b'})
        functions = {first.name: first, second.name: second}
        trie = build_name_trie(list(functions),
                               lambda text: [1] if text.startswith('fn_a')
                               else [2])
        model = object.__new__(Model)
        for scores, warned in (([3.0, 1.0, 2.0], True),
                               ([0.0, 1.0, 2.0], False)):
            with self.subTest(scores=scores):
                events: list[tuple[str, str]] = []
                with patch.object(Model, 'encode', return_value=[9]), \
                        patch.object(Model, 'decode', return_value='b'), \
                        patch.object(Model, 'logits', return_value=scores):
                    result = pipeline.select_name('request', pipeline.Run(
                        model, {0: b'x', 1: b'a', 2: b'b'}, functions, trie,
                        lambda label, text: events.append((label, text)),
                    ))
                self.assertEqual(result, second)
                self.assertIn(('value', 'b'), events)
                self.assertIn(('function', 'fn_b'), events)
                self.assertEqual(any(label == 'warning'
                                     for label, _ in events), warned)
                self.assertFalse(any(label == 'candidates'
                                     for label, _ in events))

    def test_complete_call_and_value_progress_with_observation(self) -> None:
        """Real masks and validation produce the same JSON in both modes."""
        fn = definition(a='number').model_copy(update={'name': 'f'})
        trie = build_name_trie(['f'], lambda text: [7])
        model = object.__new__(Model)
        vocab = {1: b'1', 2: b'2', 3: b'}', 7: b'f'}
        events: list[tuple[str, str]] = []
        results = []
        for observe in (False, True):
            with patch.object(Model, 'encode', return_value=[]), \
                    patch.object(Model, 'decode', side_effect=lambda ids:
                                 b''.join(vocab[i] for i in ids).decode()), \
                    patch.object(Model, 'logits', side_effect=[
                        [0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                    ]):
                results.append(pipeline.process_prompt(
                    'request', pipeline.Run(
                        model, vocab, {'f': fn}, trie,
                        (lambda label, text: events.append((label, text)))
                        if observe else None,
                    ),
                ).model_dump())
        self.assertEqual(results, [{
            'prompt': 'request', 'name': 'f', 'parameters': {'a': 1},
        }] * 2)
        self.assertIn(('value', '1'), events)
        self.assertIn(('value', '1}'), events)
        self.assertIn(('value_tokens', '1'), events)
        self.assertIn(('value_tokens', '2'), events)


class TerminalTests(unittest.TestCase):
    """Check real stderr streams and an actual pseudo-terminal."""

    def test_redirected_output_has_no_control_sequences(self) -> None:
        """Files receive readable result and warning lines, not ANSI frames."""
        from src.display import Dashboard

        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            with Dashboard(2) as dashboard:
                report = dashboard.report
                report('input', '1/2')
                report('prompt', 'request\x1b[2J')
                report('function', 'store_text')
                report('value', 'hello')
                report('value_tokens', '1')
                report('warning', 'name_selection_constrained: step=1')
                report('result', 'store_text {"text":"hello"}')
        text = output.getvalue()
        self.assertNotIn('\x1b', text)
        self.assertIn('store_text', text)
        self.assertIn('name_selection_constrained: step=1', text)
        self.assertIn('[1/2]', text)

    def test_pseudo_terminal_restores_cursor_after_exception(self) -> None:
        """The live view is cleaned up even when a prompt fails."""
        from src.display import Dashboard

        master, slave = os.openpty()
        try:
            with os.fdopen(slave, 'w', encoding='utf-8') as output:
                with contextlib.redirect_stderr(output):
                    with patch.dict(os.environ, {'TERM': 'xterm'}):
                        with self.assertRaisesRegex(RuntimeError, 'test'):
                            with Dashboard(1) as dashboard:
                                dashboard.report('input', '1/1')
                                dashboard.report('function', 'store_number')
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
            self.assertIn('1 prompt: 0 done, 0 failed, 1 not finished', text)
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
