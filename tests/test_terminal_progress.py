"""Verify token observations and terminal output without model inference."""

import contextlib
import io
import os
import unittest
from collections.abc import Sequence
from unittest.mock import patch

from src import pipeline
from src.decoder import SelectionMode
from src.display import Dashboard
from src.errors import PromptError
from src.model_port import Model
from src.name_trie import build_name_trie
from test_application import definition


class ObservationTests(unittest.TestCase):
    """Observing scores must never change greedy selection or model calls."""

    def test_diagnostic_decode_failure_preserves_name_selection(self) -> None:
        """A display-text failure cannot change a valid selected function."""
        fn = definition()
        trie = build_name_trie([fn.name], lambda text: [1, 2])
        model = object.__new__(Model)
        with patch.object(Model, 'encode', return_value=[9]), \
                patch.object(Model, 'decode',
                             side_effect=PromptError('diagnostic decode')):
            plain = pipeline.select_name('request', model=model,
                                         functions={fn.name: fn}, trie=trie)
            with contextlib.redirect_stderr(io.StringIO()):
                dashboard = Dashboard(1, model, {})
                observed = pipeline.select_name(
                    'request', model=model, functions={fn.name: fn},
                    trie=trie, progress=dashboard,
                )
        self.assertEqual(observed, plain)
        self.assertIn('unavailable', dashboard.fields['value'])
        self.assertEqual(dashboard.fields['token'], '2 (forced)')

    def test_name_warning_and_progress_without_ranking(self) -> None:
        """Warn only for a masked raw preference; keep ID/mode progress."""
        first = definition().model_copy(update={'name': 'fn_a'})
        second = definition().model_copy(update={'name': 'fn_b'})
        functions = {first.name: first, second.name: second}
        trie = build_name_trie(list(functions),
                               lambda text: [1] if text.startswith('fn_a')
                               else [2])
        model = object.__new__(Model)
        for scores, warned in (([3.0, 1.0, 2.0], True),
                               ([0.0, 1.0, 2.0], False)):
            output = io.StringIO()
            with self.subTest(scores=scores), \
                    contextlib.redirect_stderr(output), \
                    patch.object(Model, 'encode', return_value=[9]), \
                    patch.object(Model, 'decode', return_value='b'), \
                    patch.object(Model, 'logits', return_value=scores) as llm:
                dashboard = Dashboard(1, model, {0: b'x', 1: b'a', 2: b'b'})
                result = pipeline.select_name(
                    'request', model=model, functions=functions,
                    trie=trie, progress=dashboard,
                )
            self.assertEqual(result, second)
            self.assertEqual(dashboard.fields['value'], 'b')
            self.assertEqual(dashboard.fields['function'], 'fn_b')
            self.assertEqual(dashboard.fields['token'], '2 (greedy)')
            self.assertEqual('warning' in dashboard.fields, warned)
            self.assertEqual('name_selection_constrained' in output.getvalue(),
                             warned)
            self.assertNotIn('candidates', dashboard.fields)
            llm.assert_called_once_with([9])

    def test_complete_call_and_value_progress_with_observation(self) -> None:
        """Real masks/validation keep JSON and model inputs in all modes."""
        fn = definition(a='number').model_copy(update={'name': 'f'})
        trie = build_name_trie(['f'], lambda text: [7])
        model = object.__new__(Model)
        vocab = {1: b'1', 2: b'2', 3: b'}', 7: b'f'}
        resource = pipeline.GenerationResources(
            model=model, vocab=vocab, functions={'f': fn}, trie=trie,
        )
        results = []
        inputs = []
        for mode in ('none', 'redirected', 'unavailable', 'broken'):
            output = io.StringIO()
            with contextlib.redirect_stderr(output), \
                    patch.object(Model, 'encode', return_value=[]), \
                    patch.object(Model, 'decode', side_effect=lambda ids:
                                 b''.join(vocab[i] for i in ids).decode()), \
                    patch.object(Model, 'logits', side_effect=[
                        [0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                    ]) as llm:
                dashboard = Dashboard(1, model, vocab)
                dashboard.available = mode != 'unavailable'
                dashboard.interactive = mode == 'broken'
                with patch.object(dashboard, 'value_token',
                                  wraps=dashboard.value_token) as values, \
                        patch.object(output, 'write',
                                     side_effect=OSError('closed')) if (
                            mode == 'broken') else contextlib.nullcontext():
                    results.append(pipeline.process_prompt(
                        'request', resource,
                        progress=None if mode == 'none' else dashboard,
                    ).model_dump())
                    inputs.append(llm.call_args_list)
                if mode == 'redirected':
                    self.assertEqual([c.args for c in values.call_args_list],
                                     [(b'1', 1), (b'1}', 2)])
                    self.assertEqual([c.kwargs for c in values.call_args_list],
                                     [dict(token=1, mode='greedy'),
                                      dict(token=3, mode='greedy')])
                    self.assertEqual(dashboard.fields['token'], '3 (greedy)')
        self.assertEqual(results, [{
            'prompt': 'request', 'name': 'f', 'parameters': {'a': 1},
        }] * 4)
        self.assertTrue(all(item == inputs[0] for item in inputs))
        self.assertEqual(len(inputs[0]), 2)


class TerminalTests(unittest.TestCase):
    """Check real stderr streams and an actual pseudo-terminal."""

    def test_redirected_output_has_no_control_sequences(self) -> None:
        """Files receive readable result and warning lines, not ANSI frames."""
        from src.display import Dashboard

        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            with Dashboard(2, object.__new__(Model), {}) as dashboard:
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

        for error in (RuntimeError('test'), KeyboardInterrupt('test')):
            master, slave = os.openpty()
            try:
                with os.fdopen(slave, 'w', encoding='utf-8') as output:
                    with contextlib.redirect_stderr(output):
                        with patch.dict(os.environ, {'TERM': 'xterm'}):
                            with self.assertRaisesRegex(type(error), 'test'):
                                with Dashboard(
                                    1, object.__new__(Model), {},
                                ) as dashboard:
                                    dashboard.report('input', '1/1')
                                    dashboard.report(
                                        'function', 'store_number',
                                    )
                                    raise error
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
                self.assertIn(
                    '1 prompt: 0 done, 0 failed, 1 not finished', text,
                )
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


class LoopObservationTests(unittest.TestCase):
    """Stage loops expose actual decisions without taking control from data."""

    def test_name_forced_then_greedy_uses_same_borrowed_list(self) -> None:
        """A leaf at the name cap succeeds; only branching calls logits."""
        from src.name_trie import NameTrie

        first = definition().model_copy(update={'name': 'a'})
        second = definition().model_copy(update={'name': 'b'})
        trie = NameTrie(children={1: NameTrie(children={
            2: NameTrie(name='a'), 3: NameTrie(name='b'),
        })})
        model = object.__new__(Model)
        with contextlib.redirect_stderr(io.StringIO()):
            dashboard = Dashboard(1, model, {1: b'x', 2: b'a', 3: b'b'})
        seen: list[tuple[int, tuple[int, ...], str]] = []
        original = dashboard.name_token

        def observe(
            tokens: Sequence[int], *, mode: SelectionMode,
            raw_top: int | None,
        ) -> None:
            """Capture copies for assertions, then use the real receiver."""
            seen.append((id(tokens), tuple(tokens), mode))
            original(tokens, mode=mode, raw_top=raw_top)

        with patch.object(Model, 'encode', return_value=[9]), \
                patch.object(Model, 'decode', return_value='b'), \
                patch.object(Model, 'logits',
                             return_value=[9.0, 0.0, 1.0, 2.0]) as llm, \
                patch.object(dashboard, 'name_token', side_effect=observe):
            selected = pipeline.select_name(
                'request', model=model, functions={'a': first, 'b': second},
                trie=trie, progress=dashboard,
            )
        self.assertIs(selected, second)
        llm.assert_called_once_with([9, 1])
        self.assertEqual([item[1:] for item in seen],
                         [((1,), 'forced'), ((1, 3), 'greedy')])
        self.assertEqual(seen[0][0], seen[1][0])
        self.assertEqual(dashboard.fields['token'], '3 (greedy)')
        self.assertIn('step=2', dashboard.fields['warning'])

    def test_value_forced_and_duplicate_bytes_keep_actual_id(self) -> None:
        """Mode and selected ID are not inferred from displayed bytes."""
        from src.skeleton import ValueSlot

        model = object.__new__(Model)
        slot = ValueSlot(name='flag', type='boolean', terminator='}')
        for vocab, expected, mode in (
            ({1: b'true}'}, 1, 'forced'),
            ({1: b'true}', 2: b'true}'}, 2, 'greedy'),
        ):
            with self.subTest(mode=mode), \
                    contextlib.redirect_stderr(io.StringIO()):
                dashboard = Dashboard(1, model, vocab)
                with patch.object(Model, 'logits',
                                  return_value=[0.0, 1.0, 2.0]) as llm:
                    result = pipeline.generate_value(
                        slot, [9], model=model, vocab=vocab,
                        progress=dashboard,
                    )
                self.assertEqual(result, [expected])
                self.assertEqual(llm.call_count, int(mode == 'greedy'))
                self.assertEqual(dashboard.fields['value'], 'true}')
                self.assertEqual(dashboard.fields['token'],
                                 f'{expected} ({mode})')
