"""Observe committed JSON without changing inference."""

import contextlib
import io
import unittest
from unittest.mock import patch

from src import generation
from src.display import Dashboard, render_progress
from src.model_port import Model
from src.values import normalize_call_json
from test_application import definition


class PreviewTests(unittest.TestCase):
    """Exercise observers and rendering with deterministic model bytes."""

    def test_prefix_keeps_previous_values_and_split_utf8(self) -> None:
        """Starting a parameter cannot erase earlier JSON bytes."""
        with contextlib.redirect_stderr(io.StringIO()):
            dashboard = Dashboard(1, object.__new__(Model), {})
        dashboard.stage('generate_call_json')
        for data in (b'{"a": 2,"b": "', b'\xe6', b'\x9d\xb1'):
            dashboard.output_token(data, token=1, mode='forced')
        dashboard.parameter_started('c', 'string')
        self.assertEqual(dashboard.fields['json'], '{"a": 2,"b": "東')
        self.assertEqual(dashboard.fields['tokens'], '3')
        frame = render_progress(dashboard.fields, 80, 24)
        self.assertIn('2', frame)
        self.assertIn('東▌', frame)
        self.assertNotIn('}', frame)
        self.assertNotIn('\ufffd', frame)

    def test_fixed_tokens_and_no_parameters_match_unobserved_run(self) -> None:
        """Observe fixed events without adding forward calls."""
        model = object.__new__(Model)
        vocab = {i: bytes([i]) for i in range(128)}
        fn = definition().model_copy(update={'name': 'f'})
        results, contexts = [], []
        for enabled in (False, True):
            with contextlib.redirect_stderr(io.StringIO()):
                dashboard = Dashboard(1, model, vocab)
            dashboard.stage('generate_call_json')
            with patch.object(Model, 'encode', side_effect=lambda s:
                              list(s.encode())), \
                    patch.object(Model, 'decode', side_effect=lambda ids:
                                 bytes(ids).decode()), \
                    patch.object(Model, 'logits',
                                 return_value=[0.] * 128) as llm:
                results.append(generation.generate_call_json(
                    'request', fn, model=model, vocab=vocab,
                    progress=dashboard if enabled else None,
                ))
                contexts.append(llm.call_args_list)
            if enabled:
                self.assertEqual(dashboard.fields['json'], results[-1][0])
                self.assertEqual(int(dashboard.fields['tokens']),
                                 len(results[-1][0]))
        self.assertEqual(results[0], results[1])
        self.assertEqual(contexts[0], contexts[1])

    def test_small_viewport_retains_cursor_and_marks_omission(self) -> None:
        """Long strings must wrap and expose the current position on resize."""
        for width, height in ((18, 8), (40, 15), (80, 24), (1, 2)):
            frame = render_progress(
                {'json': '{"s":"' + 'x' * 600, 'stage': 'generate_call_json'},
                width, height,
            )
            self.assertIn('▌', frame)
            self.assertLessEqual(len(frame.splitlines()), max(height - 1, 1))
            self.assertTrue(all(len(line) <= width
                                for line in frame.splitlines()))

    def test_interruption_leaves_summary_without_json(self) -> None:
        """An interrupted prefix stays in the live screen only."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            dashboard = Dashboard(1, object.__new__(Model), {})
        dashboard.interactive = True
        with self.assertRaises(KeyboardInterrupt), dashboard:
            dashboard.prompt_started(1, 'p')
            dashboard.output_token(b'{"a": 2', token=1, mode='greedy')
            raise KeyboardInterrupt
        retained = output.getvalue().split('\x1b[?1049l')[-1]
        self.assertNotIn('2▌', retained)
        self.assertNotIn('JSON', retained)
        self.assertIn('not finished', retained)

    def test_wide_character_boundary_keeps_cursor(self) -> None:
        """Place the cursor according to terminal-cell width."""
        frame = render_progress({'json': '"東東東東東'}, 11, 8)
        self.assertIn('▌', frame)

    def test_validated_saved_text_is_explicitly_separate(self) -> None:
        """Saved .0 must never be counted as a model token."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            dashboard = Dashboard(1, object.__new__(Model), {})
        dashboard.interactive = True
        with dashboard:
            dashboard.prompt_started(1, 'p')
            dashboard.output_token(b'{"a": 2}', token=1, mode='greedy')
            dashboard.prompt_done('f', '{"a": 2.0}')
        self.assertEqual(dashboard.fields['json'], '{"a": 2}')
        self.assertEqual(dashboard.fields['tokens'], '1')
        self.assertIn('SAVED JSON', output.getvalue())
        self.assertIn('2.0', output.getvalue())
        retained = output.getvalue().split('\x1b[?1049l')[-1]
        self.assertIn('[1/1] done', retained)
        self.assertNotIn('JSON', retained)
        self.assertNotIn('2.0', retained)

    def test_fixed_text_clears_active_parameter(self) -> None:
        """Structural tokens must clear the active argument."""
        with contextlib.redirect_stderr(io.StringIO()):
            dashboard = Dashboard(1, object.__new__(Model), {})
        dashboard.parameter_started('a', 'number')
        dashboard.output_token(b'}', token=1, mode='forced', fixed=True)
        self.assertNotIn('parameter', dashboard.fields)

    def test_value_bytes_before_normalization(self) -> None:
        """Keep all value types and Unicode byte-exact."""
        fn = definition(a='number', s='string', flag='boolean')
        raw = ('{"prompt":"p","name":"dynamic","parameters":'
               '{"a": 2,"s": "東,\\\"x",'
               '"flag": true}}').encode()
        model = object.__new__(Model)
        vocab = {i: bytes([i]) for i in range(256)}
        with contextlib.redirect_stderr(io.StringIO()):
            dashboard = Dashboard(1, model, vocab)
        with patch.object(generation, 'build_parameters_prompt',
                          return_value='P'), \
                patch.object(Model, 'encode', side_effect=lambda s:
                             list(s.encode())), \
                patch.object(Model, 'decode', side_effect=lambda ids:
                             bytes(ids).decode()), \
                patch.object(Model, 'logits', side_effect=lambda ids:
                             [1. if i == raw[len(ids) - 1] else 0.
                              for i in range(256)]):
            text, number_slot_slices = generation.generate_call_json(
                'p', fn, model=model, vocab=vocab, progress=dashboard,
            )
        self.assertEqual(dashboard.fields['json'].encode(), raw)
        self.assertEqual(int(dashboard.fields['tokens']), len(raw))
        self.assertEqual(text, raw.decode())
        self.assertEqual(normalize_call_json(text, number_slot_slices),
                         raw.decode().replace('2,', '2.0,'))

    def test_resize_reads_terminal_size_at_each_frame(self) -> None:
        """A resized terminal changes wrapping without resetting the prefix."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            dashboard = Dashboard(1, object.__new__(Model), {})
        dashboard.interactive = True
        dashboard.output_token(b'{"s":"' + b'x' * 80,
                               token=1, mode='greedy')
        with patch.object(output, 'fileno', return_value=2), \
                patch('src.display.os.get_terminal_size',
                      side_effect=[(30, 12), (60, 20)]):
            dashboard.draw('result')
            first = output.getvalue().split('\x1b[H')[-1]
            dashboard.draw('result')
            second = output.getvalue().split('\x1b[H')[-1]
        self.assertNotEqual(first, second)
        self.assertIn('▌', first)
        self.assertIn('▌', second)

    def test_metadata_controls_are_escaped_in_json_view(self) -> None:
        """Function and parameter names cannot issue terminal commands."""
        frame = render_progress({'json': '{', 'function': 'f\x1b[2J',
                                 'parameter': 'x\n\x1b[H'}, 80, 24)
        self.assertNotIn('\x1b', frame)
        self.assertIn('\\u001b', frame)

    def test_small_screen_keeps_error_and_json_label(self) -> None:
        """Decoration must yield to errors and the meaning of the prefix."""
        frame = render_progress({'json': '{"a": 2', 'error': 'failed',
                                 'parameter': 'a (number)',
                                 'token': '2 (greedy)'}, 40, 8)
        self.assertIn('Error: failed', frame)
        self.assertIn('JSON OUTPUT', frame)
        self.assertIn('greedy', frame)

    def test_exact_cell_width_does_not_replace_cursor(self) -> None:
        """Content fitting exactly must survive final line clipping."""
        from src.display import _clip

        self.assertEqual(_clip('東▌', 3), '東▌')
        self.assertEqual(_clip('abc', 3), 'abc')
        self.assertEqual(_clip('abcd', 3), 'ab…')

    def test_name_failure_leaves_reason_without_live_state(self) -> None:
        """Keep the exception reason but omit live state after exit."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            dashboard = Dashboard(1, object.__new__(Model), {})
        dashboard.interactive = True
        with self.assertRaises(RuntimeError), dashboard:
            dashboard.prompt_started(1, 'p')
            dashboard.stage('select_function')
            dashboard.fields['value'] = 'partial_name'
            raise RuntimeError('model failure')
        retained = output.getvalue().split('\x1b[?1049l')[-1]
        self.assertNotIn('partial_name', retained)
        self.assertIn('model failure', retained)

    def test_failure_leaves_outcome_time_and_path_without_json(self) -> None:
        """A failed prompt leaves its reason and output path, not a frame."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            dashboard = Dashboard(1, object.__new__(Model), {})
        dashboard.interactive = True
        dashboard.output_path = '/tmp/calls.json'
        with dashboard:
            dashboard.prompt_started(1, 'p')
            dashboard.output_token(b'{"a": 2', token=1, mode='forced')
            dashboard.prompt_failed('token limit')
        retained = output.getvalue().split('\x1b[?1049l')[-1]
        self.assertIn('[1/1] failed', retained)
        self.assertIn('token limit', retained)
        self.assertIn('0 done, 1 failed', retained)
        self.assertIn('Output file: /tmp/calls.json', retained)
        self.assertRegex(retained, r'\d+\.\d s')
        self.assertNotIn('JSON', retained)
        self.assertNotIn('{"a"', retained)
