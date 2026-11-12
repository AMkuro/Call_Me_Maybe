"""Check token progress and the end summary (ADR-0021), no model."""

import contextlib
import io
import re
import unittest
from unittest.mock import patch

from src import display
from src.display import Dashboard, render_progress
from src.model_port import Model


class CandidateTextTests(unittest.TestCase):
    """Token text is still needed for constrained-name warnings."""

    def setUp(self) -> None:
        """A run whose model decode is used only for tokens outside vocab."""
        self.model = object.__new__(Model)
        self.vocab = {1: b',"', 2: b'\x1b[2J', 3: b'\xe6',
                      4: b'\xe6\x9d\xb1'}

    def test_token_text_is_unambiguous(self) -> None:
        """Quotes, controls and partial UTF-8 are visible as such."""
        with patch.object(Model, "decode", return_value="<|im_end|>"):
            texts = [display._token_text(token, self.model, self.vocab)
                     for token in range(1, 6)]
        self.assertEqual(texts, ["',\"'", "'\\x1b[2J'", "b'\\xe6'", "'東'",
                                 "'<|im_end|>'"])


class DashboardTests(unittest.TestCase):
    """Progress and the summary left after the run."""

    def test_progress_renders_safely_within_width(self) -> None:
        """Values remain visible and escaped."""
        frame = render_progress(
            {"input": "1/1", "stage": "generate_parameters", "result": "ok",
             "value": "東京\x1b[2J", "value_tokens": "3"},
            width=30, height=40,
        )
        self.assertNotIn("\x1b", frame)
        self.assertIn("Value: 東京", frame)
        self.assertIn("Value tokens: 3", frame)
        self.assertIn("Result: ok", frame)

    def test_summary_has_status_and_time_only(self) -> None:
        """Done/failed and seconds per prompt, then a total line."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            with Dashboard(2, object.__new__(Model), {}) as dashboard:
                report = dashboard.report
                report("input", "1/2")
                report("function", "fn_a")
                report("value", "a")
                report("result", 'fn_a {"a": 1}')
                report("input", "2/2")
                report("error", "parameter 'x': limit")
        text = output.getvalue()
        self.assertRegex(text, re.compile(r"^\[1/2\] done +fn_a +\d+\.\d s$",
                                          re.M))
        self.assertRegex(text, re.compile(
            r"^\[2/2\] failed +parameter 'x': limit +\d+\.\d s$", re.M))
        self.assertRegex(text, re.compile(
            r"^2 prompts: 1 done, 1 failed, \d+\.\d s$", re.M))
        self.assertNotIn('{"a"', text)
        self.assertNotIn("'a'", text)


class BorrowedProgressTests(unittest.TestCase):
    """Synchronous progress borrows IDs and keeps only rendered strings."""

    def test_name_sequence_is_unchanged_not_retained_and_only_list_when_needed(
        self,
    ) -> None:
        """Lists pass through; only other sequences need a decode copy."""
        model = object.__new__(Model)
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            dashboard = Dashboard(1, model, {})
        ids = [1]
        with patch.object(Model, 'decode', return_value='first') as decode:
            dashboard.name_token(ids, mode='forced', raw_top=None)
            self.assertIs(decode.call_args.args[0], ids)
            self.assertEqual(ids, [1])
            self.assertFalse(any(value is ids
                                 for value in vars(dashboard).values()))
            ids.append(2)
            self.assertEqual(dashboard.fields['value'], 'first')
            self.assertEqual(dashboard.fields['token'], '1 (forced)')
            dashboard.name_token((1, 2), mode='greedy', raw_top=None)
            self.assertIsInstance(decode.call_args.args[0], list)
            self.assertEqual(decode.call_args.args[0], [1, 2])

    def test_atomic_token_fields_and_stage_reset(self) -> None:
        """Each draw sees the ID, mode, count and value from the same step."""
        with contextlib.redirect_stderr(io.StringIO()):
            dashboard = Dashboard(1, object.__new__(Model), {})
        dashboard.interactive = True
        frames: list[dict[str, str]] = []
        with patch.object(dashboard, 'draw', side_effect=(
            lambda _: frames.append(dashboard.fields.copy())
        )):
            dashboard.value_token(b'\xe6', 1, token=42, mode='forced')
            dashboard.value_token(b'\xe6\x9d\xb1', 2, token=43, mode='greedy')
            dashboard.stage('generate_parameters')
            dashboard.parameter_started('x', 'string')
        self.assertEqual(frames[0]['token'], '42 (forced)')
        self.assertEqual(frames[0]['value_tokens'], '1')
        self.assertEqual(frames[1]['token'], '43 (greedy)')
        self.assertEqual(frames[1]['value'], '東')
        self.assertEqual(frames[2]['token'], '')
        self.assertEqual(frames[3]['value_tokens'], '0')
        self.assertEqual(frames[3]['token'], '')
        self.assertEqual(frames[3]['parameter'], 'x (string)')
        self.assertIn('Token: 43 (greedy)',
                      render_progress(frames[1], 80, 30))
