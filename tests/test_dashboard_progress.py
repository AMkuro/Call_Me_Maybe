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
        self.vocab = {1: b',"', 2: b"\x1b[2J", 3: b"\xe6", 4: b"\xe6\x9d\xb1"}

    def test_token_text_is_unambiguous(self) -> None:
        """Quotes, controls and partial UTF-8 are visible as such."""
        with patch.object(Model, "decode", return_value="<|im_end|>"):
            texts = [
                display._token_text(token, self.model, self.vocab)
                for token in range(1, 6)
            ]
        self.assertEqual(
            texts, ["',\"'", "'\\x1b[2J'", "b'\\xe6'", "'東'", "'<|im_end|>'"]
        )


class DashboardTests(unittest.TestCase):
    """Progress and the summary left after the run."""

    def test_progress_renders_safely_within_width(self) -> None:
        """Values remain visible and escaped."""
        frame = render_progress(
            {
                "input": "1/1",
                "stage": "generate_call_json",
                "result": "ok",
                "value": "東京\x1b[2J",
                "value_tokens": "3",
            },
            width=30,
            height=40,
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
                dashboard.prompt_started(1, 'first {"a": 1}')
                dashboard.function_selected("fn_a")
                dashboard.value_token(b"'a'", 1, token=1, mode="forced")
                dashboard.prompt_done("fn_a")
                dashboard.prompt_started(2, "second")
                dashboard.prompt_failed("parameter 'x': limit")
        text = output.getvalue()
        self.assertRegex(
            text, re.compile(r"^\[1/2\] done +fn_a +\d+\.\d s$", re.MULTILINE)
        )
        self.assertRegex(
            text,
            re.compile(
                r"^\[2/2\] failed +parameter 'x': limit +\d+\.\d s$",
                re.MULTILINE,
            ),
        )
        self.assertRegex(
            text,
            re.compile(
                r"^2 prompts: 1 done, 1 failed, \d+\.\d s$", re.MULTILINE
            ),
        )
        self.assertNotIn('{"a"', text)
        self.assertNotIn("'a'", text)

    def test_prompt_events_own_timing_and_counts(self) -> None:
        """Start resets rows and the timer; done/failed only count once."""
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            dashboard = Dashboard(2, object.__new__(Model), {})
        with patch(
            "src.display.time.monotonic",
            side_effect=[
                10.0,
                12.5,
                20.0,
                21.0,
            ],
        ):
            dashboard.prompt_started(1, "p1")
            dashboard.function_selected("fn_a")
            dashboard.prompt_done("fn_a")
            dashboard.prompt_started(2, "p2")
            self.assertEqual(
                dashboard.fields, {"input": "2/2", "prompt": "p2"}
            )
            dashboard.prompt_failed("bad")
        self.assertEqual(dashboard.counts, {"done": 1, "failed": 1})
        self.assertIn("[1/2] done    fn_a  2.5 s", stream.getvalue())
        self.assertIn("[2/2] failed  bad  1.0 s", stream.getvalue())

    def test_draw_skips_redirected_stream_before_rendering(self) -> None:
        """All redraw requests share the non-interactive early return."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            dashboard = Dashboard(1, object.__new__(Model), {})
        with patch(
            "src.display.time.monotonic",
            side_effect=AssertionError("must not render"),
        ):
            dashboard.draw("result")
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(dashboard.last_frame, 0.0)

    def test_frame_interval_and_forced_outcomes(self) -> None:
        """Token frames throttle at 50 ms; selections and outcomes force."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            dashboard = Dashboard(2, object.__new__(Model), {1: b"1", 2: b"2"})
        dashboard.interactive = True
        with patch("src.display.time.monotonic", return_value=10.0) as clock:
            dashboard.prompt_started(1, "first")
            output.seek(0)
            output.truncate()
            clock.return_value = 10.01
            dashboard.value_token(b"1", 1, token=1, mode="forced")
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(dashboard.fields["token"], "1 '1' (forced)")
            clock.return_value = 10.049
            dashboard.value_token(b"12", 2, token=2, mode="greedy")
            self.assertEqual(output.getvalue(), "")
            clock.return_value = 10.05
            dashboard.value_token(b"123", 3, token=3, mode="greedy")
            self.assertEqual(output.getvalue().count(display._CLEAR), 1)
            clock.return_value = 10.06
            dashboard.function_selected("f")
            clock.return_value = 10.07
            dashboard.prompt_done("f")
            clock.return_value = 10.08
            dashboard.prompt_started(2, "second")
            clock.return_value = 10.09
            dashboard.prompt_failed("bad")
        self.assertEqual(output.getvalue().count(display._CLEAR), 4)
        self.assertEqual(dashboard.counts, {"done": 1, "failed": 1})
        self.assertEqual(dashboard.last_frame, 10.09)

    def test_warning_frame_contains_consistent_token_fields(self) -> None:
        """A warning frame includes the same step's value, count and token."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            dashboard = Dashboard(1, object.__new__(Model), {1: b"a", 2: b"b"})
        dashboard.interactive = True
        with (
            patch("src.display.time.monotonic", return_value=10.0),
            patch.object(Model, "decode", return_value="a"),
        ):
            dashboard.name_token([1], mode="greedy", raw_top=2)
        text = output.getvalue()
        self.assertEqual(text.count(display._CLEAR), 1)
        self.assertIn("Value: a", text)
        self.assertIn("Value tokens: 1", text)
        self.assertIn("Token: 1 'a' (greedy)", text)
        self.assertIn("Warning: name_selection_constrained: step=1", text)

    def test_flush_failure_preserves_counts_and_stops_writes(self) -> None:
        """A failed diagnostic flush leaves prompt completion bookkeeping."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            dashboard = Dashboard(2, object.__new__(Model), {})
        with (
            patch.object(output, "write", wraps=output.write) as writes,
            patch.object(output, "flush", side_effect=OSError("closed")),
        ):
            with dashboard:
                dashboard.prompt_started(1, "first")
                dashboard.prompt_done("f")
                dashboard.prompt_started(2, "second")
                dashboard.prompt_failed("bad")
        self.assertFalse(dashboard.available)
        self.assertEqual(dashboard.counts, {"done": 1, "failed": 1})
        self.assertEqual(writes.call_count, 1)

    def test_stage_change_clears_token_rows_without_parameters(self) -> None:
        """A function with no parameters must not keep the name's count."""
        with contextlib.redirect_stderr(io.StringIO()):
            dashboard = Dashboard(1, object.__new__(Model), {})
        with patch.object(Model, "decode", return_value="fn"):
            dashboard.name_token([1, 2], mode="forced", raw_top=None)
        dashboard.stage("generate_call_json")
        for row in ("value", "value_tokens", "token"):
            self.assertNotIn(row, dashboard.fields)
        self.assertEqual(dashboard.fields["stage"], "generate_call_json")
        dashboard.value_token(b"1.0}", 2, token=3, mode="greedy")
        dashboard.stage("validate_call")
        self.assertEqual(dashboard.fields["value"], "1.0}")


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
        with patch.object(Model, "decode", return_value="first") as decode:
            dashboard.name_token(ids, mode="forced", raw_top=None)
            # Calls: the name, then its last token for the Token row.
            self.assertIs(decode.call_args_list[0].args[0], ids)
            self.assertEqual(ids, [1])
            self.assertFalse(
                any(value is ids for value in vars(dashboard).values())
            )
            ids.append(2)
            self.assertEqual(dashboard.fields["value"], "first")
            self.assertEqual(dashboard.fields["token"], "1 'first' (forced)")
            dashboard.name_token((1, 2), mode="greedy", raw_top=None)
            self.assertIsInstance(decode.call_args_list[2].args[0], list)
            self.assertEqual(decode.call_args_list[2].args[0], [1, 2])

    def test_atomic_token_fields_and_stage_reset(self) -> None:
        """Each draw sees the ID, mode, count and value from the same step."""
        with contextlib.redirect_stderr(io.StringIO()):
            dashboard = Dashboard(
                1, object.__new__(Model), {42: b"\xe6", 43: b"\x9d\xb1"}
            )
        dashboard.interactive = True
        frames: list[dict[display.Row, str]] = []
        with patch.object(
            dashboard,
            "draw",
            side_effect=(lambda _: frames.append(dashboard.fields.copy())),
        ):
            dashboard.value_token(b"\xe6", 1, token=42, mode="forced")
            dashboard.value_token(b"\xe6\x9d\xb1", 2, token=43, mode="greedy")
            dashboard.stage("generate_call_json")
            dashboard.argument_started("x", "string")
        self.assertEqual(frames[0]["token"], "42 b'\\xe6' (forced)")
        self.assertEqual(frames[0]["value_tokens"], "1")
        self.assertEqual(frames[1]["token"], "43 b'\\x9d\\xb1' (greedy)")
        self.assertEqual(frames[1]["value"], "東")
        self.assertNotIn("token", frames[2])
        self.assertEqual(frames[3]["value_tokens"], "0")
        self.assertEqual(frames[3]["token"], "")
        self.assertEqual(frames[3]["parameter"], "x (string)")
        self.assertIn(
            "Token: 43 b'\\x9d\\xb1' (greedy)",
            render_progress(frames[1], 80, 30),
        )
