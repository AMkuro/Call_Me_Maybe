"""Check token progress and the end summary (ADR-0021), no model."""

import contextlib
import io
import re
import unittest
from unittest.mock import patch

import numpy as np

from src import decoder, pipeline
from src.display import Dashboard, render_progress
from src.model_port import Model
from src.name_trie import NameTrie


class DecisionTests(unittest.TestCase):
    """Observe decisions without ranking or probability work."""

    def test_no_ranking_or_probability_calculation(self) -> None:
        """Observations do not need top-k, sorting or softmax."""
        seen: list[decoder.TokenChoice] = []
        with patch.object(np, "argpartition", side_effect=AssertionError), \
                patch.object(np, "sort", side_effect=AssertionError), \
                patch.object(np, "exp", side_effect=AssertionError):
            result = decoder.generate(
                lambda ids: [3.0, 1.0, 2.0], [],
                lambda ids: None if ids else [1, 2], 1,
                on_token=seen.append,
            )
        self.assertEqual(result, [2])
        self.assertEqual(seen[0].raw_top, 0)

    def test_fixed_scores_preserve_selection_and_calls(self) -> None:
        """Ties, masked maxima and -inf preserve greedy/forced behavior."""
        for scores, allowed, expected, raw in (
            ([3.0, 1.0, 2.0], [1, 2], 2, 0),
            ([2.0, 2.0, 0.0], [1, 0], 0, 0),
            ([-np.inf, 1.0, -np.inf], [0, 1], 1, 1),
        ):
            with self.subTest(scores=scores):
                calls: list[list[int]] = []

                def logits(ids: list[int]) -> list[float]:
                    calls.append(ids.copy())
                    return scores

                def permitted(ids: list[int]) -> list[int] | None:
                    return [allowed, [1], None][len(ids)]

                seen: list[decoder.TokenChoice] = []
                for observe in (None, seen.append):
                    calls.clear()
                    self.assertEqual(decoder.generate(
                        logits, [9], permitted, 2, on_token=observe,
                    ), [expected, 1])
                    self.assertEqual(calls, [[9]])
                self.assertEqual(seen[0].raw_top, raw)
                self.assertTrue(seen[1].forced)
                self.assertIsNone(seen[1].raw_top)


class CandidateTextTests(unittest.TestCase):
    """Token text is still needed for constrained-name warnings."""

    def setUp(self) -> None:
        """A run whose model decode is used only for tokens outside vocab."""
        self.shown: dict[str, str] = {}
        self.context = pipeline.Run(
            object.__new__(Model),
            {1: b',"', 2: b'\x1b[2J', 3: b'\xe6', 4: b'\xe6\x9d\xb1'},
            {}, NameTrie(), self.shown.__setitem__,
        )

    def test_token_text_is_unambiguous(self) -> None:
        """Quotes, controls and partial UTF-8 are visible as such."""
        with patch.object(Model, "decode", return_value="<|im_end|>"):
            texts = [pipeline._token_text(token, self.context)
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
            with Dashboard(2) as dashboard:
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
