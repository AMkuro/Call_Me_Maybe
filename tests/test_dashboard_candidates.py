"""Check dashboard candidates and the end summary (ADR-0021), no model."""

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


class CandidateTests(unittest.TestCase):
    """The top 10 come from logits already computed."""

    def test_top_ten_match_a_full_sort(self) -> None:
        """Raw top 10 with allowed marks; selection is unchanged."""
        scores = np.random.default_rng(0).normal(size=300)
        order = [int(t) for t in np.argsort(-scores)]
        allowed = [order[3], order[7]] + order[50:80]
        seen: list[decoder.TokenChoice] = []
        result = decoder.generate(
            lambda ids: list(scores), [], lambda ids: None if ids else allowed,
            1, on_token=seen.append,
        )
        (choice,) = seen
        self.assertEqual([t for t, _, _ in choice.top], order[:10])
        self.assertEqual([ok for _, _, ok in choice.top],
                         [t in allowed for t in order[:10]])
        self.assertEqual(result, [order[3]])
        self.assertAlmostEqual(choice.logit or 0.0, scores[order[3]])
        self.assertEqual(decoder.generate(
            lambda ids: list(scores), [], lambda ids: None if ids else allowed,
            1,
        ), result)

    def test_forced_tokens_have_no_candidates_or_inference(self) -> None:
        """A single allowed token never calls the model."""
        calls: list[list[int]] = []

        def logits(ids: list[int]) -> list[float]:
            calls.append(ids)
            return [0.0, 1.0, 2.0]

        seen: list[decoder.TokenChoice] = []
        decoder.generate(logits, [], lambda ids: [[0], [1, 2], None][len(ids)],
                         2, on_token=seen.append)
        self.assertEqual(len(calls), 1)
        self.assertEqual((seen[0].forced, seen[0].top), (True, []))

    def test_non_finite_logits_are_not_listed(self) -> None:
        """-inf scores are not candidates."""
        seen: list[decoder.TokenChoice] = []
        decoder.generate(lambda ids: [-np.inf, 1.0, -np.inf], [],
                         lambda ids: None if ids else [0, 1], 1,
                         on_token=seen.append)
        self.assertEqual(seen[0].top, [(1, 1.0, True)])


class CandidateTextTests(unittest.TestCase):
    """The table the pipeline hands to the dashboard."""

    def setUp(self) -> None:
        """A run whose model decode is used only for tokens outside vocab."""
        self.shown: dict[str, str] = {}
        self.context = pipeline.Run(
            object.__new__(Model),
            {1: b',"', 2: b'\x1b[2J', 3: b'\xe6', 4: b'\xe6\x9d\xb1'},
            {}, NameTrie(), self.shown.__setitem__,
        )

    def test_selected_below_top_ten_is_added_and_marked(self) -> None:
        """Masked rows say so; the selected token always shows."""
        choice = decoder.TokenChoice(
            step=1, token=50, tokens=[50], forced=False, logit=0.5,
            allowed_mass=0.25, margin=1.5,
            top=[(i, 9.0 - i, False) for i in range(10)],
        )
        with patch.object(Model, "decode", return_value="x"):
            pipeline._show_choice(choice, self.context)
        header, *rows = self.shown["candidates"].splitlines()
        self.assertEqual(header, "allowed_mass=0.25  gap=1.5")
        self.assertEqual(len(rows), 11)
        self.assertTrue(all("masked" in row for row in rows[:10]))
        self.assertTrue(rows[10].startswith(">"))
        self.assertIn("50", rows[10])

    def test_token_text_is_unambiguous(self) -> None:
        """Quotes, controls and partial UTF-8 are visible as such."""
        with patch.object(Model, "decode", return_value="<|im_end|>"):
            texts = [pipeline._token_text(token, self.context)
                     for token in range(1, 6)]
        self.assertEqual(texts, ["',\"'", "'\\x1b[2J'", "b'\\xe6'", "'東'",
                                 "'<|im_end|>'"])

    def test_forced_token_text(self) -> None:
        """No logits are shown for a forced token."""
        choice = decoder.TokenChoice(step=1, token=1, tokens=[1], forced=True)
        pipeline._show_choice(choice, self.context)
        self.assertIn("forced", self.shown["candidates"])


class DashboardTests(unittest.TestCase):
    """The candidate block and the summary left after the run."""

    def test_candidates_render_safely_within_width(self) -> None:
        """Every candidate row is escaped and clipped like other rows."""
        frame = render_progress(
            {"input": "1/1", "result": "ok",
             "candidates": "> 1 '東京' 9.000\n  2 \x1b[2J 8.000 masked"},
            width=30, height=40,
        )
        self.assertNotIn("\x1b", frame)
        self.assertIn("Candidates:", frame)
        self.assertIn("東京", frame)
        self.assertTrue(frame.index("Result") < frame.index("Candidates"))

    def test_summary_has_status_and_time_only(self) -> None:
        """Done/failed and seconds per prompt, then a total line."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            with Dashboard(2) as dashboard:
                report = dashboard.report
                report("input", "1/2")
                report("function", "fn_a")
                report("candidates", "> 1 'a' 1.000")
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
