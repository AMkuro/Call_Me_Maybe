"""Tests for the optional visual trace, without loading the model or Rich."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.decoder import generate
from src.trace import event_log
from src.viewer import _pretty_prefix, _safe


class TraceTests(unittest.TestCase):
    """Check observations and required decoding behavior separately."""

    def test_raw_top_is_not_filtered_before_masking(self) -> None:
        """Candidates rejected by the mask remain in the raw top ten."""
        observed = []
        calls = []

        def logits(ids: list[int]) -> list[float]:
            calls.append(ids.copy())
            return [1.0, 5.0, 9.0, 8.0, 7.0]

        def allowed(generated: list[int]) -> list[int] | None:
            return [0, 1] if not generated else None

        self.assertEqual(generate(logits, [10], allowed, 3,
                                  observed.append, top_k=3), [1])
        self.assertEqual(calls, [[10]])
        choice = observed[0]
        self.assertEqual(choice.token, 1)
        self.assertEqual(
            [(i, ok) for i, _, ok in choice.raw_candidates or []],
            [(2, False), (3, False), (4, False)],
        )
        self.assertEqual(
            [i for i, _ in choice.allowed_candidates or []], [1, 0],
        )
        self.assertTrue(choice.constrained)

    def test_ties_do_not_change_the_selected_token(self) -> None:
        """Observability must not change argmax tie breaking."""
        observed = []

        def allowed(generated: list[int]) -> list[int] | None:
            return [4, 2, 1] if not generated else None

        self.assertEqual(generate(lambda ids: [0, 8, 9, 0, 9],
                                  [], allowed, 2, observed.append,
                                  top_k=2), [2])
        self.assertEqual(observed[0].allowed_candidates,
                         [(2, 9.0), (4, 9.0)])

    def test_forced_token_never_runs_the_model(self) -> None:
        """Logits are genuinely unknown on single-candidate steps."""
        observed = []

        def no_logits(ids: list[int]) -> list[float]:
            self.fail("forced selection must skip the forward pass")

        def allowed(generated: list[int]) -> list[int] | None:
            return [3] if not generated else None

        self.assertEqual(generate(no_logits, [], allowed, 2,
                                  observed.append, top_k=10), [3])
        self.assertTrue(observed[0].forced)
        self.assertIsNone(observed[0].raw_candidates)
        self.assertIsNone(observed[0].allowed_candidates)

    def test_trace_is_opt_in_and_ends_after_exception(self) -> None:
        """A diagnostic stream is closed and ends on exceptional exits."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            with patch.dict(os.environ, {}, clear=True):
                with event_log() as emit:
                    self.assertIsNone(emit)
            with patch.dict(os.environ, {"CMM_TRACE_FILE": str(path)}):
                with self.assertRaisesRegex(ValueError, "stop"):
                    with event_log() as emit:
                        self.assertIsNotNone(emit)
                        if emit is not None:
                            emit({"type": "input", "prompt": "東京"})
                        raise ValueError("stop")
            events = [json.loads(line) for line in
                      path.read_text().splitlines()]
            self.assertEqual(events[0]["prompt"], "東京")
            self.assertEqual(events[-1], {"type": "end"})

    def test_pretty_json_preserves_string_punctuation(self) -> None:
        """Formatting never splits quoted commas or incomplete values."""
        fragment = '{"prompt":"a,b","parameters":{"city":"東'
        result = _pretty_prefix(fragment)
        self.assertIn('"a,b"', result)
        self.assertIn('"city": "東', result)
        self.assertNotIn('"a,\\n', result)

    def test_escape_control_characters(self) -> None:
        """Viewer text must never execute a token's terminal control codes."""
        self.assertEqual(_safe("A\\x1b[2J\\nB"), "A\\\\u001b[2J\\nB")


if __name__ == "__main__":
    unittest.main()
