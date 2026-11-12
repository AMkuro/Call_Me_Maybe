"""One-token selection preserves masks, inference counts and diagnostics."""

import unittest
from unittest.mock import patch

import numpy as np

from src import decoder
from src.errors import PromptError
from src.model_port import Model


class SelectionTests(unittest.TestCase):
    """Exercise the public selection boundary with fixed model scores."""

    def test_fixed_scores_and_optional_diagnostics(self) -> None:
        """Diagnostics preserve the selected ID, model input and call count."""
        model = object.__new__(Model)
        for scores, allowed, expected, raw in (
            ([3.0, 1.0, 2.0], [1, 2], 2, 0),
            ([2.0, 2.0, 0.0], [1, 0], 0, 0),
            ([-np.inf, 1.0, -np.inf], [0, 1], 1, 1),
        ):
            for diagnose in (False, True):
                with self.subTest(scores=scores, diagnose=diagnose):
                    ids = [9, 7]
                    with patch.object(
                        Model, 'logits', return_value=scores,
                    ) as llm:
                        choice = decoder.select_next_token(
                            model, ids, allowed, include_raw_top=diagnose,
                        )
                    self.assertEqual(choice, (
                        expected, 'greedy', raw if diagnose else None,
                    ))
                    llm.assert_called_once_with(ids)
                    self.assertEqual(ids, [9, 7])

    def test_forced_skips_model_and_all_array_work(self) -> None:
        """A unique allowed token never needs logits or diagnostic argmax."""
        model = object.__new__(Model)
        with patch.object(Model, 'logits', side_effect=AssertionError), \
                patch.object(np, 'asarray', side_effect=AssertionError), \
                patch.object(np, 'argmax', side_effect=AssertionError):
            for diagnose in (False, True):
                self.assertEqual(decoder.select_next_token(
                    model, [9], [2], include_raw_top=diagnose,
                ), (2, 'forced', None))

    def test_empty_mask_fails_without_inference(self) -> None:
        """Empty candidates preserve the existing continuation error."""
        with patch.object(Model, 'logits', side_effect=AssertionError):
            with self.assertRaisesRegex(PromptError, 'no allowed token'):
                decoder.select_next_token(object.__new__(Model), [], [])

    def test_no_ranking_or_probability_work(self) -> None:
        """Selection and warnings require neither top-k nor softmax."""
        with patch.object(Model, 'logits', return_value=[3.0, 1.0, 2.0]), \
                patch.object(np, 'argpartition', side_effect=AssertionError), \
                patch.object(np, 'sort', side_effect=AssertionError), \
                patch.object(np, 'exp', side_effect=AssertionError):
            self.assertEqual(decoder.select_next_token(
                object.__new__(Model), [], [1, 2], include_raw_top=True,
            ), (2, 'greedy', 0))
