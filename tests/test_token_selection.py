"""One-token selection preserves masks, inference counts and diagnostics."""

import unittest
from unittest.mock import patch

import numpy as np

from src import decoder, pipeline
from src.errors import PromptError
from src.model_port import Model
from src.name_trie import NameTrie
from src.schema import FunctionDef


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
                        choice = decoder.infer_and_select_next_token(
                            model, ids, allowed, include_raw_top=diagnose,
                        )
                    self.assertEqual(choice, (
                        expected, 'greedy', raw if diagnose else None,
                    ))
                    llm.assert_called_once_with(ids)
                    self.assertEqual(ids, [9, 7])

    def test_forced_runs_one_forward_without_array_work(self) -> None:
        """Method B: a unique token still gets one forward, no array work."""
        model = object.__new__(Model)
        for diagnose in (False, True):
            with self.subTest(diagnose=diagnose), \
                    patch.object(Model, 'logits',
                                 return_value=[0.0, 0.0, 0.0]) as llm, \
                    patch.object(np, 'asarray', side_effect=AssertionError), \
                    patch.object(np, 'argmax', side_effect=AssertionError):
                self.assertEqual(decoder.infer_and_select_next_token(
                    model, [9], [2], include_raw_top=diagnose,
                ), (2, 'forced', None))
                llm.assert_called_once_with([9])

    def test_name_forwards_every_token_on_full_context(self) -> None:
        """Method B: forced and branching name IDs see P plus all IDs."""
        def fn(name: str) -> FunctionDef:
            return FunctionDef.model_validate({
                'name': name, 'description': 'd', 'parameters': {},
                'returns': {'type': 'string'}})
        trie = NameTrie(paths={'ab': (1, 2, 0), 'ac': (1, 3, 0)})
        contexts: list[list[int]] = []

        def logits(ids: list[int]) -> list[float]:
            contexts.append(ids.copy())
            return [0.0, 0.0, 1.0, 2.0]
        with patch.object(Model, 'encode', return_value=[50]), \
                patch.object(Model, 'logits', side_effect=logits), \
                patch.object(pipeline, 'build_name_prompt', return_value='N'):
            chosen = pipeline.select_function(
                'x', model=object.__new__(Model),
                functions={'ab': fn('ab'), 'ac': fn('ac')}, trie=trie)
        self.assertEqual(chosen.name, 'ac')
        self.assertEqual(contexts, [[50], [50, 1], [50, 1, 3]])

    def test_empty_mask_fails_without_inference(self) -> None:
        """Empty candidates preserve the existing continuation error."""
        with patch.object(Model, 'logits', side_effect=AssertionError):
            with self.assertRaisesRegex(PromptError, 'no allowed token'):
                decoder.infer_and_select_next_token(
                    object.__new__(Model), [], [])

    def test_name_errors_say_they_come_from_the_name(self) -> None:
        """A failure while choosing a name is not reported as a value."""
        fn = FunctionDef.model_validate({
            'name': 'a', 'description': 'd', 'parameters': {},
            'returns': {'type': 'string'}})
        with patch.object(Model, 'encode', return_value=[50]), \
                patch.object(Model, 'logits',
                             side_effect=PromptError('model inference')), \
                patch.object(pipeline, 'build_name_prompt', return_value='N'):
            with self.assertRaisesRegex(
                    PromptError, r'^function name: model inference$'):
                pipeline.select_function(
                    'x', model=object.__new__(Model), functions={'a': fn},
                    trie=NameTrie(paths={'a': (1, 0)}))

    def test_a_token_off_the_tree_stops_name_selection(self) -> None:
        """Without a step limit, leaving the tree is still an error."""
        fn = FunctionDef.model_validate({
            'name': 'a', 'description': 'd', 'parameters': {},
            'returns': {'type': 'string'}})
        with patch.object(Model, 'encode', return_value=[50]), \
                patch.object(pipeline, 'build_name_prompt',
                             return_value='N'), \
                patch.object(pipeline, 'infer_and_select_next_token',
                             return_value=(9, 'greedy', None)):
            with self.assertRaisesRegex(
                    PromptError, r'^function name: invalid token path'):
                pipeline.select_function(
                    'x', model=object.__new__(Model), functions={'a': fn},
                    trie=NameTrie(paths={'a': (1, 0)}))

    def test_no_ranking_or_probability_work(self) -> None:
        """Selection and warnings require neither top-k nor softmax."""
        with patch.object(Model, 'logits', return_value=[3.0, 1.0, 2.0]), \
                patch.object(np, 'argpartition', side_effect=AssertionError), \
                patch.object(np, 'sort', side_effect=AssertionError), \
                patch.object(np, 'exp', side_effect=AssertionError):
            self.assertEqual(decoder.infer_and_select_next_token(
                object.__new__(Model), [], [1, 2], include_raw_top=True,
            ), (2, 'greedy', 0))
