"""Mock-only checks: no torch import, model download, or forward pass."""

import contextlib
import io
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from examples.token_generation import generate, main


class TokenGenerationTests(unittest.TestCase):
    """Check ID accumulation, observation, reuse, and failure reporting."""

    def setUp(self) -> None:
        """Provide an SDK-shaped mock without importing the supplied SDK."""
        self.model = MagicMock()
        self.model.encode.return_value.tolist.side_effect = lambda: [[10, 11]]
        self.model.decode.return_value = "hello"
        self.sdk = types.ModuleType("llm_sdk")
        self.constructor = MagicMock(return_value=self.model)
        setattr(self.sdk, "Small_LLM_Model", self.constructor)
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def test_greedy_loop_and_observations(self) -> None:
        """Encode once, append IDs, and decode the full generated sequence."""
        received: list[list[int]] = []

        def logits(ids: list[int]) -> list[float]:
            """Snapshot each SDK input before the caller appends to it."""
            self.assertIsInstance(ids, list)
            self.assertTrue(all(type(token) is int for token in ids))
            received.append(ids.copy())
            return [-2.0, 3.0, 1.0] if len(received) == 1 else [4.0, 4.0, 0.0]

        self.model.get_logits_from_input_ids.side_effect = logits
        with contextlib.redirect_stdout(self.stdout):
            self.assertEqual(generate(self.model, "Hi", 3), [1, 0, 0])
        self.assertEqual(received, [[10, 11], [10, 11, 1], [10, 11, 1, 0]])
        self.model.encode.assert_called_once_with(
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        self.model.decode.assert_called_once_with([1, 0, 0])
        for observation in (
            "input_ids=[10, 11]", "step=1 token_id=1 logit=3.0",
            "generated_ids=[1, 0, 0]", "generated_text='hello'",
        ):
            self.assertIn(observation, self.stdout.getvalue())

    def test_main_reuses_one_model(self) -> None:
        """Two prompts share a model and have separate input sequences."""
        self.model.get_logits_from_input_ids.return_value = [0.0, 1.0]
        with patch.dict(sys.modules, {"llm_sdk": self.sdk}), \
                contextlib.redirect_stdout(self.stdout):
            result = main(["Hi", "Hello", "--max-new-tokens", "2"])
        self.assertEqual(result, 0)
        self.constructor.assert_called_once_with(
            "Qwen/Qwen3-0.6B", trust_remote_code=False
        )
        self.assertEqual(self.model.encode.call_count, 2)
        self.assertEqual(self.model.get_logits_from_input_ids.call_count, 4)
        self.assertEqual(self.model.decode.call_count, 2)
        self.assertEqual(self.stdout.getvalue().count("input_ids=[10, 11]"), 2)

    def test_invalid_arguments_do_not_load_model(self) -> None:
        """Reject invalid CLI input before importing or loading the SDK."""
        for argv in ([""], ["Hi", "--max-new-tokens", "0"],
                     ["Hi", "--max-new-tokens", "-1"],
                     ["Hi", "--max-new-tokens", "abc"]):
            with self.subTest(argv=argv), \
                    patch.dict(sys.modules, {"llm_sdk": self.sdk}), \
                    contextlib.redirect_stderr(self.stderr), \
                    self.assertRaises(SystemExit) as caught:
                main(argv)
            self.assertEqual(caught.exception.code, 2)
        self.constructor.assert_not_called()

    def test_invalid_generation_arguments(self) -> None:
        """The function also rejects blank prompts and bad limits."""
        for prompt, limit in ((" ", 1), ("Hi", 0), ("Hi", -1)):
            with self.subTest(prompt=prompt, limit=limit), \
                    self.assertRaises(ValueError):
                generate(self.model, prompt, limit)
        self.model.encode.assert_not_called()

    def test_sdk_initialization_error(self) -> None:
        """Preserve the SDK constructor's exception type and reason."""
        self.constructor.side_effect = OSError("model unavailable")
        with patch.dict(sys.modules, {"llm_sdk": self.sdk}), \
                contextlib.redirect_stderr(self.stderr):
            self.assertEqual(main(["Hi"]), 1)
        self.assertIn("SDK import/initialization", self.stderr.getvalue())
        self.assertIn("OSError: model unavailable", self.stderr.getvalue())

    def test_sdk_import_error(self) -> None:
        """A missing SDK is a reported failure, not an uncaught traceback."""
        with patch.dict(sys.modules, {"llm_sdk": None}), \
                contextlib.redirect_stderr(self.stderr):
            self.assertEqual(main(["Hi"]), 1)
        self.assertIn("SDK import/initialization", self.stderr.getvalue())
        self.assertIn("ModuleNotFoundError", self.stderr.getvalue())

    def test_generation_errors_reach_cli(self) -> None:
        """Input, forward, invalid-logit, and decode errors reach stderr."""
        failures = (
            ("encode", "encode/input conversion", ValueError("bad input")),
            ("logits", "step 1", OSError("forward failed")),
            ("empty_input", "encode/input conversion", None),
            ("empty_logits", "step 1", None),
            ("nan", "step 1", None),
            ("inf", "step 1", None),
            ("decode", "decode", ValueError("bad IDs")),
        )
        for failure, stage, error in failures:
            with self.subTest(failure=failure):
                self.setUp()
                self.model.get_logits_from_input_ids.return_value = [0.0, 1.0]
                if failure in ("encode", "decode"):
                    getattr(self.model, failure).side_effect = error
                elif failure == "logits":
                    self.model.get_logits_from_input_ids.side_effect = error
                elif failure == "empty_input":
                    self.model.encode.return_value.tolist.side_effect = (
                        lambda: [[]]
                    )
                else:
                    values = {"empty_logits": [], "nan": [float("nan")],
                              "inf": [float("inf")]}
                    self.model.get_logits_from_input_ids.return_value = values[
                        failure
                    ]
                with patch.dict(sys.modules, {"llm_sdk": self.sdk}), \
                        contextlib.redirect_stdout(self.stdout), \
                        contextlib.redirect_stderr(self.stderr):
                    self.assertEqual(main(["Hi", "--max-new-tokens", "2"]), 1)
                self.assertIn(stage, self.stderr.getvalue())
                if error is not None:
                    self.assertIn(str(error), self.stderr.getvalue())

    def test_mid_generation_error_keeps_cause(self) -> None:
        """A failed second forward pass reports its step and original cause."""
        error = OSError("second forward failed")
        self.model.get_logits_from_input_ids.side_effect = [[0.0, 1.0], error]
        with contextlib.redirect_stdout(self.stdout), \
                self.assertRaises(RuntimeError) as caught:
            generate(self.model, "Hi", 3)
        self.assertIs(caught.exception.__cause__, error)
        self.assertIn("step 2", str(caught.exception))
        self.model.decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
