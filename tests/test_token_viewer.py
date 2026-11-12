"""Check the token viewer (ADR-0021) without the real model."""

import contextlib
import io
import json
import os
import queue
import tempfile
import threading
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np
from rich.console import Console

from src import decoder, display, pipeline, viewer
from src.model_port import Model
from src.name_trie import EOS, build_name_trie
from src.schema import FunctionDef


def _encode(text: str) -> list[int]:
    """One token per byte; the name terminator is token 127."""
    return list(text.replace(EOS, "\x7f").encode())


def _function(name: str) -> FunctionDef:
    """A definition with one number and one string parameter."""
    return FunctionDef.model_validate({
        "name": name, "description": "fixture",
        "parameters": {"a": {"type": "number"}, "b": {"type": "string"}},
        "returns": {"type": "number"},
    })


class DecoderCandidateTests(unittest.TestCase):
    """Items 1-5: candidates come from logits already computed."""

    def test_candidates_match_a_full_sort(self) -> None:
        """RAW is the vocabulary top 10, SORT the allowed top 10."""
        scores = np.random.default_rng(0).normal(size=300)
        raw_order = list(np.argsort(-scores))
        # Allow two of the raw top 10 and many low-ranked tokens.
        allowed = [int(raw_order[3]), int(raw_order[7])] + [
            int(token) for token in raw_order[50:80]]
        seen: list[decoder.TokenChoice] = []
        result = decoder.generate(
            lambda ids: list(scores), [], lambda ids: None if ids else allowed,
            1, on_token=seen.append, top_k=10,
        )
        (choice,) = seen
        self.assertEqual([token for token, _, _ in choice.raw],
                         [int(t) for t in raw_order[:10]])
        self.assertEqual([ok for _, _, ok in choice.raw],
                         [int(t) in allowed for t in raw_order[:10]])
        expected = sorted(allowed, key=lambda t: -scores[t])[:10]
        self.assertEqual([token for token, _ in choice.masked], expected)
        self.assertTrue(set(expected) - {t for t, _, _ in choice.raw})
        self.assertEqual(result, [expected[0]])
        self.assertEqual(choice.token,
                         decoder.choose_token(list(scores), allowed))
        self.assertEqual(decoder.generate(
            lambda ids: list(scores), [], lambda ids: None if ids else allowed,
            1,
        ), result)

    def test_forced_tokens_have_no_candidates_or_inference(self) -> None:
        """A single allowed token never calls the model."""
        calls: list[list[int]] = []

        def logits(ids: list[int]) -> list[float]:
            calls.append(ids)
            return [0.0, 1.0, 2.0, 3.0]

        seen: list[decoder.TokenChoice] = []
        decoder.generate(logits, [], lambda ids: [[0], [1, 2], None][len(ids)],
                         2, on_token=seen.append, top_k=10)
        self.assertEqual(len(calls), 1)
        self.assertTrue(seen[0].forced)
        self.assertEqual((seen[0].raw, seen[0].masked), ([], []))
        self.assertFalse(seen[1].forced)

    def test_non_finite_logits_are_not_listed(self) -> None:
        """Events stay standard JSON: -inf is never sent."""
        seen: list[decoder.TokenChoice] = []
        decoder.generate(lambda ids: [-np.inf, 1.0, -np.inf],
                         [], lambda ids: None if ids else [0, 1], 1,
                         on_token=seen.append, top_k=10)
        self.assertEqual(seen[0].masked, [(1, 1.0)])
        self.assertEqual([t for t, _, _ in seen[0].raw], [1])


class PipelinePreviewTests(unittest.TestCase):
    """Items 6, 7 and 9: previews follow the real record and change nothing."""

    def run_prompt(self, top_k: int) -> tuple[Any, list[dict[str, Any]], int]:
        """Process one prompt with scripted logits and collect trace events."""
        functions = {name: _function(name) for name in ("fn_a", "fn_b")}
        trie = build_name_trie(list(functions), _encode)
        vocab = {token: bytes([token]) for token in range(127)}
        script = list('a12,"hi"}')
        calls = 0

        def logits(ids: list[int]) -> list[float]:
            """Prefer the next scripted byte; token 200 is never allowed."""
            nonlocal calls
            scores = [0.0] * 256
            scores[ord(script[calls])] = 5.0
            scores[200] = 9.0
            calls += 1
            return scores

        events: list[tuple[str, str]] = []
        model = object.__new__(Model)
        with patch.object(Model, "encode", side_effect=_encode), \
                patch.object(Model, "decode",
                             side_effect=lambda ids: bytes(ids).decode(
                                 errors="replace")), \
                patch.object(Model, "logits", side_effect=logits):
            call = pipeline.process_prompt(
                "p", functions, model, trie, vocab,
                lambda label, text: events.append((label, text)), top_k=top_k,
            )
        traces = [json.loads(text) for label, text in events
                  if label == "trace"]
        return call, traces, calls

    def test_previews_end_as_the_generated_record(self) -> None:
        """The last name and value previews spell the actual output."""
        call, traces, calls = self.run_prompt(top_k=10)
        names = [t for t in traces if t["stage"] == "select_name"]
        values = [t for t in traces if t["stage"] == "generate_parameters"]
        self.assertTrue(names[-1]["done"])
        self.assertEqual(names[-1]["fixed"] + names[-1]["value"],
                         '{"prompt": "p", "name": "fn_a')
        last = values[-1]
        self.assertTrue(last["done"])
        self.assertEqual(
            last["fixed"] + last["value"] + last["rest"],
            '{"prompt": "p", "name": "fn_a", '
            '"parameters": {"a": 12, "b": "hi"}}',
        )
        self.assertEqual(values[0]["rest"], ' "b": …}}')
        self.assertEqual(call.parameters, {"a": 12, "b": "hi"})
        for trace in traces:
            if trace["forced"]:
                self.assertEqual((trace["raw"], trace["masked"]), ([], []))
            else:
                self.assertEqual(trace["raw"][0],
                                 [200, "<id 200>", 9.0, False])
                self.assertEqual(trace["masked"][0][0], trace["selected"][0])
        self.assertEqual(calls, 9)

    def test_tracing_changes_nothing(self) -> None:
        """Same call and the same model calls with and without the viewer."""
        traced, traces, traced_calls = self.run_prompt(top_k=10)
        plain, no_traces, plain_calls = self.run_prompt(top_k=0)
        self.assertEqual(traced, plain)
        self.assertEqual(traced_calls, plain_calls)
        self.assertTrue(traces)
        self.assertEqual(no_traces, [])


class SocketTests(unittest.TestCase):
    """Items 8, 10 and 11 at the process boundary."""

    def setUp(self) -> None:
        """Listen on a private path, as the viewer does."""
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "viewer.sock")
        self.server = viewer.listen(self.path)
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stderr = io.StringIO()

    def tearDown(self) -> None:
        """Close the listener and remove the directory."""
        self.server.close()
        self.directory.cleanup()

    def receive(self) -> None:
        """Run the viewer's receiving thread."""
        threading.Thread(target=viewer.receive,
                         args=(self.server, self.events), daemon=True).start()

    def drain(self) -> list[dict[str, Any]]:
        """Collect events up to and including ``end``."""
        received = [self.events.get(timeout=5)]
        while received[-1]["event"] != "end":
            received.append(self.events.get(timeout=5))
        return received

    def test_events_arrive_in_order_and_stderr_stays_plain(self) -> None:
        """Queued events outlive the generator; stderr gets plain lines."""
        self.receive()
        with patch.object(display, "VIEWER_SOCKET", self.path), \
                contextlib.redirect_stderr(self.stderr):
            with display.terminal_progress(1) as report:
                report("input", "1/1")
                self.assertEqual(report.top_k, display.TRACE_TOP_K)
                report("prompt", "東京\x1b[2J")
                report("trace", json.dumps({"event": "token", "step": 1}))
                report("function", "fn_a")
                report("result", "fn_a {}")
        self.assertEqual([e["event"] for e in self.drain()],
                         ["prompt", "token", "result", "end"])
        self.assertNotIn("\x1b", self.stderr.getvalue())
        self.assertIn("function: fn_a", self.stderr.getvalue())
        self.assertNotIn("trace", self.stderr.getvalue())

    def test_slow_viewer_does_not_block_generation(self) -> None:
        """Far more data than a socket buffer is queued, not dropped."""
        self.receive()
        event = json.dumps({"event": "token", "value": "x" * 1000})
        with patch.object(display, "VIEWER_SOCKET", self.path), \
                contextlib.redirect_stderr(self.stderr):
            with display.terminal_progress(1) as report:
                report("input", "1/1")
                for _ in range(5000):
                    report("trace", event)
                self.assertEqual(report.top_k, display.TRACE_TOP_K)
        self.assertEqual(len(self.drain()), 5001)

    def test_stopped_viewer_falls_back_to_stderr(self) -> None:
        """Losing the viewer keeps reporting results."""
        with patch.object(display, "VIEWER_SOCKET", self.path), \
                contextlib.redirect_stderr(self.stderr):
            with display.terminal_progress(1) as report:
                report("input", "1/1")
                connection, _ = self.server.accept()
                connection.close()
                for _ in range(100):
                    report("trace", "{}")
                self.assertEqual(report.top_k, 0)
                report("result", "fn_a {}")
        self.assertIn("result: fn_a {}", self.stderr.getvalue())

    def test_missing_or_foreign_viewer_is_not_used(self) -> None:
        """No socket, or another user's socket, means no tracing."""
        self.assertIsNone(display.connect_viewer(self.path + ".missing"))
        with patch.object(display.os, "getuid", return_value=-1):
            self.assertIsNone(display.connect_viewer(self.path))


class ViewerScreenTests(unittest.TestCase):
    """Phases and safe rendering (items 1-4, 12)."""

    def token(self, forced: bool = False) -> dict[str, Any]:
        """A token event with hostile and wide token texts."""
        texts = ["\x1b[2J", "\n", "東京", "x" * 300, "[bold]markup"]
        return {
            "event": "token", "stage": "generate_parameters",
            "parameter": "city (string)", "step": 3, "forced": forced,
            "selected": [2, "東京"],
            "raw": [] if forced else [
                [i, text, 9.0 - i, i % 2 == 0] for i, text in enumerate(texts)
            ],
            "masked": [] if forced else [
                [i, texts[i], 9.0 - i] for i in (0, 2, 4)
            ],
            "fixed": '{"city": "', "value": "東\\xe4", "rest": "}}",
            "done": False,
        }

    def test_phases_and_preview_timing(self) -> None:
        """Four timed phases; the JSON changes only on SELECT."""
        screen = viewer.Screen()
        holds = []
        for seconds in viewer.show(self.token(), screen):
            holds.append((screen.phase, seconds, screen.preview[1]))
        self.assertEqual(holds, [
            ("RAW", viewer.RAW_SECONDS, ""),
            ("MASK", viewer.MASK_SECONDS, ""),
            ("SORT", viewer.SORT_SECONDS, ""),
            ("SELECT", viewer.SELECT_SECONDS, "東\\xe4"),
        ])
        forced = list(viewer.show(self.token(forced=True), screen))
        self.assertEqual((forced, screen.phase),
                         ([viewer.SELECT_SECONDS], "FORCED"))

    def test_render_escapes_controls(self) -> None:
        """No raw control or markup reaches the terminal."""
        screen = viewer.Screen()
        list(viewer.show({"event": "prompt", "input": "1/1",
                          "prompt": "p\x1b[31m"}, screen))
        for phase in ("RAW", "MASK", "SORT", "SELECT", "FORCED"):
            for seconds in viewer.show(self.token(phase == "FORCED"), screen):
                output = io.StringIO()
                Console(file=output, width=100, height=30,
                        color_system=None).print(viewer.render(screen))
                text = output.getvalue()
                self.assertNotIn("\x1b", text)
                self.assertIn("東", text)
                if screen.phase == "RAW":
                    self.assertIn('"\\u001b[2J"', text)
                    self.assertIn('"\\n"', text)
                    self.assertIn("[bold]markup", text)
                if screen.phase == "MASK":
                    self.assertIn("-inf", text)
                if screen.phase == "FORCED":
                    self.assertIn("FORCED", text)
