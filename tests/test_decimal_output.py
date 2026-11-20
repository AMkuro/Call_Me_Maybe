"""Independent REQ-031 grammar, generation and exact-save regression checks."""

import json
import re
import tempfile
import unittest
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pydantic import ValidationError
from trie_fixtures import trie_from_paths

from src import generation, pipeline, runtime
from src.errors import PromptError
from src.model_port import Model
from src.schema import FunctionCallRequest, FunctionDef
from src.skeleton import ArgumentGenerationPart
from src.values import (
    allowed_tokens_for_argument,
    integer_to_float_literal,
    is_argument_part_complete,
    normalize_call_json,
)
from src.writer import save_results

VALID = (
    "0.0",
    "-0.0",
    "2.0",
    "2.75",
    "-42.125",
    "9007199254740992.0",
    "0.00000001",
    "10.0",
    "-3.0",
    "0.1",
    "2.00",
)
# The terminator may follow an integer (owner decision, 2026-11-18).
INTEGERS = ("0", "-0", "2", "-2", "10", "9007199254740993")
INVALID = (
    "2.",
    ".5",
    "+2.0",
    "01.0",
    "01",
    "-",
    "1e10",
    "1.0e10",
    "1.0E-5",
    "NaN",
    "Infinity",
    "1.0, extra",
)


def definition() -> FunctionDef:
    """Combine a number, numeric-looking string and boolean."""
    return FunctionDef.model_validate(
        {
            "name": "record",
            "description": "Record values unchanged.",
            "parameters": {
                "x": {"type": "number"},
                "s": {"type": "string"},
                "b": {"type": "boolean"},
            },
            "returns": {"type": "number"},
        }
    )


def generated(literal: str, prompt: str = "依頼") -> FunctionCallRequest:
    """Keep real validation and the production raw-text setter."""
    fn = definition()
    model = object.__new__(Model)
    resources = runtime.GenerationAssets.model_construct(
        model=model,
        vocab={},
        functions={"record": fn},
        trie=trie_from_paths({"record": (1,)}),
    )
    text = (
        '{"prompt":' + json.dumps(prompt) + ',"name":"record",'
        '"parameters":{"x":'
        + literal
        + ',"s":'
        + json.dumps('日本語 " \\ \n 1e10 2.0 <|im_end|>')
        + ',"b":false}}'
    )
    start = text.index(literal, text.index('"parameters":{"x":'))
    number_slot_slices = [slice(start, start + len(literal) + 1)]
    with (
        patch.object(pipeline, "select_function", return_value=fn),
        patch.object(
            pipeline,
            "generate_call_json",
            return_value=(text, number_slot_slices),
        ),
    ):
        return pipeline.process_prompt(prompt, resources)


class DecimalGrammarTests(unittest.TestCase):
    """Compare masks and completion against independently supplied literals."""

    def test_every_valid_prefix_reaches_both_terminators(self) -> None:
        """No legal next byte is masked before the complete decimal value."""
        vocab = {token: bytes([token]) for token in range(128)}
        for literal in (
            VALID
            + INTEGERS
            + ("9007199254740993.0", "123456789012345678901234567890.0")
        ):
            for terminator in (",", "}"):
                with self.subTest(literal=literal, terminator=terminator):
                    written = b""
                    for token in (literal + terminator).encode():
                        self.assertFalse(
                            is_argument_part_complete(
                                written, "number", terminator
                            )
                        )
                        self.assertIn(
                            token,
                            allowed_tokens_for_argument(
                                vocab, "number", written, terminator
                            ),
                        )
                        written += bytes([token])
                    self.assertTrue(
                        is_argument_part_complete(
                            written, "number", terminator
                        )
                    )
                    self.assertEqual(
                        allowed_tokens_for_argument(
                            vocab, "number", written, terminator
                        ),
                        [],
                    )

    def test_invalid_literals_cannot_be_committed_whole(self) -> None:
        """Incomplete prefixes can continue, but cannot accept a terminator."""
        for literal in INVALID:
            for terminator in (",", "}"):
                with self.subTest(literal=literal, terminator=terminator):
                    payload = (literal + terminator).encode()
                    self.assertFalse(
                        is_argument_part_complete(
                            payload, "number", terminator
                        )
                    )
                    self.assertEqual(
                        allowed_tokens_for_argument(
                            {9: payload}, "number", b"", terminator
                        ),
                        [],
                    )
        for prefix, allowed in (
            (b"2", [9]),
            (b"-2", [9]),
            (b"2.", []),
            (b"-", []),
        ):
            for terminator in (",", "}"):
                self.assertEqual(
                    allowed_tokens_for_argument(
                        {9: terminator.encode()},
                        "number",
                        prefix,
                        terminator,
                    ),
                    allowed,
                )

    def test_merged_token_candidates_obey_the_whole_token(self) -> None:
        """A token cannot hide an exponent or trailing material."""
        vocab = {
            1: b"2.0",
            2: b"2.0,",
            3: b".0}",
            4: b"1e10",
            5: b"1.0e10",
            6: b"0,",
            7: b"2.0, extra",
        }
        allowed = allowed_tokens_for_argument
        self.assertEqual(allowed(vocab, "number", b"", ","), [1, 2, 6])
        self.assertEqual(allowed(vocab, "number", b"2", "}"), [1, 3])
        self.assertEqual(allowed(vocab, "number", b"2.", ","), [6])
        self.assertTrue(is_argument_part_complete(b"2.0}", "number", "}"))
        self.assertFalse(is_argument_part_complete(b"2.0", "number", "}"))

    def test_float_text_adds_only_a_missing_fraction(self) -> None:
        """String edit, so large integers keep every digit."""
        for fragment, saved in (
            ("10,", "10.0,"),
            ("-0}", "-0.0}"),
            (" 7 }", " 7.0 }"),
            ("2.5,", "2.5,"),
            ("9007199254740993,", "9007199254740993.0,"),
        ):
            with self.subTest(fragment=fragment):
                self.assertEqual(integer_to_float_literal(fragment), saved)

    def test_fraction_selects_from_forced_context(self) -> None:
        """The model chooses 05, with the forced decimal prefix in context."""
        model = object.__new__(Model)
        # 00} and 05} cannot start a number, so 2. is the only first token.
        vocab = {1: b"2.", 2: b"00}", 3: b"05}", 4: b"1e10"}
        scores = [0.0, 0.0, 1.0, 2.0, 1000.0]
        slot = ArgumentGenerationPart(name="x", type="number", terminator="}")
        with (
            patch.object(Model, "logits", return_value=scores) as forward,
            patch.object(Model, "encode") as encode,
        ):
            ids = generation.generate_argument_part(
                slot, [90, 91], model=model, vocab=vocab
            )
        self.assertEqual(ids, [1, 3])
        self.assertEqual(b"".join(vocab[t] for t in ids), b"2.05}")
        self.assertEqual(
            [c.args for c in forward.call_args_list],
            [([90, 91],), ([90, 91, 1],)],
        )
        encode.assert_not_called()

    def test_real_number_grammar_completes_at_32_not_33(self) -> None:
        """A number stops at its own cap, so a run of digits fails early."""
        model = object.__new__(Model)
        vocab = {1: b"1", 2: b".", 3: b"0", 4: b"}"}
        for length in (32, 33):
            target = b"1" * (length - 3) + b".0}"

            def logits(ids: list[int]) -> list[float]:
                """Choose the fixture continuation among legal tokens."""
                scores = [0.0] * 5
                next_byte = target[len(ids)]
                scores[{49: 1, 46: 2, 48: 3, 125: 4}[next_byte]] = 1.0
                return scores

            with (
                self.subTest(length=length),
                patch.object(Model, "logits", side_effect=logits),
            ):
                slot = ArgumentGenerationPart(
                    name="x", type="number", terminator="}"
                )
                if length == 32:
                    ids = generation.generate_argument_part(
                        slot, [], model=model, vocab=vocab
                    )
                    self.assertEqual(b"".join(vocab[t] for t in ids), target)
                else:
                    with self.assertRaisesRegex(PromptError, "token limit 32"):
                        generation.generate_argument_part(
                            slot, [], model=model, vocab=vocab
                        )


class DecimalSaveTests(unittest.TestCase):
    """Inspect the saved file using standard JSON and Decimal hooks."""

    def test_pipeline_normalizes_integer_before_float_validation(self) -> None:
        """Integer generation succeeds only after the separate text edit."""
        for literal in INTEGERS[:-1]:
            with self.subTest(literal=literal):
                call = generated(literal)
                text = call._generated_json
                assert text is not None
                self.assertIn('"x":' + literal + ".0,", text)
                self.assertIs(type(call.parameters["x"]), float)

    def test_normalization_preserves_all_other_text(self) -> None:
        """Only supplied number-slot slices may gain a fraction."""
        head = '{"prompt":"東 2, \\u0032","parameters":{"s":"-5}","a":'
        for raw, expected in (
            ("2,", "2.0,"),
            ("-5}", "-5.0}"),
            ("0}", "0.0}"),
            ("-0}", "-0.0}"),
            ("12.2500,", "12.2500,"),
            (
                "0.12345678901234567890123456789}",
                "0.12345678901234567890123456789}",
            ),
            ("9007199254740993}", "9007199254740993.0}"),
            (" 7 }", " 7.0 }"),
            ("1e10}", "1e10}"),
        ):
            with self.subTest(raw=raw):
                text = head + raw + "}"
                self.assertEqual(
                    normalize_call_json(
                        text,
                        [slice(len(head), len(head) + len(raw))],
                    ),
                    head + expected + "}",
                )
        text = '{"prompt":"数字2,","parameters":{}}'
        self.assertEqual(normalize_call_json(text, []), text)

    def test_generated_lexemes_survive_cumulative_save(self) -> None:
        """Preserve digits, sign and trailing zeros for every value."""
        calls = [
            generated(literal, str(index))
            for index, literal in enumerate(VALID)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.json"
            for count in range(1, len(calls) + 1):
                save_results(path, calls[:count])
                lexemes: list[str] = []

                def decimal(literal: str) -> Decimal:
                    """Observe only numeric literals, never string content."""
                    lexemes.append(literal)
                    return Decimal(literal)

                rows = json.loads(path.read_text(), parse_float=decimal)
                self.assertEqual(lexemes, list(VALID[:count]))
                self.assertEqual(
                    [r["prompt"] for r in rows], [str(i) for i in range(count)]
                )
                for literal, row in zip(VALID, rows):
                    self.assertEqual(
                        row["parameters"]["x"].as_tuple(),
                        Decimal(literal).as_tuple(),
                    )
                    self.assertIs(row["parameters"]["b"], False)
                    self.assertEqual(
                        row["parameters"]["s"],
                        '日本語 " \\ \n 1e10 2.0 <|im_end|>',
                    )
        for call in calls:
            self.assertEqual(
                set(call.model_dump()), {"prompt", "name", "parameters"}
            )
        schema = FunctionCallRequest.model_json_schema()
        self.assertEqual(
            set(schema["properties"]), {"prompt", "name", "parameters"}
        )

    def test_precision_rejections_keep_existing_reasons(self) -> None:
        """Keep precision and finite-range checks."""
        for literal, reason in (
            ("9007199254740993", "precision"),
            ("9007199254740993.0", "precision"),
            ("123456789012345678901234567890", "precision"),
            ("123456789012345678901234567890.0", "precision"),
            ("0.1234567890123456789", "precision"),
            ("1." + "0" * 18 + "1", "precision"),
            ("0." + "0" * 400 + "1", "underflow"),
            ("9" * 309 + ".0", "overflow"),
        ):
            with self.subTest(literal=literal[:30]):
                self.assertTrue(
                    re.fullmatch(r"-?(0|[1-9][0-9]*)(?:\.[0-9]+)?", literal)
                )
                with self.assertRaisesRegex(PromptError, reason):
                    generated(literal)

    def test_saved_numbers_must_load_as_float(self) -> None:
        """Python's json must read every number argument as a finite float."""
        for literal in ("10", "-3"):
            with (
                self.subTest(literal=literal),
                self.assertRaisesRegex(PromptError, "integer"),
            ):
                text = (
                    '{"prompt":"p","name":"record","parameters":'
                    '{"x":' + literal + ',"s":"","b":false}}'
                )
                pipeline.validate_call(text, definition(), "p")
        saved = generated("2.0")._generated_json
        assert saved is not None
        self.assertIsInstance(json.loads(saved)["parameters"]["x"], float)

    def test_generated_text_must_pass_validation(self) -> None:
        """An unvalidated body cannot enter the normal trusted-text path."""
        for literal in ("true", '"2.0"', "NaN", '2.0,"x":3.0'):
            with self.subTest(literal=literal), self.assertRaises(PromptError):
                generated(literal)
        with self.assertRaises(ValidationError):
            FunctionCallRequest.model_validate(
                {
                    "prompt": "x",
                    "name": "record",
                    "parameters": {"x": 2.0},
                    "_generated_json": '{"unvalidated":true}',
                }
            )

    def test_saved_records_are_indented_keeping_spelling(self) -> None:
        """Only whitespace outside strings changes; 2.00 stays 2.00."""
        prompt = 'a, "b" {x}: [y]'
        call = generated("2.00", prompt=prompt)
        record = (
            '  {\n    "prompt": ' + json.dumps(prompt) + ",\n"
            '    "name": "record",\n'
            '    "parameters": {\n'
            '      "x": 2.00,\n'
            '      "s": ' + json.dumps('日本語 " \\ \n 1e10 2.0 <|im_end|>')
            + ",\n"
            '      "b": false\n'
            "    }\n  }"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.json"
            save_results(path, [call, call])
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "[\n" + record + ",\n" + record + "\n]\n",
            )

    def test_manual_call_keeps_the_documented_fallback(self) -> None:
        """Manually built calls still use ordinary JSON serialization."""
        call = FunctionCallRequest(
            prompt="手動",
            name="record",
            parameters={"x": 2.0, "s": "1e10", "b": True},
        )
        empty = FunctionCallRequest(prompt="p", name="none", parameters={})
        self.assertIs(type(call.parameters["x"]), float)
        self.assertIsNone(call._generated_json)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.json"
            save_results(path, [call, empty])
            self.assertEqual(
                path.read_text(),
                json.dumps([call.model_dump(), empty.model_dump()], indent=2)
                + "\n",
            )
            save_results(path, [])
            self.assertEqual(path.read_text(), "[]\n")

    def test_replace_failure_preserves_previous_file(self) -> None:
        """Fail replacement after writing raw generated text."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.json"
            path.write_text('["previous"]\n')
            with (
                patch("src.writer.os.replace", side_effect=OSError("replace")),
                self.assertRaisesRegex(OSError, "replace"),
            ):
                save_results(path, [generated("2.00")])
            self.assertEqual(path.read_text(), '["previous"]\n')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_raw_write_failure_preserves_previous_file(self) -> None:
        """A write error during generated text never replaces prior output."""
        real_temporary = tempfile.NamedTemporaryFile

        def temporary(**kwargs: Any) -> Any:
            """Keep real temporary-file lifecycle and fail only its write."""
            output = real_temporary(**kwargs)
            original: Callable[[str], int] = output.write

            def write(text: str) -> int:
                """Raise while writing the record."""
                if '"prompt"' in text:
                    raise OSError("write failed")
                return original(text)

            writer = patch.object(output, "write", side_effect=write)
            writer.start()
            self.addCleanup(writer.stop)
            return output

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.json"
            path.write_text('["previous"]\n')
            with (
                patch(
                    "src.writer.tempfile.NamedTemporaryFile",
                    side_effect=temporary,
                ),
                self.assertRaisesRegex(OSError, "write failed"),
            ):
                save_results(path, [generated("2.00")])
            self.assertEqual(path.read_text(), '["previous"]\n')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_success_order_after_prompt_failure(self) -> None:
        """Run the real loop and writer, skipping one failed prompt."""
        import contextlib
        import io

        from src.display import Dashboard

        model = object.__new__(Model)
        fn = definition()
        resources = runtime.GenerationAssets.model_construct(
            model=model,
            vocab={},
            functions={"record": fn},
            trie=trie_from_paths({"record": (1,)}),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.json"
            with contextlib.redirect_stderr(io.StringIO()):
                dashboard = Dashboard(3, model, {})
            outcomes = [
                generated("2.00", "first"),
                PromptError("skip"),
                generated("0.00000001", "last"),
            ]
            with patch.object(
                pipeline, "process_prompt", side_effect=outcomes
            ):
                self.assertFalse(
                    pipeline.process_all(
                        ["first", "bad", "last"], resources, path, dashboard
                    )
                )
            prompts = [r["prompt"] for r in json.loads(path.read_text())]
            self.assertEqual(prompts, ["first", "last"])
            with patch.object(
                pipeline, "process_prompt", side_effect=PromptError("skip")
            ):
                self.assertFalse(
                    pipeline.process_all(["bad"], resources, path, dashboard)
                )
            self.assertEqual(path.read_text(), "[]\n")
