"""Tokenwise fixed JSON and exact byte/SDK/context correspondence."""

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from trie_fixtures import trie_from_paths

from src import generation, pipeline, runtime
from src.errors import PromptError
from src.model_port import Model
from src.progress import GenerationProgress
from src.schema import FunctionDef
from src.values import normalize_call_json
from src.writer import save_results


class ForcedJsonTests(unittest.TestCase):
    """Use byte tokens and fixed scores, with real grammar and validation."""

    def setUp(self) -> None:
        """Replace only the external model boundary."""
        self.model = object.__new__(Model)
        self.vocab = {token: bytes([token]) for token in range(256)}
        self.encodings: list[str] = []
        self.contexts: list[list[int]] = []
        self.decodings: list[list[int]] = []
        self.target = b""
        self.special = "<|im_end|>"
        self.addCleanup(patch.stopall)
        for method, effect in (
            ("encode", self.encode),
            ("decode", self.decode),
            ("logits", self.logits),
        ):
            patch.object(Model, method, side_effect=effect).start()
        patch.object(
            generation, "build_arguments_prompt", return_value="P"
        ).start()

    def encode(self, text: str) -> list[int]:
        """Recognize a special ID outside the ordinary byte vocabulary."""
        self.encodings.append(text)
        parts = text.split(self.special)
        ids: list[int] = []
        for index, part in enumerate(parts):
            if index:
                ids.append(400)
            ids.extend(part.encode("utf-8"))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode completed sequences, dropping the recognized special ID."""
        self.decodings.append(ids.copy())
        return b"".join(self.vocab[t] for t in ids if t != 400).decode("utf-8")

    def logits(self, ids: list[int]) -> list[float]:
        """Prefer the known continuation; give an invalid token top score."""
        self.contexts.append(ids.copy())
        self.assertEqual(ids[:1], [ord("P")])
        self.assertNotIn(400, ids, "special control ID in output context")
        written = b"".join(self.vocab[t] for t in ids[1:])
        self.assertTrue(self.target.startswith(written), repr(written))
        remaining = self.target[len(written):]
        scores = [-10.0] * 512
        scores[0] = 1000.0  # NUL cannot occur in these JSON values.
        for token, data in self.vocab.items():
            if remaining.startswith(data):
                scores[token] = float(len(data)) + (token / 1000)
        return scores

    def definition(self, name: str, **parameters: str) -> FunctionDef:
        """Build unrelated function names and arbitrary parameter keys."""
        return FunctionDef.model_validate(
            {
                "name": name,
                "description": "Record supplied values.",
                "parameters": {
                    key: {"type": kind} for key, kind in parameters.items()
                },
                "returns": {"type": "string"},
            }
        )

    def run_record(
        self, prompt: str, fn: FunctionDef, values: dict[str, object]
    ) -> str:
        """Supply an independently serialized target for the fixed logits."""
        head = (
            f'{{"prompt":{json.dumps(prompt)},'
            f'"name":{json.dumps(fn.name)},"parameters":{{'
        )
        members = ",".join(
            json.dumps(key).replace("<", r"\u003c")
            + ": "
            + (
                f"{value}.0"
                if type(value) is int
                else json.dumps(value, ensure_ascii=False)
            )
            for key, value in values.items()
        )
        expected = head.replace("<", r"\u003c") + members + "}}"
        self.target = expected.encode()
        raw, _ = generation.generate_call_json(
            prompt,
            fn,
            model=self.model,
            vocab=self.vocab,
        )
        return raw

    def test_fixed_ids_commit_individually_after_one_forward(self) -> None:
        """Method B: each fixed ID follows one forward on the full prefix."""
        self.target = b'{"x": '
        context = [80]
        ids = generation._commit_fixed_text(
            '{"x": ', context, model=self.model, vocab=self.vocab
        )
        self.assertEqual(ids, [123, 34, 120, 34, 58, 32])
        self.assertEqual(context, [80], "the caller owns the output IDs")
        self.assertEqual(
            self.contexts, [[80] + list(b'{"x": '[:end]) for end in range(6)]
        )

    def test_integer_is_returned_raw_with_number_slice(self) -> None:
        """Return raw text; normalization never enters the model context."""
        fn = self.definition("timer", minutes="number", repeat="boolean")
        self.target = (
            b'{"prompt":"t","name":"timer","parameters":'
            b'{"minutes": 10,"repeat": true}}'
        )
        raw, number_slot_slices = generation.generate_call_json(
            "t",
            fn,
            model=self.model,
            vocab=self.vocab,
        )
        self.assertEqual(raw, self.target.decode())
        self.assertEqual([raw[part] for part in number_slot_slices], ["10,"])
        text = normalize_call_json(raw, number_slot_slices)
        self.assertEqual(
            text,
            '{"prompt":"t","name":"timer","parameters":'
            '{"minutes": 10.0,"repeat": true}}',
        )
        # The context keeps what the model wrote; ".0" is never forwarded.
        outputs = [bytes(ids[1:]) for ids in self.contexts]
        self.assertIn(
            b'{"prompt":"t","name":"timer","parameters":'
            b'{"minutes": 10,"repeat": ',
            outputs,
        )
        self.assertFalse(any(b"10.0" in output for output in outputs))

    def test_number_slices_use_raw_character_positions(self) -> None:
        """Unicode and earlier fractions must not shift later number slots."""
        fn = self.definition(
            "record", a="number", s="string", flag="boolean", b="number"
        )
        self.target = (
            '{"prompt":"p","name":"record","parameters":'
            '{"a": 2,"s": "日本語🙂","flag": false,"b": -5}}'
        ).encode()
        raw, number_slot_slices = generation.generate_call_json(
            "p",
            fn,
            model=self.model,
            vocab=self.vocab,
        )
        self.assertEqual(len(number_slot_slices), 2)
        for number_slice in number_slot_slices:
            self.assertIsInstance(number_slice, slice)
            self.assertIsNone(number_slice.step)
        self.assertEqual(
            [raw[part] for part in number_slot_slices], ["2,", "-5}"]
        )
        self.assertLessEqual(
            number_slot_slices[0].stop, number_slot_slices[1].start
        )
        self.assertEqual(
            normalize_call_json(raw, number_slot_slices),
            '{"prompt":"p","name":"record","parameters":'
            '{"a": 2.0,"s": "日本語🙂","flag": false,"b": -5.0}}',
        )
        out = list(self.target)
        self.assertEqual(self.decodings[-1], out)
        self.assertEqual(
            self.contexts, [[80] + out[:i] for i in range(len(out))]
        )

    def test_no_number_slots_return_no_slices(self) -> None:
        """Empty parameters and string/boolean parameters need no edits."""
        for parameters, members in (
            ({}, ""),
            (
                {"s": "string", "flag": "boolean"},
                '"s": "日本語🙂","flag": true',
            ),
        ):
            with self.subTest(parameters=parameters):
                fn = self.definition("record", **parameters)
                self.target = (
                    '{"prompt":"p","name":"record","parameters":{'
                    + members
                    + "}}"
                ).encode()
                raw, number_slot_slices = generation.generate_call_json(
                    "p",
                    fn,
                    model=self.model,
                    vocab=self.vocab,
                )
                self.assertEqual(number_slot_slices, [])
                self.assertEqual(
                    normalize_call_json(raw, number_slot_slices),
                    self.target.decode(),
                )

    def test_pipeline_preserves_raw_tokens_and_exact_saved_json(self) -> None:
        """Pin forwarding, notifications and saved bytes before refactoring."""
        fn = self.definition("record", a="number", s="string", b="number")
        prompt = '数字12 "quote" \\ newline\n <|im_end|>'
        head = (
            '{"prompt":"\\u6570\\u5b5712 \\"quote\\" \\\\ newline\\n '
            '\\u003c|im_end|>","name":"record","parameters":{"a": '
        )
        middle = ',"s": "東2, -5} \\" \\\\ \\n \\u0031","b": '
        resources = runtime.GenerationAssets.model_construct(
            model=self.model,
            vocab=self.vocab,
            functions={"record": fn},
            trie=trie_from_paths({"record": (1,)}),
        )
        cases = (
            ("2", "2.0"),
            ("-5", "-5.0"),
            ("0", "0.0"),
            ("-0", "-0.0"),
            ("12.25", "12.25"),
            ("2.00", "2.00"),
            ("9007199254740992", "9007199254740992.0"),
        )
        # A float cannot hold these after ".0" is added: precision loss.
        for raw_number in (
            "9007199254740993",
            "123456789012345678901234567890",
        ):
            with self.subTest(rejected=raw_number):
                self.target = (head + raw_number + middle + "-0}}").encode()
                with (
                    patch.object(pipeline, "select_function", return_value=fn),
                    self.assertRaisesRegex(PromptError, "precision"),
                ):
                    pipeline.process_prompt(prompt, resources)
        for raw_number, saved_number in cases:
            with self.subTest(number=raw_number):
                self.contexts.clear()
                self.decodings.clear()
                raw = head + raw_number + middle + "-0}}"
                expected = head + saved_number + middle + "-0.0}}"
                self.target = raw.encode()
                progress = Mock(spec=GenerationProgress, available=True)
                with patch.object(
                    pipeline, "select_function", return_value=fn
                ):
                    call = pipeline.process_prompt(
                        prompt,
                        resources,
                        progress=progress,
                    )
                self.assertEqual(call._generated_json, expected)
                out = list(self.target)
                self.assertEqual(self.decodings[-1], out)
                self.assertEqual(
                    self.contexts, [[80] + out[:i] for i in range(len(out))]
                )
                events = progress.output_token.call_args_list
                self.assertEqual([c.kwargs["token"] for c in events], out)
                self.assertEqual(
                    b"".join(c.args[0] for c in events), self.target
                )
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "calls.json"
                    save_results(path, [call])
                    # parse_float=str keeps each number's written spelling.
                    self.assertEqual(
                        json.loads(path.read_text(), parse_float=str),
                        [json.loads(expected, parse_float=str)],
                    )

    def test_special_prompt_name_keys_preserve_meaning(self) -> None:
        """Copied special spellings must never become control IDs."""
        fn = self.definition(
            "new" + self.special, **{"key" + self.special: "number"}
        )
        prompt = "前" + self.special + '後 "quote" \\ newline\n'
        text = self.run_record(prompt, fn, {"key" + self.special: 7})
        self.assertEqual(
            json.loads(text),
            {
                "prompt": prompt,
                "name": fn.name,
                "parameters": {"key" + self.special: 7},
            },
        )
        self.assertTrue(self.contexts)
        self.assertTrue(all(400 not in ids for ids in self.contexts))
        self.assertTrue(
            all(self.special not in text for text in self.encodings[1:])
        )

    def test_unicode_escapes_multiple_types_and_boundaries(self) -> None:
        """Values and their terminators enter the next slot's exact context."""
        fn = self.definition(
            "記録_v2", number="number", text="string", flag="boolean"
        )
        values: dict[str, object] = {
            "number": -12.5,
            "text": '𠮷🦄 日本語 <|im_end|> " \\ \n',
            "flag": False,
        }
        text = self.run_record("José あ 🙂", fn, values)
        self.assertEqual(
            json.loads(text),
            {"prompt": "José あ 🙂", "name": fn.name, "parameters": values},
        )
        call = pipeline.validate_call(text, fn, "José あ 🙂")
        self.assertEqual(call.parameters, values)
        self.assertEqual(self.decode(self.decodings[-1]), text)
        self.assertTrue(
            any(
                b'"number": -12.5,"text": ' in bytes(ids[1:])
                for ids in self.contexts
            )
        )
        self.assertTrue(
            any(b'"flag": ' in bytes(ids[1:]) for ids in self.contexts)
        )
        # Method B: one forward per output ID, on P plus every earlier ID.
        out = self.decodings[-1]
        self.assertEqual(
            self.contexts, [[ord("P")] + out[:i] for i in range(len(out))]
        )

    def test_zero_parameters_forward_every_fixed_token(self) -> None:
        """A complete fixed record preserves special-looking text."""
        fn = self.definition("heartbeat")
        prompt = "<|im_end|><|im_start|><think><|endoftext|>"
        text = self.run_record(prompt, fn, {})
        self.assertEqual(
            json.loads(text),
            {"prompt": prompt, "name": "heartbeat", "parameters": {}},
        )
        out = self.decodings[-1]
        self.assertEqual(
            self.contexts, [[ord("P")] + out[:i] for i in range(len(out))]
        )
        self.assertNotIn("<", text)

    def test_duplicate_byte_tokens_keep_the_selected_id(self) -> None:
        """Equal displayed text must not cause re-encoding of selected IDs."""
        self.vocab.update({300: b"true,", 301: b"true,"})
        fn = self.definition("changed", flag="boolean", n="number")
        text = self.run_record("set true and 7", fn, {"flag": True, "n": 7})
        self.assertEqual(
            json.loads(text)["parameters"], {"flag": True, "n": 7}
        )
        self.assertTrue(any(301 in ids for ids in self.contexts))
        self.assertTrue(all(300 not in ids for ids in self.contexts))
        self.assertIn(301, self.decodings[-1])

    def test_encode_mismatch_fails_before_value_inference(self) -> None:
        fn = self.definition("f", n="number")
        with (
            patch.object(
                Model,
                "encode",
                side_effect=lambda text: [80] if text == "P" else [120],
            ),
            self.assertRaises(PromptError),
        ):
            generation.generate_call_json(
                "x", fn, model=self.model, vocab=self.vocab
            )
        self.assertEqual(self.contexts, [])

    def test_unknown_fixed_id_is_a_prompt_error(self) -> None:
        fn = self.definition("f")
        with (
            patch.object(
                Model,
                "encode",
                side_effect=lambda text: [80] if text == "P" else [999],
            ),
            self.assertRaises(PromptError),
        ):
            generation.generate_call_json(
                "x", fn, model=self.model, vocab=self.vocab
            )

    def test_sdk_decode_loss_is_a_prompt_error(self) -> None:
        fn = self.definition("f")
        with patch.object(Model, "decode", return_value=""):
            with self.assertRaises(PromptError):
                generation.generate_call_json(
                    "x", fn, model=self.model, vocab=self.vocab
                )

    def test_completed_value_decode_must_match_grammar_bytes(self) -> None:
        fn = self.definition("f", n="number")
        original = self.decode

        def corrupt(ids: list[int]) -> str:
            """Change a number after its byte grammar passed."""
            text = original(ids)
            return "8.0}" if text == "7.0}" else text

        with patch.object(Model, "decode", side_effect=corrupt):
            with self.assertRaises(PromptError):
                self.run_record("number 7", fn, {"n": 7})

    def test_full_decode_must_match_vocabulary_bytes(self) -> None:
        fn = self.definition("f")
        original = self.decode

        def corrupt(ids: list[int]) -> str:
            """Model a context-sensitive decode cleanup on the full record."""
            text = original(ids)
            return (
                text.replace('"prompt"', '"wrong"')
                if text.endswith("}}") and text.startswith("{")
                else text
            )

        with patch.object(Model, "decode", side_effect=corrupt):
            with self.assertRaises(PromptError):
                self.run_record("x", fn, {})

    def test_fixed_noncanonical_ids_are_preserved(self) -> None:
        """Correct decoded text does not require canonical token IDs."""
        self.vocab.update({300: b'{"a":', 301: b" true}", 302: b'{"a": true}'})
        self.target = b'{"a": true}'
        output: list[int] = []
        encodings = {'{"a":': [300], " true}": [301], '{"a": true}': [302]}
        with patch.object(Model, "encode", side_effect=encodings.__getitem__):
            output += generation._commit_fixed_text(
                '{"a":', [80] + output, model=self.model, vocab=self.vocab
            )
            output += generation._commit_fixed_text(
                " true}", [80] + output, model=self.model, vocab=self.vocab
            )
            self.assertNotEqual(output, self.model.encode('{"a": true}'))
        self.assertEqual(output, [300, 301])
        self.assertEqual(
            generation._decode_tokens_with_vocab_check(
                output, model=self.model, vocab=self.vocab
            ),
            '{"a": true}',
        )
        self.assertEqual(self.contexts, [[80], [80, 300]])

    def test_llm_number_literal_survives_validation_and_save(self) -> None:
        """Save exactly the model's decimal, including trailing zeros."""
        fn = self.definition("f", n="number")
        resources = runtime.GenerationAssets.model_construct(
            model=self.model,
            vocab=self.vocab,
            functions={"f": fn},
            trie=trie_from_paths({"f": (1,)}),
        )
        text = (
            '{"prompt":"request","name":"f","parameters":'
            '{"n":9007199254740992.00}}'
        )
        with (
            patch.object(pipeline, "select_function", return_value=fn),
            patch.object(
                pipeline, "generate_call_json", return_value=(text, [])
            ),
        ):
            call = pipeline.process_prompt("request", resources)
        self.assertEqual(call.parameters["n"], 9007199254740992.0)
        self.assertIs(type(call.parameters["n"]), float)
        self.assertEqual(call._generated_json, text)
        self.assertNotIn("_generated_json", call.model_dump())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            save_results(path, [call])
            saved = path.read_text(encoding="utf-8")
        self.assertIn("9007199254740992.00", saved)
        self.assertEqual(
            json.loads(saved, parse_float=Decimal)[0]["parameters"]["n"],
            Decimal("9007199254740992.00"),
        )

    def test_process_rejects_changed_prompt(self) -> None:
        fn = self.definition("f")
        resources = runtime.GenerationAssets.model_construct(
            model=self.model,
            vocab=self.vocab,
            functions={"f": fn},
            trie=trie_from_paths({"f": (1,)}),
        )
        with (
            patch.object(pipeline, "select_function", return_value=fn),
            patch.object(
                pipeline,
                "generate_call_json",
                return_value=(
                    '{"prompt":"changed","name":"f","parameters":{}}',
                    [],
                ),
            ),
            self.assertRaisesRegex(PromptError, "prompt"),
        ):
            pipeline.process_prompt("original", resources)

    def test_invalid_normalized_json_is_rejected(self) -> None:
        """Adding .0 must not bypass validation of the completed record."""
        fn = self.definition("f", n="number")
        head = '{"prompt":"x","name":"f","parameters":{"n": '
        raw = head + "2}"  # missing the record's closing brace
        number_slot_slices = [slice(len(head), len(raw))]
        self.assertEqual(
            normalize_call_json(raw, number_slot_slices), head + "2.0}"
        )
        resources = runtime.GenerationAssets.model_construct(
            model=self.model,
            vocab=self.vocab,
            functions={"f": fn},
            trie=trie_from_paths({"f": (1,)}),
        )
        with (
            patch.object(pipeline, "select_function", return_value=fn),
            patch.object(
                pipeline,
                "generate_call_json",
                return_value=(raw, number_slot_slices),
            ),
            self.assertRaisesRegex(PromptError, "invalid output"),
        ):
            pipeline.process_prompt("x", resources)
