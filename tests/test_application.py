"""Direct checks of application contracts, without loading a model."""

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from pydantic import ValidationError

from src import __main__ as entry
from src.cli import parse_args
from src.decoder import choose_token, generate
from src.errors import PromptError, StartupError
from src.loader import load_functions, load_prompts
from src.model_port import Model
from src import pipeline
from src.pipeline import validate_call
from src.schema import FunctionCall, FunctionDef, build_parameters_model
from src.skeleton import Forced, ValueSlot, build_skeleton
from src.writer import save_results


def definition(**parameters: str) -> FunctionDef:
    """Return a definition with arbitrary parameter names and types."""
    return FunctionDef.model_validate({
        'name': 'dynamic', 'description': 'A test operation.',
        'parameters': {name: {'type': kind}
                       for name, kind in parameters.items()},
        'returns': {'type': 'string'},
    })


class InputTests(unittest.TestCase):
    """Validate inputs before any SDK load or inference."""

    def test_cli_defaults_overrides_and_bad_options(self) -> None:
        """Default paths and explicit paths implement the Subject CLI."""
        self.assertEqual(parse_args([]).output,
                         Path('data/output/function_calls.json'))
        args = parse_args(['--input', 'in.json', '--output', 'out.json',
                           '--functions_definition', 'functions.json'])
        self.assertEqual(args.input, Path('in.json'))
        self.assertEqual(args.functions_definition, Path('functions.json'))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_args(['--unknown'])
        self.assertEqual(raised.exception.code, 2)

    def test_loaders_accept_dynamic_definitions_and_prompts(self) -> None:
        """Definitions are dynamic and prompt strings preserve their input."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            fn = definition(flag='boolean', text='string', count='number')
            path.write_text(json.dumps([fn.model_dump()]), encoding='utf-8')
            self.assertEqual(load_functions(path), {fn.name: fn})
            path.write_text('[{"prompt": "日本語"}, {"prompt": ""}]',
                            encoding='utf-8')
            self.assertEqual(load_prompts(path), ['日本語', ''])

    def test_empty_prompt_array_is_insufficient_input(self) -> None:
        """An empty array is an explicit startup error, not a zero-item job."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prompts.json'
            path.write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(StartupError, 'no prompts'):
                load_prompts(path)

    def test_empty_array_stops_before_model_and_output(self) -> None:
        """Real startup validation preserves old output or leaves it absent."""
        for existing in [True, False]:
            with self.subTest(existing=existing):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    functions = root / 'functions.json'
                    functions.write_text(json.dumps([
                        definition().model_dump(),
                    ]), encoding='utf-8')
                    prompts = root / 'prompts.json'
                    prompts.write_text('[]', encoding='utf-8')
                    output = root / 'results' / 'calls.json'
                    previous = b'["previous run"]\n'
                    if existing:
                        output.parent.mkdir()
                        output.write_bytes(previous)
                    stderr = io.StringIO()
                    with patch.object(
                        entry, 'Model', side_effect=AssertionError(
                            'model must not load for empty input'
                        ),
                    ) as model, contextlib.redirect_stderr(stderr):
                        status = entry.main([
                            '--functions_definition', str(functions),
                            '--input', str(prompts), '--output', str(output),
                        ])
                    self.assertEqual(status, 1)
                    self.assertIn('no prompts', stderr.getvalue())
                    model.assert_not_called()
                    if existing:
                        self.assertEqual(output.read_bytes(), previous)
                    else:
                        self.assertFalse(output.parent.exists())

    def test_bad_function_files_are_startup_errors(self) -> None:
        """Missing, malformed or unsupported definitions fail clearly."""
        fn = definition().model_dump()
        invalid = ['{', '{}', '[]', json.dumps([fn, fn]),
                   json.dumps([{**fn, 'name': ''}]),
                   json.dumps([{**fn, 'name': '   '}]),
                   json.dumps([{**fn, 'parameters': {
                       'a': {'type': 'array'},
                   }}])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'functions.json'
            with self.assertRaises(StartupError):
                load_functions(path)
            for text in invalid:
                with self.subTest(text=text):
                    path.write_text(text, encoding='utf-8')
                    with self.assertRaises(StartupError):
                        load_functions(path)

    def test_bad_prompt_files_are_startup_errors(self) -> None:
        """Every prompt must be a string in the expected array structure."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prompts.json'
            for text in ['{', '{}', '[{}]', '[{"prompt": 3}]', '[null]']:
                with self.subTest(text=text):
                    path.write_text(text, encoding='utf-8')
                    with self.assertRaises(StartupError):
                        load_prompts(path)


class StructureTests(unittest.TestCase):
    """Check forced structure and dynamic output validation directly."""

    def test_skeleton_owns_each_delimiter_once(self) -> None:
        """Only slots supply commas or the parameters' closing brace."""
        fn = definition(**{'quote"': 'string', 'b': 'number'})
        parts = build_skeleton(fn)
        self.assertEqual(parts, [
            Forced(text=', "parameters": {"quote\\\"": '),
            ValueSlot(name='quote"', type='string', terminator=','),
            Forced(text=' "b": '),
            ValueSlot(name='b', type='number', terminator='}'),
            Forced(text='}'),
        ])
        self.assertEqual(build_skeleton(definition()),
                         [Forced(text=', "parameters": {}}')])

    def test_arbitrary_parameter_names_survive_validation(self) -> None:
        """Pydantic attribute names never replace dynamic JSON keys."""
        fn = definition(**{'_private': 'number', 'model_dump': 'string',
                           'a-b': 'boolean', '': 'string'})
        parameters = {'_private': 2, 'model_dump': 'yes', 'a-b': True, '': ''}
        result = build_parameters_model(fn).model_validate(parameters)
        self.assertEqual(result.model_dump(by_alias=True), parameters)
        call = validate_call(json.dumps({
            'prompt': 'test', 'name': fn.name, 'parameters': parameters,
        }), fn)
        self.assertEqual(call.parameters, parameters)
        self.assertIs(type(call.parameters['_private']), int)

    def test_missing_extra_and_nonfinite_values_are_rejected(self) -> None:
        """Generated records cannot omit fields or serialize NaN/Infinity."""
        fn = definition(a='number')
        for parameters in [{}, {'a': 1, 'other': 2}, {'a': math.inf}]:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValidationError):
                    build_parameters_model(fn).model_validate(parameters)
        with self.assertRaises(ValidationError):
            FunctionCall.model_validate({
                'prompt': 'x', 'name': 'f', 'parameters': {}, 'other': True,
            })


class DecoderTests(unittest.TestCase):
    """Constrain real arrays and exercise the loop with fixed logits."""

    def test_mask_uses_highest_allowed_score_without_mutation(self) -> None:
        """Invalid maxima and allowed-token order cannot change the result."""
        logits = [100.0, 3.0, 8.0, 2.0]
        self.assertEqual(choose_token(logits, [3, 1, 2]), 2)
        self.assertEqual(logits, [100.0, 3.0, 8.0, 2.0])
        self.assertEqual(choose_token([0.0, 1.0, 1.0], [2, 1]), 1)

    def test_invalid_or_impossible_masks_report_prompt_error(self) -> None:
        """A decoder never falls through to an invalid vocabulary index."""
        for logits, allowed in [([1.0], []), ([1.0], [-1]), ([1.0], [1]),
                                ([math.nan], [0]), ([math.inf], [0]),
                                ([-math.inf], [0])]:
            with self.subTest(logits=logits, allowed=allowed):
                with self.assertRaises(PromptError):
                    choose_token(logits, allowed)

    def test_malformed_sdk_scores_report_prompt_error(self) -> None:
        """A malformed array never returns a token outside the mask."""
        malformed: list[Any] = [[[1, 2], [3, 4]], ['not a number'], [[]]]
        for logits in malformed:
            with self.subTest(logits=logits):
                with self.assertRaises(PromptError):
                    choose_token(logits, [0])

    def test_forcing_completion_limit_and_input_preservation(self) -> None:
        """Forced tokens need no inference; completion wins at the limit."""
        calls: list[list[int]] = []

        def logits(ids: list[int]) -> list[float]:
            """Supply fixed scores and record the actual input sequence."""
            calls.append(ids)
            return [20.0, 1.0, 3.0]

        def allowed(ids: list[int]) -> list[int] | None:
            """Force one token, then choose, then terminate."""
            return [[1], [1, 2], None][len(ids)]

        prefix = [7]
        self.assertEqual(generate(logits, prefix, allowed, 2), [1, 2])
        self.assertEqual(calls, [[7, 1]])
        self.assertEqual(prefix, [7])
        with self.assertRaises(PromptError):
            generate(logits, prefix, lambda ids: [1], 2)
        with self.assertRaises(PromptError):
            generate(logits, prefix, lambda ids: [], 2)

    def test_repeated_boundary_cases_have_clear_limit_errors(self) -> None:
        """A value may finish at the cap; unfinished values fail explicitly."""
        for length in [1, 255, 256, 257, 1024]:
            for repeat in range(3):
                with self.subTest(length=length, repeat=repeat):
                    def allowed(ids: list[int]) -> list[int] | None:
                        """Complete only after the target's actual length."""
                        return None if len(ids) == length else [0, 1]

                    if length <= 256:
                        result = generate(lambda ids: [0.0, 1.0], [],
                                          allowed, 256)
                        self.assertEqual(len(result), length)
                    else:
                        with self.assertRaisesRegex(
                            PromptError, r'token limit 256.*terminator',
                        ):
                            generate(lambda ids: [0.0, 1.0], [], allowed, 256)

    def test_value_errors_identify_the_parameter(self) -> None:
        """Limit errors explain which typed parameter could not complete."""
        model = object.__new__(Model)
        slot = ValueSlot(name='source_string', type='string', terminator='}')
        with patch.object(pipeline, 'generate', side_effect=PromptError(
            'token limit 256 reached before a terminator'
        )):
            with self.assertRaisesRegex(
                PromptError, r"source_string.*string.*token limit 256",
            ):
                pipeline.generate_value(slot, [], model, {})


class WriterTests(unittest.TestCase):
    """Use the filesystem to check atomic cumulative output."""

    def test_save_creates_parents_and_preserves_json_strings(self) -> None:
        """Nested output paths, Unicode and empty results are supported."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'calls.json'
            call = FunctionCall(prompt='日本語\n"', name='dynamic',
                                parameters={'s': '\\', 'n': 2.0})
            save_results(path, [call])
            self.assertEqual(json.loads(path.read_text()), [call.model_dump()])
            save_results(path, [])
            self.assertEqual(json.loads(path.read_text()), [])

    def test_replace_failure_keeps_output_and_removes_temp(self) -> None:
        """A failed final replace cannot destroy the last good file."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calls.json'
            path.write_text('["previous"]', encoding='utf-8')
            with patch('os.replace', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    save_results(path, [])
            self.assertEqual(path.read_text(), '["previous"]')
            self.assertEqual(list(Path(directory).iterdir()), [path])


class ModelBoundaryTests(unittest.TestCase):
    """Mock only the external SDK to check conversion and error boundaries."""

    def test_public_sdk_methods_return_plain_data(self) -> None:
        """The wrapper flattens encode without exposing a tensor to callers."""
        sdk = Mock()
        sdk.encode.return_value.tolist.return_value = [[2, 3]]
        sdk.decode.return_value = '日本語'
        sdk.get_logits_from_input_ids.return_value = [1.0, 2.0]
        sdk.get_path_to_vocab_file.return_value = '/tmp/vocab.json'
        factory = Mock(return_value=sdk)
        with patch.dict(sys.modules, {
            'llm_sdk': SimpleNamespace(Small_LLM_Model=factory),
        }):
            model = Model()
        self.assertEqual(model.encode('text'), [2, 3])
        self.assertEqual(model.decode([2, 3]), '日本語')
        self.assertEqual(model.logits([2, 3]), [1.0, 2.0])
        self.assertEqual(model.vocab_path(), '/tmp/vocab.json')

    def test_sdk_failures_report_original_cause(self) -> None:
        """Startup and per-prompt exceptions keep the SDK's reason."""
        factory = Mock(side_effect=RuntimeError('load failed'))
        with patch.dict(sys.modules, {
            'llm_sdk': SimpleNamespace(Small_LLM_Model=factory),
        }):
            with self.assertRaisesRegex(StartupError, 'load failed'):
                Model()
        sdk = Mock()
        with patch.dict(sys.modules, {
            'llm_sdk': SimpleNamespace(Small_LLM_Model=lambda **kwargs: sdk),
        }):
            model = Model()
        sdk.encode.side_effect = RuntimeError('encode failed')
        sdk.decode.side_effect = RuntimeError('decode failed')
        sdk.get_logits_from_input_ids.side_effect = RuntimeError('forward')
        sdk.get_path_to_vocab_file.side_effect = OSError('vocab missing')
        with self.assertRaisesRegex(PromptError, 'encode failed'):
            model.encode('x')
        with self.assertRaisesRegex(PromptError, 'decode failed'):
            model.decode([1])
        with self.assertRaisesRegex(PromptError, 'forward'):
            model.logits([1])
        with self.assertRaisesRegex(StartupError, 'vocab missing'):
            model.vocab_path()
