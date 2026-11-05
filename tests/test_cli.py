"""Check sequential execution and real on-disk JSON without model inference."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.cli import main
from src.storage import save_results
from src.validation import FunctionDefinition


class CliTests(unittest.TestCase):
    """Test fail-fast behavior through the same entry point as the CLI."""

    def setUp(self) -> None:
        """Create an independent input set and output for every test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.functions = self.root / "functions.json"
        self.prompts = self.root / "prompts.json"
        self.output = self.root / "nested" / "calls.json"
        self.functions.write_text(json.dumps([{
            "name": "record", "description": "Record a numeric value.",
            "parameters": {"value": {"type": "number"}},
            "returns": {"type": "null"},
        }]), encoding="utf-8")
        self.prompts.write_text(json.dumps([
            {"prompt": "first"}, {"prompt": "second"}, {"prompt": "third"},
        ]), encoding="utf-8")
        self.argv = [
            "--functions_definition", str(self.functions),
            "--input", str(self.prompts), "--output", str(self.output),
        ]
        self.stderr = io.StringIO()
        self.calls: list[str] = []

    def generate(
        self, prompt: str, definitions: list[FunctionDefinition],
    ) -> str:
        """Stand in only for future LLM generation, keeping all I/O real."""
        self.calls.append(prompt)
        return json.dumps({
            "prompt": prompt, "name": "record", "parameters": {"value": 9},
        })

    def run_main(self, **kwargs: Any) -> int:
        """Capture diagnostics from the production main function."""
        with contextlib.redirect_stderr(self.stderr):
            return main(self.argv, **kwargs)

    def read_output(self) -> list[dict[str, Any]]:
        """Read the published file using an ordinary JSON reader."""
        value: list[dict[str, Any]] = json.loads(
            self.output.read_text(encoding="utf-8"),
        )
        return value

    def test_each_success_is_saved_before_next_generation(self) -> None:
        """The file is a complete array at every observable save boundary."""
        snapshots: list[list[dict[str, Any]]] = []

        def generate(
            prompt: str, definitions: list[FunctionDefinition],
        ) -> str:
            """Observe the file at the start of every generation."""
            snapshots.append(self.read_output())
            self.assertEqual(definitions[0].name, "record")
            return self.generate(prompt, definitions)

        self.assertEqual(self.run_main(generate=generate), 0)
        first = {"prompt": "first", "name": "record",
                 "parameters": {"value": 9}}
        second = {"prompt": "second", "name": "record",
                  "parameters": {"value": 9}}
        third = {"prompt": "third", "name": "record",
                 "parameters": {"value": 9}}
        self.assertEqual(snapshots, [[], [first], [first, second]])
        self.assertEqual(self.read_output(), [first, second, third])
        self.assertEqual(self.stderr.getvalue(), "")

    def test_generation_errors_keep_success_and_skip_later_input(self) -> None:
        """SDK, constraint, and token-limit failures are fail-fast errors."""
        for failure in (
            RuntimeError("SDK inference failed"),
            ValueError("no allowed token"),
            TimeoutError("token limit exceeded"),
        ):
            self.calls.clear()
            self.stderr.seek(0)
            self.stderr.truncate()

            def generate(
                prompt: str, definitions: list[FunctionDefinition],
            ) -> str:
                """Fail on input two, after input one has been saved."""
                text = self.generate(prompt, definitions)
                if prompt == "second":
                    raise failure
                return text

            with self.subTest(failure=failure):
                self.assertEqual(self.run_main(generate=generate), 1)
                self.assertEqual(self.calls, ["first", "second"])
                self.assertEqual(self.read_output(), [{
                    "prompt": "first", "name": "record",
                    "parameters": {"value": 9},
                }])
                error = self.stderr.getvalue()
                self.assertIn("input 2", error)
                self.assertIn("generation", error)
                self.assertIn(str(failure), error)
                self.assertNotIn("Traceback", error)

    def test_invalid_generated_results_are_never_published(self) -> None:
        """All final validation failures retain only the preceding success."""
        valid = {"prompt": "second", "name": "record",
                 "parameters": {"value": 9}}
        cases: list[object] = [
            "{", "prose", [],
            {**valid, "prompt": "changed"},
            {**valid, "name": "unknown"},
            {**valid, "parameters": {}},
            {**valid, "parameters": {"value": "9"}},
            {**valid, "parameters": {"value": 9, "extra": 1}},
            {**valid, "metric": 1},
        ]
        for invalid in cases:
            self.calls.clear()

            def generate(
                prompt: str, definitions: list[FunctionDefinition],
            ) -> str:
                """Return a malformed second result, using real validation."""
                text = self.generate(prompt, definitions)
                if prompt == "second":
                    return invalid if isinstance(invalid, str) else json.dumps(
                        invalid,
                    )
                return text

            with self.subTest(invalid=invalid):
                self.assertEqual(self.run_main(generate=generate), 1)
                self.assertEqual(self.calls, ["first", "second"])
                self.assertEqual(len(self.read_output()), 1)
                self.assertEqual(self.read_output()[0]["prompt"], "first")
        self.assertIn("result validation", self.stderr.getvalue())

    def test_later_invalid_input_keeps_prior_success(self) -> None:
        """Input records are checked sequentially, including type errors."""
        self.prompts.write_text(
            '[{"prompt":"first"},{"prompt":42},{"prompt":"third"}]',
            encoding="utf-8",
        )
        self.assertEqual(self.run_main(generate=self.generate), 1)
        self.assertEqual(self.calls, ["first"])
        self.assertEqual(len(self.read_output()), 1)
        self.assertIn("input 2", self.stderr.getvalue())
        self.assertIn("input validation", self.stderr.getvalue())

    def test_previous_run_results_do_not_survive_initialization(self) -> None:
        """Even a first-item failure leaves an empty current-run output."""
        self.output.parent.mkdir()
        self.output.write_text('[{"old":"result"}]', encoding="utf-8")
        self.assertEqual(self.run_main(), 1)
        self.assertEqual(self.read_output(), [])
        self.assertIn("not implemented", self.stderr.getvalue().lower())

    def test_bad_files_fail_before_generation_with_clear_errors(self) -> None:
        """Bad input files report paths and stop before generation."""
        for path in (self.functions, self.prompts):
            original = path.read_bytes()
            for content in (None, b"{", b"{}", b"\xff"):
                self.calls.clear()
                self.stderr.seek(0)
                self.stderr.truncate()
                if content is None:
                    path.unlink()
                else:
                    path.write_bytes(content)
                with self.subTest(path=path.name, content=content):
                    self.assertEqual(self.run_main(generate=self.generate), 1)
                    self.assertEqual(self.calls, [])
                    self.assertEqual(self.read_output(), [])
                    self.assertIn(str(path), self.stderr.getvalue())
                path.write_bytes(original)
        with patch.object(Path, "read_text", side_effect=PermissionError(
            "input denied",
        )):
            self.assertEqual(self.run_main(generate=self.generate), 1)
        self.assertIn("PermissionError", self.stderr.getvalue())

    def test_empty_prompt_list_is_successful_without_generation(self) -> None:
        """An empty batch initializes a valid empty output array."""
        self.prompts.write_text("[]", encoding="utf-8")
        self.assertEqual(self.run_main(generate=self.generate), 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.read_output(), [])

    def test_save_failure_stops_and_preserves_published_array(self) -> None:
        """Failed replacement on input two retains only input one's output."""
        replace = os.replace
        replacements = 0

        def fail_second_result(source: str, destination: Path) -> None:
            """Allow initialization and input one; deny the next replace."""
            nonlocal replacements
            replacements += 1
            if replacements == 3:
                raise OSError("disk write denied")
            replace(source, destination)

        with patch("src.storage.os.replace", side_effect=fail_second_result):
            self.assertEqual(self.run_main(generate=self.generate), 1)
        self.assertEqual(self.calls, ["first", "second"])
        self.assertEqual(self.read_output(), [{
            "prompt": "first", "name": "record", "parameters": {"value": 9},
        }])
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])
        error = self.stderr.getvalue()
        self.assertIn("save", error)
        self.assertIn("input 2", error)
        self.assertIn(str(self.output), error)
        self.assertIn("disk write denied", error)

    def test_initialization_failure_prevents_generation(self) -> None:
        """An unwritable output reports its path and stops before the model."""
        self.output.parent.write_text("a file", encoding="utf-8")
        self.assertEqual(self.run_main(generate=self.generate), 1)
        self.assertEqual(self.calls, [])
        self.assertIn("initialize output", self.stderr.getvalue())
        self.assertIn(str(self.output), self.stderr.getvalue())

    def test_partial_temporary_write_never_corrupts_output(self) -> None:
        """A mid-write error cleans the temporary file and preserves output."""
        save_results(self.output, [])

        def fail_during_dump(
            value: object, stream: Any, **kwargs: Any,
        ) -> None:
            """Simulate a device failing after accepting part of the JSON."""
            stream.write("[")
            raise OSError("disk full")

        with patch("src.storage.json.dump", side_effect=fail_during_dump), \
                self.assertRaisesRegex(OSError, "disk full"):
            save_results(self.output, [{"prompt": "first"}])
        self.assertEqual(self.read_output(), [])
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_interrupt_preserves_results_and_returns_130(self) -> None:
        """Ctrl-C never appends a failed generation or processes input 3."""
        def generate(
            prompt: str, definitions: list[FunctionDefinition],
        ) -> str:
            """Interrupt after saving input one."""
            text = self.generate(prompt, definitions)
            if prompt == "second":
                raise KeyboardInterrupt
            return text

        self.assertEqual(self.run_main(generate=generate), 130)
        self.assertEqual(self.calls, ["first", "second"])
        self.assertEqual(len(self.read_output()), 1)
        self.assertIn("input 2", self.stderr.getvalue())
        self.assertIn("interrupted", self.stderr.getvalue())

    def test_interrupt_during_save_preserves_published_file(self) -> None:
        """Interrupting replacement leaves the prior JSON and no temp file."""
        save_results(self.output, [])
        with patch("src.storage.os.replace", side_effect=KeyboardInterrupt), \
                self.assertRaises(KeyboardInterrupt):
            save_results(self.output, [{"prompt": "first"}])
        self.assertEqual(self.read_output(), [])
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_output_cannot_alias_an_input(self) -> None:
        """Direct, relative, symlink, and hardlink aliases preserve inputs."""
        for source in (self.functions, self.prompts):
            original = source.read_bytes()
            symlink = self.root / "symlink.json"
            hardlink = self.root / "hardlink.json"
            symlink.symlink_to(source)
            hardlink.hardlink_to(source)
            for output in (
                source, source.parent / "nested" / ".." / source.name,
                symlink, hardlink,
            ):
                argv = [*self.argv[:-1], str(output)]
                with self.subTest(output=output), \
                        contextlib.redirect_stderr(self.stderr):
                    self.assertEqual(main(argv, generate=self.generate), 1)
                self.assertEqual(source.read_bytes(), original)
            symlink.unlink()
            hardlink.unlink()
        self.assertEqual(self.calls, [])
        self.assertIn("input", self.stderr.getvalue())

    def test_generator_cannot_modify_validation_definitions(self) -> None:
        """Copied definitions prevent mutation bypassing final validation."""
        def generate(
            prompt: str, definitions: list[FunctionDefinition],
        ) -> str:
            """Try to substitute a new candidate name while generating."""
            definitions[0].name = "substituted"
            return json.dumps({
                "prompt": prompt, "name": "substituted", "parameters": {},
            })

        self.assertEqual(self.run_main(generate=generate), 1)
        self.assertEqual(self.read_output(), [])

    def test_default_paths_and_subject_output_alias(self) -> None:
        """Both the documented --output and synopsis -output options work."""
        (self.root / "data/input").mkdir(parents=True)
        (self.root / "data/input/functions_definition.json").write_bytes(
            self.functions.read_bytes(),
        )
        (self.root / "data/input/function_calling_tests.json").write_bytes(
            self.prompts.read_bytes(),
        )
        previous_directory = Path.cwd()
        try:
            os.chdir(self.root)
            self.assertEqual(main([], generate=self.generate), 0)
            result = json.loads(Path(
                "data/output/function_calls.json",
            ).read_text(encoding="utf-8"))
            self.assertEqual(len(result), 3)
        finally:
            os.chdir(previous_directory)
        self.argv[-2] = "-output"
        self.assertEqual(self.run_main(generate=self.generate), 0)

    def test_module_entrypoint_help_errors_and_unimplemented_generation(
        self,
    ) -> None:
        """Real subprocess exit codes need neither an SDK nor model weights."""
        project_root = Path(__file__).resolve().parents[1]
        for argv, status, message in (
            (["--help"], 0, "--functions_definition"),
            (["--unknown"], 2, "unrecognized arguments"),
            (self.argv, 1, "not implemented"),
        ):
            result = subprocess.run(
                [sys.executable, "-m", "src", *argv], cwd=project_root,
                capture_output=True, text=True, check=False,
            )
            with self.subTest(argv=argv):
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertIn(message, result.stdout + result.stderr)
                self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(self.read_output(), [])
