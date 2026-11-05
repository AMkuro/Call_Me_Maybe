"""Exercise reusable execution components independently of the CLI."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.inputs import InputFiles
from src.processing import CallProcessor
from src.storage import ResultStore
from src.validation import FunctionCall, FunctionDefinition
from src.workflow import Application


class ComponentTests(unittest.TestCase):
    """Check component ownership and failure behavior with real file I/O."""

    def setUp(self) -> None:
        """Provide a dynamic function and independent input/output paths."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.functions = self.root / "functions.json"
        self.prompts = self.root / "prompts.json"
        self.output = self.root / "results.json"
        self.definition = FunctionDefinition.model_validate({
            "name": "remember", "description": "Remember a string.",
            "parameters": {"text": {"type": "string"}},
            "returns": {"type": "null"},
        })
        self.functions.write_text(json.dumps([
            self.definition.model_dump(),
        ]), encoding="utf-8")
        self.prompts.write_text('[{"prompt":"one"}]', encoding="utf-8")

    def result(self, prompt: str) -> FunctionCall:
        """Create a typed result for testing storage without generation."""
        return FunctionCall(
            prompt=prompt, name="remember", parameters={"text": prompt},
        )

    def read_output(self) -> list[dict[str, Any]]:
        """Read the published JSON using an ordinary reader."""
        result: list[dict[str, Any]] = json.loads(
            self.output.read_text(encoding="utf-8"),
        )
        return result

    def test_inputs_load_without_generating_or_creating_output(self) -> None:
        """Input loading is usable without a CLI or result store."""
        inputs = InputFiles(
            functions_path=self.functions, input_path=self.prompts,
        )
        definitions, records = inputs.load()
        self.assertEqual(definitions[0].name, "remember")
        self.assertEqual(records, [{"prompt": "one"}])
        self.assertFalse(self.output.exists())
        self.prompts.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "load prompts") as caught:
            inputs.load()
        self.assertIsInstance(caught.exception.__cause__, json.JSONDecodeError)
        self.assertIn(str(self.prompts), str(caught.exception))

    def test_processor_is_reusable_without_files_or_cli(self) -> None:
        """One processor isolates mutable definitions on each generation."""
        names: list[str] = []

        def generate(
            prompt: str, definitions: list[FunctionDefinition],
        ) -> str:
            """Observe then modify the received copy of the function set."""
            names.append(definitions[0].name)
            definitions[0].name = "mutated"
            return json.dumps({
                "prompt": prompt, "name": "remember",
                "parameters": {"text": prompt},
            })

        processor = CallProcessor(
            definitions=[self.definition], generate=generate,
        )
        for index, prompt in enumerate(("one", "two"), start=1):
            result = processor.process({"prompt": prompt}, index)
            self.assertEqual(result.prompt, prompt)
            self.assertEqual(result.parameters, {"text": prompt})
        self.assertEqual(names, ["remember", "remember"])
        self.assertEqual(self.definition.name, "remember")
        self.assertFalse(self.output.exists())

    def test_processor_errors_preserve_stage_and_original_cause(self) -> None:
        """Context is attached at the failing operation, outside the batch."""
        failure = ValueError("no permitted token")

        def generate(
            prompt: str, definitions: list[FunctionDefinition],
        ) -> str:
            """Simulate a constrained generator that cannot complete."""
            raise failure

        processor = CallProcessor(
            definitions=[self.definition], generate=generate,
        )
        with self.assertRaisesRegex(RuntimeError, "input 7 generation") as ctx:
            processor.process({"prompt": "seven"}, 7)
        self.assertIs(ctx.exception.__cause__, failure)
        with self.assertRaisesRegex(RuntimeError, "input 8 input validation"):
            processor.process({"prompt": False}, 8)

    def test_failed_append_does_not_advance_store_state(self) -> None:
        """A subsequent save must not publish a previously failed result."""
        store = ResultStore(path=self.output)
        store.initialize((self.functions, self.prompts))
        store.append(self.result("one"), 1)
        failure = OSError("disk full")
        with patch("src.storage.os.replace", side_effect=failure), \
                self.assertRaisesRegex(RuntimeError, "input 2 save output"):
            store.append(self.result("failed"), 2)
        self.assertEqual(self.read_output(), [{
            "prompt": "one", "name": "remember",
            "parameters": {"text": "one"},
        }])
        store.append(self.result("three"), 3)
        self.assertEqual(
            [result["prompt"] for result in self.read_output()],
            ["one", "three"],
        )

    def test_store_initialization_resets_disk_and_owned_state(self) -> None:
        """Reusing a store for a new batch must not bring back old results."""
        store = ResultStore(path=self.output)
        paths = (self.functions, self.prompts)
        store.initialize(paths)
        store.append(self.result("old"), 1)
        store.initialize(paths)
        self.assertEqual(self.read_output(), [])
        store.append(self.result("fresh"), 1)
        self.assertEqual(self.read_output(), [{
            "prompt": "fresh", "name": "remember",
            "parameters": {"text": "fresh"},
        }])

    def test_application_does_not_reuse_prior_batch(self) -> None:
        """The application composes fresh inputs with reset output state."""
        def generate(
            prompt: str, definitions: list[FunctionDefinition],
        ) -> str:
            """Return a test-only completed JSON result."""
            return self.result(prompt).model_dump_json()

        application = Application(
            inputs=InputFiles(
                functions_path=self.functions, input_path=self.prompts,
            ),
            output=ResultStore(path=self.output), generate=generate,
        )
        self.assertEqual(application.run(), 0)
        self.prompts.write_text('[{"prompt":"two"}]', encoding="utf-8")
        self.assertEqual(application.run(), 0)
        self.assertEqual(self.read_output(), [{
            "prompt": "two", "name": "remember", "parameters": {"text": "two"},
        }])

    def test_application_reports_save_interruption_and_stops(self) -> None:
        """Save interruption follows the same failure policy as generation."""
        processed: list[str] = []

        def generate(
            prompt: str, definitions: list[FunctionDefinition],
        ) -> str:
            """Record which prompts reached generation."""
            processed.append(prompt)
            return self.result(prompt).model_dump_json()

        replace = os.replace

        def interrupt_replace(source: Path, destination: Path) -> None:
            """Interrupt publication of the two-result array."""
            values = json.loads(source.read_text(encoding="utf-8"))
            if len(values) == 2:
                raise KeyboardInterrupt
            replace(source, destination)

        self.prompts.write_text(
            '[{"prompt":"one"},{"prompt":"two"},{"prompt":"three"}]',
            encoding="utf-8",
        )
        application = Application(
            inputs=InputFiles(
                functions_path=self.functions, input_path=self.prompts,
            ),
            output=ResultStore(path=self.output), generate=generate,
        )
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), \
                patch("src.storage.os.replace", side_effect=interrupt_replace):
            self.assertEqual(application.run(), 130)
        self.assertEqual(processed, ["one", "two"])
        self.assertEqual(len(self.read_output()), 1)
        self.assertIn("interrupted: input 2 save output", stderr.getvalue())
        self.assertFalse(list(self.root.glob(".results.json.*.tmp")))
