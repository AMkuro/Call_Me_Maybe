"""Coordinate input, per-record processing, and committed result storage."""

import sys

from pydantic import BaseModel, ConfigDict

from src.inputs import InputFiles
from src.processing import CallProcessor, GenerateCall, generate_call
from src.storage import ResultStore


class Application(BaseModel):
    """Compose execution components and apply the fail-fast batch policy."""

    model_config = ConfigDict(strict=True, extra="forbid")

    inputs: InputFiles
    output: ResultStore
    generate: GenerateCall = generate_call

    def run(self) -> int:
        """Report failures at the application boundary and return status."""
        try:
            self._run_batch()
        except KeyboardInterrupt as exc:
            print(f"interrupted: {exc}", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    def _run_batch(self) -> None:
        """Initialize output and coordinate collaborators for each record."""
        self.output.initialize((
            self.inputs.functions_path, self.inputs.input_path,
        ))
        definitions, records = self.inputs.load()
        processor = CallProcessor(
            definitions=definitions, generate=self.generate,
        )
        for index, record in enumerate(records, start=1):
            result = processor.process(record, index)
            self.output.append(result, index)
