"""Own input file locations and load the shared dynamic data contract."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from src.operations import operation
from src.validation import FunctionDefinition, load_definitions, load_prompts


class InputFiles(BaseModel):
    """Load definitions and raw prompt records from a pair of input paths."""

    model_config = ConfigDict(strict=True, extra="forbid")

    functions_path: Path
    input_path: Path

    def load(self) -> tuple[list[FunctionDefinition], list[JsonValue]]:
        """Load both files before generation, retaining sequential prompts."""
        with operation(f"load function definitions {self.functions_path}"):
            definitions = load_definitions(self.functions_path)
        with operation(f"load prompts {self.input_path}"):
            records = load_prompts(self.input_path)
        return definitions, records
