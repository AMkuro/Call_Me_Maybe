"""Own input file locations and load the shared dynamic data contract."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from src.validation import FunctionDefinition, load_definitions, load_prompts


class InputFiles(BaseModel):
    """Load definitions and raw prompt records from a pair of input paths."""

    model_config = ConfigDict(strict=True, extra="forbid")

    functions_path: Path
    input_path: Path

    def load(self) -> tuple[list[FunctionDefinition], list[JsonValue]]:
        """Load both files before generation, retaining sequential prompts."""
        stage = f"load function definitions {self.functions_path}"
        try:
            definitions = load_definitions(self.functions_path)
            stage = f"load prompts {self.input_path}"
            records = load_prompts(self.input_path)
        except KeyboardInterrupt as exc:
            raise KeyboardInterrupt(stage) from exc
        except Exception as exc:
            raise RuntimeError(
                f"{stage}: {type(exc).__name__}: {exc}"
            ) from exc
        return definitions, records
