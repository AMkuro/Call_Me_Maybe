"""Process one prompt independently of the CLI, batch policy, and storage."""

from typing import Callable

from pydantic import BaseModel, ConfigDict, JsonValue

from src.validation import (
    FunctionCall,
    FunctionDefinition,
    Prompt,
    validate_result,
)

GenerateCall = Callable[[str, list[FunctionDefinition]], str]


def generate_call(prompt: str, definitions: list[FunctionDefinition]) -> str:
    """Fail explicitly until a real constrained LLM generator is integrated.

    Future adapters must return only a successfully completed final JSON text.
    SDK failures, early EOS, impossible constraints, or exceeded limits must
    raise exceptions, even if partial text happens to parse as valid JSON.
    The adapter owns model loading/reuse and internal generation metrics.
    """
    raise NotImplementedError(
        "LLM generation is not implemented; integrate a real constrained "
        "generation method before running nonempty batches"
    )


class CallProcessor(BaseModel):
    """Own the generator and function set for processing single records."""

    model_config = ConfigDict(strict=True, extra="forbid")

    definitions: list[FunctionDefinition]
    generate: GenerateCall = generate_call

    def process(self, record: JsonValue, index: int) -> FunctionCall:
        """Return a fully validated call or fail with its operation context."""
        stage = f"input {index} input validation"
        try:
            prompt = Prompt.model_validate(record).prompt
            stage = f"input {index} generation"
            text = self.generate(
                prompt,
                [item.model_copy(deep=True) for item in self.definitions],
            )
            stage = f"input {index} result validation"
            return validate_result(text, prompt, self.definitions)
        except KeyboardInterrupt as exc:
            raise KeyboardInterrupt(stage) from exc
        except Exception as exc:
            raise RuntimeError(
                f"{stage}: {type(exc).__name__}: {exc}"
            ) from exc
