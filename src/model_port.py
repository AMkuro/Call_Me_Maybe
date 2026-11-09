"""The model boundary. Only this file imports llm_sdk."""

from typing import Protocol


class ModelPort(Protocol):
    """What the rest of the program needs from a model.

    A test can pass a fake with fixed logits instead of the real model.
    """

    def encode(self, text: str) -> list[int]:
        """Return the token ids of ``text``."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Return the text of ``ids``."""
        ...

    def logits(self, ids: list[int]) -> list[float]:
        """Return the next-token logits after ``ids`` (one per token)."""
        ...

    def vocab_path(self) -> str:
        """Return the path of the vocabulary file."""
        ...


def load_model() -> ModelPort:
    """Create the real model from llm_sdk and check it (ADR-0007)."""
    raise NotImplementedError
