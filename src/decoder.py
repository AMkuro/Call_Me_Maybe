"""The constrained-decoding loop shared by name and value generation."""

from collections.abc import Callable, Collection, Sequence

from .model_port import ModelPort


def choose_token(logits: Sequence[float], allowed: Collection[int]) -> int:
    """Set every token outside ``allowed`` to -inf and return the best one."""
    raise NotImplementedError


def generate(
    model: ModelPort,
    ids: list[int],
    allowed_next: Callable[[list[int]], list[int] | None],
    max_steps: int,
) -> list[int]:
    """Append tokens to ``ids`` until ``allowed_next`` returns None.

    ``allowed_next`` gets the tokens generated so far. One allowed token is
    forced without asking the model. Raises PromptError after ``max_steps``.
    """
    raise NotImplementedError
