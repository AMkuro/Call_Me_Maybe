"""The constrained-decoding loop shared by name and value generation."""

from collections.abc import Callable, Collection, Sequence


def choose_token(logits: Sequence[float], allowed: Collection[int]) -> int:
    """Set every token outside ``allowed`` to -inf and return the best one."""
    raise NotImplementedError


def generate(
    next_logits: Callable[[list[int]], list[float]],
    ids: list[int],
    allowed_next: Callable[[list[int]], list[int] | None],
    max_steps: int,
) -> list[int]:
    """Append tokens to ``ids`` until ``allowed_next`` returns None.

    ``next_logits`` is the model's ``logits``. ``allowed_next`` gets the
    tokens generated so far. One allowed token is forced without asking
    the model. Raises PromptError after ``max_steps``.

    Judgment: takes only the logits function, not the whole model: that is
    all the loop needs. When only one token is allowed, force it without
    calling the model (saves compute). Selection is greedy (largest
    logit). It stops when allowed_next returns None.
    """
    raise NotImplementedError
