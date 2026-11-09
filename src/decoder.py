"""The constrained-decoding loop shared by name and value generation."""

from collections.abc import Callable, Collection, Sequence

from .errors import PromptError


def choose_token(logits: Sequence[float], allowed: Collection[int]) -> int:
    """Set every token outside ``allowed`` to -inf and return the best one."""
    raise NotImplementedError


def generate(
    next_logits: Callable[[list[int]], list[float]],
    ids: list[int],
    allowed_next: Callable[[list[int]], list[int] | None],
    max_steps: int,
) -> list[int]:
    """Generate tokens after ``ids`` until ``allowed_next`` returns None.

    Returns only the generated tokens. ``next_logits`` is the model's
    ``logits``. ``allowed_next`` gets the tokens generated so far. One
    allowed token is forced without asking the model. Raises PromptError
    after ``max_steps`` tokens.

    Effect: only what ``next_logits`` does. It changes none of its arguments.

    Judgment: takes only the logits function, not the whole model: that is
    all the loop needs. When only one token is allowed, force it without
    calling the model (saves compute). Selection is greedy (largest
    logit). It stops when allowed_next returns None.
    """
    generated: list[int] = []
    while True:
        allowed = allowed_next(generated)
        if allowed is None:
            return generated
        if len(generated) >= max_steps:
            raise PromptError(f"no end after {max_steps} tokens")
        if len(allowed) == 1:
            generated.append(allowed[0])
        else:
            logits = next_logits(ids + generated)
            generated.append(choose_token(logits, allowed))
