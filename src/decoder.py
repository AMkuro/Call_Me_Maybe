"""The constrained-decoding loop shared by name and value generation."""

from collections.abc import Callable, Collection, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from .errors import PromptError
from .model_port import Model

SelectionMode = Literal["forced", "greedy"]


class TokenChoice(BaseModel):
    """One observed token decision; forced steps have no raw preference."""

    model_config = ConfigDict(frozen=True)

    step: int
    token: int
    tokens: list[int]
    forced: bool
    # Unconstrained argmax, only to warn when name selection is masked.
    raw_top: int | None = None


def choose_token(logits: Sequence[float], allowed: Collection[int]) -> int:
    """Set every token outside ``allowed`` to -inf and return the best one."""
    try:
        scores = np.asarray(logits, dtype=float)
    except (TypeError, ValueError) as e:
        raise PromptError(f"model returned malformed logits: {e}") from e
    if scores.ndim != 1:
        raise PromptError("model logits must be one-dimensional")
    if not allowed or any(token < 0 or token >= len(scores)
                          for token in allowed):
        raise PromptError("no valid token in the decoding mask")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise PromptError("model returned invalid logits")
    masked = np.full(scores.shape, -np.inf)
    indices = list(allowed)
    masked[indices] = scores[indices]
    if not np.isfinite(masked).any():
        raise PromptError("all allowed tokens have non-finite logits")
    return int(np.argmax(masked))


def generate(
    next_logits: Callable[[list[int]], list[float]],
    ids: list[int],
    allowed_next: Callable[[list[int]], list[int] | None],
    max_steps: int,
    on_token: Callable[[TokenChoice], None] | None = None,
) -> list[int]:
    """Generate tokens after ``ids`` until ``allowed_next`` returns None.

    Returns only the generated tokens. ``next_logits`` is the model's
    ``logits``. ``allowed_next`` gets the tokens generated so far. One
    allowed token is forced without asking the model. Raises PromptError
    after ``max_steps`` tokens. With ``on_token``, report each selected
    token and the unconstrained argmax from the same logits for warnings.
    Observations are not used for selection.

    Effect: calls ``next_logits`` and the optional observation callback.

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
            raise PromptError(
                f"token limit {max_steps} reached before a terminator"
            )
        if not allowed:
            raise PromptError("no allowed token can continue this value")
        raw_top = None
        if len(allowed) == 1:
            token = allowed[0]
        else:
            logits = next_logits(ids + generated)
            token = choose_token(logits, allowed)
            if on_token is not None:
                raw_top = int(np.argmax(logits))
        generated.append(token)
        if on_token is not None:
            on_token(TokenChoice.model_validate({
                "step": len(generated), "token": token,
                "tokens": generated.copy(), "forced": len(allowed) == 1,
                "raw_top": raw_top,
            }))


def select_next_token(
    model: Model,
    ids: list[int],
    allowed: list[int],
    *,
    include_raw_top: bool = False,
) -> tuple[int, SelectionMode, int | None]:
    """Select one ID without updating the sequence or controlling its loop.

    A unique candidate needs no inference. Otherwise request logits once,
    use the constrained argmax, and optionally report the raw argmax from
    those same scores. Diagnostics never affect the selected token.
    """
    if not allowed:
        raise PromptError("no allowed token can continue this value")
    if len(allowed) == 1:
        return allowed[0], "forced", None
    logits = model.logits(ids)
    token = choose_token(logits, allowed)
    raw_top = int(np.argmax(logits)) if include_raw_top else None
    return token, "greedy", raw_top
