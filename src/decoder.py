"""Constrained argmax and one-token selection without loop control."""

from collections.abc import Collection, Sequence
from typing import Literal

import numpy as np

from .errors import PromptError
from .model_port import Model

SelectionMode = Literal["forced", "greedy"]


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
