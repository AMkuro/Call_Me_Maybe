"""Constrained argmax and one-token selection without loop control."""

from collections.abc import Collection, Sequence
from typing import Literal

import numpy as np

from .errors import PromptError
from .model_port import Model

SelectionMode = Literal["forced", "greedy"]


def constrained_argmax(
    logits: Sequence[float], allowed: Collection[int],
) -> int:
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


def infer_and_select_next_token(
    model: Model,
    ids: list[int],
    allowed: list[int],
    *,
    include_raw_top: bool = False,
) -> tuple[int, SelectionMode, int | None]:
    """Select one ID without updating the sequence or controlling its loop.

    Method B (owner decision, 2026-11-18): every token gets exactly one
    forward on ``ids``, the full context. A unique candidate is then
    returned as is, with no array, mask or argmax work on the scores.
    Otherwise use the constrained argmax, and optionally report the raw
    argmax from those same scores. Diagnostics never affect the selection.
    """
    if not allowed:
        # The caller adds what was being generated (name or parameter).
        raise PromptError("no allowed token can continue the output")
    logits = model.logits(ids)
    if len(allowed) == 1:
        return allowed[0], "forced", None
    token = constrained_argmax(logits, allowed)
    raw_top = int(np.argmax(logits)) if include_raw_top else None
    return token, "greedy", raw_top
