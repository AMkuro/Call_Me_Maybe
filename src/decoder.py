"""Choose one token: run the model, mask disallowed tokens, take the best."""

from collections.abc import Collection, Sequence
from typing import Literal

import numpy as np

from .errors import PromptError
from .model_port import Model

SelectionMode = Literal["forced", "greedy"]


def constrained_argmax(
    logits: Sequence[float], allowed: Collection[int],
) -> int:
    """Return the best-scoring token among ``allowed``.

    Every other token counts as -inf; ``logits`` is not modified. On a tie
    the lowest token ID wins.

    Args:
        logits: One score per vocabulary ID.
        allowed: The token IDs that may be chosen.

    Returns:
        The chosen token ID.

    Raises:
        PromptError: The logits are not a one-dimensional sequence of numbers
            or contain NaN or +inf, ``allowed`` is empty or holds an ID
            outside the logits, or every allowed score is -inf.
    """
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
    """Run one forward pass on ``ids`` and choose the next ID from ``allowed``.

    The forward pass runs even when ``allowed`` holds a single ID; that ID
    is then chosen without using the scores. ``ids`` is not modified.

    Args:
        model: The model to run.
        ids: The context: the prompt and the output so far.
        allowed: The token IDs that may come next.
        include_raw_top: Also return the unconstrained best token, for
            diagnostics. It never changes the choice.

    Returns:
        The chosen ID; its mode, "forced" for a single candidate and
        "greedy" otherwise; and the unconstrained best ID when
        ``include_raw_top`` is set and the mode is greedy, else None.

    Raises:
        PromptError: ``allowed`` is empty (raised before any forward pass),
            or model inference fails. With multiple candidates, malformed
            scores or no finite allowed score also raise PromptError.
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
