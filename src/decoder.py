"""The constrained-decoding loop shared by name and value generation."""

from collections.abc import Callable, Collection, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from .errors import PromptError


class TokenChoice(BaseModel):
    """One observed token decision; uncomputed diagnostics stay None."""

    model_config = ConfigDict(frozen=True)

    step: int
    token: int
    tokens: list[int]
    forced: bool
    raw_top: int | None = None
    allowed_mass: float | None = None
    margin: float | None = None
    constrained: bool = False
    raw_candidates: list[tuple[int, float, bool]] | None = None
    allowed_candidates: list[tuple[int, float]] | None = None


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


def _top_tokens(scores: np.ndarray, candidates: Sequence[int], count: int
                ) -> list[tuple[int, float]]:
    """Select a small ranked view without sorting the whole vocabulary."""
    indices = np.asarray(candidates, dtype=int)
    if not len(indices):
        return []
    limit = min(count, len(indices))
    # The cutoff is found in O(V) expected time, then ties are broken by ID.
    threshold = float(np.partition(scores[indices], -limit)[-limit])
    better = indices[scores[indices] > threshold]
    ties = np.sort(indices[scores[indices] == threshold])
    chosen = list(map(int, better)) + list(map(int, ties[:limit - len(better)]))
    chosen.sort(key=lambda i: (-float(scores[i]), i))
    return [(i, float(scores[i])) for i in chosen]


def generate(
    next_logits: Callable[[list[int]], list[float]],
    ids: list[int],
    allowed_next: Callable[[list[int]], list[int] | None],
    max_steps: int,
    on_token: Callable[[TokenChoice], None] | None = None,
    top_k: int = 0,
) -> list[int]:
    """Generate tokens after ``ids`` until ``allowed_next`` returns None.

    Returns only the generated tokens. ``next_logits`` is the model's
    ``logits``. ``allowed_next`` gets the tokens generated so far. One
    allowed token is forced without asking the model. Raises PromptError
    after ``max_steps`` tokens.

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
        if len(allowed) == 1:
            token = allowed[0]
            diagnostics: dict[str, object] = {}
        else:
            logits = next_logits(ids + generated)
            token = choose_token(logits, allowed)
            diagnostics = {}
            if on_token is not None:
                scores = np.asarray(logits, dtype=float)
                weights = np.exp(scores - np.max(scores))
                indices = list(set(allowed))
                ordered = np.sort(scores[indices])
                raw_top = int(np.argmax(scores))
                diagnostics = {
                    "raw_top": raw_top,
                    "allowed_mass": float(weights[indices].sum()
                                          / weights.sum()),
                    "margin": float(ordered[-1] - ordered[-2]),
                    "constrained": raw_top not in allowed,
                }
                if top_k > 0:
                    permitted = set(indices)
                    raw = _top_tokens(scores, range(len(scores)), top_k)
                    kept = _top_tokens(scores, indices, top_k)
                    diagnostics["raw_candidates"] = [
                        (i, logit, i in permitted) for i, logit in raw
                    ]
                    diagnostics["allowed_candidates"] = kept
        generated.append(token)
        if on_token is not None:
            on_token(TokenChoice.model_validate({
                "step": len(generated), "token": token,
                "tokens": generated.copy(), "forced": len(allowed) == 1,
                **diagnostics,
            }))
