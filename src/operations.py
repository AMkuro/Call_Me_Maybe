"""Attach operation context to failures while preserving their causes."""

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def operation(stage: str) -> Iterator[None]:
    """Preserve the failing stage for exceptions and keyboard interruptions."""
    try:
        yield
    except KeyboardInterrupt as exc:
        raise KeyboardInterrupt(stage) from exc
    except Exception as exc:
        raise RuntimeError(f"{stage}: {type(exc).__name__}: {exc}") from exc
