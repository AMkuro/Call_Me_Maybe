"""A live stderr dashboard, with ordinary logs when stderr is redirected."""

import json
import os
import sys
import time
import unicodedata
from collections.abc import Callable, Generator
from contextlib import contextmanager

_ROWS = (
    ("input", "Input"), ("stage", "Stage"), ("prompt", "Prompt"),
    ("function", "Function"), ("parameter", "Parameter"),
    ("value", "Value"), ("value_tokens", "Value tokens"),
    ("choice", "Token choice"),
    ("metrics", "Diagnostics"), ("warning", "Warning"),
    ("result", "Result"), ("error", "Error"),
)


def _visible(text: str) -> str:
    """Escape control characters so input cannot issue terminal commands."""
    return "".join(
        char if char.isprintable() else json.dumps(char)[1:-1]
        for char in text
    )


def _clip(text: str, width: int) -> str:
    """Limit terminal cells, counting wide CJK characters as two cells."""
    result = ""
    used = 0
    for char in text:
        size = 0 if unicodedata.combining(char) else (
            2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        )
        if used + size > max(width - 1, 0):
            return result + "…" if width else ""
        result += char
        used += size
    return result


def render_progress(fields: dict[str, str], width: int, height: int) -> str:
    """Render validated runtime observations as a compact, safe frame."""
    lines = ["Call Me Maybe — generation"]
    for key, label in _ROWS:
        if key in fields:
            lines.append(f"{label}: {_visible(fields[key])}")
    return "\n".join(_clip(line, width) for line in lines[:max(height - 1, 1)])


@contextmanager
def terminal_progress(
    total: int, live: bool = True,
) -> Generator[Callable[[str, str], None], None, None]:
    """Yield a reporter and restore the terminal even after an exception.

    Effect: writes stderr; reads its terminal size and the terminal type.

    Judgment: observe without keyboard interaction or extra model calls.
    Redirected stderr gets function choices, warnings and final results.
    """
    stream = sys.stderr
    interactive = live and stream.isatty() and os.environ.get("TERM") != "dumb"
    fields = {"input": f"0/{total}"}
    summaries: list[str] = []
    last_frame = 0.0
    available = True

    def write(text: str) -> None:
        """A closed diagnostic stream must not discard generated results."""
        nonlocal available
        if available:
            try:
                stream.write(text)
                stream.flush()
            except OSError:
                available = False

    def report(label: str, text: str) -> None:
        """Update one field, retaining only the current prompt's state."""
        nonlocal last_frame
        if label == "input":
            fields.clear()
        fields[label] = text
        line = f"[{fields['input']}] {label}: {_visible(text)}\n"
        if label in ("result", "error"):
            summaries.append(line)
        if not interactive:
            if label in ("function", "warning", "result", "error"):
                write(line)
            return
        now = time.monotonic()
        if now - last_frame < 0.05 and label not in (
            "function", "result", "error",
        ):
            return
        try:
            width, height = os.get_terminal_size(stream.fileno())
        except OSError:
            width, height = 100, 24
        if not width or not height:
            width, height = 100, 24
        write("\x1b[H\x1b[2J" + render_progress(fields, width, height))
        last_frame = now

    try:
        if interactive:
            write("\x1b[?1049h\x1b[?25l")
        yield report
    finally:
        if interactive:
            write("\x1b[?25h\x1b[?1049l")
            for summary in summaries:
                write(summary)
