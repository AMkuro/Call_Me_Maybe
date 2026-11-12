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
    ("candidates", "Candidates"),  # last: a small terminal cuts it first
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
        if key == "candidates" and key in fields:
            lines.append(f"{label}:")
            lines.extend("  " + _visible(row)
                         for row in fields[key].split("\n"))
        elif key in fields:
            lines.append(f"{label}: {_visible(fields[key])}")
    return "\n".join(_clip(line, width) for line in lines[:max(height - 1, 1)])


@contextmanager
def terminal_progress(
    total: int,
) -> Generator[Callable[[str, str], None], None, None]:
    """Yield a reporter and restore the terminal even after an exception.

    Effect: writes stderr; reads its terminal size and the terminal type.

    Judgment: observe without keyboard interaction or extra model calls.
    Redirected stderr gets warnings as they come.
    Both modes end with one line per prompt (done or failed, seconds)
    and a total; a terminal gets them after leaving the dashboard.
    """
    stream = sys.stderr
    interactive = stream.isatty() and os.environ.get("TERM") != "dumb"
    fields = {"input": f"0/{total}"}
    summaries: list[str] = []
    last_frame = 0.0
    available = True
    started = prompt_started = time.monotonic()
    counts = {"done": 0, "failed": 0}

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
        nonlocal last_frame, prompt_started
        if label == "input":
            fields.clear()
            prompt_started = time.monotonic()
        fields[label] = text
        line = f"[{fields['input']}] {label}: {_visible(text)}\n"
        if label in ("result", "error"):
            status = "done" if label == "result" else "failed"
            counts[status] += 1
            what = fields.get("function", "") if label == "result" else text
            line = (f"[{fields['input']}] {status:<7} {_visible(what)}  "
                    f"{time.monotonic() - prompt_started:.1f} s\n")
            if interactive:
                summaries.append(line)
        if not interactive:
            if label in ("warning", "result", "error"):
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
        write(f"{total} prompt{'' if total == 1 else 's'}: "
              f"{counts['done']} done, {counts['failed']} failed, "
              f"{time.monotonic() - started:.1f} s\n")
