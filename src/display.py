"""A live stderr dashboard, with ordinary logs when stderr is redirected.

Both modes end with one line per prompt (done or failed, with seconds)
and a total line (ADR-0014, ADR-0021).
"""

import json
import os
import sys
import time
import unicodedata

_ROWS = (
    ("input", "Input"), ("stage", "Stage"), ("prompt", "Prompt"),
    ("function", "Function"), ("parameter", "Parameter"),
    ("value", "Value"), ("value_tokens", "Value tokens"),
    ("warning", "Warning"), ("result", "Result"), ("error", "Error"),
)
_FRAME_SECONDS = 0.05  # at most one redraw per interval ...
# ... except for completed selections and prompt outcomes.
_DRAW_NOW = ("function", "result", "error")
_FALLBACK_SIZE = (100, 24)
_ENTER, _LEAVE = "\x1b[?1049h\x1b[?25l", "\x1b[?25h\x1b[?1049l"
_CLEAR = "\x1b[H\x1b[2J"


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
        if key not in fields:
            continue
        lines.append(f"{label}: {_visible(fields[key])}")
    return "\n".join(_clip(line, width) for line in lines[:max(height - 1, 1)])


class Dashboard:
    """The stderr output of one run; ``report`` is what callers get.

    Use it in ``with``: leaving the block restores the terminal and prints
    the summary, even after an exception or Ctrl-C.

    Effect: writes stderr; reads its terminal size and the terminal type.

    Judgment: observe without keyboard interaction or extra model calls.
    """

    def __init__(self, total: int) -> None:
        """Nothing written yet."""
        self.stream = sys.stderr
        self.interactive = (self.stream.isatty()
                            and os.environ.get("TERM") != "dumb")
        self.total = total
        self.fields = {"input": f"0/{total}"}
        self.counts = {"done": 0, "failed": 0}
        self.summary: list[str] = []  # held while the dashboard covers it
        self.started = self.prompt_started = time.monotonic()
        self.last_frame = 0.0
        self.available = True

    def __enter__(self) -> "Dashboard":
        """Switch a terminal to the dashboard screen."""
        if self.interactive:
            self.write(_ENTER)
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Restore the terminal, then print the summary and the total."""
        if self.interactive:
            self.write(_LEAVE)
            for line in self.summary:
                self.write(line)
        done, failed = self.counts["done"], self.counts["failed"]
        left = self.total - done - failed  # an interrupted run
        self.write(
            f"{self.total} prompt{'' if self.total == 1 else 's'}: "
            f"{done} done, {failed} failed, "
            + (f"{left} not finished, " if left else "")
            + f"{time.monotonic() - self.started:.1f} s\n"
        )

    def report(self, label: str, text: str) -> None:
        """Record one observation; ``input`` starts a new prompt."""
        if label == "input":
            self.fields.clear()
            self.prompt_started = time.monotonic()
        self.fields[label] = text
        if label in ("result", "error"):
            self.end_prompt(label, text)
        if self.interactive:
            self.draw(label)
        elif label == "warning":
            self.write(f"[{self.fields['input']}] warning: {_visible(text)}\n")

    def end_prompt(self, label: str, text: str) -> None:
        """One summary line: done (function) or failed (reason), seconds."""
        status = "done" if label == "result" else "failed"
        self.counts[status] += 1
        what = self.fields.get("function", "") if label == "result" else text
        seconds = time.monotonic() - self.prompt_started
        line = (f"[{self.fields['input']}] {status:<7} {_visible(what)}  "
                f"{seconds:.1f} s\n")
        if self.interactive:
            self.summary.append(line)
        else:
            self.write(line)

    def draw(self, label: str) -> None:
        """Redraw the dashboard, at most once per interval."""
        now = time.monotonic()
        if now - self.last_frame < _FRAME_SECONDS and label not in _DRAW_NOW:
            return
        try:
            width, height = os.get_terminal_size(self.stream.fileno())
        except OSError:
            width, height = _FALLBACK_SIZE
        if not width or not height:
            width, height = _FALLBACK_SIZE
        self.write(_CLEAR + render_progress(self.fields, width, height))
        self.last_frame = now

    def write(self, text: str) -> None:
        """A closed diagnostic stream must not discard generated results."""
        if self.available:
            try:
                self.stream.write(text)
                self.stream.flush()
            except OSError:
                self.available = False
