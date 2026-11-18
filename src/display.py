"""A live stderr dashboard, with ordinary logs when stderr is redirected.

Both modes end with one line per prompt (done or failed, with seconds)
and a total line (docs/architecture.md §6).
"""

import json
import os
import sys
import time
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Literal

from .decoder import SelectionMode
from .errors import PromptError
from .model_port import Model
from .progress import Stage
from .schema import ParamType

Row = Literal["input", "stage", "prompt", "function", "parameter", "value",
              "value_tokens", "token", "warning", "result", "error"]
_ROWS: tuple[tuple[Row, str], ...] = (
    ("input", "Input"), ("stage", "Stage"), ("prompt", "Prompt"),
    ("function", "Function"), ("parameter", "Parameter"),
    ("value", "Value"), ("value_tokens", "Value tokens"),
    ("token", "Token"),
    ("warning", "Warning"), ("result", "Result"), ("error", "Error"),
)
# The rows that show the latest generated token. A stage that generates
# tokens starts them empty; validation keeps them, so a failure shows the
# last generated value.
_TOKEN_ROWS: tuple[Row, ...] = ("value", "value_tokens", "token")
_GENERATING_STAGES: tuple[Stage, ...] = ("select_function",
                                         "generate_call_json")
_FRAME_SECONDS = 0.05  # at most one redraw per interval ...
# ... except for completed selections and prompt outcomes.
_DRAW_NOW: tuple[Row, ...] = ("function", "result", "error")
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


def render_progress(fields: Mapping[Row, str], width: int, height: int) -> str:
    """Render runtime progress as a compact, safe frame."""
    lines = ["Call Me Maybe — generation"]
    for key, label in _ROWS:
        if key not in fields:
            continue
        lines.append(f"{label}: {_visible(fields[key])}")
    return "\n".join(_clip(line, width) for line in lines[:max(height - 1, 1)])


class Dashboard:
    """Own one run's terminal state, prompt timings and outcome counts.

    Use it in ``with``: leaving the block restores the terminal and prints
    the summary, even after an exception or Ctrl-C. Each prompt goes
    through prompt_started, then prompt_done or prompt_failed; the
    GenerationProgress methods between them only change displayed rows.

    Effect: writes stderr; reads its terminal size and the terminal type.
    It reads no keyboard input and makes no model calls beyond decoding.
    """

    def __init__(
        self, total: int, model: Model, vocab: Mapping[int, bytes],
    ) -> None:
        """Nothing written yet."""
        self.model = model
        self.vocab = vocab
        self.stream = sys.stderr
        self.interactive = (self.stream.isatty()
                            and os.environ.get("TERM") != "dumb")
        self.total = total
        self.fields: dict[Row, str] = {"input": f"0/{total}"}
        self.counts = {"done": 0, "failed": 0}
        self.summary: list[str] = []  # held while the dashboard covers it
        self.started = self.prompt_started_at = time.monotonic()
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

    def prompt_started(self, index: int, prompt: str) -> None:
        """Clear the previous prompt's rows and start timing this one."""
        self.fields.clear()
        self.prompt_started_at = time.monotonic()
        self.fields["input"] = f"{index}/{self.total}"
        self._show("prompt", prompt)

    def prompt_done(self, name: str) -> None:
        """Count a saved result and add its summary line."""
        self._show("result", name)
        self._end_prompt("done", name)

    def prompt_failed(self, reason: str) -> None:
        """Count a skipped prompt and add its summary line."""
        self._show("error", reason)
        self._end_prompt("failed", reason)

    def stage(self, stage: Stage) -> None:
        """Show the stage, clearing token rows when it generates tokens."""
        if stage in _GENERATING_STAGES:
            for row in _TOKEN_ROWS:
                self.fields.pop(row, None)
        self._show("stage", stage)

    def function_selected(self, name: str) -> None:
        """Display the selected function, forcing a completed-name frame."""
        self._show("function", name)

    def parameter_started(self, name: str, param_type: ParamType) -> None:
        """Clear the previous value before showing this parameter."""
        self.fields.update({"value": "", "value_tokens": "0", "token": ""})
        self._show("parameter", f"{name} ({param_type})")

    def name_token(
        self, tokens: Sequence[int], *, mode: SelectionMode,
        raw_top: int | None,
    ) -> None:
        """Decode borrowed IDs now; never change or retain the sequence."""
        token = tokens[-1]
        self.fields.update({"value": _decoded(self.model, tokens),
                            "value_tokens": str(len(tokens)),
                            "token": f"{token} ({mode})"})
        if mode == "greedy" and raw_top is not None and raw_top != token:
            raw = _token_text(raw_top, self.model, self.vocab)
            selected = _token_text(token, self.model, self.vocab)
            self._warn("name_selection_constrained: "
                       f"step={len(tokens)} raw_top={raw} "
                       f"selected={selected}")
        elif self.interactive:
            self.draw("token")

    def value_token(
        self, written: bytes, step: int, *, token: int, mode: SelectionMode,
    ) -> None:
        """Update all fields for one step before drawing its latest frame."""
        self.fields.update({"value": written.decode("utf-8", errors="ignore"),
                            "value_tokens": str(step),
                            "token": f"{token} ({mode})"})
        if self.interactive:
            self.draw("token")

    def _show(self, row: Row, text: str) -> None:
        """Set one row; only a terminal shows rows (redirected logs do not)."""
        self.fields[row] = text
        if self.interactive:
            self.draw(row)

    def _warn(self, text: str) -> None:
        """Show a warning; a redirected stream also gets it as a log line."""
        self._show("warning", text)
        if not self.interactive:
            self.write(f"[{self.fields['input']}] warning: {_visible(text)}\n")

    def _end_prompt(self, status: Literal["done", "failed"],
                    what: str) -> None:
        """One summary line: status, function or reason, and seconds."""
        self.counts[status] += 1
        seconds = time.monotonic() - self.prompt_started_at
        line = (f"[{self.fields['input']}] {status:<7} {_visible(what)}  "
                f"{seconds:.1f} s\n")
        if self.interactive:
            self.summary.append(line)
        else:
            self.write(line)

    def draw(self, row: Row) -> None:
        """Redraw the dashboard, at most once per interval."""
        now = time.monotonic()
        if now - self.last_frame < _FRAME_SECONDS and row not in _DRAW_NOW:
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


def _token_text(token: int, model: Model, vocab: Mapping[int, bytes]) -> str:
    """Quote token bytes, preserving partial UTF-8 and special-token text."""
    data = vocab.get(token)
    if data is None:
        return repr(_decoded(model, [token]))
    try:
        return repr(data.decode("utf-8"))
    except UnicodeDecodeError:
        return repr(data)


def _decoded(model: Model, tokens: Sequence[int]) -> str:
    """Decode synchronously; only non-list inputs need a boundary copy."""
    ids = tokens if isinstance(tokens, list) else list(tokens)
    try:
        return model.decode(ids) or "<EOS/special>"
    except PromptError:
        return f"<text unavailable; ids={ids}>"
