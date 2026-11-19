"""A live stderr dashboard, with ordinary logs when stderr is redirected.

Both modes end with one line per prompt (done or failed, with seconds)
and a total line.
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


def _decoded(model: Model, tokens: Sequence[int]) -> str:
    """Return the text of ``tokens`` for display, never raising.

    Args:
        model: Decodes the IDs.
        tokens: The token IDs.

    Returns:
        The decoded text; "<EOS/special>" when it is empty (special tokens
        only), or a placeholder listing the IDs when decoding fails.
    """
    ids = tokens if isinstance(tokens, list) else list(tokens)
    try:
        return model.decode(ids) or "<EOS/special>"
    except PromptError:
        return f"<text unavailable; ids={ids}>"


def _token_text(token: int, model: Model, vocab: Mapping[int, bytes]) -> str:
    """Return one token as a quoted literal for a warning.

    Args:
        token: The token ID.
        model: Decodes a token that is missing from ``vocab``.
        vocab: The bytes of each token.

    Returns:
        The repr of the token's text, or of its bytes when they are not
        complete UTF-8 (part of a character).
    """
    data = vocab.get(token)
    if data is None:
        return repr(_decoded(model, [token]))
    try:
        return repr(data.decode("utf-8"))
    except UnicodeDecodeError:
        return repr(data)


def _visible(text: str) -> str:
    """Escape control characters so input cannot issue terminal commands.

    Args:
        text: Text to show.

    Returns:
        ``text`` with each non-printable character written as its JSON
        escape.
    """
    return "".join(
        char if char.isprintable() else json.dumps(char)[1:-1]
        for char in text
    )


def _clip(text: str, width: int) -> str:
    """Shorten ``text`` to ``width`` terminal cells.

    Wide East Asian characters count as two cells and combining characters
    as none.

    Args:
        text: One line.
        width: The number of cells available.

    Returns:
        ``text`` when it fits in ``width - 1`` cells, otherwise its longest
        start that fits there followed by "…"; "" when ``width`` is 0.
    """
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
    """Return one screen of text: a title, then one line per present row.

    Rows keep a fixed order and control characters are escaped.

    Args:
        fields: The text of each row to show.
        width: The terminal width; every line is clipped to it.
        height: The terminal height; at most ``height - 1`` lines (at least
            one) are kept.

    Returns:
        The lines, joined by line breaks.
    """
    lines = ["Call Me Maybe — generation"]
    for key, label in _ROWS:
        if key not in fields:
            continue
        lines.append(f"{label}: {_visible(fields[key])}")
    return "\n".join(_clip(line, width) for line in lines[:max(height - 1, 1)])


class Dashboard:
    """One run's terminal state, prompt timings and outcome counts.

    Use it in ``with``: leaving the block restores the terminal and prints
    the summary, even after an exception or Ctrl-C. Each prompt goes
    through prompt_started, then prompt_done or prompt_failed; the
    GenerationProgress methods between them only change displayed rows.

    On a terminal (stderr is a TTY and TERM is not "dumb") it draws a live
    screen; otherwise it writes one log line per prompt and per warning.
    It writes only to stderr, reads the terminal size, reads no keyboard
    input and uses the model only to decode token IDs for display.

    Attributes:
        model: Decodes token IDs for display.
        vocab: The bytes of each token.
        stream: Where everything is written (sys.stderr).
        interactive: True when a live screen is drawn.
        total: The number of prompts.
        fields: The text of each displayed row.
        counts: The numbers of done and failed prompts.
        summary: Summary lines held until the live screen is left.
        started: When the run started (time.monotonic).
        prompt_started_at: When the current prompt started.
        last_frame: When the screen was last drawn.
        available: False once writing to ``stream`` has failed.
    """

    def __init__(
        self, total: int, model: Model, vocab: Mapping[int, bytes],
    ) -> None:
        """Prepare for ``total`` prompts; nothing is written until ``with``.

        Args:
            total: The number of prompts.
            model: Decodes token IDs for display.
            vocab: The bytes of each token.
        """
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
        """Switch a terminal to the dashboard screen.

        Returns:
            This dashboard.
        """
        if self.interactive:
            self.write(_ENTER)
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Restore the terminal, then print the summary and the total.

        The total line also counts the prompts that were not finished.

        Args:
            *exc_info: The exception, if any; it is not suppressed.
        """
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
        """Clear the previous prompt's rows and start timing this one.

        Args:
            index: The prompt's position, from 1.
            prompt: The prompt text.
        """
        self.fields.clear()
        self.prompt_started_at = time.monotonic()
        self.fields.update({"input": f"{index}/{self.total}",
                            "prompt": prompt})
        self.draw("prompt")

    def prompt_done(self, name: str) -> None:
        """Count a saved result and add its summary line.

        Args:
            name: The selected function.
        """
        self.fields["result"] = name
        self.draw("result")
        self._end_prompt("done", name)

    def prompt_failed(self, reason: str) -> None:
        """Count a skipped prompt and add its summary line.

        Args:
            reason: Why the prompt failed.
        """
        self.fields["error"] = reason
        self.draw("error")
        self._end_prompt("failed", reason)

    def _end_prompt(self, status: Literal["done", "failed"],
                    what: str) -> None:
        """Count the outcome and add its summary line with the seconds taken.

        The line is written now when logging, or held until the live screen
        is left.

        Args:
            status: The outcome.
            what: The function name or the failure reason.
        """
        self.counts[status] += 1
        seconds = time.monotonic() - self.prompt_started_at
        line = (f"[{self.fields['input']}] {status:<7} {_visible(what)}  "
                f"{seconds:.1f} s\n")
        if self.interactive:
            self.summary.append(line)
        else:
            self.write(line)

    def stage(self, stage: Stage) -> None:
        """Show the stage, clearing token rows when it generates tokens.

        Args:
            stage: The stage that starts.
        """
        if stage in _GENERATING_STAGES:
            for row in _TOKEN_ROWS:
                self.fields.pop(row, None)
        self.fields["stage"] = stage
        self.draw("stage")

    def function_selected(self, name: str) -> None:
        """Show the selected function at once.

        Args:
            name: The function name.
        """
        self.fields["function"] = name
        self.draw("function")

    def parameter_started(self, name: str, param_type: ParamType) -> None:
        """Show the parameter whose value starts, clearing the last value.

        Args:
            name: The parameter name.
            param_type: Its type.
        """
        self.fields.update({"value": "", "value_tokens": "0", "token": "",
                            "parameter": f"{name} ({param_type})"})
        self.draw("parameter")

    def name_token(
        self, tokens: Sequence[int], *, mode: SelectionMode,
        raw_top: int | None,
    ) -> None:
        """Show the name written so far and how its last token was chosen.

        When a greedy choice differs from the unconstrained best token, a
        warning names both. ``tokens`` is read now and not kept.

        Args:
            tokens: The name's token IDs so far.
            mode: How the last token was chosen.
            raw_top: The unconstrained best token, or None when it was not
                computed.
        """
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
        else:
            self.draw("token")

    def value_token(
        self, written: bytes, step: int, *, token: int, mode: SelectionMode,
    ) -> None:
        """Show the value written so far, its token count and last token.

        Args:
            written: The value's bytes so far.
            step: The number of its tokens.
            token: The last token ID.
            mode: How the last token was chosen.
        """
        self.fields.update({"value": written.decode("utf-8", errors="ignore"),
                            "value_tokens": str(step),
                            "token": f"{token} ({mode})"})
        self.draw("token")

    def _warn(self, text: str) -> None:
        """Show a warning; when logging, also write it as a line.

        Args:
            text: The warning.
        """
        self.fields["warning"] = text
        self.draw("warning")
        if not self.interactive:
            self.write(f"[{self.fields['input']}] warning: {_visible(text)}\n")

    def draw(self, row: Row) -> None:
        """Redraw the live screen, at most once per _FRAME_SECONDS.

        A row in _DRAW_NOW is drawn at once. Nothing is drawn when logging.

        Args:
            row: The row that changed.
        """
        if not self.interactive:
            return
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
        """Write and flush ``text``; after an OSError, stop writing.

        The failure only sets ``available`` to False, so a closed stderr
        does not stop generation or saving.

        Args:
            text: The text to write.
        """
        if self.available:
            try:
                self.stream.write(text)
                self.stream.flush()
            except OSError:
                self.available = False
