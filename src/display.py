"""A live stderr dashboard, with ordinary logs when stderr is redirected.

Both modes report each completed or failed prompt with seconds and a total
line. The total also counts unfinished prompts after an interruption.
"""

import codecs
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
from .schema import JsonValueTypes

Row = Literal["input", "stage", "prompt", "function", "parameter", "value",
              "value_tokens", "token", "warning", "result", "error",
              "json", "saved_json", "tokens", "elapsed"]
_ROWS: tuple[tuple[Row, str], ...] = (
    ("input", "Input"), ("stage", "Stage"), ("elapsed", "Elapsed"),
    ("prompt", "Prompt"),
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
_CLEAR = "\x1b[H"


def _decoded(model: Model, tokens: Sequence[int]) -> str:
    """Decode ``tokens`` for display, substituting text on PromptError.

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
        ``text`` when it fits, otherwise a prefix and ellipsis within width.
    """
    result = ""
    sizes: list[int] = []
    used = 0
    for char in text:
        size = 0 if unicodedata.combining(char) else (
            2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        )
        if used + size > max(width, 0):
            while sizes and used > width - 1:
                used -= sizes.pop()
                result = result[:-1]
            return result + "…" if width > 0 else ""
        result += char
        sizes.append(size)
        used += size
    return result


def _json_lines(text: str, width: int) -> list[str]:
    """Indent and wrap an actual JSON prefix without completing its syntax.

    Args:
        text: A possibly incomplete JSON prefix.
        width: Available terminal cells (at least one).

    Returns:
        Lines containing only committed characters and display whitespace.
    """
    lines = [""]
    depth = 0
    quoted = escaped = False
    for char in text:
        if not quoted and char in "}]":
            depth = max(0, depth - 1)
            if lines[-1].strip():
                lines.append("  " * depth)
        lines[-1] += char
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            depth += 1
            lines.append("  " * depth)
        elif char == ",":
            lines.append("  " * depth)
    wrapped: list[str] = []
    for line in lines:
        current = ""
        cells = 0
        for char in _visible(line):
            size = 0 if unicodedata.combining(char) else (
                2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
            )
            if cells + size > width and current:
                wrapped.append(current)
                current, cells = "", 0
            current += char if size <= width else "…"
            cells += min(size, width)
        wrapped.append(current)
    return wrapped


def render_progress(fields: Mapping[Row, str], width: int, height: int) -> str:
    """Render metadata above a bounded viewport of committed JSON.

    Args:
        fields: Observed state; JSON is never parsed or completed here.
        width: Terminal width in cells.
        height: Terminal height in rows.

    Returns:
        A frame leaving the terminal's final row unused.
    """
    width = max(1, width - 1)
    budget = max(height - 1, 1)
    if "json" not in fields:
        lines = ["Call Me Maybe — generation"]
        lines.extend(f"{label}: {_visible(fields[key])}"
                     for key, label in _ROWS if key in fields)
        lines.append("JSON OUTPUT: waiting for JSON tokens")
        return "\n".join(_clip(line, width)
                         for line in lines[:budget])
    saved = fields.get("saved_json")
    label = ("SAVED JSON — validated" if saved is not None else
             "JSON OUTPUT — committed prefix")
    # The cursor is display-only and never claims an ungenerated brace.
    text = saved if saved is not None else fields["json"] + "▌"
    preview = _json_lines(text, width)
    header = [f"Call Me Maybe    {fields.get('input', '')}",
              f"Stage: {fields.get('stage', '')}",
              f"Function: {fields.get('function', 'selecting')}"]
    if fields.get("parameter"):
        header.append(f"Parameter: {fields['parameter']}")
    header.append(f"Elapsed: {fields.get('elapsed', '0.0 s')}  "
                  f"Tokens: {fields.get('tokens', '0')}")
    if fields.get("token"):
        header.append(f"Token: {fields['token']}")
    for key in ("error", "warning", "result"):
        if fields.get(key):
            header.append(f"{key.title()}: {_visible(fields[key])}")
    header += ["─" * width, label]
    # Small screens keep errors and token activity before decoration.
    if len(header) > budget - 3:
        compact = []
        if fields.get("error"):
            compact.append(f"Error: {fields['error']}")
        compact.append(f"Stage: {fields.get('stage', '')}")
        if fields.get("token"):
            compact.append(f"Token: {fields['token']}")
        if fields.get("parameter"):
            compact.append(f"Parameter: {fields['parameter']}")
        compact.extend([header[0],
                        f"Tokens: {fields.get('tokens', '0')}  "
                        f"Elapsed: {fields.get('elapsed', '0.0 s')}"])
        header_room = max(0, budget - min(2, budget))
        header = (compact[:header_room - 1] + [label]
                  if header_room else [])
    room = budget - len(header)
    if len(preview) > room:
        preview = (["… earlier JSON omitted …"] + preview[-(room - 1):]
                   if room > 1 else preview[-1:])
    return "\n".join(_clip(_visible(line), width)
                     for line in header + preview)


class Dashboard:
    """One run's terminal state, prompt timings and outcome counts.

    Use it in ``with``: leaving the block restores the terminal and prints
    the summary, even after an exception or Ctrl-C. Each prompt goes
    through prompt_started, then prompt_done or prompt_failed on a normal
    outcome; an interruption can leave it unfinished. Progress methods
    update displayed rows and may emit diagnostic warnings.

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
        self.json_decoder = codecs.getincrementaldecoder("utf-8")()
        self.output_tokens = 0
        self.output_path: str | None = None
        self.active = False

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
            if self.active:
                reason = str(exc_info[1]) if len(exc_info) > 1 else ""
                self.write("Interrupted / not finished: "
                           f"{_visible(reason)}\n")
            if self.output_path:
                self.write("Output file: "
                           f"{_visible(self.output_path)}\n")
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
        self.json_decoder.reset()
        self.output_tokens = 0
        self.active = True
        self.prompt_started_at = time.monotonic()
        self.fields.update({"input": f"{index}/{self.total}",
                            "prompt": prompt})
        self.draw("prompt")

    def prompt_done(self, name: str, saved_json: str | None = None) -> None:
        """Count a saved result and add its summary line.

        Args:
            name: The selected function.
            saved_json: Validated, successfully saved text, when available.
        """
        if saved_json is not None:
            self.fields["saved_json"] = saved_json
        self.fields["stage"] = "done"
        self.fields["result"] = name
        self.draw("result")
        self._end_prompt("done", name)

    def prompt_failed(self, reason: str) -> None:
        """Count a skipped prompt and add its summary line.

        Args:
            reason: Why the prompt failed.
        """
        self.fields["error"] = reason
        self.fields["stage"] = "failed"
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
        self.active = False
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
        if stage == "generate_call_json":
            self.fields.update({"json": "", "tokens": "0"})
            self.json_decoder.reset()
            self.output_tokens = 0
        self.fields["stage"] = stage
        self.draw("stage")

    def function_selected(self, name: str) -> None:
        """Show the selected function at once.

        Args:
            name: The function name.
        """
        self.fields["function"] = name
        self.draw("function")

    def argument_started(self, name: str, param_type: JsonValueTypes) -> None:
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
                            "token": self._token_label(token, mode)})
        if mode == "greedy" and raw_top is not None and raw_top != token:
            raw = _token_text(raw_top, self.model, self.vocab)
            selected = _token_text(token, self.model, self.vocab)
            self._warn("name_selection_constrained: "
                       f"step={len(tokens)} raw_top={raw} "
                       f"selected={selected}")
        else:
            self.draw("token")

    def output_token(
        self, data: bytes, *, token: int, mode: SelectionMode,
        fixed: bool = False,
    ) -> None:
        """Accumulate actual committed bytes; decode split UTF-8 safely.

        Args:
            data: Vocabulary bytes of one committed JSON token.
            token: Its actual ID.
            mode: Its selection mode.
            fixed: True for structural text outside argument slots.
        """
        if fixed:
            self.fields.pop("parameter", None)
        self.fields["json"] = (self.fields.get("json", "")
                               + self.json_decoder.decode(data))
        self.output_tokens += 1
        self.fields.update({"tokens": str(self.output_tokens),
                            "token": self._token_label(token, mode)})
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
                            "token": self._token_label(token, mode)})
        self.draw("token")

    def _token_label(self, token: int, mode: SelectionMode) -> str:
        """Return a token's ID, its text and how it was chosen.

        Args:
            token: The token ID.
            mode: How it was chosen.

        Returns:
            For example ``2 '.0}' (greedy)``.
        """
        return f"{token} {_token_text(token, self.model, self.vocab)} ({mode})"

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
        width, height = self.terminal_size()
        self.fields["elapsed"] = f"{now - self.prompt_started_at:.1f} s"
        frame = render_progress(self.fields, width, height)
        self.write(_CLEAR + frame.replace("\n", "\x1b[K\n") + "\x1b[K\x1b[J")
        self.last_frame = now

    def terminal_size(self) -> tuple[int, int]:
        """Read current terminal dimensions, falling back when unavailable.

        Returns:
            The positive width and height, including after a resize.
        """
        try:
            width, height = os.get_terminal_size(self.stream.fileno())
        except OSError:
            return _FALLBACK_SIZE
        return (width, height) if width > 0 and height > 0 else _FALLBACK_SIZE

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
