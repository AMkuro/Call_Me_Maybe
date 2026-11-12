"""A live stderr dashboard, with ordinary logs when stderr is redirected.

A running viewer (``python -m src.viewer``, ADR-0021) takes the dashboard's
place: it receives one event per token through a Unix socket.
"""

import json
import os
import socket
import sys
import tempfile
import time
import unicodedata
from collections.abc import Generator
from contextlib import contextmanager

# Candidates per token the decoder extracts while a viewer is connected.
TRACE_TOP_K = 10
VIEWER_SOCKET = os.path.join(tempfile.gettempdir(),
                             f"call_me_maybe-{os.getuid()}.sock")
# A stalled viewer may delay generation this many seconds, once, before
# it is dropped.
SEND_TIMEOUT_SECONDS = 1.0
_PLAIN = ("function", "warning", "result", "error")

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


def connect_viewer(path: str) -> socket.socket | None:
    """Connect to this user's viewer; None when none is listening.

    Effect: connects a Unix socket.
    """
    try:
        if os.stat(path).st_uid != os.getuid():
            return None  # another user's socket must not receive the prompts
    except OSError:
        return None
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(SEND_TIMEOUT_SECONDS)
    try:
        sock.connect(path)
    except OSError:
        sock.close()
        return None
    return sock


class Progress:
    """The reporter: ``progress(label, text)`` records one observation.

    Each prompt (label ``input``) tries to reach a viewer. While one is
    connected, events go to it, ``top_k`` asks the decoder for candidates
    and stderr gets only the ordinary lines; otherwise the dashboard (or
    ordinary lines when stderr is redirected) shows them.

    Effect: writes stderr and the viewer socket; reads the terminal size
    and type.

    Judgment: observe without keyboard interaction or extra model calls.
    A failing stderr or viewer must not discard generated results.
    """

    def __init__(self, total: int) -> None:
        """Start with no viewer and the terminal untouched."""
        self._stream = sys.stderr
        self._interactive = (self._stream.isatty()
                             and os.environ.get("TERM") != "dumb")
        self._fields = {"input": f"0/{total}"}
        self._summaries: list[str] = []
        self._last_frame = 0.0
        self._available = True
        self._screen = False
        self._viewer: socket.socket | None = None

    @property
    def top_k(self) -> int:
        """Candidates per token the decoder should extract (0 = none)."""
        return TRACE_TOP_K if self._viewer is not None else 0

    def __call__(self, label: str, text: str) -> None:
        """Update one field, retaining only the current prompt's state."""
        if label == "input":
            self._fields.clear()
            if self._viewer is None:
                self._viewer = connect_viewer(VIEWER_SOCKET)
                if self._viewer is not None:
                    self._leave_screen()
        if self._viewer is not None:
            self._send(label, text)
        if label == "trace":
            return
        self._fields[label] = text
        line = f"[{self._fields['input']}] {label}: {_visible(text)}\n"
        if not self._interactive or self._viewer is not None:
            if label in _PLAIN:
                self._write(line)
            return
        if label in ("result", "error"):
            self._summaries.append(line)
        now = time.monotonic()
        if now - self._last_frame < 0.05 and label not in (
            "function", "result", "error",
        ):
            return
        try:
            width, height = os.get_terminal_size(self._stream.fileno())
        except OSError:
            width, height = 100, 24
        if not width or not height:
            width, height = 100, 24
        if not self._screen:
            self._write("\x1b[?1049h\x1b[?25l")
            self._screen = True
        self._write("\x1b[H\x1b[2J"
                    + render_progress(self._fields, width, height))
        self._last_frame = now

    def close(self) -> None:
        """Tell the viewer the run ended and restore the terminal."""
        if self._viewer is not None:
            self._send("end", "")
            if self._viewer is not None:
                self._viewer.close()
            self._viewer = None
        self._leave_screen()

    def _send(self, label: str, text: str) -> None:
        """Send one event line; a lost viewer falls back to the dashboard."""
        if label == "trace":
            line = text
        elif label == "prompt":
            line = json.dumps({"event": "prompt",
                               "input": self._fields["input"], "prompt": text})
        elif label in ("result", "error", "end"):
            line = json.dumps({"event": label, "text": text})
        else:
            return
        assert self._viewer is not None
        try:
            self._viewer.sendall(line.encode() + b"\n")
        except OSError:
            self._viewer.close()
            self._viewer = None

    def _leave_screen(self) -> None:
        """Leave the dashboard and print the results it was hiding."""
        if self._screen:
            self._write("\x1b[?25h\x1b[?1049l")
            self._screen = False
            for summary in self._summaries:
                self._write(summary)
            self._summaries.clear()

    def _write(self, text: str) -> None:
        """A closed diagnostic stream must not discard generated results."""
        if self._available:
            try:
                self._stream.write(text)
                self._stream.flush()
            except OSError:
                self._available = False


@contextmanager
def terminal_progress(total: int) -> Generator[Progress, None, None]:
    """Yield a reporter and restore the terminal even after an exception."""
    progress = Progress(total)
    try:
        yield progress
    finally:
        progress.close()
