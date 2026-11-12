"""Watch each token decision step by step: ``uv run python -m src.viewer``.

Start it before ``python -m src`` (ADR-0021). The generator finds it through
a Unix socket and sends one JSON line per token. A thread queues the lines
as they arrive, so the generator never waits for the screen; the screen then
plays each token as RAW -> MASK -> SORT -> SELECT. Queued events are shown
to the end even after the generator exits. Ctrl-C quits.

Reading order: the durations, ``main``, ``listen``, ``receive``, ``show``
(what each event does to the screen), ``render`` (how the screen looks).
"""

import contextlib
import json
import os
import queue
import socket
import stat
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .display import VIEWER_SOCKET, _visible

# How long each phase of one token stays on screen, in seconds. A forced
# token (no model call, no logits) is shown for SELECT_SECONDS only.
RAW_SECONDS = 0.4
MASK_SECONDS = 0.4
SORT_SECONDS = 0.3
SELECT_SECONDS = 0.4

PHASES = (("RAW", RAW_SECONDS), ("MASK", MASK_SECONDS),
          ("SORT", SORT_SECONDS), ("SELECT", SELECT_SECONDS))


def main() -> int:
    """Listen, queue events in a thread and play them one by one.

    Effect: creates and removes the socket file; draws on the terminal.
    """
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    try:
        server = listen(VIEWER_SOCKET)
    except OSError as e:  # e.g. another user's file at this path
        print(f"cannot listen on {VIEWER_SOCKET}: {e}", file=sys.stderr)
        return 1
    threading.Thread(target=receive, args=(server, events),
                     daemon=True).start()
    screen = Screen()
    try:
        with Live(render(screen), auto_refresh=False, screen=True) as live:
            while True:
                event = events.get()
                for seconds in show(event, screen):
                    live.update(render(screen), refresh=True)
                    time.sleep(seconds)
                live.update(render(screen), refresh=True)
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()
        with contextlib.suppress(OSError):
            os.unlink(VIEWER_SOCKET)


def listen(path: str) -> socket.socket:
    """Listen at ``path``, replacing a stale socket; only this user connects.

    Raises OSError when another viewer is already listening there.

    Effect: removes an old socket file and creates a new one (mode 0600).
    """
    with contextlib.suppress(FileNotFoundError):
        if stat.S_ISSOCK(os.lstat(path).st_mode):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                try:
                    probe.connect(path)
                except OSError:
                    os.unlink(path)  # stale: its viewer has exited
                else:
                    raise OSError("another viewer is already running")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    mask = os.umask(0o177)
    try:
        server.bind(path)
    finally:
        os.umask(mask)
    server.listen()
    return server


def receive(server: socket.socket,
            events: queue.Queue[dict[str, Any]]) -> None:
    """Queue every event line as soon as it arrives, in arrival order.

    One connection is read to its end before the next is accepted, so the
    order of one run is kept. Memory grows with the queued events (ADR-0021).
    """
    while True:
        try:
            connection, _ = server.accept()
        except OSError:
            return  # the listener was closed
        with connection, connection.makefile(
                encoding="utf-8", errors="replace") as lines:
            try:
                for line in lines:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        events.put(event)
            except (OSError, ValueError):
                continue  # a broken connection ends that run's events


class Screen:
    """What is on screen; changed only by ``show``."""

    def __init__(self) -> None:
        """Nothing received yet."""
        self.title = "waiting for python -m src"
        self.status = ""
        self.preview = ("", "", "")  # fixed, value, rest of the record
        self.token: dict[str, Any] | None = None
        self.phase = ""


def show(event: dict[str, Any], screen: Screen) -> Iterator[float]:
    """Apply one event, yielding how long to hold each frame it makes."""
    kind = event.get("event")
    if kind == "token":
        screen.token = event
        preview = (event["fixed"], event["value"], event["rest"])
        if event["forced"]:
            screen.phase = "FORCED"
            screen.preview = preview
            yield SELECT_SECONDS
            return
        for phase, seconds in PHASES:
            screen.phase = phase
            if phase == "SELECT":
                screen.preview = preview
            yield seconds
    elif kind == "prompt":
        screen.title = f"[{event['input']}] {event['prompt']}"
        screen.status = ""
        screen.preview = ("", "", "")
        screen.token = None
        screen.phase = ""
    elif kind in ("result", "error"):
        screen.status = f"{kind}: {event['text']}"
    elif kind == "end":
        screen.status += "  (generator finished)"


def render(screen: Screen) -> Layout:
    """JSON being written on the left, the candidates on the right.

    All received text goes through ``_visible`` and ``Text`` so control
    characters and Rich markup are shown, not interpreted.
    """
    fixed, value, rest = screen.preview
    record = Text(_visible(fixed))
    record.append(_visible(value), style="bold yellow")
    record.append("▌", style="yellow")
    record.append(_visible(rest), style="dim")
    left = Panel(Group(record, Text(_visible(screen.status), style="cyan")),
                 title=Text(_visible(screen.title)), title_align="left")
    layout = Layout()
    layout.split_row(Layout(left), Layout(_candidates(screen)))
    return layout


def _candidates(screen: Screen) -> Panel:
    """The candidate table for the current phase."""
    token = screen.token
    if token is None:
        return Panel(Text(""), title="candidates")
    title = Text(_visible(
        f"{token['stage']} {token['parameter']} step {token['step']}"
        f" — {screen.phase}"))
    selected_id, selected_text = token["selected"]
    if screen.phase == "FORCED":
        return Panel(Text(
            f"FORCED {_quoted(selected_text)} (id {selected_id})\n"
            "only one token is allowed: the model is not called and no"
            " logits exist"), title=title, title_align="left")
    table = Table("ID", "TOKEN", "LOGIT", expand=True)
    if screen.phase in ("RAW", "MASK"):
        for token_id, text, logit, allowed in token["raw"]:
            masked = screen.phase == "MASK" and not allowed
            table.add_row(str(token_id), Text(_quoted(text)),
                          "-inf" if masked else f"{logit:.3f}",
                          style="dim strike" if masked else None)
        note = "all tokens, highest raw logits"
    else:
        for token_id, text, logit in token["masked"]:
            chosen = screen.phase == "SELECT" and token_id == selected_id
            table.add_row(str(token_id), Text(_quoted(text)), f"{logit:.3f}",
                          style="bold reverse green" if chosen else None)
        note = ("allowed tokens only; this order is for display, the"
                " decoder takes the largest logit")
    return Panel(table, title=title, title_align="left",
                 subtitle=note, subtitle_align="left")


def _quoted(text: str) -> str:
    """Quote a token so spaces and escaped newlines stay visible."""
    return f'"{_visible(text)}"'


if __name__ == "__main__":
    raise SystemExit(main())
