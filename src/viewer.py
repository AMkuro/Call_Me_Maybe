"""Slow, read-only Rich TUI for JSONL token decisions from CMM_TRACE_FILE.

Run in a second terminal; this process never slows model inference.
Rich is optional and is imported only by this viewer.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _read_events(path: Path) -> Iterator[dict[str, Any]]:
    """Read appended JSONL records, including a previously completed trace."""
    while not path.exists():
        time.sleep(0.05)
    with path.open("r", encoding="utf-8") as stream:
        while True:
            line = stream.readline()
            if not line:
                time.sleep(0.05)
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            yield event
            if event.get("type") == "end":
                return


def _pretty_prefix(raw: str) -> str:
    """Indent structural JSON without requiring the prefix to be complete."""
    output: list[str] = []
    depth = 0
    quoted = False
    escaped = False
    for char in raw:
        if quoted:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            output.append(char)
            quoted = True
        elif char in "[{":
            output.append(char)
            depth += 1
            output.append("\n" + "  " * depth)
        elif char in "]}":
            depth = max(depth - 1, 0)
            output.append("\n" + "  " * depth + char)
        elif char == ",":
            output.append(",\n" + "  " * depth)
        elif char == ":":
            output.append(": ")
        elif char not in " \t\r\n":
            output.append(char)
    return "".join(output).rstrip()


def _safe(text: str) -> str:
    """Show untrusted tokens without interpreting terminal control codes."""
    return "".join(
        char if char.isprintable() or char == "\n"
        else json.dumps(char)[1:-1] for char in text
    )


def _score(value: Any) -> str:
    """Format one observed logit; None means it was not computed."""
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


def _frame(state: dict[str, Any], phase: str,
           decision: dict[str, Any] | None) -> Any:
    """Construct the latest screen. There is intentionally no event history."""
    from rich.columns import Columns
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    count = f"{state.get('index', '—')}/{state.get('total', '—')}"
    header = Text("Call Me Maybe", style="bold")
    header.append(f"   Processing Prompt {count}", style="dim")
    prompt = Text(_safe(str(state.get("prompt", ""))))
    stage = str(state.get("stage", "waiting"))
    parameter = str(state.get("parameter", ""))
    heading = stage + (f" · {parameter}" if parameter else "")
    preview = _pretty_prefix(str(state.get("preview", "")))
    if state.get("stage") not in ("complete", "error"):
        preview += "▌"
    json_panel = Panel(
        Text(_safe(preview)), title="Output JSON", border_style="cyan",
        expand=True,
    )

    table = Table(expand=True, box=None, show_edge=False, pad_edge=False)
    table.add_column("ID", no_wrap=True)
    table.add_column("TOKEN", overflow="ellipsis")
    table.add_column("LOGIT", justify="right", no_wrap=True)

    rows: list[list[Any]] = []
    selected_id: int | None = None
    if decision is not None:
        selected = decision.get("selected", [])
        selected_id = selected[0] if selected else None
        if decision.get("forced"):
            rows = [[selected[0], selected[1], None]]
        elif phase in ("raw", "mask"):
            for token, text, logit, permitted in decision.get("raw", []):
                rows.append([token, text, logit if phase == "raw" or permitted
                             else "-∞"])
        else:
            rows = [list(item) for item in decision.get("masked", [])]

    for row in rows[:10]:
        token_id, token_text, logit = row
        is_selected = phase == "select" and token_id == selected_id
        rejected = logit == "-∞"
        style = "bold green" if is_selected else "red dim" if rejected else ""
        value = "-∞" if rejected else _score(logit)
        table.add_row(
            Text(str(token_id)), Text(_safe(str(token_text))),
            Text(value), style=style,
        )
    for _ in range(max(0, 10 - len(rows))):
        table.add_row("", "", "")

    name = {
        "raw": "Raw logits · Top 10",
        "mask": "Masked logits · Top 10 raw",
        "sort": "Allowed logits · Top 10",
        "select": "Selected · constrained greedy",
        "forced": "Forced · logits not computed",
    }.get(phase, "Token candidates")
    right = Panel(table, title=name, border_style="magenta", expand=True)
    footer = Text(heading, style="dim")
    if decision is not None:
        footer.append(f"   · token step {decision.get('step', 0)}")
    if state.get("done"):
        footer.append("   · Finished — Ctrl+C to close", style="bold")
    if state.get("error"):
        footer.append(f"   · ERROR: {_safe(str(state['error']))}",
                      style="bold red")
    return Group(header, prompt, Columns([json_panel, right], expand=True,
                                       equal=True), footer)


def main(argv: list[str] | None = None) -> int:
    """Replay the live trace at a human-readable pace."""
    parser = argparse.ArgumentParser(description="Slow visual token debugger")
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--delay", type=float, default=0.35,
                        help="Seconds per visual phase (default: 0.35)")
    parser.add_argument("--exit-on-end", action="store_true",
                        help="Exit instead of holding the final frame")
    args = parser.parse_args(argv)
    if not 0 <= args.delay <= 10:
        parser.error("--delay must be between 0 and 10 seconds")
    if not sys.stdout.isatty() or os.environ.get("TERM") == "dumb":
        print("viewer requires an interactive terminal", file=sys.stderr)
        return 2
    try:
        from rich.console import Console
        from rich.live import Live
    except ImportError:
        print("Rich is missing. Run with: uv run --with rich "
              "python -m src.viewer --events PATH", file=sys.stderr)
        return 2

    state: dict[str, Any] = {}
    decision: dict[str, Any] | None = None
    console = Console()
    try:
        with Live(_frame(state, "", None), console=console, screen=True,
                  auto_refresh=False) as live:
            for event in _read_events(args.events):
                kind = event.get("type")
                if kind == "input":
                    state = dict(index=event.get("index"),
                                 total=event.get("total"),
                                 prompt=event.get("prompt", ""),
                                 stage="select_name", parameter="",
                                 preview=json.dumps({
                                     "prompt": event.get("prompt", ""),
                                     "name": "", "parameters": {},
                                 }, ensure_ascii=False))
                    decision = None
                elif kind == "state":
                    new_stage = event.get("stage")
                    if (new_stage is not None and
                            new_stage != state.get("stage")):
                        decision = None
                    if new_stage is not None:
                        state["stage"] = new_stage
                        state["parameter"] = ""
                    state.update({k: v for k, v in event.items()
                                  if k in ("preview", "function", "error")})
                elif kind == "token":
                    state["stage"] = event.get("stage", "")
                    state["parameter"] = event.get("parameter", "")
                    decision = event
                    phases = (["forced"] if event.get("forced") else
                              ["raw", "mask", "sort", "select"])
                    for phase in phases:
                        if phase in ("select", "forced"):
                            state["preview"] = event.get("preview", "")
                        live.update(_frame(state, phase, decision),
                                    refresh=True)
                        time.sleep(args.delay)
                    continue
                elif kind == "end":
                    state["done"] = True
                live.update(_frame(state, "select", decision), refresh=True)
            if not args.exit_on_end:
                while True:
                    time.sleep(1)
    except KeyboardInterrupt:
        return 0
    except OSError as error:
        print(f"viewer: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
