"""Optional JSONL event stream for the separate visual debugger.

Enabled only with CMM_TRACE_FILE. A broken diagnostic sink must not stop
function-call generation or change the required JSON output.
"""

import json
import os
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager

Trace = Callable[[dict[str, object]], None]


@contextmanager
def event_log() -> Generator[Trace | None, None, None]:
    """Yield an optional event writer and close it even on an exception."""
    path = os.environ.get("CMM_TRACE_FILE")
    if not path:
        yield None
        return
    try:
        stream = open(path, "w", encoding="utf-8", buffering=1)
    except OSError as error:
        print(f"trace disabled: {error}", file=sys.stderr)
        yield None
        return

    available = True

    def emit(event: dict[str, object]) -> None:
        """Write one JSON record; disable logging if its destination fails."""
        nonlocal available
        if available:
            try:
                stream.write(json.dumps(
                    event, ensure_ascii=False, allow_nan=False,
                ) + "\n")
            except (OSError, ValueError, TypeError):
                available = False

    try:
        yield emit
    finally:
        emit({"type": "end"})
        try:
            stream.close()
        except OSError:
            pass
