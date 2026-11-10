"""Save the results."""

import json
import os
import stat
import tempfile
from pathlib import Path

from .errors import StartupError
from .schema import FunctionCall


def prepare_output(path: Path) -> None:
    """Prepare and probe an atomic-save destination before model startup.

    Effect: creates parent directories and writes/removes a sibling probe.
    Existing output bytes are untouched; no final output file is created.
    Raises StartupError with the destination and underlying failure reason.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(mode):
                raise OSError("output destination is not a regular file")
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp",
        ) as probe:
            probe.write("[]\n")
            probe.flush()
    except (OSError, ValueError) as e:
        raise StartupError(f"cannot prepare output at {path}: {e}") from e


def save_results(path: Path, calls: list[FunctionCall]) -> None:
    """Write ``calls`` as a JSON array, replacing the file atomically.

    Called after every successful prompt with all results so far, or with
    an empty list when the run produces no results. Raises OSError on a
    failed write; main reports it and stops (ADR-0012).

    Effect: writes a file.

    Judgment: rewrite the whole file after every prompt (temporary file,
    then replace), so a crash keeps the results so far. Slow when there
    are many prompts (not measured).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump([call.model_dump() for call in calls], output,
                      ensure_ascii=True, allow_nan=False, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
