"""Save the results."""

import json
import os
import tempfile
from pathlib import Path

from .schema import FunctionCall


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
