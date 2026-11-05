"""Publish each cumulative result array without exposing partial JSON."""

import json
import os
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import JsonValue


def save_results(path: Path, results: list[dict[str, JsonValue]]) -> None:
    """Atomically replace output with a complete UTF-8 JSON array.

    A failed write or replacement leaves the previously published file intact.
    The temporary file lives beside the output to keep replacement on the same
    filesystem. Cleanup also runs on interruption; removal is best effort if
    filesystem permissions change during a failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                results, stream, ensure_ascii=False, allow_nan=False, indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
