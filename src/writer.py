"""Save the results."""

import json
import os
import stat
import tempfile
from pathlib import Path

import simplejson

from .errors import StartupError
from .schema import FunctionCallRequest


def _prepare_destination(path: Path) -> None:
    """Create parents; reject an existing destination resolving to a non-file.

    Args:
        path: The output file.

    Raises:
        OSError: A directory cannot be created, or ``path`` exists and is
            not a regular file after following a symbolic link.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISREG(mode):
            raise OSError("output destination is not a regular file")


def prepare_output(path: Path) -> None:
    """Check that results can be saved to ``path``, without changing it.

    Creates missing parent directories, then writes and removes a
    temporary file next to ``path``. An existing file at ``path`` is left
    unchanged, and none is created.

    Args:
        path: The output file.

    Raises:
        StartupError: The output cannot be prepared, also when ``path``
            resolves to a non-regular file. The message names ``path``
            and the cause.
    """
    try:
        _prepare_destination(path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as probe:
            probe.write("[]\n")
            probe.flush()
    except (OSError, ValueError) as e:
        raise StartupError(f"cannot prepare output at {path}: {e}") from e


def save_results(path: Path, calls: list[FunctionCallRequest]) -> None:
    """Write ``calls`` as an indented JSON array, replacing it atomically.

    The array is written to a temporary file next to ``path``, which then
    replaces ``path``, so a failure or interruption leaves the previous
    file whole. Each record is its ``_generated_json`` text, or its
    model_dump when that is None, written by simplejson with ``indent=2``.
    Numbers in ``_generated_json`` keep the spelling the LLM wrote.
    Creates missing parent directories.
    A final symbolic link is replaced, leaving its target unchanged. No
    fsync is performed, so crash or power-loss durability is not guaranteed.

    Args:
        path: The output file.
        calls: All the records so far, in input order.

    Raises:
        OSError: Writing fails, or ``path`` exists and is not a regular
            file after following symbolic links. A failure before replacement
            leaves the previous output intact; cleanup is attempted in finally.
    """
    _prepare_destination(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            # validate_call set _generated_json only after checking it.
            # RawJSON writes each number as the LLM spelled it; a float
            # could add exponent notation or drop its decimal digits.
            records = [
                (
                    call.model_dump()
                    if call._generated_json is None
                    else json.loads(
                        call._generated_json, parse_float=simplejson.RawJSON
                    )
                )
                for call in calls
            ]
            simplejson.dump(records, output, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
