"""Save the results."""

import json
import os
import stat
import tempfile
from pathlib import Path

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
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp",
        ) as probe:
            probe.write("[]\n")
            probe.flush()
    except (OSError, ValueError) as e:
        raise StartupError(f"cannot prepare output at {path}: {e}") from e


def save_results(path: Path, calls: list[FunctionCallRequest]) -> None:
    """Write ``calls`` as a JSON array, replacing the file atomically.

    The array is written to a temporary file next to ``path``, which then
    replaces ``path``, so a failure or interruption leaves the previous
    file whole. Each record is written as its ``_generated_json`` text, or
    with json.dump when that is None. Creates missing parent directories.
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
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            if not calls:
                output.write("[]\n")
            else:
                output.write("[\n")
                for index, call in enumerate(calls):
                    if index:
                        output.write(",\n")
                    output.write("  ")
                    # validate_call set _generated_json only after checking
                    # it, so it is written without parsing it again.
                    # Re-encoding its numbers could add exponent notation
                    # or lose the decimal digits selected by the LLM.
                    if call._generated_json is not None:
                        output.write(call._generated_json)
                    else:
                        json.dump(call.model_dump(), output,
                                  ensure_ascii=True, allow_nan=False,
                                  indent=2)
                output.write("\n]\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
