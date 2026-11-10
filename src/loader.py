"""Read and validate the two input files."""

from pathlib import Path

from pydantic import TypeAdapter

from .errors import StartupError
from .schema import FunctionDef


def load_functions(path: Path) -> list[FunctionDef]:
    """Read function definitions.

    Raises StartupError on a missing or invalid file, an empty list,
    duplicate names, or an unsupported parameter type.

    Effect: reads a file.
    """
    try:
        functions = TypeAdapter(list[FunctionDef]).validate_json(
            path.read_text(encoding="utf-8")
        )
        if not functions:
            raise ValueError("function definitions must not be empty")
        names = [fn.name for fn in functions]
        if any(not name.strip() for name in names):
            raise ValueError("function names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("duplicate function names")
        return functions
    except (OSError, ValueError) as e:
        raise StartupError(f"cannot load functions from {path}: {e}") from e


def load_prompts(path: Path) -> list[str]:
    """Read the prompts (the ``prompt`` key of each input entry).

    Raises StartupError for insufficient input when the array is empty.

    Effect: reads a file.
    """
    try:
        entries = TypeAdapter(list[dict[str, str]]).validate_json(
            path.read_text(encoding="utf-8")
        )
        if not entries:
            raise ValueError("insufficient input: no prompts supplied")
        if any(set(entry) != {"prompt"} for entry in entries):
            raise ValueError('each input entry must contain only "prompt"')
        return [entry["prompt"] for entry in entries]
    except (OSError, ValueError) as e:
        raise StartupError(f"cannot load prompts from {path}: {e}") from e
