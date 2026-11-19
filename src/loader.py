"""Read and validate the two input files."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import TypeAdapter

from .errors import StartupError
from .json_io import load
from .schema import FunctionDef


def index_functions(
    functions: Iterable[FunctionDef],
) -> dict[str, FunctionDef]:
    """Key validated definitions by their names, keeping their order.

    Args:
        functions: Validated definitions.

    Returns:
        A dict from each name to its definition, in input order.

    Raises:
        ValueError: A name is repeated, or there is no definition.
    """
    indexed: dict[str, FunctionDef] = {}
    for fn in functions:
        if fn.name in indexed:
            raise ValueError("duplicate function names")
        indexed[fn.name] = fn
    if not indexed:
        raise ValueError("function definitions must not be empty")
    return indexed


def load_functions(path: Path) -> dict[str, FunctionDef]:
    """Read the function definitions file.

    Args:
        path: The JSON file: an array of function definitions.

    Returns:
        The definitions keyed by name, in input order.

    Raises:
        StartupError: The file is missing or invalid, the array is empty,
            a name is repeated, or a parameter type is unsupported.
    """
    try:
        return index_functions(
            TypeAdapter(list[FunctionDef]).validate_python(
                load(path), strict=True,
            )
        )
    except (OSError, ValueError) as e:
        raise StartupError(f"cannot load functions from {path}: {e}") from e


def load_prompts(path: Path) -> list[str]:
    """Read the prompts file.

    Args:
        path: The JSON file: an array of objects, each with only a
            "prompt" string.

    Returns:
        The prompts in input order.

    Raises:
        StartupError: The file is missing or invalid, the array is empty,
            or an entry has another key or a non-string prompt.
    """
    try:
        entries = TypeAdapter(list[dict[str, str]]).validate_python(
            load(path), strict=True,
        )
        if not entries:
            raise ValueError("insufficient input: no prompts supplied")
        if any(set(entry) != {"prompt"} for entry in entries):
            raise ValueError('each input entry must contain only "prompt"')
        return [entry["prompt"] for entry in entries]
    except (OSError, ValueError) as e:
        raise StartupError(f"cannot load prompts from {path}: {e}") from e
