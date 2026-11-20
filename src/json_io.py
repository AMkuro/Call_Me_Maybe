"""Read and parse JSON without duplicate-key overwrite or decimal rounding."""

import json
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any, NoReturn


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object, rejecting a repeated key.

    Args:
        pairs: The object's keys and values, in order.

    Returns:
        The object.

    Raises:
        ValueError: A key occurs twice.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    """Reject NaN, Infinity and -Infinity, which are not standard JSON.

    Args:
        value: The constant as written.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"non-standard JSON number {value!r}")


def loads(text: str) -> Any:
    """Parse JSON ``text`` strictly, keeping exact number values.

    Numbers with a fraction or an exponent become Decimal, others int.
    Shape and types are checked later, by Pydantic.

    Args:
        text: JSON text.

    Returns:
        The parsed value.

    Raises:
        ValueError: The text is not JSON, repeats an object key, contains
            NaN or Infinity, or exceeds the parser's limits.
    """
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_float=Decimal,
            parse_constant=_reject_constant,
        )
    except DecimalException as e:
        raise ValueError(
            f"JSON number exceeds Decimal parser limits: {e}"
        ) from e
    except RecursionError as e:
        raise ValueError(
            "JSON nesting exceeds the parser recursion limit"
        ) from e


def load(path: Path) -> Any:
    """Read the UTF-8 JSON file at ``path`` and parse it with loads.

    The errors are not wrapped; callers add which file it was.

    Args:
        path: The file.

    Returns:
        The parsed value.

    Raises:
        OSError: The file cannot be read.
        ValueError: The file is not UTF-8 or not valid JSON (see loads).
    """
    return loads(path.read_text(encoding="utf-8"))
