"""Read and parse JSON without duplicate-key overwrite or decimal rounding."""

import json
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any, NoReturn


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject a duplicate before inserting it into the decoded object."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    """NaN and Infinity are extensions, not standard JSON numbers."""
    raise ValueError(f"non-standard JSON number {value!r}")


def loads(text: str) -> Any:
    """Parse once, preserving exact numeric values for later validation.

    Shape and type checks belong to Pydantic at the consuming boundary.
    This module uses only the standard library, including for dev tools.
    """
    try:
        return json.loads(text, object_pairs_hook=_unique_object,
                          parse_float=Decimal, parse_constant=_reject_constant)
    except DecimalException as e:
        raise ValueError(f"JSON number exceeds Decimal parser limits: {e}") \
            from e
    except RecursionError as e:
        raise ValueError("JSON nesting exceeds the parser recursion limit") \
            from e


def load(path: Path) -> Any:
    """Read a UTF-8 JSON file once, using the same strict parser as loads.

    Effect: reads a file. I/O and JSON errors retain their original types;
    the caller supplies the function-definition or prompt error context.
    """
    return loads(path.read_text(encoding="utf-8"))
