"""Validate Subject input and final JSON independently of generation."""

import json
import math
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    create_model,
)


class JsonType(BaseModel):
    """Describe one flat JSON type; nested schema keywords are unsupported."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal[
        "number", "integer", "string", "boolean", "null", "array", "object"
    ]


class FunctionDefinition(BaseModel):
    """Describe a dynamic function with all listed parameters required."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, JsonType]
    returns: JsonType


class Prompt(BaseModel):
    """Validate a single input record without interpreting its meaning."""

    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str


class FunctionCall(BaseModel):
    """Represent exactly the three Subject output fields."""

    model_config = ConfigDict(
        strict=True, extra="forbid", allow_inf_nan=False,
    )

    prompt: str
    name: str
    parameters: dict[str, JsonValue]


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    """Reject duplicate JSON keys instead of silently overwriting values."""
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> float:
    """Reject the non-JSON NaN and Infinity extensions of Python's parser."""
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _finite_float(value: str) -> float:
    """Reject JSON exponents that overflow the runtime's finite float range."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"JSON number exceeds finite float range: {value}")
    return result


def parse_json(text: str) -> JsonValue:
    """Parse exactly one JSON value, rejecting ambiguous and nonfinite data."""
    return cast(JsonValue, json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    ))


def load_prompts(path: Path) -> list[JsonValue]:
    """Read an array; validate each record immediately before generation."""
    value = parse_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("prompt input must be a JSON array")
    return value


def load_definitions(path: Path) -> list[FunctionDefinition]:
    """Read a nonempty dynamic function set with unambiguous unique names."""
    definitions = TypeAdapter(list[FunctionDefinition]).validate_python(
        parse_json(path.read_text(encoding="utf-8")),
    )
    if not definitions:
        raise ValueError("function definitions must contain a function")
    names: set[str] = set()
    for definition in definitions:
        if definition.name in names:
            raise ValueError(f"duplicate function name: {definition.name!r}")
        names.add(definition.name)
    return definitions


def validate_result(
    text: str, prompt: str, definitions: list[FunctionDefinition],
) -> FunctionCall:
    """Validate generated JSON and dynamic arguments, without judging accuracy.

    Completion and token-time constraints must be guaranteed by the generator.
    This function can only establish properties of its finished JSON text.
    """
    if not isinstance(text, str):
        raise ValueError("generator must return JSON text")
    result = FunctionCall.model_validate(parse_json(text))
    if result.prompt != prompt:
        raise ValueError("result prompt does not exactly match input prompt")
    definition = next(
        (item for item in definitions if item.name == result.name), None,
    )
    if definition is None:
        raise ValueError(f"unknown function name: {result.name!r}")

    types: dict[str, Any] = {
        "number": int | float,
        "integer": int,
        "string": str,
        "boolean": bool,
        "null": type(None),
        "array": list[JsonValue],
        "object": dict[str, JsonValue],
    }
    # Internal names avoid collisions with BaseModel methods or private fields.
    fields: dict[str, Any] = {
        f"argument_{index}": (types[schema.type], Field(alias=name))
        for index, (name, schema) in enumerate(definition.parameters.items())
    }
    parameter_model = create_model(
        "Parameters",
        __config__=ConfigDict(
            strict=True, extra="forbid", allow_inf_nan=False,
        ),
        **fields,
    )
    parameter_model.model_validate(result.parameters)
    return result
