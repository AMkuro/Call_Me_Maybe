"""Pydantic models: function definitions and one output record."""

import json
import logging
import math
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel, BeforeValidator, ConfigDict, Field, create_model,
    field_validator, model_validator,
)

ParamType = Literal["number", "string", "boolean"]
_LOGGER = logging.getLogger(__name__)


def validate_number(value: object) -> int | float:
    """Keep a JSON number's decimal value, or reject unsafe conversion."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a JSON number argument")
    if isinstance(value, int):
        # A Python int may also have been constructed without parsing JSON.
        try:
            str(value)
        except ValueError as e:
            raise ValueError(
                f"integer exceeds JSON serialization limit: {e}",
            ) from e
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("number must be finite")
        candidate = float(value)
        if not math.isfinite(candidate):
            raise ValueError("decimal overflows the finite float range")
        if candidate == 0 and value != 0:
            raise ValueError("decimal underflows to zero")
        if Decimal(json.dumps(candidate, allow_nan=False)) == value:
            return candidate
        # A finite float bounds the size of this exact integral fallback.
        if value == value.to_integral_value():
            return int(value)
        raise ValueError("decimal precision would be lost in JSON output")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("number must be finite")
        return value
    raise ValueError("expected a JSON number, not a string or another type")


Number = Annotated[int | float, BeforeValidator(validate_number)]
Scalar = str | bool | Number


class ParamSpec(BaseModel):
    """The type of one parameter (or of a return value)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    type: ParamType


class FunctionDef(BaseModel):
    """One entry of functions_definition.json."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    description: str
    parameters: dict[str, ParamSpec]
    returns: ParamSpec

    @field_validator("name")
    @classmethod
    def require_name(cls, value: str) -> str:
        """Reject blank names without stripping valid identifiers."""
        if not value.strip():
            raise ValueError("function names must not be empty")
        return value

    @model_validator(mode="after")
    def warn_blank_metadata(self) -> "FunctionDef":
        """Warn about incomplete metadata while preserving its exact value."""
        if not self.description.strip():
            _LOGGER.warning("function %r: description is blank", self.name)
        for position, name in enumerate(self.parameters, 1):
            if not name.strip():
                _LOGGER.warning("function %r: parameter #%d name %r is blank",
                                self.name, position, name)
        return self


class FunctionCall(BaseModel):
    """One output record: exactly the keys prompt, name and parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    prompt: str
    name: str
    parameters: dict[str, Scalar]


def build_parameters_model(fn: FunctionDef) -> type[BaseModel]:
    """Build a model that checks the generated parameters of ``fn``."""
    types: dict[ParamType, Any] = {
        "number": Number, "string": str, "boolean": bool,
    }
    # Judgment: aliases preserve arbitrary JSON keys, including model_dump
    # and leading underscores, without creating conflicting attributes.
    fields: dict[str, Any] = {
        f"parameter_{index}": (types[spec.type], Field(alias=name))
        for index, (name, spec) in enumerate(fn.parameters.items())
    }
    return create_model(
        "Parameters",
        __config__=ConfigDict(
            frozen=True, extra="forbid", strict=True, allow_inf_nan=False,
        ),
        **fields,
    )
