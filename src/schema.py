"""Pydantic models: function definitions and one output record."""

import json
import logging
import math
from collections.abc import Mapping
from decimal import Decimal
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    create_model,
    field_serializer,
    field_validator,
    model_validator,
)

JsonValueTypes = Literal["number", "string", "boolean"]
_LOGGER = logging.getLogger(__name__)


def validate_number(value: object) -> float:
    """Return a finite float preserving the JSON decimal value.

    Already supplied floats are checked only for finiteness; their source
    decimal digits are not available for a precision comparison.

    Args:
        value: A finite float or an exactly parsed Decimal. Python ints,
            booleans and other types are rejected; no int fallback is used.

    Returns:
        The checked finite float.

    Raises:
        ValueError: The type is invalid, the value is nonfinite, overflows,
            underflows to zero, or loses precision in JSON float output.
    """
    if isinstance(value, bool):
        raise ValueError("boolean is not a JSON number argument")
    if isinstance(value, int):
        value = Decimal(value)
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
        raise ValueError("decimal precision would be lost in JSON output")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("number must be finite")
        return value
    raise ValueError("expected a JSON number, not a string or another type")


Number = Annotated[float, BeforeValidator(validate_number)]
Scalar = str | bool | Number


class ValueTypeSpec(BaseModel):
    """The type of one parameter (or of a return value).

    Attributes:
        type: "number", "string" or "boolean".
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    type: JsonValueTypes


class FunctionDef(BaseModel):
    """One entry of functions_definition.json, validated strictly.

    Unknown keys and values of another type are rejected. A blank
    description or parameter name is allowed but logged as a warning.

    Attributes:
        name: The function name; not blank.
        description: What the function does.
        parameters: The type of each parameter, read-only, in input order.
        returns: The type of the return value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    description: str
    parameters: Mapping[str, ValueTypeSpec]
    returns: ValueTypeSpec

    @field_validator("name")
    @classmethod
    def require_name(cls, value: str) -> str:
        """Reject an empty or whitespace-only name; keep others unchanged.

        Args:
            value: The name.

        Returns:
            ``value``.

        Raises:
            ValueError: ``value`` is blank.
        """
        if not value.strip():
            raise ValueError("function names must not be empty")
        return value

    @field_validator("parameters")
    @classmethod
    def _read_only(
        cls,
        parameters: Mapping[str, ValueTypeSpec],
    ) -> Mapping[str, ValueTypeSpec]:
        """Wrap the new dict Pydantic built, so no outside alias exists.

        Args:
            parameters: The validated parameters.

        Returns:
            A read-only view of ``parameters``.
        """
        return MappingProxyType(parameters)

    @field_serializer("parameters")
    def _plain_parameters(
        self,
        parameters: Mapping[str, ValueTypeSpec],
    ) -> dict[str, ValueTypeSpec]:
        """Serialize the read-only view as an ordinary dict.

        Args:
            parameters: The stored parameters.

        Returns:
            A dict with the same items.
        """
        return dict(parameters)

    @model_validator(mode="after")
    def warn_blank_metadata(self) -> "FunctionDef":
        """Log a warning for a blank description or parameter name.

        The values are kept unchanged.

        Returns:
            This definition.
        """
        if not self.description.strip():
            _LOGGER.warning("function %r: description is blank", self.name)
        for position, name in enumerate(self.parameters, 1):
            if not name.strip():
                _LOGGER.warning(
                    "function %r: parameter #%d name %r is blank",
                    self.name,
                    position,
                    name,
                )
        return self


class FunctionCallRequest(BaseModel):
    """One output record: exactly the keys prompt, name and parameters.

    Attributes:
        prompt: The user's request.
        name: The selected function name; no function is executed.
        parameters: String, boolean or checked float values by name.
            Matching a particular definition is checked by validate_call.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    prompt: str
    name: str
    parameters: dict[str, Scalar]

    # The generated JSON of this record, set only by pipeline.validate_call
    # after every check passed; None for a record built directly. It is not
    # part of the output schema and model_dump leaves it out.
    _generated_json: str | None = PrivateAttr(default=None)


def build_argument_validation_model(fn: FunctionDef) -> type[BaseModel]:
    """Return a strict model of ``fn``'s parameters, in their order.

    It requires exactly those names, each of its declared type (numbers
    through Number). Fields are named parameter_0, parameter_1, ... with
    the parameter names as aliases, so any name works, even one that is not
    an identifier or is a BaseModel attribute such as ``model_dump``.

    Args:
        fn: The function.

    Returns:
        The model class.
    """
    types: dict[JsonValueTypes, Any] = {
        "number": Number,
        "string": str,
        "boolean": bool,
    }
    fields: dict[str, Any] = {
        f"parameter_{index}": (types[spec.type], Field(alias=name))
        for index, (name, spec) in enumerate(fn.parameters.items())
    }
    return create_model(
        "Parameters",
        __config__=ConfigDict(
            frozen=True,
            extra="forbid",
            strict=True,
            allow_inf_nan=False,
        ),
        **fields,
    )
