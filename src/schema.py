"""Pydantic models: function definitions and one output record."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

ParamType = Literal["number", "string", "boolean"]


class ParamSpec(BaseModel):
    """The type of one parameter (or of a return value)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ParamType


class FunctionDef(BaseModel):
    """One entry of functions_definition.json."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParamSpec]
    returns: ParamSpec


class FunctionCall(BaseModel):
    """One output record: exactly the keys prompt, name and parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]


def build_parameters_model(fn: FunctionDef) -> type[BaseModel]:
    """Build a model that checks the generated parameters of ``fn``."""
    types: dict[ParamType, type] = {
        "number": float, "string": str, "boolean": bool,
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
            frozen=True, extra="forbid", allow_inf_nan=False,
        ),
        **fields,
    )
