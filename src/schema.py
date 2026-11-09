"""Pydantic models: function definitions and one output record."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

ParamType = Literal["number", "string", "boolean"]


class ParamSpec(BaseModel):
    """The type of one parameter (or of a return value)."""

    model_config = ConfigDict(frozen=True)

    type: ParamType


class FunctionDef(BaseModel):
    """One entry of functions_definition.json."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, ParamSpec]
    returns: ParamSpec


class FunctionCall(BaseModel):
    """One output record: exactly the keys prompt, name and parameters."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    name: str
    parameters: dict[str, Any]


def build_parameters_model(fn: FunctionDef) -> type[BaseModel]:
    """Build a model that checks the generated parameters of ``fn``."""
    raise NotImplementedError
