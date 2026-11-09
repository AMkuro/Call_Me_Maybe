"""The forced part of the output and the places the model fills in."""

from pydantic import BaseModel

from .schema import FunctionDef, ParamType


class Forced(BaseModel):
    """Text written as is: keys, quotes, colons, braces."""

    text: str


class ValueSlot(BaseModel):
    """A place where the model generates the value of one parameter."""

    name: str
    type: ParamType


def build_skeleton(fn: FunctionDef) -> list[Forced | ValueSlot]:
    """Lay out what follows the chosen name: ``", "parameters": {...}}``."""
    raise NotImplementedError
