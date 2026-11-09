"""The forced part of the output and the places the model fills in."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .schema import FunctionDef, ParamType


class Forced(BaseModel):
    """Text written as is: keys, quotes, colons, braces."""

    model_config = ConfigDict(frozen=True)

    text: str


class ValueSlot(BaseModel):
    """A parameter value, including its following JSON delimiter."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: ParamType
    terminator: Literal[",", "}"]


def build_skeleton(fn: FunctionDef) -> list[Forced | ValueSlot]:
    """Lay out what follows the JSON-escaped name: ``, "parameters": {...}}``.

    Judgment: lays out only what follows the name; the name is chosen
    through the tree, so it is not part of the skeleton. The head
    ``{"prompt": ..., "name": "..."`` is built by pipeline (the prompt is
    copied from the input, never generated). Open: should the skeleton
    include the head?

    Each slot owns its following comma, or the parameters' closing brace
    for the last slot. Forced text must not repeat that delimiter
    (ADR-0012). With no parameters, force ``, "parameters": {}}``.
    """
    raise NotImplementedError
