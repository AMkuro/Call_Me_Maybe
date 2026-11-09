"""The forced part of the output and the places the model fills in."""

from pydantic import BaseModel, ConfigDict

from .schema import FunctionDef, ParamType


class Forced(BaseModel):
    """Text written as is: keys, quotes, colons, braces."""

    model_config = ConfigDict(frozen=True)

    text: str


class ValueSlot(BaseModel):
    """A place where the model generates the value of one parameter."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: ParamType


def build_skeleton(fn: FunctionDef) -> list[Forced | ValueSlot]:
    """Lay out what follows the chosen name: ``", "parameters": {...}}``.

    Judgment: lays out only what follows the name; the name is chosen
    through the tree, so it is not part of the skeleton. The head
    ``{"prompt": ..., "name": "`` is built by pipeline (the prompt is
    copied from the input, never generated). Open: should the skeleton
    include the head?
    """
    raise NotImplementedError
