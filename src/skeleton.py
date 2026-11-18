"""The layout of one output record: fixed JSON text and values to generate."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .schema import FunctionDef, ParamType


class FixedText(BaseModel):
    """JSON text written as is: braces, keys, the prompt and the name."""

    model_config = ConfigDict(frozen=True)

    text: str


class ValueSlot(BaseModel):
    """A parameter value, including its following terminator ("," or "}")."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: ParamType
    terminator: Literal[",", "}"]


def json_string(text: str) -> str:
    """Return ``text`` as a JSON string literal with ``<`` escaped.

    The SDK encodes text such as ``<|im_end|>`` as one special token, and
    its decode drops that token. Written as ``\\u003c`` the same JSON value
    stays in ordinary tokens. ``<`` occurs only inside strings here.
    """
    return json.dumps(text).replace("<", r"\u003c")


def build_call_skeleton(
    user_prompt: str, fn: FunctionDef,
) -> list[FixedText | ValueSlot]:
    """Lay out ``{"prompt":...,"name":...,"parameters":{"k": ...}}``.

    The prompt and the name are fixed text: the prompt is copied from the
    input and the name was already chosen through the name tree. Only the
    parameter values are generated. Each slot owns its terminator (a comma,
    or the parameters' closing brace for the last slot), so no fixed text
    repeats it. With no parameters, everything is fixed text.
    """
    head = FixedText(text=f'{{"prompt":{json_string(user_prompt)},'
                          f'"name":{json_string(fn.name)}')
    if not fn.parameters:
        return [head, FixedText(text=',"parameters":{}}')]
    parts: list[FixedText | ValueSlot] = [head]
    for index, (name, spec) in enumerate(fn.parameters.items()):
        prefix = ',"parameters":{' if index == 0 else ''
        # Keep ": ": without it Qwen3 never ended "2.0" (2026-11-18 GPU run).
        parts.append(FixedText(text=prefix + json_string(name) + ': '))
        parts.append(ValueSlot(
            name=name, type=spec.type,
            terminator='}' if index == len(fn.parameters) - 1 else ',',
        ))
    parts.append(FixedText(text='}'))
    return parts
