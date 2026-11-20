"""The layout of one output record: fixed JSON text and values to generate."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .schema import FunctionDef, ParamType


class FixedTextPart(BaseModel):
    """JSON text written as is: braces, keys, the prompt and the name.

    Attributes:
        text: The JSON text.
    """

    model_config = ConfigDict(frozen=True)

    text: str


class ArgumentGenerationPart(BaseModel):
    """A parameter value, including its following terminator ("," or "}").

    Attributes:
        name: The parameter name.
        type: The parameter type.
        terminator: "," before the next parameter, "}" after the last.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    type: ParamType
    terminator: Literal[",", "}"]


JsonPart = FixedTextPart | ArgumentGenerationPart


def escape_lt(text: str) -> str:
    """Write every ``<`` in ``text`` as ``\\u003c``.

    The SDK encodes text such as ``<|im_end|>`` as one special token, and
    its decode drops that token. Every special token starts with ``<``, so
    escaped text stays in ordinary tokens. In JSON text, where ``<`` occurs
    only inside strings, the escape keeps the same JSON value.

    Args:
        text: Any text, usually JSON.

    Returns:
        ``text`` without ``<``.
    """
    return text.replace("<", "\\u003c")


def json_string(text: str) -> str:
    """Return ``text`` as a JSON string literal with ``<`` escaped.

    Args:
        text: Any text.

    Returns:
        The JSON string literal, quotes included (see escape_lt).
    """
    return escape_lt(json.dumps(text))


def build_call_skeleton(
    user_prompt: str,
    fn: FunctionDef,
) -> list[JsonPart]:
    """Lay out ``{"prompt":...,"name":...,"parameters":{"k": ...}}``.

    The prompt and the name are fixed text: the prompt is copied from the
    input and the name was already chosen through the name tree. Only the
    parameter values are generated. Each slot owns its terminator (a comma,
    or the parameters' closing brace for the last slot), so no fixed text
    repeats it. With no parameters, everything is fixed text.

    Args:
        user_prompt: The user's request.
        fn: The selected function.

    Returns:
        The parts in output order.
    """
    head = FixedTextPart(
        text=f'{{"prompt":{json_string(user_prompt)},'
        f'"name":{json_string(fn.name)}'
    )
    if not fn.parameters:
        return [head, FixedTextPart(text=',"parameters":{}}')]
    parts: list[FixedTextPart | ArgumentGenerationPart] = [head]
    for index, (name, spec) in enumerate(fn.parameters.items()):
        prefix = ',"parameters":{' if index == 0 else ""
        # Keep ": ": without the space, Qwen3 never ended "2.0" in our runs.
        parts.append(FixedTextPart(text=prefix + json_string(name) + ": "))
        parts.append(
            ArgumentGenerationPart(
                name=name,
                type=spec.type,
                terminator="}" if index == len(fn.parameters) - 1 else ",",
            )
        )
    parts.append(FixedTextPart(text="}"))
    return parts
