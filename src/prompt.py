"""Replaceable instruction settings and Qwen's non-thinking chat format."""

import json

from .schema import FunctionDef, Scalar
from .skeleton import Forced, build_skeleton

NAME_INSTRUCTIONS = (
    "Choose the function matching the request. Reply with its exact name only."
    " Use the descriptions to match the requested action."
)
PARAMETER_INSTRUCTIONS = (
    "The user request will be fulfilled by calling the given function. "
    "Extract the arguments of that call from the user request: copy each "
    "value as it appears in the request; do not answer the request or "
    "compute a result. "
    "Keep the requested input values and strings exact. Return only a JSON "
    "object with prompt, name, parameters. "
    "Use JSON numbers, strings, true/false."
    " Convert descriptions of patterns into actual regular expressions; "
    "do not list the matched substrings. Express replacement symbols as "
    "their literal characters."
    " Do not add leading or trailing whitespace unless it belongs to the "
    "requested value."
    " Preserve numeric signs and decimal points."
)
# Worked examples shown before the request: (function, request, parameters).
PARAMETER_EXAMPLES: list[tuple[FunctionDef, str, dict[str, Scalar]]] = []


def _chat(instructions: str, content: str,
          examples: list[tuple[str, str]] | None = None) -> str:
    """Use the installed tokenizer's non-thinking ChatML completion prefix.

    Earlier (user, assistant) turns have no think block, as in the template.
    """
    turns = "".join(
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>\n"
        for user, answer in examples or []
    )
    return (
        f"<|im_start|>system\n{instructions}<|im_end|>\n{turns}"
        f"<|im_start|>user\n{content}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def build_name_prompt(
    user_prompt: str, functions: dict[str, FunctionDef]
) -> str:
    """Prompt for choosing the function name."""
    definitions = "\n\n".join(
        f"{fn.name}: {fn.description}\nArguments:" + "".join(
            f"\n    {name} (type={spec.type})"
            for name, spec in fn.parameters.items()
        ) for fn in functions.values()
    )
    return _chat(
        NAME_INSTRUCTIONS,
        f"Available functions:\n{definitions}\n\n"
        f"Request: {json.dumps(user_prompt)}",
    ) + "Function name:\n"


def _parameters_request(user_prompt: str, fn: FunctionDef) -> str:
    """User turn content: the user's request, then the call that fulfils it."""
    definition = json.dumps(fn.model_dump(), ensure_ascii=False,
                            separators=(",", ":"))
    return (f"User request: {json.dumps(user_prompt)}\n"
            f"Function to call: {definition}")


def example_answer(user_prompt: str, fn: FunctionDef,
                   parameters: dict[str, Scalar]) -> str:
    """The JSON text the pipeline would produce: same head and skeleton."""
    text = (f'{{"prompt": {json.dumps(user_prompt)}, '
            f'"name": {json.dumps(fn.name)}')
    for part in build_skeleton(fn):
        text += (part.text if isinstance(part, Forced) else
                 json.dumps(parameters[part.name]) + part.terminator)
    return text


def build_parameters_prompt(user_prompt: str, fn: FunctionDef) -> str:
    """Prompt for generating the parameter values of ``fn``."""
    examples = [(_parameters_request(request, example_fn),
                 example_answer(request, example_fn, values))
                for example_fn, request, values in PARAMETER_EXAMPLES]
    return _chat(PARAMETER_INSTRUCTIONS, _parameters_request(user_prompt, fn),
                 examples)
