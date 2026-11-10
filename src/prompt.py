"""Replaceable instruction settings and Qwen's non-thinking chat format."""

import json

from .schema import FunctionDef

NAME_INSTRUCTIONS = (
    "Choose the function matching the request. Reply with its exact name only."
    " Use the descriptions to match the requested action."
)
PARAMETER_INSTRUCTIONS = (
    "Extract arguments for the selected function; do not execute it. "
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


def _chat(instructions: str, content: str) -> str:
    """Use the installed tokenizer's non-thinking ChatML completion prefix."""
    return (
        f"<|im_start|>system\n{instructions}<|im_end|>\n"
        f"<|im_start|>user\n{content}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def build_name_prompt(
    user_prompt: str, functions: list[FunctionDef]
) -> str:
    """Prompt for choosing the function name."""
    definitions = "\n".join(
        f"{fn.name}: {fn.description} Arguments: " + ", ".join(
            f"{name} ({spec.type})" for name, spec in fn.parameters.items()
        ) for fn in functions
    )
    return _chat(
        NAME_INSTRUCTIONS,
        f"Available functions:\n{definitions}\n\n"
        f"Request: {json.dumps(user_prompt)}",
    ) + "Function name:\n"


def build_parameters_prompt(user_prompt: str, fn: FunctionDef) -> str:
    """Prompt for generating the parameter values of ``fn``."""
    definition = json.dumps(fn.model_dump(), ensure_ascii=False,
                            separators=(",", ":"))
    return _chat(PARAMETER_INSTRUCTIONS,
                 f"Function: {definition}\nRequest: {json.dumps(user_prompt)}")
