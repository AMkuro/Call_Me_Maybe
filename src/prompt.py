"""Replaceable instruction settings and Qwen's non-thinking chat format."""

import json
from collections.abc import Mapping

from .schema import FunctionDef, Scalar
from .skeleton import FixedText, build_call_skeleton

NAME_INSTRUCTIONS = (
    "Choose the function matching the request. Reply with its exact name only."
    " Use the descriptions to match the requested action."
)
# Kept short: five more sentences changed no answer on 119 test requests.
PARAMETER_INSTRUCTIONS = (
    "Extract the arguments of that call from the user request: copy each "
    "value as it appears in the request; do not answer the request or "
    "compute a result. "
    "Return only a JSON object with prompt, name, parameters. "
    "Use JSON numbers, strings, true/false."
    " Convert descriptions of patterns into actual regular expressions; "
    "do not list the matched substrings."
)
# Worked examples shown before the request: (function, request, parameters).
# Empty: the application shows none. Filling it adds few-shot examples
# without changing how the prompt is built.
PARAMETER_EXAMPLES: list[tuple[FunctionDef, str, dict[str, Scalar]]] = []


def _format_chatml(instructions: str, content: str,
                   examples: list[tuple[str, str]] | None = None) -> str:
    """Return a ChatML prompt that ends where the assistant's answer starts.

    The text equals the Qwen3 tokenizer's chat template with thinking
    turned off (apply_chat_template with enable_thinking=False): the
    answer starts after an empty think block.

    Args:
        instructions: The system message.
        content: The user message.
        examples: Earlier (user, assistant) turns, shown without a think
            block, or None.

    Returns:
        The prompt text.
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
    user_prompt: str, functions: Mapping[str, FunctionDef]
) -> str:
    """Return the prompt for choosing a function name.

    It lists every definition and ends with "Function name:" and a line
    break, so the model writes a name next.

    Args:
        user_prompt: The user's request.
        functions: The definitions, in the order to list them.

    Returns:
        The prompt text.
    """
    definitions = "\n".join(
        f"{fn.name}: {fn.description} Arguments: " + ", ".join(
            f"{name} ({spec.type})" for name, spec in fn.parameters.items()
        ) for fn in functions.values()
    )
    return _format_chatml(
        NAME_INSTRUCTIONS,
        f"Available functions:\n{definitions}\n\n"
        f"Request: {json.dumps(user_prompt)}",
    ) + "Function name:\n"


def _parameters_request(user_prompt: str, fn: FunctionDef) -> str:
    """Return the user message: the request, then the function to call.

    Args:
        user_prompt: The user's request.
        fn: The selected function, shown as JSON without "returns".

    Returns:
        The message text.
    """
    # "returns" is left out: it does not help extract the arguments.
    definition = json.dumps(fn.model_dump(exclude={"returns"}),
                            ensure_ascii=False, separators=(",", ":"))
    return (f"User request: {json.dumps(user_prompt)}\n"
            f"Function to call: {definition}")


def build_example_call_json(user_prompt: str, fn: FunctionDef,
                            parameters: dict[str, Scalar]) -> str:
    """Return the JSON record of a worked example, laid out like the output.

    The fixed text comes from the same skeleton as generate_call_json.

    Args:
        user_prompt: The example request.
        fn: The example function.
        parameters: The example values, each written with json.dumps.

    Returns:
        The JSON text.
    """
    text = ""
    for part in build_call_skeleton(user_prompt, fn):
        if isinstance(part, FixedText):
            text += part.text
        else:
            text += json.dumps(parameters[part.name]) + part.terminator
    return text


def build_parameters_prompt(user_prompt: str, fn: FunctionDef) -> str:
    """Return the prompt for generating the parameter values of ``fn``.

    The worked examples in PARAMETER_EXAMPLES, if any, come first.

    Args:
        user_prompt: The user's request.
        fn: The selected function.

    Returns:
        The prompt text.
    """
    examples = [(_parameters_request(request, example_fn),
                 build_example_call_json(request, example_fn, values))
                for example_fn, request, values in PARAMETER_EXAMPLES]
    return _format_chatml(PARAMETER_INSTRUCTIONS,
                          _parameters_request(user_prompt, fn), examples)
