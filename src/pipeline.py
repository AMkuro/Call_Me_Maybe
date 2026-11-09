"""Handle one prompt: name -> skeleton -> values -> check."""

import json

from .decoder import generate
from .errors import PromptError
from .model_port import Model
from .name_trie import NameTrie, leaf_name, next_tokens
from .prompt import build_name_prompt, build_parameters_prompt
from .schema import FunctionCall, FunctionDef, build_parameters_model
from .skeleton import Forced, ValueSlot, build_skeleton
from .values import allowed_value_tokens, is_value_complete

# Open: the limits are not decided (ADR-0003). These are placeholders.
MAX_NAME_STEPS = 32
MAX_VALUE_STEPS = 64


def select_name(
    user_prompt: str,
    functions: list[FunctionDef],
    model: Model,
    trie: NameTrie,
) -> FunctionDef:
    """Let the model choose a function, limited by the name tree.

    Effect: uses the model.
    """
    prompt_ids = model.encode(build_name_prompt(user_prompt, functions))

    def allowed_next(generated: list[int]) -> list[int] | None:
        if leaf_name(trie, generated) is not None:
            return None
        return next_tokens(trie, generated)

    generated = generate(
        model.logits, prompt_ids, allowed_next, MAX_NAME_STEPS
    )
    name = leaf_name(trie, generated)
    return next(fn for fn in functions if fn.name == name)


def generate_value(
    slot: ValueSlot,
    ids: list[int],
    model: Model,
    vocab: dict[int, str],
) -> list[int]:
    """Generate the tokens of one parameter value.

    Effect: uses the model.
    """

    def allowed_next(generated: list[int]) -> list[int] | None:
        written = model.decode(generated)
        if is_value_complete(written, slot.type):
            return None
        return allowed_value_tokens(vocab, slot.type, written)

    return generate(model.logits, ids, allowed_next, MAX_VALUE_STEPS)


def generate_parameters(
    user_prompt: str,
    fn: FunctionDef,
    model: Model,
    vocab: dict[int, str],
) -> str:
    """Generate the parameters, forcing the skeleton.

    Returns the full JSON text of the record.

    Effect: uses the model.

    Judgment: this function builds the head ``{"prompt": ..., "name": "...``
    itself and adds the rest from the skeleton and the generated values.
    Open: should the skeleton include the head?
    """
    ids = model.encode(build_parameters_prompt(user_prompt, fn))
    text = f'{{"prompt": {json.dumps(user_prompt)}, "name": "{fn.name}'
    ids = ids + model.encode(text)
    for part in build_skeleton(fn):
        if isinstance(part, Forced):
            text = text + part.text
            ids = ids + model.encode(part.text)
        else:
            value_ids = generate_value(part, ids, model, vocab)
            text = text + model.decode(value_ids)
            ids = ids + value_ids
    return text


def validate_call(text: str, fn: FunctionDef) -> FunctionCall:
    """Parse ``text`` as JSON and check it with the dynamic model."""
    try:
        call = FunctionCall.model_validate(json.loads(text))
        build_parameters_model(fn).model_validate(call.parameters)
    except ValueError as e:  # JSON errors and pydantic's are ValueErrors
        raise PromptError(f"invalid output: {e}") from e
    return call


def process_prompt(
    user_prompt: str,
    functions: list[FunctionDef],
    model: Model,
    trie: NameTrie,
    vocab: dict[int, str],
) -> FunctionCall:
    """Run the whole flow for one prompt. Raises PromptError on failure.

    Effect: uses the model.
    """
    fn = select_name(user_prompt, functions, model, trie)
    text = generate_parameters(user_prompt, fn, model, vocab)
    return validate_call(text, fn)
