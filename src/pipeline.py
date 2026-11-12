"""Handle one prompt: name -> skeleton -> values -> validation."""

import json

from pydantic import BaseModel, ConfigDict, InstanceOf

from .decoder import select_next_token
from .errors import PromptError
from .json_io import loads
from .model_port import Model
from .name_trie import NameTrie, leaf_name, max_name_steps, next_tokens
from .progress import GenerationProgress
from .prompt import build_name_prompt, build_parameters_prompt
from .schema import FunctionCall, FunctionDef, build_parameters_model
from .skeleton import Forced, ValueSlot, build_skeleton
from .values import allowed_value_tokens, is_value_complete

MAX_VALUE_STEPS = 256


class GenerationResources(BaseModel):
    """Hold already validated resources by reference for one application run.

    Validate reference types, not dictionary contents or SDK initialization.
    Frozen prevents field replacement; shared dictionaries remain mutable
    and must not be changed after startup.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: InstanceOf[Model]
    vocab: InstanceOf[dict[int, bytes]]
    functions: InstanceOf[dict[str, FunctionDef]]
    trie: InstanceOf[NameTrie]


def process_prompt(
    user_prompt: str,
    resources: GenerationResources,
    *,
    progress: GenerationProgress | None = None,
) -> FunctionCall:
    """Select a name, generate parameters, then validate one JSON record."""
    if progress is not None and progress.available:
        progress.stage("select_name")
    fn = select_name(user_prompt, model=resources.model,
                     functions=resources.functions, trie=resources.trie,
                     progress=progress)
    if progress is not None and progress.available:
        progress.stage("generate_parameters")
    text = generate_parameters(user_prompt, fn, model=resources.model,
                               vocab=resources.vocab, progress=progress)
    if progress is not None and progress.available:
        progress.stage("validate_call")
    return validate_call(text, fn)


def select_name(
    user_prompt: str,
    *,
    model: Model,
    functions: dict[str, FunctionDef],
    trie: NameTrie,
    progress: GenerationProgress | None = None,
) -> FunctionDef:
    """Follow the name tree until a leaf, selecting one token at a time."""
    prompt_ids = model.encode(build_name_prompt(user_prompt, functions))
    generated: list[int] = []
    limit = max_name_steps(trie)
    while True:
        name = leaf_name(trie, generated)
        if name is not None:
            if progress is not None and progress.available:
                progress.function_selected(name)
            return functions[name]
        allowed = next_tokens(trie, generated)
        if len(generated) >= limit:
            raise PromptError(
                f"token limit {limit} reached before a terminator"
            )
        ids = prompt_ids + generated if len(allowed) > 1 else prompt_ids
        token, mode, raw_top = select_next_token(
            model, ids, allowed,
            include_raw_top=progress is not None and progress.available,
        )
        generated.append(token)
        if progress is not None and progress.available:
            progress.name_token(generated, mode=mode, raw_top=raw_top)


def generate_parameters(
    user_prompt: str,
    fn: FunctionDef,
    *,
    model: Model,
    vocab: dict[int, bytes],
    progress: GenerationProgress | None = None,
) -> str:
    """Append the forced JSON skeleton and each generated value in order."""
    ids = model.encode(build_parameters_prompt(user_prompt, fn))
    text = (
        f'{{"prompt": {json.dumps(user_prompt)}, '
        f'"name": {json.dumps(fn.name)}'
    )
    ids = ids + model.encode(text)
    for part in build_skeleton(fn):
        if isinstance(part, Forced):
            text = text + part.text
            ids = ids + model.encode(part.text)
        else:
            if progress is not None and progress.available:
                progress.parameter_started(part.name, part.type)
            value_ids = generate_value(part, ids, model=model, vocab=vocab,
                                       progress=progress)
            text = text + model.decode(value_ids)
            ids = ids + value_ids
    return text


def generate_value(
    slot: ValueSlot,
    ids: list[int],
    *,
    model: Model,
    vocab: dict[int, bytes],
    progress: GenerationProgress | None = None,
) -> list[int]:
    """Generate one value and delimiter, checking completion before the cap."""
    generated: list[int] = []
    try:
        while True:
            written = b"".join(vocab[token] for token in generated)
            if is_value_complete(written, slot.type, slot.terminator):
                return generated
            allowed = allowed_value_tokens(vocab, slot.type, written,
                                           slot.terminator)
            if len(generated) >= MAX_VALUE_STEPS:
                raise PromptError(
                    f"token limit {MAX_VALUE_STEPS} "
                    "reached before a terminator"
                )
            context = ids + generated if len(allowed) > 1 else ids
            token, mode, _ = select_next_token(model, context, allowed)
            generated.append(token)
            if progress is not None and progress.available:
                written = b"".join(vocab[item] for item in generated)
                progress.value_token(written, len(generated),
                                     token=token, mode=mode)
    except PromptError as e:
        raise PromptError(
            f"parameter {slot.name!r} ({slot.type}): {e}"
        ) from e


def validate_call(text: str, fn: FunctionDef) -> FunctionCall:
    """Parse and validate a record against the selected function."""
    try:
        call = FunctionCall.model_validate(loads(text))
        if call.name != fn.name:
            raise ValueError(
                f"function name mismatch: expected {fn.name!r}, "
                f"got {call.name!r}",
            )
        parameters = build_parameters_model(fn).model_validate(call.parameters)
    except ValueError as e:  # JSON errors and pydantic's are ValueErrors
        raise PromptError(f"invalid output: {e}") from e
    return FunctionCall(
        prompt=call.prompt, name=call.name,
        parameters=parameters.model_dump(by_alias=True),
    )
