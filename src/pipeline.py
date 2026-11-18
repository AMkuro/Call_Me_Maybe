"""Handle one prompt: name -> skeleton -> values -> validation."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeGuard

from pydantic import (
    BaseModel, ConfigDict, InstanceOf, field_validator, model_validator,
)

from .decoder import infer_and_select_next_token
from .errors import PromptError
from .json_io import loads
from .model_port import Model
from .name_trie import NameTrie, leaf_name, max_name_steps, next_tokens
from .progress import GenerationProgress
from .prompt import build_name_prompt, build_parameters_prompt
from .schema import FunctionCall, FunctionDef, build_parameters_model
from .skeleton import FixedText, ValueSlot, build_call_skeleton
from .values import allowed_slot_tokens, is_slot_complete

MAX_VALUE_STEPS = 256


def _observing(
    progress: GenerationProgress | None,
) -> TypeGuard[GenerationProgress]:
    """Whether to notify ``progress`` and build what only it uses.

    Checked at every notification, because ``available`` turns False when
    the diagnostic stream fails during a prompt.
    """
    return progress is not None and progress.available


class GenerationResources(BaseModel):
    """Combine validated resources for one application run.

    model, vocab and trie are kept by reference. The trie guarantees its
    own paths and immutability; this class only checks that its names are
    exactly the function names. functions is held as a read-only copy, so
    that match cannot be broken later. vocab contents and SDK state are not
    checked, and the shared vocab dict must not be changed after startup.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: InstanceOf[Model]
    vocab: InstanceOf[dict[int, bytes]]
    functions: Mapping[str, InstanceOf[FunctionDef]]
    trie: InstanceOf[NameTrie]

    @field_validator("functions")
    @classmethod
    def _read_only(
        cls, functions: Mapping[str, FunctionDef],
    ) -> Mapping[str, FunctionDef]:
        """Wrap the new dict Pydantic built, so no outside alias exists."""
        return MappingProxyType(functions)

    @model_validator(mode="after")
    def _names_match_trie(self) -> "GenerationResources":
        """Require each key to be its definition's name and a trie name."""
        for key, fn in self.functions.items():
            if fn.name != key:
                raise ValueError(f"functions[{key!r}] is named {fn.name!r}")
        if self.functions.keys() != self.trie.names:
            raise ValueError("trie names differ from the function names")
        return self


def process_prompt(
    user_prompt: str,
    resources: GenerationResources,
    *,
    progress: GenerationProgress | None = None,
) -> FunctionCall:
    """Select a name, generate parameters, then validate one JSON record."""
    if _observing(progress):
        progress.stage("select_function")
    fn = select_function(user_prompt, model=resources.model,
                         functions=resources.functions, trie=resources.trie,
                         progress=progress)
    if _observing(progress):
        progress.stage("generate_call_json")
    text = generate_call_json(user_prompt, fn, model=resources.model,
                              vocab=resources.vocab, progress=progress)
    if _observing(progress):
        progress.stage("validate_call")
    call = validate_call(text, fn)
    if call.prompt != user_prompt:
        raise PromptError("invalid output: prompt differs from the input")
    # Preserve the LLM-selected number lexemes after successful validation.
    call._generated_json = text
    return call


def select_function(
    user_prompt: str,
    *,
    model: Model,
    functions: Mapping[str, FunctionDef],
    trie: NameTrie,
    progress: GenerationProgress | None = None,
) -> FunctionDef:
    """Follow the name tree until a leaf, selecting one token at a time."""
    prompt_ids = model.encode(build_name_prompt(user_prompt, functions))
    generated: list[int] = []
    limit = max_name_steps(trie)
    try:
        while True:
            name = leaf_name(trie, generated)
            if name is not None:
                if _observing(progress):
                    progress.function_selected(name)
                return functions[name]
            allowed = next_tokens(trie, generated)
            if len(generated) >= limit:
                raise PromptError(
                    f"token limit {limit} reached before a terminator"
                )
            token, mode, raw_top = infer_and_select_next_token(
                model, prompt_ids + generated, allowed,
                include_raw_top=_observing(progress),
            )
            generated.append(token)
            if _observing(progress):
                progress.name_token(generated, mode=mode, raw_top=raw_top)
    except PromptError as e:
        raise PromptError(f"function name: {e}") from e


def _decode_and_verify_output(
    ids: list[int], *, model: Model, vocab: dict[int, bytes],
) -> str:
    """Require SDK decoding to preserve the grammar's exact UTF-8 bytes."""
    try:
        expected = b"".join(vocab[token] for token in ids).decode("utf-8")
    except (KeyError, UnicodeDecodeError) as e:
        raise PromptError(f"invalid output token bytes: {e}") from e
    text = model.decode(ids)
    if text != expected:
        raise PromptError("output token bytes differ from SDK decoding")
    return text


def _commit_fixed_text(
    text: str, context: list[int], *, model: Model, vocab: dict[int, bytes],
) -> list[int]:
    """Return the IDs of a fixed JSON fragment, each after one forward.

    The fragment is encoded once and must decode back to ``text``. Method B
    (owner decision): although each ID is already fixed and its scores are
    unused, the model runs once before it, on ``context`` plus the IDs of
    this fragment committed so far, as before every output token.
    """
    tokens = model.encode(text)
    if _decode_and_verify_output(tokens, model=model, vocab=vocab) != text:
        raise PromptError("fixed JSON text differs from its encoded tokens")
    committed: list[int] = []
    for token in tokens:
        chosen, _, _ = infer_and_select_next_token(
            model, context + committed, [token])
        committed.append(chosen)
    return committed


def generate_call_json(
    user_prompt: str,
    fn: FunctionDef,
    *,
    model: Model,
    vocab: dict[int, bytes],
    progress: GenerationProgress | None = None,
) -> str:
    """Commit JSON IDs in skeleton order and verify the decoded record.

    Only this function extends ``output_ids``. Each part receives the
    prompt and the output so far as its context, and returns its own IDs.
    ``fragments`` holds each part's expected text to check the whole.
    """
    prompt_ids = model.encode(build_parameters_prompt(user_prompt, fn))
    output_ids: list[int] = []
    fragments: list[str] = []
    for part in build_call_skeleton(user_prompt, fn):
        context = prompt_ids + output_ids
        if isinstance(part, FixedText):
            part_ids = _commit_fixed_text(part.text, context, model=model,
                                          vocab=vocab)
            fragments.append(part.text)
        else:
            if _observing(progress):
                progress.parameter_started(part.name, part.type)
            part_ids = generate_slot(part, context, model=model, vocab=vocab,
                                     progress=progress)
            fragments.append(_decode_and_verify_output(part_ids, model=model,
                                                       vocab=vocab))
        output_ids.extend(part_ids)
    text = _decode_and_verify_output(output_ids, model=model, vocab=vocab)
    if text != "".join(fragments):
        raise PromptError("full JSON decoding differs from its fragments")
    return text


def generate_slot(
    slot: ValueSlot,
    ids: list[int],
    *,
    model: Model,
    vocab: dict[int, bytes],
    progress: GenerationProgress | None = None,
) -> list[int]:
    """Generate one ValueSlot: a value and its terminator, within the cap."""
    generated: list[int] = []
    written = b""  # the bytes of generated, extended with each token
    try:
        while True:
            if is_slot_complete(written, slot.type, slot.terminator):
                return generated
            allowed = allowed_slot_tokens(vocab, slot.type, written,
                                          slot.terminator)
            if len(generated) >= MAX_VALUE_STEPS:
                raise PromptError(
                    f"token limit {MAX_VALUE_STEPS} "
                    "reached before a terminator"
                )
            token, mode, _ = infer_and_select_next_token(
                model, ids + generated, allowed)
            generated.append(token)
            written += vocab[token]
            if _observing(progress):
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
