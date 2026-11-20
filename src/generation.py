"""Generate constrained name and JSON tokens with explicit model resources."""

from collections.abc import Mapping

from .decoder import infer_and_select_next_token
from .errors import PromptError
from .model_port import Model
from .name_trie import NameTrie
from .progress import GenerationProgress, observing
from .prompt import build_arguments_prompt, build_name_prompt
from .schema import FunctionDef
from .skeleton import (
    ArgumentGenerationPart,
    FixedTextPart,
    build_call_skeleton,
)
from .values import (
    allowed_tokens_for_arguments,
    is_argument_generation_complete,
)

MAX_VALUE_STEPS = 256
# Bound repeated digit generation before it costs 256 forwards;
# this token budget is separate from numeric precision and range checks.
MAX_NUMBER_STEPS = 32


def select_function(
    user_prompt: str,
    *,
    model: Model,
    functions: Mapping[str, FunctionDef],
    trie: NameTrie,
    progress: GenerationProgress | None = None,
) -> FunctionDef:
    """Let the model write a defined function name; return its definition.

    Tokens are chosen one at a time among those the name tree allows,
    until a complete name (a leaf) is reached. No step limit is needed:
    every path of the finite tree ends at a leaf.

    Args:
        user_prompt: The user's request.
        model: The model.
        functions: The definitions keyed by name.
        trie: The name tree of ``functions``.
        progress: Receives each step, or None.

    Returns:
        The definition of the chosen name.

    Raises:
        PromptError: Generation fails; the message starts with
            "function name: ".
    """
    prompt_ids = model.encode(build_name_prompt(user_prompt, functions))
    generated: list[int] = []
    try:
        while True:
            name = trie.leaf_name(generated)
            if name is not None:
                if observing(progress):
                    progress.function_selected(name)
                return functions[name]
            allowed = trie.next_tokens(generated)
            token, mode, raw_top = infer_and_select_next_token(
                model,
                prompt_ids + generated,
                allowed,
                include_raw_top=observing(progress),
            )
            generated.append(token)
            if observing(progress):
                progress.name_token(generated, mode=mode, raw_top=raw_top)
    except PromptError as e:
        raise PromptError(f"function name: {e}") from e


def _decode_tokens_with_vocab_check(
    ids: list[int], *, model: Model, vocab: Mapping[int, bytes]
) -> str:
    """Return the text of ``ids``, checked against their vocabulary bytes.

    Args:
        ids: Output token IDs.
        model: Decodes the IDs.
        vocab: The bytes of each token.

    Returns:
        The decoded text.

    Raises:
        PromptError: An ID has no vocabulary bytes, the bytes are not
            UTF-8, or the SDK's decoding differs from them (it drops
            special tokens, for example).
    """
    try:
        expected = b"".join(vocab[token] for token in ids).decode("utf-8")
    except (KeyError, UnicodeDecodeError) as e:
        raise PromptError(f"invalid output token bytes: {e}") from e
    text = model.decode(ids)
    if text != expected:
        raise PromptError("output token bytes differ from SDK decoding")
    return text


def _commit_fixed_text(
    text: str,
    context: list[int],
    *,
    model: Model,
    vocab: Mapping[int, bytes],
    progress: GenerationProgress | None = None,
) -> list[int]:
    """Return the token IDs of the fixed JSON ``text``.

    ``text`` is encoded once and must decode back to itself. Like every
    output token, each ID is preceded by one forward pass on ``context``
    plus the IDs of ``text`` committed so far, although its scores are not
    used.

    Args:
        text: Fixed JSON text from the skeleton.
        context: The prompt and the output before ``text``; not modified.
        model: The model.
        vocab: The bytes of each token.
        progress: Receives committed JSON tokens, or None.

    Returns:
        The token IDs of ``text``.

    Raises:
        PromptError: ``text`` does not survive encoding and decoding, or
            the model fails.
    """
    tokens = model.encode(text)
    if (
        _decode_tokens_with_vocab_check(tokens, model=model, vocab=vocab)
        != text
    ):
        raise PromptError("fixed JSON text differs from its encoded tokens")
    committed: list[int] = []
    for token in tokens:
        chosen, _, _ = infer_and_select_next_token(
            model, context + committed, [token]
        )
        committed.append(chosen)
        if observing(progress):
            progress.output_token(
                vocab[chosen], token=chosen, mode="forced", fixed=True
            )
    return committed


def generate_call_json(
    user_prompt: str,
    fn: FunctionDef,
    *,
    model: Model,
    vocab: Mapping[int, bytes],
    progress: GenerationProgress | None = None,
) -> tuple[str, list[slice]]:
    """Generate raw JSON and locate its number slots for normalization.

    Fixed parts of the layout (build_call_skeleton) are committed as is,
    and the model generates each parameter value.

    Args:
        user_prompt: The user's request, copied into the record.
        fn: The selected function.
        model: The model.
        vocab: The bytes of each token.
        progress: Receives each step, or None.

    Returns:
        The unmodified raw_json and ordered, nonoverlapping number-slot
        slices into that pre-normalization Python string (not bytes or
        tokens). Starts are inclusive, stops exclusive, and steps None.
        Each slice includes the slot's terminator ("," or "}").

    Raises:
        PromptError: A value cannot be generated, or SDK decoding differs
            from the vocabulary bytes of a fixed part, slot or full output.
    """
    prompt_ids = model.encode(build_arguments_prompt(user_prompt, fn))
    # Only this loop extends output_ids; each part gets the prompt and the
    # output so far as context and returns its own IDs.
    output_ids: list[int] = []
    number_slot_slices: list[slice] = []
    generated_char_count = 0
    for part in build_call_skeleton(user_prompt, fn):
        context = prompt_ids + output_ids
        if isinstance(part, FixedTextPart):
            part_ids = _commit_fixed_text(
                part.text, context, model=model, vocab=vocab, progress=progress
            )
            part_text = part.text
        else:
            if observing(progress):
                progress.parameter_started(part.name, part.type)
            part_ids = generate_argument_part(
                part, context, model=model, vocab=vocab, progress=progress
            )
            part_text = _decode_tokens_with_vocab_check(
                part_ids, model=model, vocab=vocab
            )
            if part.type == "number":
                number_slot_slices.append(
                    slice(
                        generated_char_count,
                        generated_char_count + len(part_text),
                    )
                )
        generated_char_count += len(part_text)
        output_ids.extend(part_ids)
    text = _decode_tokens_with_vocab_check(
        output_ids, model=model, vocab=vocab
    )
    return text, number_slot_slices


def generate_argument_part(
    slot: ArgumentGenerationPart,
    ids: list[int],
    *,
    model: Model,
    vocab: Mapping[int, bytes],
    progress: GenerationProgress | None = None,
) -> list[int]:
    """Generate one value and its terminator; return their token IDs.

    A number may take up to MAX_NUMBER_STEPS tokens and other types up to
    MAX_VALUE_STEPS.

    Args:
        slot: The parameter and its terminator.
        ids: The context; not modified.
        model: The model.
        vocab: The bytes of each token.
        progress: Receives each step, or None.

    Returns:
        The token IDs of the value and its terminator.

    Raises:
        PromptError: No token can continue the value, the limit is reached
            before the terminator, or the model fails. The message names
            the parameter and its type.
    """
    generated: list[int] = []
    written = b""  # the bytes of generated, extended with each token
    limit = MAX_NUMBER_STEPS if slot.type == "number" else MAX_VALUE_STEPS
    try:
        while True:
            if is_argument_generation_complete(
                written, slot.type, slot.terminator
            ):
                return generated
            allowed = allowed_tokens_for_arguments(
                vocab, slot.type, written, slot.terminator
            )
            if len(generated) >= limit:
                raise PromptError(
                    f"token limit {limit} reached before a terminator"
                )
            token, mode, _ = infer_and_select_next_token(
                model, ids + generated, allowed
            )
            generated.append(token)
            written += vocab[token]
            if observing(progress):
                progress.output_token(vocab[token], token=token, mode=mode)
                progress.value_token(
                    written, len(generated), token=token, mode=mode
                )
    except PromptError as e:
        raise PromptError(f"parameter {slot.name!r} ({slot.type}): {e}") from e
