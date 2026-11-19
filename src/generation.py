"""Generate constrained name and JSON tokens with explicit model resources."""

from collections.abc import Mapping

from .decoder import infer_and_select_next_token
from .errors import PromptError
from .model_port import Model
from .name_trie import NameTrie
from .progress import GenerationProgress, observing
from .prompt import build_name_prompt, build_parameters_prompt
from .schema import FunctionDef
from .skeleton import FixedText, ValueSlot, build_call_skeleton
from .values import allowed_slot_tokens, float_text, is_slot_complete

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


def _decode_and_verify_output(
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
    text: str, context: list[int], *, model: Model, vocab: Mapping[int, bytes]
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

    Returns:
        The token IDs of ``text``.

    Raises:
        PromptError: ``text`` does not survive encoding and decoding, or
            the model fails.
    """
    tokens = model.encode(text)
    if _decode_and_verify_output(tokens, model=model, vocab=vocab) != text:
        raise PromptError("fixed JSON text differs from its encoded tokens")
    committed: list[int] = []
    for token in tokens:
        chosen, _, _ = infer_and_select_next_token(
            model, context + committed, [token]
        )
        committed.append(chosen)
    return committed


def generate_call_json(
    user_prompt: str,
    fn: FunctionDef,
    *,
    model: Model,
    vocab: Mapping[int, bytes],
    progress: GenerationProgress | None = None,
) -> str:
    """Write the JSON record of one call to ``fn``; return its saved text.

    Fixed parts of the layout (build_call_skeleton) are committed as is,
    and the model generates each parameter value.

    Args:
        user_prompt: The user's request, copied into the record.
        fn: The selected function.
        model: The model.
        vocab: The bytes of each token.
        progress: Receives each step, or None.

    Returns:
        The decoded output, except that a number written as an integer
        gains ".0" (float_text); the model's context keeps the integer it
        wrote.

    Raises:
        PromptError: A value cannot be generated, or the decoded output
            differs from the parts it was built from.
    """
    prompt_ids = model.encode(build_parameters_prompt(user_prompt, fn))
    # Only this loop extends output_ids; each part gets the prompt and the
    # output so far as context and returns its own IDs.
    output_ids: list[int] = []
    fragments: list[str] = []  # each part's expected text, to check the whole
    saved: list[str] = []  # the same, with ".0" added to integer numbers
    for part in build_call_skeleton(user_prompt, fn):
        context = prompt_ids + output_ids
        if isinstance(part, FixedText):
            part_ids = _commit_fixed_text(
                part.text, context, model=model, vocab=vocab
            )
            fragments.append(part.text)
            saved.append(part.text)
        else:
            if observing(progress):
                progress.parameter_started(part.name, part.type)
            part_ids = generate_slot(
                part, context, model=model, vocab=vocab, progress=progress
            )
            fragments.append(
                _decode_and_verify_output(part_ids, model=model, vocab=vocab)
            )
            saved.append(
                float_text(fragments[-1])
                if part.type == "number"
                else fragments[-1]
            )
        output_ids.extend(part_ids)
    text = _decode_and_verify_output(output_ids, model=model, vocab=vocab)
    if text != "".join(fragments):
        raise PromptError("full JSON decoding differs from its fragments")
    return "".join(saved)


def generate_slot(
    slot: ValueSlot,
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
            if is_slot_complete(written, slot.type, slot.terminator):
                return generated
            allowed = allowed_slot_tokens(
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
                progress.value_token(
                    written, len(generated), token=token, mode=mode
                )
    except PromptError as e:
        raise PromptError(f"parameter {slot.name!r} ({slot.type}): {e}") from e
