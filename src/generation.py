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
    """Follow the name tree until a leaf, selecting one token at a time.

    No step limit is needed: each selected token is one of the allowed
    ones, so the path grows by one known step, and every path of the
    finite tree ends at a leaf.
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
    ids: list[int], *, model: Model, vocab: dict[int, bytes]
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
    text: str, context: list[int], *, model: Model, vocab: dict[int, bytes]
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
            model, context + committed, [token]
        )
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
    ``saved`` is the returned text: the same, except that an integer number
    gains ".0" (float_text). The context keeps the model's own integer.
    """
    prompt_ids = model.encode(build_parameters_prompt(user_prompt, fn))
    output_ids: list[int] = []
    fragments: list[str] = []
    saved: list[str] = []
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
    vocab: dict[int, bytes],
    progress: GenerationProgress | None = None,
) -> list[int]:
    """Generate one ValueSlot: a value and its terminator, within the cap."""
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
