"""Handle one prompt: name -> skeleton -> values -> check."""

import json
from collections.abc import Callable

from .decoder import TokenChoice, generate
from .errors import PromptError
from .json_io import loads
from .model_port import Model
from .name_trie import NameTrie, leaf_name, max_name_steps, next_tokens
from .prompt import build_name_prompt, build_parameters_prompt
from .schema import FunctionCall, FunctionDef, build_parameters_model
from .skeleton import Forced, ValueSlot, build_skeleton
from .values import allowed_value_tokens, is_value_complete

# Provisional value cap, to be re-evaluated with stress cases (ADR-0013).
MAX_VALUE_STEPS = 256


def _diagnostic_text(model: Model, ids: list[int]) -> str:
    """Render token text without letting a diagnostic discard a result.

    Effect: uses SDK decode; failures are shown as unavailable text.
    """
    try:
        return model.decode(ids) or "<EOS/special>"
    except PromptError:
        return f"<text unavailable; ids={ids}>"


def _report_choice(
    choice: TokenChoice, model: Model, report: Callable[[str, str], None],
    warn: bool,
) -> None:
    """Attach readable token text to an already-computed observation.

    Effect: decodes through the SDK and reports to the terminal/log sink.
    """
    mode = "forced" if choice.forced else "greedy"
    text = _diagnostic_text(model, [choice.token])
    report("choice", f"step={choice.step} {mode} id={choice.token} "
           f"text={text!r}")
    if choice.allowed_mass is not None:
        raw_text = _diagnostic_text(model, [choice.raw_top]) if (
            choice.raw_top is not None
        ) else ""
        metrics = (f"allowed_mass={choice.allowed_mass:.6g} "
                   f"gap={choice.margin:.6g} raw_top={choice.raw_top} "
                   f"raw_text={raw_text!r} selected={choice.token}")
        report("metrics", metrics)
        if warn and choice.constrained:
            report("warning", "name_selection_constrained: "
                   f"step={choice.step} " + metrics)
    else:
        report("metrics", "not computed (forced token)")


def _text(vocab: dict[int, bytes], tokens: list[int]) -> str:
    """Token bytes as text; partial UTF-8 stays visible as ``\\xe6``."""
    return b"".join(vocab.get(token, b"") for token in tokens).decode(
        "utf-8", "backslashreplace")


def _trace(
    choice: TokenChoice, vocab: dict[int, bytes], stage: str,
    parameter: str, preview: tuple[str, str, str], done: bool,
) -> str:
    """One viewer event (ADR-0021) from an already-computed decision.

    ``preview`` is (fixed, value, rest) of the record being written.
    """

    def name(token: int) -> str:
        return _text(vocab, [token]) if token in vocab else f"<id {token}>"

    fixed, value, rest = preview
    return json.dumps({
        "event": "token", "stage": stage, "parameter": parameter,
        "step": choice.step, "forced": choice.forced,
        "selected": [choice.token, name(choice.token)],
        "raw": [[token, name(token), logit, allowed]
                for token, logit, allowed in choice.raw],
        "masked": [[token, name(token), logit]
                   for token, logit in choice.masked],
        "fixed": fixed, "value": value, "rest": rest, "done": done,
    })


def _rest(parts: list[Forced | ValueSlot]) -> str:
    """The skeleton still to come; ungenerated values show as ``…``."""
    return "".join(part.text if isinstance(part, Forced)
                   else "…" + part.terminator for part in parts)


def select_name(
    user_prompt: str,
    functions: dict[str, FunctionDef],
    model: Model,
    trie: NameTrie,
    report: Callable[[str, str], None] | None = None,
    vocab: dict[int, bytes] | None = None,
    top_k: int = 0,
) -> FunctionDef:
    """Let the model choose a function, limited by the name tree.

    ``top_k`` > 0 also reports viewer events, with token text from ``vocab``.

    Effect: uses the model.
    """
    prompt_ids = model.encode(build_name_prompt(user_prompt, functions))
    head = f'{{"prompt": {json.dumps(user_prompt)}, "name": "'

    def allowed_next(generated: list[int]) -> list[int] | None:
        if leaf_name(trie, generated) is not None:
            return None
        return next_tokens(trie, generated)

    def observed(choice: TokenChoice) -> None:
        """Show name prefixes and diagnostics without another forward pass."""
        if report is not None:
            report("value", _diagnostic_text(model, choice.tokens))
            _report_choice(choice, model, report, warn=True)
            if top_k:
                prefix = _text(vocab or {}, choice.tokens)
                report("trace", _trace(
                    choice, vocab or {}, "select_name", "",
                    (head, prefix, '", "parameters": {…}}'),
                    leaf_name(trie, choice.tokens) is not None,
                ))

    generated = generate(
        model.logits, prompt_ids, allowed_next, max_name_steps(trie),
        on_token=observed if report is not None else None, top_k=top_k,
    )
    name = leaf_name(trie, generated)
    if name is None:
        raise PromptError("name generation ended without a function name")
    if report is not None:
        report("function", name)
    return functions[name]


def generate_value(
    slot: ValueSlot,
    ids: list[int],
    model: Model,
    vocab: dict[int, bytes],
    report: Callable[[str, str], None] | None = None,
    top_k: int = 0,
    preview: tuple[str, str] = ("", ""),
) -> list[int]:
    """Generate one parameter value and its following JSON delimiter.

    ``top_k`` > 0 also reports viewer events; ``preview`` is the record
    text before and after this value.

    Effect: uses the model.
    """

    def allowed_next(generated: list[int]) -> list[int] | None:
        written = b"".join(vocab[token] for token in generated)
        if is_value_complete(written, slot.type, slot.terminator):
            return None
        return allowed_value_tokens(vocab, slot.type, written, slot.terminator)

    def observed(choice: TokenChoice) -> None:
        """Preview full UTF-8 characters and show the chosen token."""
        if report is not None:
            written = b"".join(vocab[token] for token in choice.tokens)
            report("value", written.decode("utf-8", errors="ignore"))
            report("value_tokens", str(choice.step))
            _report_choice(choice, model, report, warn=False)
            if top_k:
                fixed, rest = preview
                report("trace", _trace(
                    choice, vocab, "generate_parameters",
                    f"{slot.name} ({slot.type})",
                    (fixed, _text(vocab, choice.tokens), rest),
                    is_value_complete(written, slot.type, slot.terminator),
                ))

    try:
        return generate(
            model.logits, ids, allowed_next, MAX_VALUE_STEPS,
            on_token=observed if report is not None else None, top_k=top_k,
        )
    except PromptError as e:
        raise PromptError(
            f"parameter {slot.name!r} ({slot.type}): {e}"
        ) from e


def generate_parameters(
    user_prompt: str,
    fn: FunctionDef,
    model: Model,
    vocab: dict[int, bytes],
    report: Callable[[str, str], None] | None = None,
    top_k: int = 0,
) -> str:
    """Generate the parameters, forcing the skeleton.

    Returns the full JSON text of the record.

    Effect: uses the model.

    Judgment: this function builds the head ``{"prompt": ..., "name": "..."``
    itself and adds the rest from the skeleton and the generated values.
    Open: should the skeleton include the head?
    """
    ids = model.encode(build_parameters_prompt(user_prompt, fn))
    text = (
        f'{{"prompt": {json.dumps(user_prompt)}, '
        f'"name": {json.dumps(fn.name)}'
    )
    ids = ids + model.encode(text)
    parts = build_skeleton(fn)
    for index, part in enumerate(parts):
        if isinstance(part, Forced):
            text = text + part.text
            ids = ids + model.encode(part.text)
        else:
            if report is not None:
                report("parameter", f"{part.name} ({part.type})")
                report("value", "")
                report("value_tokens", "0")
            value_ids = generate_value(
                part, ids, model, vocab, report, top_k,
                (text, _rest(parts[index + 1:]) if top_k else ""),
            )
            text = text + model.decode(value_ids)
            ids = ids + value_ids
    return text


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


def process_prompt(
    user_prompt: str,
    functions: dict[str, FunctionDef],
    model: Model,
    trie: NameTrie,
    vocab: dict[int, bytes],
    report: Callable[[str, str], None] | None = None,
    top_k: int = 0,
) -> FunctionCall:
    """Run the whole flow for one prompt. Raises PromptError on failure.

    ``top_k`` > 0 (a viewer is connected) adds viewer events to ``report``.

    Effect: uses the model.
    """
    if report is not None:
        report("stage", "select_name")
    fn = select_name(user_prompt, functions, model, trie, report, vocab, top_k)
    if report is not None:
        report("stage", "generate_parameters")
        report("value", "")
    text = generate_parameters(user_prompt, fn, model, vocab, report, top_k)
    if report is not None:
        report("stage", "validate_call")
    return validate_call(text, fn)
