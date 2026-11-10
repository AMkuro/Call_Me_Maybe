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


def select_name(
    user_prompt: str,
    functions: dict[str, FunctionDef],
    model: Model,
    trie: NameTrie,
    report: Callable[[str, str], None] | None = None,
) -> FunctionDef:
    """Let the model choose a function, limited by the name tree.

    Effect: uses the model.
    """
    prompt_ids = model.encode(build_name_prompt(user_prompt, functions))

    def allowed_next(generated: list[int]) -> list[int] | None:
        if leaf_name(trie, generated) is not None:
            return None
        return next_tokens(trie, generated)

    def observed(choice: TokenChoice) -> None:
        """Show name prefixes and diagnostics without another forward pass."""
        if report is not None:
            report("value", _diagnostic_text(model, choice.tokens))
            _report_choice(choice, model, report, warn=True)

    generated = generate(
        model.logits, prompt_ids, allowed_next, max_name_steps(trie),
        on_token=observed if report is not None else None,
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
) -> list[int]:
    """Generate one parameter value and its following JSON delimiter.

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

    try:
        return generate(
            model.logits, ids, allowed_next, MAX_VALUE_STEPS,
            on_token=observed if report is not None else None,
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
    for part in build_skeleton(fn):
        if isinstance(part, Forced):
            text = text + part.text
            ids = ids + model.encode(part.text)
        else:
            if report is not None:
                report("parameter", f"{part.name} ({part.type})")
                report("value", "")
                report("value_tokens", "0")
            value_ids = generate_value(part, ids, model, vocab, report)
            text = text + model.decode(value_ids)
            ids = ids + value_ids
    return text


def validate_call(text: str, fn: FunctionDef) -> FunctionCall:
    """Parse ``text`` as JSON and check it with the dynamic model."""
    try:
        call = FunctionCall.model_validate(loads(text))
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
) -> FunctionCall:
    """Run the whole flow for one prompt. Raises PromptError on failure.

    Effect: uses the model.
    """
    if report is not None:
        report("stage", "select_name")
    fn = select_name(user_prompt, functions, model, trie, report)
    if report is not None:
        report("stage", "generate_parameters")
        report("value", "")
    text = generate_parameters(user_prompt, fn, model, vocab, report)
    if report is not None:
        report("stage", "validate_call")
    return validate_call(text, fn)
