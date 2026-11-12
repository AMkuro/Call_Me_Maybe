"""Handle one prompt: name -> skeleton -> values -> check."""

import json
from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass
class Run:
    """What stays the same for every prompt of one run.

    Judgment: a plain dataclass, not pydantic: it holds the model handle
    and a callback, nothing to validate (as ``Model``, ADR-0011).
    ``report`` receives dashboard observations (ADR-0014/0021); without it
    the decoder computes no diagnostics. It is set once the dashboard
    exists, after the model is loaded.
    """

    model: Model
    vocab: dict[int, bytes]
    functions: dict[str, FunctionDef]
    trie: NameTrie
    report: Callable[[str, str], None] | None = None

    def say(self, label: str, text: str) -> None:
        """Report one observation, if anyone is listening."""
        if self.report is not None:
            self.report(label, text)


def process_prompt(user_prompt: str, run: Run) -> FunctionCall:
    """Run the whole flow for one prompt. Raises PromptError on failure.

    Effect: uses the model.
    """
    run.say("stage", "select_name")
    fn = select_name(user_prompt, run)
    run.say("stage", "generate_parameters")
    run.say("value", "")
    text = generate_parameters(user_prompt, fn, run)
    run.say("stage", "validate_call")
    return validate_call(text, fn)


def select_name(user_prompt: str, run: Run) -> FunctionDef:
    """Let the model choose a function, limited by the name tree.

    Effect: uses the model.
    """
    trie = run.trie

    def allowed_next(generated: list[int]) -> list[int] | None:
        if leaf_name(trie, generated) is not None:
            return None
        return next_tokens(trie, generated)

    def observed(choice: TokenChoice) -> None:
        run.say("value", _decoded(run.model, choice.tokens))
        _show_choice(choice, run)
        if choice.top and not choice.top[0][2]:  # raw best was masked
            raw_top = _token_text(choice.top[0][0], run)
            selected = _token_text(choice.token, run)
            run.say("warning", f"name_selection_constrained: step="
                    f"{choice.step} raw_top={raw_top} selected={selected}")

    prompt_ids = run.model.encode(build_name_prompt(user_prompt,
                                                    run.functions))
    generated = generate(
        run.model.logits, prompt_ids, allowed_next, max_name_steps(trie),
        on_token=observed if run.report is not None else None,
    )
    name = leaf_name(trie, generated)
    if name is None:
        raise PromptError("name generation ended without a function name")
    run.say("function", name)
    return run.functions[name]


def generate_parameters(user_prompt: str, fn: FunctionDef, run: Run) -> str:
    """Generate the parameters, forcing the skeleton.

    Returns the full JSON text of the record.

    Effect: uses the model.

    Judgment: this function builds the head ``{"prompt": ..., "name": "..."``
    itself and adds the rest from the skeleton and the generated values.
    Open: should the skeleton include the head?
    """
    model = run.model
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
            run.say("parameter", f"{part.name} ({part.type})")
            run.say("value", "")
            run.say("value_tokens", "0")
            value_ids = generate_value(part, ids, run)
            text = text + model.decode(value_ids)
            ids = ids + value_ids
    return text


def generate_value(slot: ValueSlot, ids: list[int], run: Run) -> list[int]:
    """Generate one parameter value and its following JSON delimiter.

    Effect: uses the model.
    """
    vocab = run.vocab

    def allowed_next(generated: list[int]) -> list[int] | None:
        written = b"".join(vocab[token] for token in generated)
        if is_value_complete(written, slot.type, slot.terminator):
            return None
        return allowed_value_tokens(vocab, slot.type, written, slot.terminator)

    def observed(choice: TokenChoice) -> None:
        written = b"".join(vocab[token] for token in choice.tokens)
        run.say("value", written.decode("utf-8", errors="ignore"))
        run.say("value_tokens", str(choice.step))
        _show_choice(choice, run)

    try:
        return generate(
            run.model.logits, ids, allowed_next, MAX_VALUE_STEPS,
            on_token=observed if run.report is not None else None,
        )
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


# Dashboard text for one token decision (ADR-0021). Display only: a failure
# here must never discard a generated result.

def _show_choice(choice: TokenChoice, run: Run) -> None:
    """Report the candidate table of an already-computed decision."""
    if choice.forced:
        run.say("candidates", "forced: the only allowed token "
                f"{_token_text(choice.token, run)}; the model is not called")
        return
    rows = list(choice.top)
    if choice.token not in [token for token, _, _ in rows]:
        rows.append((choice.token, choice.logit or 0.0, True))
    lines = [f"allowed_mass={choice.allowed_mass:.3g}  "
             f"gap={choice.margin:.3g}"]
    for token, logit, allowed in rows:
        mark = ">" if token == choice.token else " "
        masked = "" if allowed else "  masked"
        lines.append(f"{mark} {token:>6} {_token_text(token, run):<16} "
                     f"{logit:9.3f}{masked}")
    run.say("candidates", "\n".join(lines))


def _token_text(token: int, run: Run) -> str:
    """Quote one token so spaces, quotes and controls are visible.

    Partial UTF-8 shows as bytes (``b'\\xe6'``); tokens outside the byte
    vocabulary (special tokens) use the SDK's text.
    """
    data = run.vocab.get(token)
    if data is None:
        return repr(_decoded(run.model, [token]))
    try:
        return repr(data.decode("utf-8"))
    except UnicodeDecodeError:
        return repr(data)


def _decoded(model: Model, ids: list[int]) -> str:
    """SDK text of ``ids``; a failure becomes a placeholder, not an error."""
    try:
        return model.decode(ids) or "<EOS/special>"
    except PromptError:
        return f"<text unavailable; ids={ids}>"
