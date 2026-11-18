"""Orchestrate prompts, validate completed calls, and persist batch results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import PromptError
from .generation import generate_call_json, select_function
from .json_io import loads
from .progress import GenerationProgress, observing
from .runtime import GenerationAssets
from .schema import FunctionCall, FunctionDef, build_parameters_model
from .writer import save_results

if TYPE_CHECKING:
    from .display import Dashboard


def process_all(
    prompts: list[str], resources: GenerationAssets, output: Path,
    dashboard: Dashboard,
) -> bool:
    """Process every prompt; True when none failed.

    A failed prompt is reported and skipped. The results so far are saved
    after each success, so an interruption keeps them. Raises OSError when
    saving fails.

    Effect: uses the model and writes ``output``.
    """
    calls: list[FunctionCall] = []
    for index, user_prompt in enumerate(prompts, 1):
        dashboard.prompt_started(index, user_prompt)
        try:
            call = process_prompt(user_prompt, resources, progress=dashboard)
        except PromptError as e:
            dashboard.prompt_failed(str(e))
            continue
        calls.append(call)
        save_results(output, calls)
        # Only after the save: if it fails, the summary shows this prompt
        # as not finished rather than done.
        dashboard.prompt_done(call.name)
    if not calls:
        save_results(output, calls)  # all failed: still write "[]"
    return len(calls) == len(prompts)


def process_prompt(
    user_prompt: str,
    resources: GenerationAssets,
    *,
    progress: GenerationProgress | None = None,
) -> FunctionCall:
    """Select a name, generate parameters, then validate one JSON record."""
    if observing(progress):
        progress.stage("select_function")
    fn = select_function(
        user_prompt,
        model=resources.model,
        functions=resources.functions,
        trie=resources.trie,
        progress=progress,
    )
    if observing(progress):
        progress.stage("generate_call_json")
    text = generate_call_json(
        user_prompt,
        fn,
        model=resources.model,
        vocab=resources.vocab,
        progress=progress,
    )
    if observing(progress):
        progress.stage("validate_call")
    return validate_call(text, fn, user_prompt)


def validate_call(
    text: str, fn: FunctionDef, user_prompt: str
) -> FunctionCall:
    """Turn generated JSON into a checked record that keeps that JSON.

    The record must parse, copy ``user_prompt`` exactly, name ``fn`` and
    match its parameters, and Python's json must read every number
    argument as a finite float. Only then is ``text`` kept as the record's
    _generated_json, which save_results writes unchanged so the number
    spelling survives (REQ-031; an integer only gained ".0").
    """
    try:
        call = FunctionCall.model_validate(loads(text))
        if call.prompt != user_prompt:
            raise ValueError("prompt differs from the input")
        if call.name != fn.name:
            raise ValueError(
                f"function name mismatch: expected {fn.name!r}, "
                f"got {call.name!r}"
            )
        parameters = build_parameters_model(fn).model_validate(call.parameters)
        # The saved output must be float: Python's json reads each number so.
        loaded = json.loads(text)["parameters"]
        for name, spec in fn.parameters.items():
            value = loaded[name]
            if spec.type == "number" and not (
                isinstance(value, float) and math.isfinite(value)
            ):
                raise ValueError(f"{name!r} does not load as a finite float")
    except ValueError as e:  # JSON errors and pydantic's are ValueErrors
        raise PromptError(f"invalid output: {e}") from e
    checked = FunctionCall(
        prompt=call.prompt,
        name=call.name,
        parameters=parameters.model_dump(by_alias=True),
    )
    checked._generated_json = text
    return checked
