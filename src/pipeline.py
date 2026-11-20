"""Orchestrate prompts, validate completed calls, and persist batch results."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import PromptError
from .generation import generate_call_json, select_function
from .json_io import loads
from .progress import GenerationProgress, observing
from .runtime import GenerationAssets
from .schema import (
    FunctionCallRequest,
    FunctionDef,
    build_argument_validation_model,
)
from .values import normalize_call_json
from .writer import save_results

if TYPE_CHECKING:
    from .display import Dashboard


def process_all(
    prompts: Sequence[str],
    resources: GenerationAssets,
    output: Path,
    dashboard: Dashboard,
) -> bool:
    """Process the prompts in order; return True when none failed.

    A failed prompt is reported to ``dashboard`` and skipped. After each
    success, all results so far are saved to ``output``, so an interruption
    keeps them; when none succeeds, an empty array is saved.

    Args:
        prompts: The prompts.
        resources: The model and the data shared by every prompt.
        output: The output file.
        dashboard: Shows the progress and counts the outcomes.

    Returns:
        True when every prompt succeeded.

    Raises:
        OSError: Saving fails.
    """
    dashboard.output_path = str(output)
    calls: list[FunctionCallRequest] = []
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
        dashboard.prompt_done(call.name, call._generated_json)
    if not calls:
        save_results(output, calls)  # all failed: still write "[]"
    return len(calls) == len(prompts)


def process_prompt(
    user_prompt: str,
    resources: GenerationAssets,
    *,
    progress: GenerationProgress | None = None,
) -> FunctionCallRequest:
    """Select a function, generate raw JSON, normalize and validate it.

    Pass generation's ordered, nonoverlapping number-slot slices directly
    to normalization. They use pre-normalization raw_json Python string
    indices: inclusive starts, exclusive stops, steps None, and include
    each slot's terminator ("," or "}").

    Args:
        user_prompt: The user's request.
        resources: The model and the data shared by every prompt.
        progress: Receives each stage and step, or None.

    Returns:
        The validated record.

    Raises:
        PromptError: Selection, generation or validation fails.
    """
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
    raw_json, raw_json_number_part_slices = generate_call_json(
        user_prompt,
        fn,
        model=resources.model,
        vocab=resources.vocab,
        progress=progress,
    )
    normalized_json = normalize_call_json(
        raw_json, raw_json_number_part_slices
    )
    if observing(progress):
        progress.stage("validate_call")
    return validate_call(normalized_json, fn, user_prompt)


def validate_call(
    text: str, fn: FunctionDef, user_prompt: str
) -> FunctionCallRequest:
    """Turn generated JSON into a checked record that keeps that JSON.

    The record must parse, copy ``user_prompt`` exactly, name ``fn`` and
    match its parameters, and Python's json must read every number
    argument as a finite float. Only then is ``text`` kept as the record's
    ``_generated_json``; save_results writes it unchanged, so each number
    is saved with the digits the model wrote.

    Args:
        text: The generated JSON record.
        fn: The selected function.
        user_prompt: The user's request.

    Returns:
        The record, with ``text`` attached.

    Raises:
        PromptError: A check fails.
    """
    try:
        call = FunctionCallRequest.model_validate(loads(text))
        if call.prompt != user_prompt:
            raise ValueError("prompt differs from the input")
        if call.name != fn.name:
            raise ValueError(
                f"function name mismatch: expected {fn.name!r}, "
                f"got {call.name!r}"
            )
        parameters = build_argument_validation_model(fn).model_validate(
            call.parameters
        )
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
    checked = FunctionCallRequest(
        prompt=call.prompt,
        name=call.name,
        parameters=parameters.model_dump(by_alias=True),
    )
    checked._generated_json = text
    return checked
