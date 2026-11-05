# A05 responsibility separation

The user requested object-oriented responsibility separation after two
refactors merely moved the procedural workflow. Routine design decisions
remain authorized; preserve the confirmed Subject/#24 behavior.

## Design

Use composition with four Pydantic-backed objects, as required by Subject:

- `InputFiles` owns the two input paths and loading/definition validation.
- `CallProcessor` owns a dynamic function set and the injected callable;
  `process(record, index)` validates one input, generates, and validates output.
- `ResultStore` owns the destination and committed results. Initialization
  prevents input aliases and resets output; append updates memory only after
  the atomic write succeeds.
- `Application` composes these objects, controls sequential execution, and
  translates contextual failures/interruption into diagnostics and exit codes.

A shared `operation(stage)` context manager attaches stage information and
preserves original causes. It removes the mutable stage variable scattered
through the batch function. Keep `save_results` as the atomic I/O primitive
and the existing Pydantic validation functions as reusable validation logic.
Keep generation as a callable; introduce no decoder hierarchy or abstract
interfaces. `main` parses arguments, builds the application, and calls it.

This replaces the single procedural `run`, rather than wrapping it in a class.
JSON formats, CLI options, initialization order, strict types, fail-fast
behavior, and the explicit unimplemented generator remain unchanged.

## Implementation steps

- [x] Add independent component tests: reusable one-item processor, error
  stage/cause preservation, output store retry after a failed save, store
  reinitialization, and application reuse/interruption. Run them before code.
- [x] Implement `src/inputs.py`, `src/processing.py`, `src/operations.py` and
  `ResultStore` in `src/storage.py`. Replace procedural workflow with
  `Application`; update CLI composition and integration documentation.
- [x] Run all tests and scoped flake8/mypy with Subject flags. Self-review
  initialization order, generator isolation, committed storage state, and
  interruption. Obtain one read-only review of the final responsibilities.
- [x] Commit locally and report the concrete responsibilities and checks.
  Do not push, create a PR, merge, change SDK/input fixtures, or change issues.
