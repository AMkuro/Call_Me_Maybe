# Output preflight implementation plan

> **For agentic workers:** Use superpowers:executing-plans with tests first,
> one implementation batch and one final independent review.

**Goal:** Fail before model initialization when the atomic output destination
cannot be prepared or a temporary file cannot be opened/written there.

**Architecture:** `main` validates both inputs, calls `writer.prepare_output`,
then constructs Model. The helper preserves existing output, probes the parent
with a temporary file and raises contextual StartupError on failure.

**Spec:** docs/design/adr/0007-error-policy-and-model-check.md, output-preflight update.

## Constraints

- Stay on feat/implementation. Local commit when complete; no push/PR mutation.
- Public SDK and existing dependencies only; no model runs needed for filesystem behavior.
- Keep input-error output preservation and the atomic writer's save-time error handling.
- Do not implement the independent ADR-0015 value-record proposal.

## Review focus

Existing output bytes must survive; missing output must not be created by the
probe. Directory targets, non-directory parents, denied writes and invalid paths
must fail before Model. Temporary probes must be removed, and malformed inputs
must fail before any output preparation. The probe must match atomic-save semantics.

### Task 1: Prepare the destination before Model

**Files:** src/writer.py, src/__main__.py, tests/test_output_preflight.py,
docs/architecture.md, docs/development.md, docs/diagrams/skeleton-flow.d2/.svg,
the ADR update above and the verification record.

**Interface:** `prepare_output(path: Path) -> None`; creates parent directories,
checks target kind and opens/writes/flushes/removes a sibling temporary file;
raises StartupError with the path and underlying reason.

- [x] Write failing main-level filesystem tests, mocking only model startup and I/O failures.
- [x] Run the tests and confirm the old code reaches Model or skips preparation.
- [x] Implement the helper and startup call, then synchronize the design and diagram.
- [x] Run make install, make lint, make test; verify real CLI failure without model loading.
- [x] Run one fresh independent review; fix important findings with regression tests.
- [x] Commit intended local changes and verify a clean worktree; do not push.

## Evidence

Base: 69a67fb. Nine initial regression/preservation tests produced seven failures
against the old startup flow, then passed after implementation. A separate
long-filename regression failed before matching the atomic writer's probe prefix.
All 10 focused tests and the whole 101-test suite pass. Subject lint passes for
32 files, and make install passed. Two real CLI failures preserved existing data
without creating the isolated model cache. No model generation ran.

One fresh independent reviewer found no Critical/Important issues, confirmed the
10 tests and independently checked FIFO and malformed-input behavior. A minor
test docstring now explicitly identifies injected I/O failure. The existing
atomic-write semantics (including read-only regular targets), existing filename
limits, retained save-time errors and exclusion of ADR-0015 were confirmed;
no additional architecture choices were made during review.
