# JSON validation and numeric integrity implementation plan

> **For agentic workers:** Use superpowers:executing-plans and tests first.
> Implement one coherent batch, then required checks and one independent review.

**Goal:** Reject duplicate/nonstandard JSON, enforce strict Pydantic shapes and
types, warn for empty metadata, and never silently change numeric JSON values.

**Spec:** User's detailed JSON/input/precision request and ADR-0017. ADR-0004 and
the numeric part of future ADR-0015 are superseded; do not implement LF records.

## Constraints

- Same feat/implementation branch, local commit when done; no push or PR operations.
- No SDK/private API, CI workflow, dependency or unrelated generation changes.
- Pydantic owns shape/type validation; stdlib JSON hooks own duplicate/constants
  and preserve decimal source values. No handwritten JSON parser/number writer.
- Exact int, checked float after Decimal parsing; reject unsupported precision
  rather than replacing values or serializing numbers as strings.

## Review focus

Nested/escaped duplicate keys; bool/number confusion; decimal rounding before
validation; overflow/underflow and huge ints; warning-only metadata preservation;
full saved JSON numeric identity; all owned parse sites and auxiliary entry points.

### Task 1: Implement the integrity contract

**Files:** src/json_io.py (new), schema.py, loader.py, pipeline.py, vocab.py;
owned JSON-reading tools and associated tests; README and architecture/ADRs.

**Interfaces:** `loads(text: str) -> Any`, exact source ints/Decimals; strict
`Number` scalar validator; strict FunctionDef/ParamSpec/FunctionCall and dynamic
parameters. Existing loader/pipeline return types and error classes remain.

- [x] Add failing JSON/metadata/strict-type/numeric-roundtrip and regression tests.
- [x] Implement shared parsing, strict Pydantic validators/logging, all owned
  parse sites and the independent measurement comparison together.
- [x] Update README and superseded design records to describe actual limits.
- [x] Run make install, make lint, make test; fixed-logit/CLI integration and
  serial JSON-only timing. Record observations separately from model accuracy.
- [x] Run one independent full-change review; fix important findings with regressions.
- [x] Commit only authorized local changes and verify a clean worktree; no push.

## Progress

Base: b9a62da, 105 tests. All owned non-test JSON parse sites inventoried: loader
(two), pipeline, vocabulary, measurement, subject checker and campus survey (two).
No owned CI script has another parser; third-party SDK internals are excluded.
Local API probe confirmed strict int|float alone converts Decimal and that
Pydantic's default Decimal JSON output is a string, so an explicit value guard
is necessary. No arbitrary digit/exponent cap will be introduced.

Added 24 regression tests (129 total); initial red run reproduced the defects.
All 129 tests and flake8/Subject mypy pass after implementation. Serial parsing
measurements record increased CPU costs, separately from any model inference.
The existing direct tool entry points were checked and preserved.

Independent review: no Critical/Important/Minor findings. Reviewer reran all
24 new tests and separately probed 458 numeric literals: 368 exact roundtrips,
90 explicit rejections, no changed accepted value. No model was loaded.

Final checks: make install, make lint, make test (129 tests), strict development
input audit, both auxiliary invocation forms, D2 compilation, Markdown links,
SVG XML and git diff --check. Changes are recorded in one local task commit;
remote publication remains unauthorized and is not performed.
