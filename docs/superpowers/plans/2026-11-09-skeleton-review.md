# Skeleton Review Corrections Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Write all regression tests first, then implement the approved corrections in one batch, as AGENTS.md requires.

**Goal:** Correct the five reviewed skeleton defects without filling in the application stubs.

**Architecture:** Values include their JSON delimiter; forced skeleton text does not repeat it. The pipeline returns validated parameters, and the CLI handles save failures and zero successful results.

**Tech Stack:** Python 3.10+, standard-library unittest, existing Pydantic.

**Spec:** `docs/design/adr/0012-skeleton-review-corrections.md` (user-approved five-point proposal).

## Global Constraints

- No new dependencies, abstract model interfaces, direct torch imports, or SDK edits.
- Preserve leaf stubs; change their contracts where needed for these corrections.
- Use English docstrings and the Subject mypy flags.
- Run `make install`, `make lint`, `make test`; one independent review at the end.
- Work on `fix/skeleton-review`, cut from fetched `origin/develop`; the existing skeleton is a prerequisite.

## Review Focus

- Multi-token integers, signs and decimals continue until the selected delimiter.
- Both comma and closing-brace delimiters appear once in the assembled JSON.
- Function names containing quotes, backslashes and control characters round-trip through JSON.
- Save failure stops processing with a clear error and nonzero exit.
- Empty input and all-failed input replace old results with an empty array.

## Task 1: Regression tests, then the five corrections

**Files:** `tests/test_skeleton_flow.py`; `src/pipeline.py`, `src/skeleton.py`, `src/values.py`, `src/__main__.py`, `src/writer.py`; `docs/architecture.md`, `docs/design/README.md`.

**Interfaces:** `ValueSlot.terminator: Literal[",", "}"]`; value mask/completion functions accept this terminator. `generate_value` returns IDs including the delimiter. Skeleton text starts after the complete JSON-escaped name and excludes slot-owned delimiters.

- [x] Write tests for numeric continuation, delimiter ownership, name escaping, validated parameters, save failure, zero successes and partial success. Replace only unavailable leaves and model operations; test actual control flow and observable results.
- [x] Run `.venv/bin/python -m unittest discover -s tests -p test_skeleton_flow.py -v`; expect the reviewed defects to fail before implementation.
- [x] Change contracts and orchestration in one batch; update architecture and ADR index.
- [x] Run required checks and inspect the exact diff. Expect lint and all tests to pass; no SDK change.
- [x] Request one independent whole-change review using Ponytail. Correct substantive findings and retest if needed.
- [x] Commit the completed correction with only its files. Push/PR handling remains separate from implementation approval.

## Progress

- `origin/develop` fetched: `7c2686e`. Prerequisite skeleton: `c1475f1`.
- Existing baseline verified earlier in this session: install and lint passed; 45 tests passed.
- RED: eight regression test methods ran before implementation; all five reviewed defects failed (13 failing subcases). The two preservation checks already passed.
- GREEN: `UV_CACHE_DIR=.cache/uv UV_OFFLINE=1 make install`, `make lint` and `make test` passed; 53 tests in total. Offline installation used the dependency cache after online metadata lookup failed under the sandbox's DNS restriction.
- Independent review found a duplicate Mermaid node ID in the updated diagram; the error exit now uses its own node `N`.
- Final independent review found no further code defects in the five approved corrections; its sole finding was the diagram ID corrected above. The reviewer also ran all 53 tests successfully.
