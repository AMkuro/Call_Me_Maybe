# ADR-0026 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. The owner explicitly requires local develop, preservation of unrelated changes, and no commits or publication.

**Goal:** Confirm fixed JSON tokens individually and verify correspondence between output IDs, model context and JSON.

**Architecture:** Keep name selection and value loops. `generate_parameters` owns prompt IDs P and completed output IDs O; the value loop owns V. Inference uses P + O + V. Fixed fragments are encoded once and appended by ID.

**Tech Stack:** Python, existing NumPy/Pydantic, public llm_sdk APIs, cached Qwen3-0.6B float32 on GPU.

**Spec:** ADR-0026 on remote develop (local copy absent), owner request and preceding design review; current `docs/architecture.md`.

## Global Constraints

- No SDK edits, commits, pushes, PR/Issue mutations, or dependency additions.
- Do not edit `src/name_trie.py` or `src/schema.py`; both have unrelated changes.
- Keep name EOS, value grammar, 256-token limit, prompts, progress and output schema.
- Confirm SDK hypotheses before application edits; keep accuracy and timing runs separate.

## Review Focus

- Special-token spellings in copied prompt, function name and parameter key: preserve JSON meaning.
- Partial UTF-8 token bytes: decode completed sequences, not individual tokens.
- Different IDs for identical text: preserve selected IDs in later inference.
- Decode cleanup or unknown vocabulary IDs: report failure rather than silent corruption.
- Observer enabled/disabled: identical output and inference inputs.

## Task 1: SDK characterization and baseline

- [x] Run scratch public-SDK characterization and existing original/changed/boundary fixtures on GPU; save IDs, byte correspondence, next-token scores and forward counts.
- [x] Run the unchanged CLI through `make gpu-run` separately for timing.
- [x] Record current `make test` / `make lint` results and protected file hashes.

## Task 2: Tokenwise output state (tests first)

**Files:** `src/pipeline.py`, `tests/test_forced_json.py`, existing mock fixtures as necessary.

**Interfaces:** Preserve `generate_value`, `select_name`, `generate_parameters` public signatures. Add private fixed append / checked decode helpers only as needed.

- [x] Add tests for incremental fixed appends/no forwards, byte/decode mismatch, boundary context, noncanonical IDs, Unicode/escape/special spellings, 0/multiple parameters and prompt preservation; run RED.
- [x] Use measured special-token behavior to choose the smallest meaning-preserving encoding or explicit failure.
- [x] Implement P/O state, checked fixed appends, completed-value/full-output checks and input prompt equality; run GREEN.
- [x] Repair old SDK mocks to provide real ID/byte correspondence without weakening their assertions.

## Task 3: Regression, evidence and design documentation

- [x] Re-run scratch accuracy against unchanged fixtures and save before/after raw evidence.
- [x] Run CLI GPU timing separately; compare JSON, schema, correctness, completion and forward counts.
- [x] Run `make install`, `make lint`, `make test`, `make subject` and `make subject-audit` as applicable; distinguish environmental and pre-existing findings.
- [x] Update architecture and a concise evidence report; preserve historical measurements and ADR.
- [x] Independently review the final diff, resolve substantive findings with regression tests, verify protected files and no commits.

## Execution record

Baseline: local develop `176e958ee4b230284a01d36c50a103788b1ae9b1`; 210 tests pass (10.287s), lint passes (48 files). Protected file hashes and pre-existing patch saved under `/tmp/cmm-0026*`. GPU is hidden in sandbox, host ROCm 2.14.1+rocm7.2 detects Radeon 8060S.

Final: implementation and tests complete without commits. Fixed special spellings use JSON \u003c escaping, selected after the public-SDK loss/roundtrip experiment; no ADR condition changed. 222 tests, lint, subject and subject-audit pass after make install. Independent review found no substantive issues and ran 28 relevant tests. Final artifact checks match the pipeline source hash and confirm equal before/after calls and forward counts. Accuracy remains 13/19 joint correct; timing is a separate one-run comparison (25.25s / 25.16s), not a speed claim. Evidence and reproduction commands are in docs/research/2026-11-17-tokenwise-json.md. Protected file hashes are unchanged. No deferred minor findings.

Follow-up (2026-11-18 JST): the owner authorized local commits in semantic units. Implementation and tests are grouped separately from architecture and verification evidence. This supersedes the original no-commit constraint; unrelated changes and remote publication remain excluded. Fresh pre-commit verification: make lint passes, make test 222/222 passes (10.307s).
