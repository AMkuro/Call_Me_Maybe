# Function registry implementation plan

> **For agentic workers:** Use superpowers:executing-plans, tests first,
> one implementation batch and one final independent review.

**Goal:** Use `dict[str, FunctionDef]` throughout function-set consumers,
preserving input order and rejecting duplicates before insertion.

**Spec:** docs/design/adr/0016-function-registry.md, approved by the owner.

## Constraints and review focus

Keep the input JSON array and output format unchanged. Share `index_functions`
between file loading and measurement fixtures. No persistent list plus index,
no name-selection heuristic, no SDK/CI/dependency or unrelated value-grammar changes.
Stay on feat/implementation; commit locally when done, no push/PR mutation.
Check input order, arbitrary names, duplicate rejection, dictionary keys used
for trie construction and original FunctionDef objects passed downstream.

### Task 1: Replace list-based function-set contracts

**Files:** src/loader.py, pipeline.py, prompt.py, __main__.py;
tools/measure_application.py; affected tests; architecture, ADR index, diagrams
and development/verification notes.

**Interfaces:** `index_functions(Iterable[FunctionDef]) -> dict[str, FunctionDef]`;
`load_functions(Path) -> dict[str, FunctionDef]`; name prompt, selection and
process_prompt consume that dictionary. Input list validation remains at loading.

- [x] Write and run failing loader/order/selection/fixture tests before changing src.
- [x] Implement the shared index and update all consumers/test fixtures together.
- [x] Synchronize design and diagrams. Run make install, make lint, make test.
- [x] Run one independent review, fix important findings with regressions if needed.
- [x] Record results and make a local commit; verify a clean worktree, do not push.

## Evidence

Base: 902f007. The four new tests failed on the old loader/list consumers and
measurement fixture validation, then passed after migration. Required install,
Subject lint (33 files) and all 105 tests passed. Fixed-logits SDK-boundary smoke
runs completed the actual application and measurement entry points through
JSON saving without loading a model. The independent reviewer found no issues,
confirmed the four tests and separately verified different-schema selection
and exact-name/order/object-identity preservation. No additional inconsistent-
trie checks were introduced: production builds trie and registry together.
No real-model accuracy, timing, push or PR operation was part of this change.
