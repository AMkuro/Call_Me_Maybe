# Local prompt improvement implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Implement the approved design in one batch, then run one independent review, as AGENTS.md requires.

**Goal:** Run a local GPU editor with only prompt-save/evaluate tools, then conditionally verify one improved candidate on the specified CPU profile and fixed calibrated data.

**Architecture:** JSON instruction candidates are data, never executable Python. A controller owns evaluation, scoring, synchronization and promotion. Separate worker processes use the existing application and public SDK APIs.

**Tech Stack:** Python, existing Pydantic/numpy/SDK environments, stdlib subprocess and Git.

**Spec:** docs/superpowers/specs/2026-11-11-local-prompt-improvement-design.md (owner approved 2026-11-11).

## Global constraints

- No SDK/private API or decoding changes; no new dependencies, push, PR or automatic application adoption.
- Only two instruction strings can change. Baseline and candidate use the same source SHA, inputs and inference settings.
- CPU affinity 0-15, six threads, 16G/no swap, GPU disabled; raw runtime and raw+8 estimate separate.
- Select one candidate using development data only; calibrated data stay outside editor context.

## Review focus

- Forged tools, file paths, code and extra JSON fields cannot expand editor authority.
- Errors and missing outputs count against input totals; bool is distinct from number.
- Parallel workers never overwrite candidates or reports; regex checks have a process timeout.
- Code changes invalidate comparisons and trigger checks plus baseline/candidate re-evaluation.
- Failed proposals, model timeouts, OOM, interrupted runs and resumed caches retain provenance.

### Task 1: Candidate protocol, scoring and worker tests

Files: `tools/prompt_candidates.py`, `tools/prompt_trial.py`, `tests/test_prompt_experiments.py`, `tests/fixtures/prompt_development.json`.

- [ ] Write failing tests for strict candidate/action validation, immutable candidate IDs, exact/regex scoring, promotion/no-regression, fixture consistency and missing results.
- [ ] Verify those tests fail before implementation.
- [ ] Implement candidate models and storage; use fixed gold labels, not model predictions.
- [ ] Implement trial execution with instruction-only overrides, fixed SDK pipeline and CLI mode.
- [ ] Run focused tests; verify unchanged default prompts and bool/number distinction.

### Task 2: Bounded local editor and automatic controller

Files: `tools/prompt_editor.py`, `tools/prompt_experiment.py`, `research/evo-x3/measure.sh`, `tests/test_cpu_measurement.py`, `dev.mk`.

- [ ] Test fixed tool dispatch, proposal caps, CPU module allowlist, raw+8 calculation, update selection and checkpoint invalidation.
- [ ] Implement public-SDK editor generation and two validated operations, without arbitrary commands/paths.
- [ ] Implement two-worker GPU rounds (maximum eight candidates), fixed baseline and candidate selection.
- [ ] Implement local merge synchronization at idle boundaries; rerun checks and evaluations after updates.
- [ ] Implement serial CPU CLI verification then one baseline/candidate calibrated comparison; retain artifacts and failures.
- [ ] Provide a development-only make target and documentation, without changing submission README.

### Task 3: Verify and run the experiment

- [ ] Run `make install`, `make lint`, `make test` in the worktree with existing caches; review the complete diff independently.
- [ ] Commit the verified tooling locally, so source SHA and later local merges are traceable.
- [ ] Run a real two-worker GPU smoke/baseline/finite search. If no candidate improves, report this without fabricating promotion.
- [ ] If improved, automatically execute the prescribed CPU check and fixed supplementary comparisons.
- [ ] Record measured outcomes separately from projections; update design/ADR/development docs and commit the final record locally.

## Checkpoint on owner stop instruction

- Task 1: candidate/scoring/trial drafts and eight direct tests written; focused tests pass.
- Task 2: controller/editor/CPU-entrypoint drafts written; CPU allowlist test passes.
- Task 3: make install and make test pass (159); mypy45files passes; full make lint fails on inherited src/display.py:52 E501. Independent draft review is recorded in handoff.
- No optimizer/target inference, parallel GPU pilot, CPU candidate trial or calibrated model test has run.
- Initial optimizer0.6B was rejected; agent research selects Qwen3.5-9B BF16 for the first pilot. Download interrupted; cache retained.
- Resume from the untracked local handoff under .cache/handoff/, fixing the documented review gaps before executing the draft.

## Resumed implementation

Owner resumed the task. The seven checkpoint issues and three further independent
review findings have regression coverage and fixes. Full install/lint/test pass:
174 tests, mypy47files. The source display signature was wrapped to fix the inherited
E501 only. Optimizer download and GPU/BF16 short JSON smoke passed. Calibrated data
grew to8suites88cases and were statically audited at8d63548, without exposing them
to the optimizer. Next: commit this verified code, run the bounded GPU search,
and execute the CPU/heldout gates only if development correctness improves.

## Owner-directed OSS harness change

Use installed OpenCode for agent execution and native Transformers Serve for
KV-cached optimizer inference. Remove the custom prompt_editor loop; configure
only tools, prompts, workspaces and the existing trusted evaluation workflow.
Two SDK attempts were stopped unscored and recorded; the next three parallel
rounds use the remaining six slots. Native two-tool permissions and serving
function-call parsing have been checked. See ADR-0019 and research/prompt-improvement.
