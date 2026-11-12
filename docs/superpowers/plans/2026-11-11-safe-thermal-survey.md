# Safe thermal survey implementation plan

> Execute in one batch, tests first, then one independent review, as AGENTS.md requires.

**Goal:** Automatically record heating under repeated CPU SDK work and stop on unsafe or unobservable conditions.

**Architecture:** `campus_survey thermal` delegates to one thermal measurement module. The parent owns safety monitoring and an exclusive JSONL file; one isolated child loads the cached model once and cycles fixed input lengths.

**Tech Stack:** Python 3.10+ standard library and existing public llm_sdk API; unittest.

**Spec:** `docs/superpowers/specs/2026-11-11-safe-thermal-survey.md`.

## Global constraints

- Linux; no SDK/src changes, downloads, dependencies, or hardware setting changes.
- Default 300 s workload, 360 s total budget, 0.5 s polling, 1 s TERM grace.
- Stop on thermal counter increase; CPU crit remains an emergency ceiling. No fixed 85 C cap or crit-minus-10 C margin. Optional lower cap cannot raise crit. Lost safety data, low memory, interruption and budget still stop.
- Preserve each timing and sensor sample; do not mix workload lengths or call this an application deadline certificate.
- VM remains a configuration candidate; guest-only sensor-less execution is refused.

## Review focus

- Frequent child messages must not starve safety checks; blocked native work must still be killable.
- Stop must kill descendants too; child termination must not inherit a parent-only signal handler.
- Missing/disappearing safety inputs must stop, while optional power/frequency fields may be unavailable.
- Existing/symlink report paths must stay untouched; interruption/failure must retain valid partial records.
- Refusal must precede model work; raw identities and paths must not leak through metadata or failures.

## Task 1: continuous measurement and integration (one batch)

**Files:** add `tools/thermal_survey.py`, `tests/test_thermal_survey.py`; modify `tools/campus_survey.py`, related research/development/design docs.

**Interfaces:** thermal `main(argv: list[str] | None = None) -> int`; pure safety evaluation from numeric snapshots; supervised `measure(args, write_event)`; fixed SDK worker emits stage/warmup/forward/done messages.

- [x] Write safety fixture and real child tests: thermal rise/counter increase and reset, lost sensors, low memory, absent guards, fixed-work repetition, budget, SIGTERM/KILL, report refusal and partial preservation, privacy.
- [x] Run the new tests and observe expected failure because the thermal module/entry point is absent.
- [x] Implement the complete module and dispatch; preserve inspect/benchmark behavior.
- [x] Run targeted and full tests; fix actual failures from evidence.
- [x] Update design/ADR/research instructions including automatic warming and VM configuration merits.
- [x] Run make install, make lint, make test; smoke real host preflight, with no heavy workload if guards are unavailable.
- [x] Independent whole-change review; fix material findings with regression tests.
- [x] Commit all authorized changes locally; record actual validation and unverified campus/VM behavior.

Pre-flight: one task; no shared interfaces between tasks. Scope refinement: fixed SDK work gives comparable within-run timings; arbitrary application repetition and VM lifecycle control are deferred.

Validation: 22 thermal tests and 75 full-suite tests pass; flake8 and required mypy flags pass for 27 files.
Independent review found the root cgroup memory.max assumption; a realistic hierarchy regression failed before the fix and passed after it.
A startup-SIGTERM regression also failed before moving parent signal registration ahead of child.start, then passed.
Recorded partial/refusal CLI smoke on the real AMD environment: exit 3, no model work, private valid JSONL.
No real heating, campus inference or VM execution was claimed. No deferred review findings remain.
