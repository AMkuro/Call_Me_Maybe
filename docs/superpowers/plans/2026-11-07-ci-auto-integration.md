# CI and auto-integration implementation plan

> Execute inline using superpowers:executing-plans. The user requested autonomous
> implementation, testing, and self-review.

**Goal:** CI-gated native auto-merge for explicitly selected PRs into main.
**Architecture:** read-only PR CI plus GitHub Ruleset and native Auto-merge.
**Tech Stack:** uv, Python 3.10/3.14, unittest, flake8, mypy, GitHub CLI.
**Spec:** ../specs/2026-11-07-ci-auto-integration-design.md

## Global constraints

- No edits to Fly-in, distributed llm_sdk, existing branches, or user edits.
- No unchecked main push, bypass actor, release approvals, or bulk auto-merge.
- No temporary changes merged to main without explicit approval.

## Review focus

- New SDK errors must fail while baseline errors stay fully visible.
- Failure/cancellation/missing required checks must block merging.
- A green but non-enrolled experimental PR must stay open.
- main updates require retesting the merged revision.
- Development checks must not claim full Subject compliance.

### Task 1: Local CI contract

- [x] Add regression tests for malformed sample input, forbidden dependencies,
  private SDK imports, generated output tracking, and missing submission src.
- [x] Run the new tests before implementing tools (expect missing module).
- [x] Add Subject development preflight and strict submission audit, exact SDK
  diagnostic baseline, locked dev tools, Makefile, and English README.
- [x] Run unittest, app lint, SDK audit, preflight, and negative full lint/audit.

### Task 2: GitHub execution and protection

- [x] Add pinned, read-only PR/main CI with stable job names and no path filters.
- [x] Validate YAML and execute the real PR CI on a dedicated infra branch.
- [x] Snapshot settings; enable squash/Auto-merge and active main Ruleset after
  verifying exact check names and app identity; record changes.

### Task 3: Live verification and review

- [x] Use a disposable base with equivalent protection to verify failed PR
  blocking, passing explicit auto-merge, non-enrolled PR, and direct-push denial.
- [x] Close temporary PRs and remove only branches created for this task.
- [x] Self-review and request a fresh whole-branch review; record outcomes.
- [x] Publish reviewable infrastructure PR, run evidence and agent instructions.
