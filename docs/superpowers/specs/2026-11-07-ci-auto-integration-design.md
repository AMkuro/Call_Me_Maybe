# CI and agent auto-integration design

Requested outcome: selected agent PRs merge into main after required CI, without
release approval. Existing feature branches and local edits remain untouched.
Base: main e3dd4ec (SDK PoC); feat/a05-common-cli is NOT included.

## Approaches

1. Native auto-merge, explicitly enabled per PR by the authenticated agent
   (`gh pr merge --auto --squash --match-head-commit SHA`). Chosen: no privileged
   workflow, labels, bot identity allowlist, custom merge logic, or bulk enrollment.
2. A label-triggered privileged workflow: rejected as unnecessary token exposure
   and configuration complexity. PR creation already happens in the agent session.
3. Fly-in Release Candidate: rejected; manual approval and branch export conflict
   with this project's main-centered integration goal.

## CI contract

Python 3.10 and 3.14, locked uv dependencies, unittest, flake8 and all mandatory
mypy flags on owned code. Three stable required contexts: Quality (Python 3.10),
Quality (Python 3.14), Subject development checks. SDK diagnostics are shown and
compared against exact committed baselines and source hash; new differences fail.
`make lint` retains the Subject's full-tree commands and fails on upstream SDK
errors. `make lint-app` is the explicitly scoped development gate.

The Subject job checks dependencies, sample input shape, tracked output absence,
README/Makefile and public SDK usage. It reports the missing src entry point on
the current PoC; once src exists, its help command must work without a model.
`make subject-audit` additionally requires src and exits nonzero on the PoC.
Neither mode claims model accuracy, constrained decoding correctness, runtime,
all-class Pydantic coverage, or complete submission compliance. Those require
implementation-specific tests and peer review. No heavyweight inference in PR CI.

## Protection

An active main-only ruleset, no bypass actors: PR required, zero approving
reviews, resolve threads, squash only, required checks from GitHub Actions app
15368, strict up-to-date checks, block deletion/force push. Enable native
Auto-merge, disable merge/rebase methods. Keep Actions read-only and unable to
approve reviews. No branch-protection duplication. No PAT stored in Actions.

## Verification boundaries

Dedicated passing and deliberately failing PRs; a non-enrolled PR remains open.
Use a disposable protected base branch to verify actual native auto-merge while
leaving main unchanged. No temporary main change without explicit approval.
The permanent infrastructure PR is left reviewable until final integration is
settled. Record before/after settings, run URLs, and unverified claims.

## Execution decisions

The user explicitly delegated ordinary design and implementation choices, so
no intermediate design approval gate is added. Native worktree creation placed
its checkout outside writable roots; implementation uses an independent local
clone under /tmp instead. The original checkout is read only throughout.
