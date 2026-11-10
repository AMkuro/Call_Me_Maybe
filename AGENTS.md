# Development and integration

Subject v1.7 requirements are summarized in docs/subject-v1.7.md. Read it
instead of fetching the Subject from GitHub; re-check the original only if the
version changes or the summary cannot settle an important question.

Read docs/development.md and docs/ci-auto-integration.md before changing CI,
the build environment or merging. README.md is a submission artifact: keep
development-only content (PoC, CI, GPU setup, SDK patches, branches, status)
out of it and put it in docs/development.md.
Work on a new branch cut from the latest integration branch (currently
origin/develop; fetch first), and open the PR against that branch, so merges
stay small. Do not push directly to main or develop.
Do not merge existing experimental/PoC branches as part of unrelated work.

Run make install, make lint, make test. CI is disabled (manual runs only)
until the implementation is done and will be reworked then. Use the Subject
mypy flags. The SDK is excluded from lint through .flake8 and the mypy
settings. The repository copy of llm_sdk has one documented local change
(float32 on every device, commented in the code); make sdk-install unpacks
the distributed version from llm_sdk.zip (or SDK_ZIP=<zip>). Do not
otherwise edit llm_sdk. Development checks are not a submission certificate.

Remote publication is separate from local edits, verification and commits.
A request to fix something does not authorize a push or a PR mutation: obtain
a separate explicit instruction before pushing, creating/updating a PR, or
enabling its merge. Remote history persists and is costly to revise; do not
publish intermediate local work automatically.

When publication and integration are explicitly authorized, create or update
a PR with only the authorized task's changes. After inspecting its diff and
required checks, opt in ONLY that intended integration PR using:

    gh pr merge NUMBER --repo AMkuro/Call_Me_Maybe --auto --squash --match-head-commit SHA

Use its current full head SHA. Never use --admin, bypass protection, create fake
statuses, enable auto-merge on all PRs, or opt in an experiment/draft. Resolve
conflicts and failed checks on the branch and retest. If the base branch moves,
merge it (origin/develop or origin/main) into the branch, push, and wait for
new checks. Report permissions
or settings errors; do not weaken protection. Existing enrollment is not proof
of completion: verify state=MERGED, mergedAt and mergeCommit on GitHub.

# Principles

The human owner requires these five. They decide how to choose between
otherwise equal options.

1. Understandability: the owner must be able to understand the mechanism,
   the design and the meaning of the code. Prefer what can be explained;
   explain any unfamiliar construct when introducing it.
2. Explainability: for each important decision, be able to say why, with
   evidence. Record it (ADR) and keep facts apart from guesses.
3. Verifiability: check AI output and implementations by an independent means
   (run it, measure it, diff it). Say what was not verified.
4. Responsibility and authority: the agent may do the work, but the human
   makes important decisions and holds final responsibility. The agent
   proposes options with a recommendation and never marks a decision as made.
5. Rational design choice: neither simplicity nor sophistication is a goal.
   Choose the method that fits the requirements, performance, maintainability
   and the cost of understanding it. Do not add variables, comments,
   mechanisms or checks that nothing needs.

# Design and implementation process

Complete corrections in one pass within the requested kind of artifact.
For feedback on a document, inspect and fix the related documentation together;
do not automatically expand it into code changes or tests. An explicit code-fix
request includes the necessary tests and verification. Once the authorized
local work and its relevant checks are complete, make a local commit without
waiting for another instruction, unless the owner explicitly asks not to.
Local commits and remote publication are separate: do not infer permission to
push or mutate PRs from a correction or local commit.
Ask only when a material decision is missing, and preserve the owner's
design-approval rules.

Do not start implementing from a vague request. Design first.

1. Write the design under docs/ before any code in src/: requirements (from
   docs/subject-v1.7.md), components and data flow, and the decisions still
   open. In Claude Code, use the engineering:system-design skill for the
   structure and the doc-coauthoring skill for the document.
2. Record each architectural decision as an ADR: context, options, trade-offs,
   decision, and what to revisit. The human decides. The agent proposes options
   with a recommendation and never marks a decision as made.
3. Start implementing only after the human approves the design.
4. Write tests first. Test the pure parts (name tree, masks, skeleton,
   validation) directly with data, the decode loop with a function that
   returns fixed logits, and the end-to-end properties (every output parses
   and matches the schema) on the real model. Do not add an abstraction
   only to make tests possible.
5. Implement the approved design in one batch, not unit by unit: this keeps
   the AI's context whole and avoids reworking changes that were already
   audited. Run the checks (lint, tests) and a reviewer once at the end and
   show the result to the human. Keep docs/architecture.md in sync.
6. Keep facts apart from guesses. If something cannot be verified, say so and
   say what was tried.
7. Keep accuracy measurements and timing measurements separate in every
   note and report. Accuracy runs may be parallel. Timing runs are ideally
   single-run; if run in parallel, record the variance and the outlier rule
   (see docs/design/adr/0010-measurement-policy.md).
