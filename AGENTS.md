# Development and integration

Subject v1.7 requirements are summarized in docs/subject-v1.7.md. Read it
instead of fetching the Subject from GitHub; re-check the original only if the
version changes or the summary cannot settle an important question.

Read README.md and docs/ci-auto-integration.md before changing CI or merging.
Work on a new branch from current origin/main. Do not push directly to main.
Do not merge existing experimental/PoC branches as part of unrelated work.

Run uv sync --locked, make lint-app, make test, make lint-sdk, make subject.
Use the Subject mypy flags. Never edit the supplied llm_sdk or refresh its
reviewed diagnostic baseline just to make CI pass. Full make lint still exposes
upstream failures; development CI is not a submission certificate.

Create or update a PR with only the authorized task's changes. After inspecting
its diff and required checks, opt in ONLY that intended integration PR using:

    gh pr merge NUMBER --repo AMkuro/Call_Me_Maybe --auto --squash --match-head-commit SHA

Use its current full head SHA. Never use --admin, bypass protection, create fake
statuses, enable auto-merge on all PRs, or opt in an experiment/draft. Resolve
conflicts and failed checks on the branch and retest. If main moves, merge
origin/main into the branch, push, and wait for new checks. Report permissions
or settings errors; do not weaken protection. Existing enrollment is not proof
of completion: verify state=MERGED, mergedAt and mergeCommit on GitHub.

# Design and implementation process

Do not start implementing from a vague request. Design first.

1. Write the design under docs/ before any code in src/: requirements (from
   docs/subject-v1.7.md), components and data flow, and the decisions still
   open. In Claude Code, use the engineering:system-design skill for the
   structure and the doc-coauthoring skill for the document.
2. Record each architectural decision as an ADR: context, options, trade-offs,
   decision, and what to revisit. The human decides. The agent proposes options
   with a recommendation and never marks a decision as made.
3. Start implementing only after the human approves the design.
4. Write tests first. Put the model behind a small port so tests can use a
   fake model with fixed logits, and check the constraint properties (every
   output parses and matches the schema) without loading Qwen3.
5. Work in small units. Run a reviewer on each unit and show the result and
   the checks to the human before starting the next unit.
6. Keep facts apart from guesses. If something cannot be verified, say so and
   say what was tried.
7. Keep accuracy measurements and timing measurements separate in every
   note and report. Accuracy runs may be parallel. Timing runs are ideally
   single-run; if run in parallel, record the variance and the outlier rule
   (see docs/design/adr/0010-measurement-policy.md).
