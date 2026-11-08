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
