# Development and integration

Subject v1.7 requirements are summarized in docs/subject-v1.7.md. Read it
instead of fetching the Subject from GitHub; re-check the original only if the
version changes or the summary cannot settle an important question.

Read README.md and docs/ci-auto-integration.md before changing CI or merging.
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

Create or update a PR with only the authorized task's changes. After inspecting
its diff and required checks, opt in ONLY that intended integration PR using:

    gh pr merge NUMBER --repo AMkuro/Call_Me_Maybe --auto --squash --match-head-commit SHA

Use its current full head SHA. Never use --admin, bypass protection, create fake
statuses, enable auto-merge on all PRs, or opt in an experiment/draft. Resolve
conflicts and failed checks on the branch and retest. If the base branch moves,
merge it (origin/develop or origin/main) into the branch, push, and wait for
new checks. Report permissions
or settings errors; do not weaken protection. Existing enrollment is not proof
of completion: verify state=MERGED, mergedAt and mergeCommit on GitHub.
