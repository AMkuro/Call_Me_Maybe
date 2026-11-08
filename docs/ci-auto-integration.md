# CI and native agent auto-integration

> **Status (2026-11-09): the workflow is disabled** (manual runs only) and
> `make lint-app` and `make lint-sdk` (with `tools/audit_sdk.py` and
> `.github/ci/sdk-mypy.ini`) were removed, so the sections below describe the
> 2026-11-07 design, not the current state. CI is reworked after the
> implementation. The SDK is now excluded from `make lint` through `.flake8`
> and the mypy settings, and its integrity is checked by `make sdk-install`
> (`llm_sdk.sha256`). The main Ruleset's required checks cannot report while the
> workflow is disabled. `.github/ci/sdk-baseline.json` is kept only because
> `tools/campus_survey.py` reads it.

## Scope and reference analysis (2026-11-07)

Reference-only inspection of [Fly-in](https://github.com/AMkuro/Fly-in), default
branch develop, found:

- [verify-and-sync.yml](https://github.com/AMkuro/Fly-in/blob/develop/.github/workflows/verify-and-sync.yml)
  runs Python 3.10/3.14 quality and exported-submission checks on develop
  push/PR plus manual dispatch. It has read-only permissions, timeouts and
  concurrency cancellation. Despite the filename, it does not synchronize refs.
- [release-candidate.yml](https://github.com/AMkuro/Fly-in/blob/develop/.github/workflows/release-candidate.yml)
  manually plans an exact develop SHA, verifies source/export, compares a plan
  hash, uses the release environment, creates release/SHA and opens a main PR.
  A separate manual verification checks the branch again. It never merges.
- [Ruleset 25780975](https://github.com/AMkuro/Fly-in/rules/25780975) protects main
  with PR, squash, resolved threads, no force push/deletion, zero required
  approvals, and **no required status checks**. Auto-merge was disabled.
- Its Makefile uses a pip virtualenv; clean even deletes uv.lock. None of that
  dependency installation/cleanup behavior is copied into this uv project.

Transferred: Python compatibility matrix, real lint/tests, Subject-aware
checks, read-only checkout, concurrency and timeouts. Simplified: one main,
ordinary task branches, no export manifest or release workflow. Not adopted:
manual dispatch/release environment, SHA/plan approval, release branches,
develop synchronization, and full model benchmarks in PR CI.

## Development checks and submission limits

Required contexts (exact job names, GitHub Actions app ID 15368):

| Check | What it establishes |
| --- | --- |
| Quality (Python 3.10) | Locked uv install, owned-code flake8, mandatory mypy flags, unittest |
| Quality (Python 3.14) | Same checks on the newer interpreter |
| Subject development checks | SDK hash/diagnostic baseline, declared numpy/Pydantic, sample file shape, README/Makefile presence, static import restrictions, untracked generated output; CLI help when src exists |

No path filters or workflow-level skip conditions: documentation-only PRs run
the same required checks. Python failures do not cancel the other matrix job.
PR checkout tests GitHub's merge revision, not only an isolated head commit.
Strict up-to-date protection requires new checks after main advances.

`make lint-app` excludes only the distributed SDK. Mypy resolves the SDK's
actual public annotations via mypy_path and suppresses vendor-body diagnostics
only; the SDK is separately audited with isolated linter configuration and
pinned versions. `make lint-sdk` prints all 11 flake8 and 4 mypy baseline errors
and compares exact diagnostics plus hashes of all tracked SDK files. Changes
fail. This is a visible, narrow upstream exception, not an ignore-all policy.
`make lint` runs the full-tree Subject commands and remains failing. Upstream
lint debt needs resolution/clarification before submission; do not edit vendor
code to satisfy this integration task.

The Subject's full lint invocation is flake8 . and mypy . with:
`--warn-return-any --warn-unused-ignores --ignore-missing-imports
--disallow-untyped-defs --check-untyped-defs`.

This PoC has no src entry point. `make subject` reports that explicitly while
checking the existing development contract. `make subject-audit` fails on it.
Even a successful preflight cannot establish constrained decoding, all-class
Pydantic inheritance, exception handling on every path, dynamic private SDK
access, unseen-function semantic accuracy, 100% schema-valid generated JSON,
or a sub-five-minute evaluation. These need application tests, human review
and the separate model benchmark. Existing feature work is not auto-integrated
to fill those gaps. README documents the current implementation honestly.

## Protection and settings

Desired settings are reviewable in [.github/ci](../.github/ci/). Main uses one
active Ruleset with no bypass actors, PR required, all three required checks
bound to GitHub Actions, strict up-to-date checks, zero approving reviews,
resolved review threads, squash only, no force push or branch deletion. Thus
there is no human release approval in the normal merge path. Check failures,
missing checks, conflicts, unresolved discussions, or insufficient rights stop
merging. Do not duplicate this with legacy Branch Protection; applicable rules
would accumulate and could accidentally require approval.

The existing default `require_extra_approval_for_unattributed_changes=true`
is retained. GitHub REST documentation limits it to unattributed Copilot PRs
when the configured approval count is nonzero; our ordinary count is zero.
No change to this extra protection is needed for the gh agent flow.

Repository Auto-merge is enabled; merge/rebase methods are disabled. Actions
keeps read-only default permissions and cannot approve PR reviews. CI explicitly
uses contents:read and never pushes, opens PRs, or enables auto-merge itself.

Auto-merge is an **explicit per-PR opt-in**, not a rule that enrolls every PR.
The authenticated agent enables it only for its authorized task, after checking
the diff and full SHA. Existing PRs, experiments, drafts and independent PoCs
remain unenrolled. Labels/author allowlists are unnecessary for this model.
Use draft PRs for unfinished work. Stop enrollment with `gh pr merge NUMBER
--disable-auto` before repurposing a PR. A writable collaborator can still
explicitly request an ordinary merge after required checks; branch names alone
are not an authorization boundary. Repository admins can edit settings, so no
repository rule can protect against a malicious administrator.

## Agent session procedure

Use an authenticated user or GitHub App with repository contents and pull
request write access. Fine-grained tokens also need workflows:write to change
workflow files; an OAuth/classic token needs repo and workflow scopes for those
file changes. SSH push uses its own authorized SSH identity. Administration
write is needed only to configure the Ruleset/settings, not normal PR work.
Check `gh auth status` without displaying or logging tokens. Never save tokens
in repository files, Issues, PR bodies or job logs.

```sh
git fetch origin
git switch -c feat/authorized-task origin/main
# edit and review only the requested task
make install
make lint test
git add path/to/changed/files
git commit -m 'feat: describe the task'
git push -u origin HEAD
# Prepare the exact PR description in a file, then:
gh pr create --repo AMkuro/Call_Me_Maybe --base main \
  --head feat/authorized-task --title 'Describe the change' --body-file /tmp/pr.md
gh pr diff NUMBER --repo AMkuro/Call_Me_Maybe
git rev-parse HEAD
# Substitute that full SHA; do not use --admin.
gh pr merge NUMBER --repo AMkuro/Call_Me_Maybe --auto --squash \
  --match-head-commit FULL_HEAD_SHA
gh pr checks NUMBER --repo AMkuro/Call_Me_Maybe --required --watch
gh pr view NUMBER --repo AMkuro/Call_Me_Maybe \
  --json state,mergedAt,mergeCommit,autoMergeRequest
```

The `--auto` command may merge immediately if all requirements already pass.
On updates: commit/push to the same branch and inspect current checks; re-enroll
only if native Auto-merge was disabled (for example after an unauthorized
contributor updates it). If main advances, merge origin/main into the task
branch and push to rerun the merge revision. An out-of-date PR waits safely;
no separate background updater is installed. Recheck the final merged state.

Do not create PRs from Actions for this workflow. The current GitHub docs say
GITHUB_TOKEN-created/updated PRs require approval to run their pull_request
workflows; GITHUB_TOKEN pushes generally do not trigger subsequent workflows.
That would defeat unattended integration. Local Claude Code/gh sessions or an
explicitly configured GitHub App token avoid this restriction without a new
privileged workflow. Do not assume `gh pr merge --auto` can cure absent CI.

Sources: [native Auto-merge](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/automatically-merging-a-pull-request),
[CLI merge flags](https://cli.github.com/manual/gh_pr_merge),
[Ruleset rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets),
[Rules REST API](https://docs.github.com/en/rest/repos/rules),
[workflow token trigger limitations](https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-when-your-workflow-runs/triggering-a-workflow).

## Settings audit and live evidence

See [verification record](ci-verification.md) for before/after settings, run
URLs, failure tests, opt-in isolation, and the limits of main verification.
