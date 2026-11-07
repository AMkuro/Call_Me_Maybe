# CI and integration verification record

Date: 2026-11-07 (Asia/Tokyo). Base main: e3dd4ec4bbd5008d28ea23d1e7037b95f3d63bc4.
Existing feature branch: dee18b978b0444bbf6e3855398795272d005847f (not integrated).
Original checkout modifications in src/storage.py and tests/test_cli.py remain
untouched. All implementation and test commits are on dedicated new branches.

## Local evidence

- Baseline: 8 unittest tests passed.
- New contract tests first failed with missing tools.check_subject, then all
  13 tests passed after implementation; the review regression brought the
  final total to 14 passing tests.
- Locked uv sync, flake8, mandatory-flag mypy, SDK audit and development Subject
  checks passed locally on Python 3.12.3. Live matrix evidence follows below.
- actionlint passed (no syntax or expression errors).
- An altered SDK diagnostic baseline was rejected with exit 1; baseline restored.
- Full Subject lint failed on the unchanged SDK's 11 flake8 and 4 mypy errors.
- Submission preflight failed for missing src/__main__.py, as expected on main.
- No SDK/model import, heavyweight inference dependency, or model download used.

## Settings before

Default branch main. No Rulesets; legacy Branch Protection returned HTTP 404
(Branch not protected). Auto-merge false; squash, merge commit and rebase true.
Actions enabled, allowed_actions all, default_workflow_permissions read,
can_approve_pull_request_reviews false. Authenticated actor AMkuro has admin.
GitHub CLI authentication works outside the network-restricted sandbox; the
sandbox's DNS failure was not an invalid credential. Token material was never
read or saved. OAuth scopes include repo but not workflow; SSH is available
for workflow-file pushes. No existing open PRs were found.

## Settings after

[Before](../.github/ci/settings-before.json) and
[after](../.github/ci/settings-after.json) are non-secret API snapshots.
[Main Ruleset 26740194](https://github.com/AMkuro/Call_Me_Maybe/rules/26740194)
is active, no bypass actors, PR/squash, 0 approvals, resolved threads, strict
latest-main checks from GitHub Actions app 15368, no force push/deletion.
Auto-merge is true; merge commit/rebase false, squash true. Default branch stays
main. Actions permissions/PR review approval rights are unchanged (read/false).
Legacy Branch Protection remains unused; Ruleset is the enforcement mechanism.

An attempted change to require_extra_approval_for_unattributed_changes=false
was rejected by the automatic approval reviewer as weakening a security
protection without explicit approval. It was abandoned; the effective and
committed setting is true. Official REST docs state this only adds approval
for unattributed Copilot PRs when the base approval count is nonzero. Native gh
integration was tested with the default retained; no bypass or retry was used.

## Live verification

All checks below are actual GitHub Actions runs, not synthetic commit statuses.

| Control | PR / run | Observed result |
| --- | --- | --- |
| Normal infra PR | [#30](https://github.com/AMkuro/Call_Me_Maybe/pull/30), [initial run](https://github.com/AMkuro/Call_Me_Maybe/actions/runs/42327134633), [parser fix](https://github.com/AMkuro/Call_Me_Maybe/actions/runs/42327558655) | All 3 checks passed, app ID 15368 confirmed |
| Deliberate unittest failure | [#31](https://github.com/AMkuro/Call_Me_Maybe/pull/31), [run](https://github.com/AMkuro/Call_Me_Maybe/actions/runs/42327332128) | Both Quality jobs failed, Subject passed; Auto-merge enabled but PR stayed OPEN/BLOCKED, then disabled and closed unmerged |
| Green, non-enrolled experiment | [#32](https://github.com/AMkuro/Call_Me_Maybe/pull/32), [run](https://github.com/AMkuro/Call_Me_Maybe/actions/runs/42327362405) | 3 checks passed, state OPEN/CLEAN, autoMergeRequest null; closed unmerged |
| Explicit native Auto-merge | [#33](https://github.com/AMkuro/Call_Me_Maybe/pull/33), [run](https://github.com/AMkuro/Call_Me_Maybe/actions/runs/42327597540) | Enabled at 02:07:06Z while checks were absent/pending and state BLOCKED; automatically squash merged at 02:07:40Z, reviews=[] |

PRs #31-33 targeted only disposable ci-validation/base-20261107, protected by
an otherwise identical Ruleset. #33 merged as
1e679e8d574bdaf7f529517abcd02f397ede2283. No temporary test marker reached main.

An actual unchecked, identical-tree commit push to the protected disposable
base was rejected with GH013: "Changes must be made through a pull request"
and "3 of 3 required status checks are expected". Remote base SHA was verified
unchanged. No actual main direct push/force push/deletion was attempted; the
effective main Ruleset was read and checked instead. The dry-run push mechanism
is not claimed as proof of server protection.

## Review and compatibility

Independent read-only review found one Important issue: URL dependencies and
PEP 503 equivalent names could bypass naive dependency parsing. Replaced it
with packaging.Requirement/canonicalize_name, added the failing regression,
and reran the green suite (14 tests). No Critical or Minor issues were found.
The reviewer also reproduced SDK rejection by modifying only a temporary copy.

On an isolated copy of committed feat/a05-common-cli dee18b9 with these CI
files overlaid, 45 tests, application lint/type checking (18 source files), SDK
audit and development Subject checks passed. python -m src --help worked
without a model. This compatibility probe neither integrated that branch nor
copied the user's uncommitted changes. No algorithm or SDK changes were made.

## Main integration and remaining limits

The permanent infrastructure [PR #30](https://github.com/AMkuro/Call_Me_Maybe/pull/30)
was enrolled at 02:10:05Z while state was OPEN/BLOCKED and no checks had yet
reported. After all 3 checks passed in
[run 42327851401](https://github.com/AMkuro/Call_Me_Maybe/actions/runs/42327851401),
GitHub automatically squash merged it at **2026-11-07 02:10:33 UTC / 11:10:33 JST**,
with reviews=[], as **2b4f9c8e74bb386c46334cb6b18efb68ef994a62**.
There was no release environment or human approval step.

The resulting main push also passed all 3 jobs in
[run 42327896120](https://github.com/AMkuro/Call_Me_Maybe/actions/runs/42327896120).
Only permanent CI/configuration/documentation was integrated. The application,
greedy algorithm, distributed SDK and existing unit test file are unchanged.
The final evidence is preserved by this permanent documentation follow-up.
The original main SHA and existing feature SHA were confirmed unchanged
through all scratch-base tests. No existing feature branch/PR was enrolled.

Temporary main-targeted [PR #34](https://github.com/AMkuro/Call_Me_Maybe/pull/34)
contained only a deterministic failing unittest on top of the integrated main.
In [run 42327959196](https://github.com/AMkuro/Call_Me_Maybe/actions/runs/42327959196),
both Quality jobs failed and Subject passed. With Auto-merge enabled,
state remained **OPEN/BLOCKED**, mergedAt=null. Enrollment was then disabled,
the PR closed unmerged and its temporary branch deleted. Main stayed at
2b4f9c8 throughout this negative test.

Temporary scratch PRs #31/#32 were closed unmerged; #33 merged only to the
scratch base. All 4 scratch remote branches and the temporary Ruleset26740196
were removed after evidence collection. Main Ruleset26740194 remains active. The permanent evidence update is also
submitted through PR and all required checks; no direct main push is used.

Not demonstrated: actual main direct push rejection, a live conflict or
cancellation case, a deliberately misconfigured permission,
and heavyweight Subject evaluation. Required-check/strict-merge configuration
is evidence for the unexercised stop conditions, not a claim of live proof.
