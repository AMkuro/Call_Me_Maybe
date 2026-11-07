# CI and integration verification record

Date: 2026-11-07 (Asia/Tokyo). Base main: e3dd4ec4bbd5008d28ea23d1e7037b95f3d63bc4.
Existing feature branch: dee18b978b0444bbf6e3855398795272d005847f (not integrated).
Original checkout modifications in src/storage.py and tests/test_cli.py remain
untouched. All implementation and test commits are on dedicated new branches.

## Local evidence

- Baseline: 8 unittest tests passed.
- New contract tests first failed with missing tools.check_subject, then all
  13 tests passed after implementation.
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

## Live verification

Pending: dedicated PR runs, exact check app identity, settings after, protected
scratch-base native auto-merge and failure/direct-push isolation evidence.
Main is unchanged at this point; no auto-merge is enabled for existing work.
