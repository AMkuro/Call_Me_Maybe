# Development and integration

Subject v1.7 requirements are summarized in docs/subject-v1.7.md. Read it
instead of fetching the Subject from GitHub; re-check the original only if the
version changes or the summary cannot settle an important question.

Read docs/development.md and docs/ci-auto-integration.md before changing CI,
the build environment or merging. README.md is a submission artifact: keep
development-only content (PoC, CI, GPU setup, SDK patches, branches, status)
out of it and put it in docs/development.md.
Do ordinary development on local develop. Do not create a branch per task.
Use exp/* only for experiments that explicitly need isolation. Parallel AI
investigation is allowed, but keep ordinary code and design work together.
Branch creation and PR creation are not standard development steps.
Do not merge existing experimental/PoC branches as part of unrelated work.

Run make install, make lint, make test. CI is disabled (manual runs only)
until the implementation is done and will be reworked then. Use the Subject
mypy flags. The SDK is excluded from lint through .flake8 and the mypy
settings. The repository copy of llm_sdk has one documented local change
(float32 on every device, commented in the code); make sdk-install unpacks
the distributed version from llm_sdk.zip (or SDK_ZIP=<zip>). Do not
otherwise edit llm_sdk. Development checks are not a submission certificate.

Use the GPU for real-model development runs through make gpu-run and the
separate .venv-gpu environment (owner instruction, 2026-11-10). Reserve CPU
runs for explicit CPU measurements or submission checks; do not default to
four CPU threads for development inference. Verify GPU availability in the
actual execution environment, since the sandbox can hide GPU device nodes.

For provisional campus application-time estimates, use the measured phys16-t6
total application time plus 8 seconds (owner decision, 2026-11-11). Preserve raw
measurements separately; add no actual sleep and do not apply the offset per
prompt/token/forward or to hardware-probe kernels. This is based on the known 11
inputs, not inference-profile equivalence. The SDK phys16-t8 default is separate.

Remote publication is separate from local edits, verification and commits.
A request to fix something does not authorize a push or a PR mutation: obtain
a separate explicit instruction before pushing, creating/updating a PR, or
enabling its merge. Remote merges also require explicit authorization.
Remote history persists and is costly to revise; do not publish intermediate
local work automatically.

Only when PR creation or updates and integration are explicitly authorized,
include only the authorized changes. Push permission alone does not authorize
PR operations. For an explicitly authorized PR merge, inspect its diff and
required checks, then opt in ONLY that intended integration PR using:

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
   evidence. Record important design reasoning in the relevant section of
   docs/architecture.md and keep facts apart from guesses.
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

Apply the owner's three scope rules:

- For documentation edits, resolve related contradictions within the authorized
  scope. Do not automatically expand into code or other document changes.
- For code edits, include the necessary tests and verification in that request.
- The owner normally handles push and PR operations. Perform them only when
  explicitly requested; local edits and commits do not authorize publication.

For each request that changes files, complete the authorized work and checks, then
make a local commit before the final response that ends that request. Do not
wait for another commit instruction or for the whole project to finish, unless
the owner explicitly requests otherwise for that request.
Local commits and remote publication are separate: do not infer permission to
push or mutate PRs from a correction or local commit.
Ask only when a material decision is missing, and preserve the owner's
design-approval rules.

docs/architecture.md is the source of truth for the current application design.
Keep it consistent with implemented code when the design changes. Keep proposed
changes separate from the current design; do not describe proposals as implemented.
For important decisions, record the background, adopted rationale, alternatives,
constraints and verification method in the relevant design section.

Do not create new ADRs or automatically update existing ADRs. Preserve existing
ADR files as history: do not delete, renumber or reorganize them. Do not consult
them as authoritative sources during ordinary investigation, design or
implementation. Read them only when the user explicitly requests investigation
of historical design decisions.

1. For small fixes, do not add unnecessary design documents or approval stages.
   Clarify vague requirements only when a material decision is missing.
2. For important specification or structural changes, present a design proposal
   and obtain human approval before implementation. Do not mark proposals as
   human decisions until approved.
3. Divide work into auditable changes and run the necessary checks for each.
   Implementation does not have to happen in one batch.
4. Write tests first for code changes. Test the pure parts (name tree, masks,
   skeleton, validation) directly with data, the decode loop with a function that
   returns fixed logits, and the end-to-end properties (every output parses
   and matches the schema) on the real model. Do not add an abstraction
   only to make tests possible.
5. Preserve existing functionality, performance requirements and the Subject's
   Pydantic validation requirement. Verify AI output independently; do not
   assume an implementation is correct because an AI produced it. Review
   substantive changes and report verification results to the human.
6. Keep facts apart from guesses. If something cannot be verified, say so and
   say what was tried.
7. Keep accuracy measurements and timing measurements separate in every
   note and report. Accuracy runs may be parallel. Timing runs are ideally
   single-run; if run in parallel, record the variance and the outlier rule.
