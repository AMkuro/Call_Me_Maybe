# Development notes

This file holds what only the development repository needs. `README.md` is a
submission artifact and must not carry any of it: this file is removed or kept
out of the submitted version. Read it (and `docs/ci-auto-integration.md`)
before changing CI, the build environment or the SDK.

## Local develop integration (2026-11-12)

Ordinary development and design work now converge on local `develop`; independent
experiments remain separate. Publication still needs explicit authorization.
See the [branch inventory and integration record](integration-2026-11-12.md).
AGENTS.md still describes the previous branch/PR workflow; this task's explicit
owner instruction overrides it. Updating that policy file is a separate task.

The eight calibrated suites now live in `tests/fixtures/calibrated/`, with their
static consistency test. The literature survey is in `docs/literature/`.
The observation-boundary proposal is ADR-0024, unapproved and unimplemented;
its ranking-related parts are superseded by ADR-0023. Earlier dated sections
below record historical branch status, not the current checkout.

The legacy CPU watch on `feat/cpu-watch` is **not integrated**: review found
that descendants can escape monitoring when their group leader exits, and
that final paused time is omitted from summaries. Its historical design is
preserved as ADR-0025, with adoption deferred pending a separate repair.
The existing thermal survey remains available; GPU development remains required.

The dev group pins uv 0.11.21 from feat/cpu-watch, making `.venv/bin/uv` available
for sandbox checks when the host uv is unavailable. `make install` deletes that
venv, so installation must use a uv executable outside `.venv`.

## Token progress update (2026-11-12)

The token ranking and candidate table have been removed at the owner's request
([ADR-0023](design/adr/0023-remove-token-ranking.md)). Token-by-token value
progress, constrained-name warnings, completion/failure summaries and terminal
restoration remain. Diagnostics no longer compute softmax, allowed logit gaps
or top-k rankings. This does not change constrained argmax selection or the CLI.

## Status (2026-11-10)

The design now describes LF-terminated typed value records, quoted newlines and
grouped-number normalization (ADR-0015). That protocol is a documentation update only:
the application and tests still use the existing JSON-delimiter protocol. The
measurements below do not validate the new design. Publication is a separate
owner instruction from local edits and commits (see AGENTS.md).

The owner approved the application skeleton, merged into `develop` in PR #43.
Implementation on `feat/implementation` includes all skeleton leaves, byte-level
value masks, the EOS trie, atomic output and token-level terminal diagnostics.
The provisional value cap is 256 tokens, with parameter-specific errors when
generation fails to end. The current tests pass (150 tests), and independent
review findings have regression fixes. The independent PoC stays in `examples/`.

PR #44 reviews the implementation foundation: component boundaries, decoding
constraints, validation, persistence, error handling and terminal observation.
The owner explicitly separated this approval from semantic-accuracy tuning.
The measured accuracy remains reference evidence for later parameter/component
changes; reaching 90% is not an approval condition for this foundation PR.
Evidence of an infrastructure defect still belongs in this review.

The owner clarified that an empty prompt array is insufficient input. It now
fails startup validation before the model loads, leaving existing output
untouched and creating no new output. A nonempty input whose prompts all fail
still saves `[]` and exits nonzero (ADR-0012, 2026-11-10 update).

Output preparation now runs after both inputs validate and before `Model()`.
It creates required parent directories, rejects existing non-file destinations,
and opens/writes/flushes/removes a sibling temporary file using the same naming
limits as the atomic writer. It preserves existing output and does not create
the final file. Preparation failures stop startup with the path and reason;
save-time errors remain handled because a probe cannot guarantee later writes.

Function sets now use `dict[str, FunctionDef]`, indexed once by exact name after
the input JSON array is validated. Application loading and measurement fixtures
share `index_functions`, which rejects duplicate names and keeps input order.
`FunctionDef` rejects blank names through Pydantic validation. Name prompts enumerate `values()` and the trie uses keys; selection
retrieves the original definition directly. This removes the final linear
lookup, without changing forward calls or claiming a measured runtime gain
(ADR-0016).

JSON readers now share stdlib hooks that reject duplicate keys before overwrite,
reject NaN/Infinity and retain Decimal source values. Pydantic owns strict
shape/type validation and metadata warnings. Integers remain exact ints;
decimals use a float only when JSON serialization retains the source value,
with exact integral fallback or a clear precision/range error (ADR-0017).
Blank descriptions/argument names warn on stderr without rewriting input.
The independent measurement comparison no longer accepts bool/number equality
or rounded expected numbers. The historical model measurements below predate
this validation change. Subsequent CPU/GPU checks of the same 11 samples
completed with 11 valid records and 10 correct calls; they do not establish
unseen accuracy or measure the effect of stricter validation. Serial parsing
costs, GPU timing and test results are in the verification record.

Real-model samples produced valid typed JSON in 11/11 cases and correct calls
in 10/11. Changed definitions produced valid typed JSON in 5/5 and correct calls
in 3/5, so semantic accuracy is below the Subject target. Separate serial CPU
timing took 98.94 seconds for the 11 samples (cached model, four threads).
See [verification record](implementation-verification.md) for the actual cases,
prompt changes, repeated limit tests, interrupted measurements and limitations.

A user-provided campus application summary reports65.74/71.41/71.49seconds
and exit0; raw evidence and exact campus workload/environment remain unverified.
Provisional local trials gave medians63.45s at6threads and81.92s at5threads,
so neither matches the reported campus median71.41s. The timing harness now
accepts CMM_OMP_THREADS while retaining its default8-thread SDK profile.
No inference-equivalent profile is adopted. The owner later chose a practical
application-time estimate: measured phys16-t6 total time plus 8 seconds. This gives
71.45 s from the 63.45 s median; raw times remain intact and no sleep is added.
The rule applies once per application run, based on the known 11 inputs;
normalized work and CPU temperature curves remain future checks. See ADR-0010
and the verification record for provenance and limits.

[Prompt experiment hypotheses](research/prompt-hypotheses.md) collects primary
research, proposed comparisons and evaluation criteria for later accuracy tuning.
It records proposals only; the research task did not run model experiments or
change application behavior.

The Qwen3-0.6B cache used for the terminator probe is under the ignored
`.cache/huggingface/`; set `HF_HOME="$PWD/.cache/huggingface"` to reuse it.
Inside the Claude Code sandbox, which blocks huggingface.co, also set
`HF_HUB_OFFLINE=1`: otherwise loading fails on the update check. Both can go
in the ignored `.local.mk` (`export HF_HOME := $(CURDIR)/.cache/huggingface`,
`export HF_HUB_OFFLINE := 1`), which `make gpu-run` reads.
This is an existing development-machine cache, not a campus download location.
Campus timing instructions reuse an existing cache offline and stop when it
is missing; they do not download model weights into the repository.
The cache-detection unit test now isolates its repository path as well as its
home directory, so a real downloaded cache cannot affect its fixture.

`make run` and `make debug` target the Subject entry point `python -m src`;
they work once `src` exists. For the independent inference experiment, see
[examples](../examples/README.md).

[Hardware comparison probe](research/hardware-probe.md) investigates CPU and
memory workloads without loading or downloading models. It compares installed
NumPy/BLAS kernels across machines and thread settings; it does not certify
campus performance reproduction or predict the application's runtime.

## Isolated prompt-improvement checkpoint (2026-11-11)

A separate `exp/local-prompt-improvement` worktree now holds a
workflow design proposal (`docs/superpowers/specs/2026-11-11-local-prompt-improvement-design.md`
on that branch) and a [static fixture review](research/calibrated-fixture-review.md).
The experiment tooling (`tools/prompt_*.py`, `research/prompt-improvement/`,
their tests and fixture, the `prompt-improve` targets) stays on that branch;
`develop` receives only the prompt changes in `src/` and the recorded results.
The proposed local model has only prompt-saving and fixed-evaluation tools;
GPU improvements would be checked serially on phys16-t6 (raw total time plus
8 seconds as a separate estimate), then on the fixed calibrated fixtures.
Updates from `feat/implementation` would be merged between rounds and the
baseline and candidate re-evaluated on the same code. The owner approved the
design, then requested a stronger optimizer and agent-based model research.
The selected first pilot is Qwen3.5-9B BF16, while the target stays Qwen3-0.6B
float32. The tooling now passes 174 unit tests and lint after the resumed review fixes.
The 9B GPU/BF16 loader and a short JSON generation passed; prompt-search
outcomes are recorded separately.
See the [model research](research/local-prompt-editor-models.md) and
local handoff (untracked, under `.cache/handoff/`) before running it.
The inherited line-length error in `src/display.py` was fixed by wrapping
its signature only; application behavior is unchanged. Model snapshots and
actual installed package/SDK versions are bound to the evidence fingerprint.
The calibrated branch has now been statically checked at eight suites (88
cases), pinned to commit `8d63548988a37bd1862b7192076fea3b1e946154`.
The calibrated data have 5 functions and 11 requests per suite, but unequal
function frequencies and a distinct boolean type mix. They are supplemental
checks, not evidence of equal difficulty or unseen accuracy.

The owner subsequently requested an existing OSS agent harness (Ponytail full).
Prompt improvement now configures installed OpenCode1.15.10 and Transformers
Serve, exposing only `write_prompt` and `run` in isolated workspaces. The old
custom agent loop was removed. Two unscored SDK attempts were stopped; six
proposal slots remain. See the OSS setup (`research/prompt-improvement/README.md`
on `exp/local-prompt-improvement`) and ADR-0019. Server-only dependencies remain in the development GPU environment.

## Environments

The owner requires GPU use for real-model development work (2026-11-10).
Use `make gpu-run` by default, and `make gpu-install` when `.venv-gpu` is
missing. CPU runs are for explicit CPU measurements or submission checks;
four CPU threads are not the default development profile. Verify that the
execution environment exposes the GPU before running: this sandbox hides
`/dev/kfd` and `/dev/dri`, so GPU runs require host execution.

[Campus runtime feasibility and proposed measurements](research/campus-runtime-feasibility.md)
records the CPU/memory reports pulled from `feat/implementation` on 2026-11-11.
The measured workload ratios do not establish a reproduced campus machine or
the Subject's five-minute result. The owner approved automated continuous SDK
work with safety stops. See [thermal survey](research/safe-thermal-survey.md)
and ADR-0022. VM configuration management is a selection criterion alongside
cgroup calibration; the VM choice remains open. The AMD development host has
no readable thermal counters, so this tool refuses to load the model there.

- `make install` recreates `.venv` from `uv.lock` (CPU torch). `uv sync` alone
  also works, as the reviewers require. torch comes from the default `cpu`
  dependency group, which uses the PyTorch CPU index.
- GPU development (Linux with an AMD GPU, or a Mac) uses a separate
  environment so `.venv` stays identical to the reviewer's:

  ```sh
  make gpu-install  # recreate .venv-gpu (ROCm torch if /dev/kfd exists, plus accelerate)
  make gpu-run      # run in .venv-gpu
  ```

- `make sdk-install` unpacks `llm_sdk.zip` over `llm_sdk` (another zip:
  `SDK_ZIP=<zip>`).
- The repository copy of `llm_sdk` has one local change: float32 on every
  device instead of float16 on GPU (commented in the code), so GPU results
  match those of the CPU-only review machine.

## Submission

`Makefile` is submitted, so it holds only what the Subject and reviewers need.
Development targets (`gpu-*`, `subject*`, `dev-help`, the `.local.mk` hook) live
in `dev.mk`, which `Makefile` includes with `-include` and the submission does
not contain. `submission-files.txt` is the allowlist of submitted files (the
method of the Fly-in repository). Export one commit without touching the
checkout:

```sh
.venv/bin/python -m tools.export_submission <directory> [--ref <commit>]
```

It fails while a manifest entry matches nothing (today `src/**/*.py`).
`llm_sdk.zip` is not submitted (the Subject lists the `llm_sdk/` directory only).
The repository copy of `llm_sdk/` carries the local float32 change, so before
the final submission run `make sdk-install` to restore the distributed
version, commit it, and only then export. No automated check (decided: not
worth it); the manual steps are in [submission-checklist.txt](submission-checklist.txt). `sdk-install` stays in the submitted `Makefile`: the zip is not
submitted, so a reviewer passes theirs with `SDK_ZIP=<zip>`.
Fly-in also has a release-candidate flow
(`tools/prepare_submission.py`, workflows); it is not adopted here yet.

## Checks

- `make lint` runs flake8 and mypy with all Subject flags and passes (checked
  2026-11-09). The distributed SDK is excluded by `.flake8` and the mypy
  settings in `pyproject.toml`, not by the Makefile commands.
- `make subject` passes with a notice that `src` is missing, and
  `make subject-audit` is a stricter preflight that fails while `src` is
  missing. Neither replaces functional evaluation or peer review.
- Unit tests use SDK-shaped mocks and no model imports, downloads or forward
  passes.
- CI is **disabled** for now: the workflow runs only when started manually. It
  is reworked after the implementation (see
  [CI and agent workflow](ci-auto-integration.md), which describes the earlier
  design).

## Integration

Keep inference experiments separate from the common CLI and validation work,
and development integration separate from final submission certification. Opt
in to native Auto-merge per PR; no custom merger or release branches.

## Known limits

Static import checks cannot prove that every SDK attribute used is public or
that every runtime class uses Pydantic; those remain explicit review
requirements. Application accuracy and completion time are not established for
this integration baseline. Recorded SDK and NumPy benchmarks do not certify
the application's runtime.
A complete submission needs unseen prompts/functions, 90%+ semantic accuracy,
100% schema-valid JSON and the Subject's runtime evaluation, measured on a
separately provisioned, recorded model benchmark.
