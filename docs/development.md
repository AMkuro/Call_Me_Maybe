# Development notes

This file holds what only the development repository needs. `README.md` is a
submission artifact and must not carry any of it: this file is removed or kept
out of the submitted version. Read it (and `docs/ci-auto-integration.md`)
before changing CI, the build environment or the SDK.

## Status (2026-11-10)

The design now describes LF-terminated typed value records, quoted newlines and
grouped-number normalization (ADR-0015). This is a documentation update only:
the application and tests still use the existing JSON-delimiter protocol. The
measurements below do not validate the new design. Publication is a separate
owner instruction from local edits and commits (see AGENTS.md).

The owner approved the application skeleton, merged into `develop` in PR #43.
Implementation on `feat/implementation` includes all skeleton leaves, byte-level
value masks, the EOS trie, atomic output and token-level terminal diagnostics.
The provisional value cap is 256 tokens, with parameter-specific errors when
generation fails to end. The current tests pass (101 tests), and independent
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

Real-model samples produced valid typed JSON in 11/11 cases and correct calls
in 10/11. Changed definitions produced valid typed JSON in 5/5 and correct calls
in 3/5, so semantic accuracy is below the Subject target. Separate serial CPU
timing took 98.94 seconds for the 11 samples (cached model, four threads).
See [verification record](implementation-verification.md) for the actual cases,
prompt changes, repeated limit tests, interrupted measurements and limitations.

[Prompt experiment hypotheses](research/prompt-hypotheses.md) collects primary
research, proposed comparisons and evaluation criteria for later accuracy tuning.
It records proposals only; the research task did not run model experiments or
change application behavior.

The Qwen3-0.6B cache used for the terminator probe is under the ignored
`.cache/huggingface/`; set `HF_HOME="$PWD/.cache/huggingface"` to reuse it.
The cache-detection unit test now isolates its repository path as well as its
home directory, so a real downloaded cache cannot affect its fixture.

`make run` and `make debug` target the Subject entry point `python -m src`;
they work once `src` exists. For the independent inference experiment, see
[examples](../examples/README.md).

## Environments

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
requirements. Accuracy, generation speed and CPU benchmarks are unmeasured.
A complete submission needs unseen prompts/functions, 90%+ semantic accuracy,
100% schema-valid JSON and the Subject's runtime evaluation, measured on a
separately provisioned, recorded model benchmark.
