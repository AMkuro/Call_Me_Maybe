# Development notes

This file holds what only the development repository needs. `README.md` is a
submission artifact and must not carry any of it: this file is removed or kept
out of the submitted version. Read it (and `docs/ci-auto-integration.md`)
before changing CI, the build environment or the SDK.

## Status (2026-11-09)

The application (`src`) is not implemented yet. The branch holds a
public-SDK greedy-token PoC (`examples/`), mock tests and the build
environment. The experimental generation methods are not integrated and no
accuracy or JSON-compliance claim is made for this revision. The design is in
`docs/architecture.md` and `docs/design/`.

`make run` and `make debug` target the Subject entry point `python -m src`;
they work once `src` exists. For the independent inference experiment, see
[examples](../examples/README.md).

## Environments

[Campus runtime feasibility and proposed measurements](research/campus-runtime-feasibility.md)
records the CPU/memory reports pulled from `feat/implementation` on 2026-11-11.
Those reports and the hardware probe are not part of this integration baseline.
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
