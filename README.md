*This project has been created as part of the 42 curriculum by miwasaki.*

## Description

Call Me Maybe explores constrained decoding for LLM function calling. This
branch contains a public-SDK greedy-token PoC, mock tests and the build
environment (Makefile, CPU and GPU setups). It is **not a completed Subject
v1.7 submission**: the application (`src`) is not implemented yet, and the
experimental generation methods are not integrated.

## Instructions

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```sh
make install      # recreate .venv from uv.lock (CPU torch)
make test
make lint
```

`uv sync` alone also works, as the reviewers require. The dependencies are
numpy, Pydantic and the local `llm_sdk` (a path dependency). torch comes from
the default `cpu` dependency group, which uses the PyTorch CPU index.

Development on a GPU machine (Linux with an AMD GPU, or a Mac) uses a separate
environment so `.venv` stays identical to the reviewer's:

```sh
make gpu-install  # recreate .venv-gpu (ROCm torch if /dev/kfd exists, plus accelerate)
make gpu-run      # run in .venv-gpu
```

`make sdk-install` restores `llm_sdk` from `llm_sdk.zip` and checks its sha256
against `llm_sdk.sha256`. The repository copy of `llm_sdk` has one local change
(float32 on every device instead of float16 on GPU; commented in the code), so
GPU results match those of the CPU-only review machine.

`make run` and `make debug` target the Subject entry point `python -m src`.
They become usable once the application exists; on this PoC, `src` is missing.

For the independent inference experiment, see [examples](examples/README.md).
The SDK downloads the Qwen/Qwen3-0.6B model itself on first use.

## Testing strategy

Unittest uses SDK-shaped mocks and no model imports, downloads, or forward
passes. Future src tests are discovered automatically.

CI is **disabled** for now: the workflow runs only when started manually. It
will be reworked after the implementation (see
[CI and agent workflow](docs/ci-auto-integration.md), which describes the
earlier design).

`make lint` runs flake8 and mypy with all Subject flags and passes (checked
2026-11-09). The distributed SDK is excluded by `.flake8` and the mypy settings
in `pyproject.toml`, not by the Makefile commands. `make subject` passes with a
notice that `src` is missing, and `make subject-audit` is a stricter preflight
that fails on this PoC. Neither replaces functional evaluation or peer review.

## Algorithm explanation

The PoC observes greedy token selection through public SDK methods. It does not
implement the required constrained function selection or argument generation.
No accuracy or JSON-compliance guarantee is claimed for this revision.

## Design decisions

Keep inference experiments separate from the common CLI and validation work.
Keep development integration separate from final submission certification.
Opt in to native Auto-merge per PR; no custom merger or release branches.

## Performance analysis

Accuracy, generation speed, schema reliability and CPU benchmarks remain
unmeasured here. A complete submission needs unseen prompts/functions, 90%+
semantic accuracy, 100% schema-valid JSON, and the Subject's runtime evaluation.
These require a separately provisioned, recorded model benchmark.

## Challenges faced

The supplied SDK depends internally on torch and transformers despite the
application's direct-import restrictions. The repository therefore pins the
distributed SDK by sha256 (`llm_sdk.sha256`) and can restore it with
`make sdk-install`. Static import checks cannot prove that every SDK attribute
used is public or that every runtime class uses Pydantic; those remain explicit
review requirements.

## Example usage

The intended application invocation after the CLI and generator are integrated:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Output must contain exactly `prompt`, `name`, and `parameters` for each input.
Generated `data/output/` files must remain untracked.

## Resources

- [Subject v1.7](https://github.com/AMkuro/pdf/blob/milestone3/Call_Me_Maybe/Call-Me-Maybe.en.subject.txt)
- [Python unittest](https://docs.python.org/3/library/unittest.html)
- [Pydantic](https://docs.pydantic.dev/latest/)
- [GitHub Auto-merge](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/automatically-merging-a-pull-request)

AI assisted the SDK experiment, tests, CI design, configuration, documentation,
and verification. The learner must understand the changes and seek peer review;
a green development CI is not a substitute for the 42 defense.
