*This project has been created as part of the 42 curriculum by miwasaki.*

## Description

Call Me Maybe explores constrained decoding for LLM function calling. The
current main branch contains a public SDK greedy-token PoC and mock tests.
It is **not a completed Subject v1.7 submission**. The shared CLI lives on an
independent, unmerged feature branch; this infrastructure change does not
integrate it or the experimental generation methods.

## Instructions

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
make test
make lint-app
make lint-sdk
make subject
```

`uv sync` also works as required by the reviewer. Root dependencies are numpy
and Pydantic; SDK inference dependencies are deliberately separate. `make run`
and `make debug` target the Subject entry point `python -m src`; they become
usable when an application is integrated. On this PoC, src is missing.

For the independent inference experiment, see [examples](examples/README.md).
Running it requires a separately provisioned SDK environment and may download
the Qwen/Qwen3-0.6B model. PR CI never provisions this environment.

## Testing strategy

Unittest uses SDK-shaped mocks and no model imports, downloads, or forward
passes. CI runs Python 3.10 and 3.14, locked dependencies, owned-code flake8,
the mandatory mypy flags, development Subject checks, and an exact SDK audit.
Future src tests are discovered automatically. An absent/failed required check
blocks merging. See [CI and agent workflow](docs/ci-auto-integration.md).

`make lint` executes full-tree flake8 and mypy using all Subject flags. It
currently **fails**: the unchanged upstream SDK has 11 flake8 and 4 mypy errors.
`make lint-sdk` prints these diagnostics and compares exact outputs and SDK
hashes with [.github/ci/sdk-baseline.json](.github/ci/sdk-baseline.json).
Any difference fails; do not refresh that baseline to hide problems. Development
CI checks owned code separately, preserving the SDK's public type annotations.
`make subject-audit` is a stricter submission preflight and also fails on this
PoC. It does not replace functional evaluation or peer review.

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
application's direct-import restrictions. CI fingerprints and audits that
upstream code without installing its heavy dependencies or changing it.
Static import checks cannot prove all SDK attributes are public or that all
runtime classes use Pydantic; those remain explicit review requirements.

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
