*This project has been created as part of the 42 curriculum by miwasaki.*

## Description

Call Me Maybe translates natural-language prompts into structured function
calls: for each prompt it outputs the name of the function to call and its
arguments as JSON, never executing the function. It uses the small
Qwen/Qwen3-0.6B model through the supplied `llm_sdk`, and relies on constrained
decoding so that the output is always valid JSON that matches the function
definitions.

## Instructions

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```sh
make install      # create .venv from uv.lock
make run          # uv run python -m src
make debug        # run under pdb
make test
make lint
make clean
make sdk-install  # unpack llm_sdk.zip over llm_sdk (another zip: SDK_ZIP=<zip>)
```

`uv sync` alone also works. The dependencies are numpy, Pydantic and the
local `llm_sdk` (a path dependency); torch is installed as a dependency of the
SDK. The SDK downloads the Qwen/Qwen3-0.6B model itself on first use.

The program reads the function definitions and the prompts from
`data/input/` and writes `data/output/function_calls.json`:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

## Algorithm explanation

*To be written once the implementation is complete.*

## Design decisions

*To be written once the implementation is complete.*

## Performance analysis

*To be written once the implementation is complete.*

## Challenges faced

The supplied SDK depends on torch and transformers internally, while the
project's own code must not import them. The application therefore only uses
the public methods of the SDK (`encode`, `decode`,
`get_logits_from_input_ids`, `get_path_to_vocab_file`).

## Testing strategy

The unit tests use SDK-shaped fakes and never load the model, so they run
without a download or a forward pass. `make lint` runs flake8 and mypy with the
flags required by the Subject.

## Example usage

See the command in Instructions. The output contains exactly `prompt`, `name`
and `parameters` for each input. Generated files in `data/output/` are not
tracked.

## Resources

- [Python unittest](https://docs.python.org/3/library/unittest.html)
- [Pydantic](https://docs.pydantic.dev/latest/)
- [uv](https://docs.astral.sh/uv/)

AI (Claude Code) was used for design discussion, tests, build configuration and
documentation drafts. The author reviewed the changes and is responsible for
them.
