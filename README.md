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
make sdk-install  # unpack an SDK zip over llm_sdk (SDK_ZIP=<zip>, default llm_sdk.zip)
```

`uv sync` alone also works. The dependencies are numpy, Pydantic and the
local `llm_sdk` (a path dependency); torch is installed as a dependency of the
SDK. The SDK downloads the Qwen/Qwen3-0.6B model itself on first use.

When stderr is a terminal, a live dashboard shows the selected function,
current argument, token choices and diagnostic scores. Redirecting stderr
produces ordinary result and warning lines instead of terminal controls.

The program reads the function definitions and the prompts from
`data/input/` and writes `data/output/function_calls.json`:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

## Algorithm explanation

Function names and the model's end-of-sequence token form a token trie.
At each branch, invalid logits are set to negative infinity and the
highest-scoring allowed token is selected. A single allowed token is forced
without a model forward pass. This also distinguishes a function name from
another name that extends it.

The program then forces the JSON keys and punctuation from the selected
definition, generating only parameter values and their following delimiter.
Number, string and boolean masks recognize valid JSON prefixes. String masks
operate on bytes so a UTF-8 character split across tokens is preserved.
The completed record is parsed as JSON and checked against a dynamically
constructed Pydantic model.

## Design decisions

The input prompt is copied, and function names and parameter keys are escaped
with `json.dumps`. Parameters follow definition order. The SDK is wrapped by
one concrete class; application code only receives plain token-ID lists.

Numbers are normalized to Python floats after validation. This can lose
precision for large integers and long decimals; non-finite results are
rejected. A value has a provisional limit of 256 tokens. Failure to finish
reports the parameter and limit, skips that prompt and returns a nonzero
exit status. Successful results are saved after each prompt by atomic
replacement of the output file.

## Performance analysis

The SDK recomputes the whole input prefix at each forward pass, so both input
length and generated length affect runtime. Forced structure and unambiguous
name paths avoid forward passes. String masks parse the shared prefix once,
then inspect only each candidate's new bytes. The dashboard reuses existing
logits and performs no additional model inference.

Semantic accuracy and total runtime depend on the function definitions,
requests and hardware. Valid JSON alone does not establish correct arguments.

## Challenges faced

The supplied SDK depends on torch and transformers internally, while the
project's own code must not import them. The application therefore only uses
the public methods of the SDK (`encode`, `decode`,
`get_logits_from_input_ids`, `get_path_to_vocab_file`).

## Testing strategy

Unit tests check tries, masks, forced structure, validation and file handling
directly. Fixed logits exercise the decode loop, and boundary tests mock only
the external SDK. They do not download or load the model. Real-model checks
are needed separately for schema compliance and semantic accuracy. `make lint`
runs flake8 and mypy with the flags required by the Subject.

## Example usage

See the command in Instructions. The output contains exactly `prompt`, `name`
and `parameters` for each input. Generated files in `data/output/` are not
tracked.

## Resources

- [Python unittest](https://docs.python.org/3/library/unittest.html)
- [Pydantic](https://docs.pydantic.dev/latest/)
- [uv](https://docs.astral.sh/uv/)

AI assistance (Claude Code and Codex) was used for design discussion,
implementation, tests, build configuration and documentation. The human owner
makes architectural decisions and reviews the changes.
