# Issue #9: public SDK token generation PoC

This standalone experiment is separate from the future `src` application.
It implements neither function calling nor constrained decoding.
The supplied SDK, project dependencies, and input data are unchanged.

## Run with the real model

From the repository root, with Python 3.10+ and uv available:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" \
  uv run --no-project --with ./llm_sdk python -m examples.token_generation \
  'What is 2 + 3?' 'Say hello.' --max-new-tokens 8
```

This command installs the local SDK and its declared dependencies into uv's
temporary environment. It does not modify `pyproject.toml` or the SDK lockfile.
It downloads Qwen/Qwen3-0.6B if necessary. Allow for model memory and download
space before running; this is not a performance benchmark. Cache locations
above stay inside the ignored workspace `.cache/` directory.

The constructor is called once; both prompts reuse that model. The SDK selects
MPS, CUDA, or CPU automatically and uses its own default precision. No private
attributes are read to discover the device. Remote model code is disabled via
the public constructor's `trust_remote_code=False` argument.

The SDK declares `transformers>=4.40.0`, but Qwen3 requires at least 4.51.0
according to the [official model card](https://huggingface.co/Qwen/Qwen3-0.6B).
If using an existing environment, check this version before running. The command
above resolves dependencies afresh; a successful installation does not establish
that inference works. Dependency versions are not pinned for this standalone PoC.

## Input and observations

`token_generation.py:generate` contains the exact input serialization: one user
message, an assistant generation prefix, and the empty thinking block used by
the official template with `enable_thinking=False`. There is no system message,
tool schema, history, or prompt optimization. The reference is Qwen's public
[tokenizer configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json).
The SDK has no public chat-template helper, so this string is passed to `encode`.

For each prompt the program prints:

- `input_text` (escaped representation) and `input_ids`;
- each step's selected `token_id` and its raw `logit`;
- `generated_ids` and `generated_text` (only the generated continuation).

`encode(text).tolist()[0]` converts the SDK's `[1, N]` Tensor to `list[int]`.
Each forward pass receives that list, and its `list[float]` output is searched
with Python `max`. Equal maxima select the lowest ID. The selected ID is appended
directly to the input; generated text is never re-encoded. Empty or nonfinite
logits are reported as errors rather than choosing an arbitrary token.

Generation stops only after `max_new_tokens` (default 8). EOS is not a stop
condition. Qwen's tokenizer configuration identifies `<|im_end|>` (151645) as
EOS; raw generated IDs remain visible if it is selected. The SDK's `decode`
removes special tokens, so decoded text alone cannot reveal EOS. Output after
EOS may be meaningless. The public SDK recomputes the entire prefix on each
call; this experiment adds no cross-call KV cache.

Blank prompts and nonpositive token limits are rejected before model loading.
Import/initialization, encoding/conversion, per-step logits/selection, and decoding
failures produce contextual stderr messages with the original exception type and
reason. A failed prompt stops the command with exit status 1. Invalid CLI input
uses status 2; interruption uses 130. The generation function preserves exception
chaining for debugging. Partial step observations may remain after a failure;
they must not be interpreted as a successful completed generation.

## Mock verification

No SDK installation or model download is required:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
python3 -m examples.token_generation --help
```

The tests replace the SDK module/constructor and forward passes. They check ID
accumulation, greedy selection and ties, one encode per prompt, generated-only
decode, observations, reuse of one model, invalid inputs, missing SDK, constructor
failure, empty/nonfinite logits, encoding/decoding failure, and a second-step
forward failure with its cause preserved. These tests prove Python control flow,
not actual Tensor conversion, tokenizer compatibility, or model inference.

Static checks for the new Python files:

```sh
uv run --no-project --with flake8 flake8 examples tests
MYPYPATH=llm_sdk uv run --no-project --with mypy mypy examples tests \
  --warn-return-any --warn-unused-ignores --ignore-missing-imports \
  --disallow-untyped-defs --check-untyped-defs
```

## Verification record (2026-11-04)

Environment: macOS/Darwin arm64, Python 3.12.12, uv 0.12.9. No installed torch,
transformers, or huggingface-hub was found in this Python environment, and the
default Qwen3-0.6B cache directory was absent.

- Eight mock tests pass. `--help` succeeds without importing the SDK.
- flake8 passes for `examples` and `tests`.
- mypy reports four `no-any-return` errors in the unchanged SDK at lines 94,
  116, 125, and 134 when its torch/transformers dependencies are absent. The
  SDK-only baseline produces the same errors; none are reported in the new
  files. This is not a fully passing repository type check.
- `git diff --check` passes; no tracked SDK, dependency, or data file changed.
- `python3 -m examples.token_generation 'What is 2 + 3?' --max-new-tokens 8`
  returns status 1 with a caught SDK import error (SDK not installed).
- With `PYTHONPATH=llm_sdk`, the same command returns status 1 with a caught
  `ModuleNotFoundError: No module named 'torch'`.
- The real-model command above has **not** been run. No weights were downloaded,
  no real forward pass executed, and no real generated IDs/text, EOS behavior,
  accelerator choice, memory consumption, or latency has been verified.

Review `generate` first (input serialization, Tensor/list boundary, ID append,
argmax, finite-logit check), then `main` (single constructor and error boundary),
and finally the mock assertions. Passing a real-model run remains necessary
before claiming that public-SDK inference has been demonstrated on this machine.

Context: [Issue #9](https://github.com/AMkuro/Call_Me_Maybe/issues/9),
[design discussion #8](https://github.com/AMkuro/Call_Me_Maybe/issues/8), and
[Subject v1.7](https://github.com/AMkuro/pdf/blob/milestone3/Call_Me_Maybe/Call-Me-Maybe.en.subject.txt).
