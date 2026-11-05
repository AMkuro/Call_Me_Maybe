# A05 common CLI and incremental output

Implemented scope: shared CLI, input validation, final result validation,
sequential fail-fast execution, atomic incremental saving, and diagnostics.
The LLM selection, argument generation, and constrained decoding algorithms
are not implemented. This is not a completed function-calling application.

Sources: [Subject v1.7](https://github.com/AMkuro/pdf/blob/milestone3/Call_Me_Maybe/Call-Me-Maybe.en.subject.txt),
[contract #24](https://github.com/AMkuro/Call_Me_Maybe/issues/24),
[implementation #25](https://github.com/AMkuro/Call_Me_Maybe/issues/25),
[dynamic schema #4](https://github.com/AMkuro/Call_Me_Maybe/issues/4), and
[evaluation #27](https://github.com/AMkuro/Call_Me_Maybe/issues/27).
The confirmed #24 contract takes priority over unresolved design details.

## Execution

```sh
uv sync --locked
uv run python -m src --help
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

All three paths above are defaults, relative to the current working directory.
`-output` is also accepted because it appears in Subject IV.3.2's synopsis.
The default generator raises `NotImplementedError` on the first valid prompt;
the CLI prints the failing stage to stderr and returns 1. No SDK is imported,
no model is downloaded, and no placeholder function call is produced.

Exit status is 0 on batch success, 1 on operational failure, 2 for argparse
usage errors, and 130 for a caught keyboard interruption. An empty prompt
array succeeds with `[]` without invoking generation. The output file alone
does not indicate whether the entire batch succeeded: inspect the exit status.

## Integrating a generation method

The single integration seam is `src.cli.GenerateCall`:

```python
from src.cli import main
from src.validation import FunctionDefinition

# Signature required of a real adapter:
# generate(prompt: str, definitions: list[FunctionDefinition]) -> str
# main(generate=generate)
```

An adapter receives the original prompt and deep copies of the validated
function definitions. Models provide `.name`, `.description`, `.parameters`,
`.returns`, and `.model_dump()` for serialization. All listed parameters are
required. The returned string must be the complete JSON object with exactly
`prompt`, `name`, and `parameters`; no Markdown, explanation, or metrics.
Generation methods can assemble their internal stages differently.

The adapter must raise an exception on SDK loading/inference errors, early
EOS, impossible constraints, or exhausted token limits. A normally returned
string is the adapter's declaration that generation completed successfully.
The common validator cannot discover a concealed termination failure or prove
that token-time constrained decoding was applied. Those guarantees belong to
A02/A04 and remain necessary before Subject compliance can be claimed.

A closure or callable may own one lazily initialized model and reuse it across
prompts. No base Decoder, model factory hierarchy, CLI method registry, or
mandatory metadata return object is introduced. When a real adapter is ready,
connect it at `generate_call` for `python -m src`, or inject it into `main`
for comparison runs. Internal metrics and termination reasons belong to the
A06 harness, outside the Subject output file.

## Validation and saving decisions

- Input format is the Subject array of prompt objects and array of function
  definitions. Definitions require name, description, parameters, and returns;
  names must be nonempty and unique. Unknown definition fields are rejected.
- Pydantic v2 validates strict types and forbids extra fields. Each function's
  parameter model is built with `create_model`; aliases support arbitrary
  argument keys, including names that collide with Pydantic methods.
- The flat type vocabulary is number, integer, string, boolean, null, array,
  and object. A number accepts JSON integers and finite floats, preserving
  integer precision. Booleans and numeric strings do not count as numbers.
  Integer rejects floating-point values such as `7.0`. Nested schema keywords
  such as items/properties/enum are rejected; structural nested schemas are
  outside this implementation. Arrays/objects accept arbitrary JSON contents.
- JSON parsing rejects duplicate keys, NaN/Infinity, overflowing floats,
  trailing prose, comments, and multiple values. Empty strings are preserved;
  ambiguous meaning and incorrect but well-typed values belong to A06.
- Output/input aliases, including symlinks and hardlinks, are rejected before
  writing. Output is then initialized to `[]` before reading the input files.
  Both files must parse before any generation. Each prompt record is validated
  just before processing, so a later malformed record preserves prior results.
- Every validated result triggers a complete-array write to a temporary sibling,
  flush, file `fsync`, close, and `os.replace`. Only replacement publishes it.
  Write/flush/replace errors preserve the last published array and stop the
  loop. Temporary-file removal is attempted in `finally`, including on Ctrl-C.

Pydantic's standard dynamic models avoid a separate schema framework.
Atomic replacement was chosen over JSON Lines or in-place appending because
the published file must always be a complete JSON array. The full-array writes
cost quadratic total serialization work over a batch; optimize only after
measurement. There is no resume or concurrent-writer coordination. Directory
`fsync`, hard process termination, power loss, and network-filesystem semantics
are not guaranteed or tested. If initialization itself fails, an existing
output may remain; no generation starts, and the nonzero exit is authoritative.

## Contract #24 coverage

| Confirmed contract | Implementation / evidence |
| --- | --- |
| Subject paths, options, module invocation | `src/__main__.py`, `main`; CLI/subprocess tests |
| Exact three keys and original prompt | `FunctionCall`, `validate_result`; validation tests |
| Dynamic candidate membership and exact arguments/types | `load_definitions`, `validate_result`; strict-type tests |
| No semantic correctness gate | Well-typed incorrect reading is accepted in tests |
| Fail-fast and explicit nonzero errors | `main`; generation, input, validation, save, interruption tests |
| Save after each success; keep valid JSON and successful prefix | `save_results`; real file snapshots, partial-write/replace failure tests |
| No prior-run contamination | Initial empty save; repeat-run and stale-output tests |
| No fabricated failures or evaluation fields | Default generator errors; exact output schema |
| Successful generation and token-time constraints | Adapter obligation; actual LLM method remains unimplemented |
| Evaluation metadata separate | No metadata added; A06 instrumentation remains future work |

## Verification record (2026-11-05)

Environment: Linux x86_64, Python 3.12.3, uv 0.11.21. Runtime and development
dependencies are recorded in the root `uv.lock`; the SDK and its lockfile are
unchanged. The existing eight SDK-PoC tests pass without importing the SDK.

Executed:

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run flake8 src tests examples
uv run mypy src tests/test_cli.py tests/test_validation.py \
  --warn-return-any --warn-unused-ignores --ignore-missing-imports \
  --disallow-untyped-defs --check-untyped-defs
uv run python -m src --help
uv run python -m src
git diff --check
```

- Sync, 32 tests (24 new, 8 existing), scoped lint/type checks, help, and diff
  whitespace checks pass. The real default CLI intentionally returns 1 with
  `input 1 generation: NotImplementedError`, leaving valid `[]`.
- The same scoped mypy command also passes with `--python-version 3.10`.
  This is static compatibility checking, not a Python 3.10 runtime test.
- `uv tree --no-dev` shows only NumPy, Pydantic, and Pydantic's dependencies;
  importing the CLI loads no torch, transformers, or SDK modules. The tracked
  SDK, input data, examples, and original tests match baseline `e3dd4ec`.
- `uv run flake8 .` fails with 11 existing findings confined to
  `llm_sdk/llm_sdk/__init__.py` (E501 and F401). `.flake8` excludes only local
  dependency/cache directories; SDK diagnostics are not suppressed.
- `uv run mypy . --warn-return-any --warn-unused-ignores
  --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` fails
  with four existing SDK `no-any-return` errors at lines 94, 116, 125, and 134.
  The application and new tests have no reported errors. The SDK is not edited.
- Disk errors and Ctrl-C are injected deterministically. No physical disk-full,
  real-model inference, weights download, semantic accuracy, generation-time
  constraints, performance target, or alternate-Python/platform test was run.

## Review order

1. `src/cli.py:main`: initialization order, stage diagnostics, sequential loop,
   callable completion contract, and validation before saving.
2. `src/storage.py:save_results`: publication boundary and failure cleanup.
3. `src/validation.py:validate_result`: exact output and strict dynamic types.
4. `tests/test_cli.py`: incremental snapshots and injected failure paths;
   `tests/test_validation.py`: valid/invalid dynamic data.

Author self-review and one independent read-only review of code range
`e3dd4ec..fc3706b` found no Critical, Important, or Minor defects in this scope.
Review checked the confirmed contract, failure preservation, strict dynamic
types, alias protection, and the explicit incomplete generation route.
The generator and submission-wide obligations listed below remain separate
from that review verdict. No review suggestions are deferred.

This scope completes the shared execution infrastructure. The real adapter,
token-time constraints, SDK/model reuse verification, A06 measurements, and
the remaining submission-wide README/Makefile requirements still need work.
