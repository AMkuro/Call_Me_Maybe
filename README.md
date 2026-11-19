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

`uv sync` alone also works. The dependencies are numpy, Pydantic,
orderedsets (an insertion-ordered set used by the name tree) and the local
`llm_sdk` (a path dependency); torch is installed as a dependency of the SDK. The SDK downloads the Qwen/Qwen3-0.6B model itself on first use.

When stderr is a terminal, a live dashboard shows the selected function,
current argument, token choices and diagnostic scores. For each token the
model chooses, it also lists the ten highest raw logits, marking the tokens
the constraints mask out and the selected token. Forced tokens are shown
as forced, without logits, because the model is not called for them.
Redirecting stderr produces warning lines instead of terminal controls.
Either way, the run ends with one line per prompt, done or failed with
its time in seconds, and a total.

The program reads the function definitions and the prompts from
`data/input/` and writes `data/output/function_calls.json`.

The prompt array must contain at least one entry. An empty array is reported
as insufficient input before loading the model and leaves existing output
unchanged. Function definitions must be a nonempty array with unique,
nonblank function names, all required fields, no extra fields and supported
parameter/return types (`number`, `string`, `boolean`). A blank description
or parameter name produces a warning on stderr and is preserved. Functions
with no parameters are supported.

All JSON readers reject duplicate keys within an object, including nested
objects, and nonstandard `NaN`/`Infinity` values. Pydantic validates shapes
and types strictly: booleans and numbers are distinct, and numeric strings
such as `"2"` or `"２"` are not automatically converted to numbers. This
validation of structured JSON is separate from interpreting numbers in the
user's natural-language request.

Specify paths with:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

## Algorithm explanation

Each prompt is processed in two stages, and every output token is chosen
the same way: the model runs one forward pass on the whole context (the
prompt plus all output tokens so far), the logits of every token that would
break the structure or the schema are set to negative infinity, and the
highest-scoring remaining token is selected. When exactly one token is
allowed, it is taken without using the scores (the forward pass still runs).

1. Function name. The tokens of each function name followed by the model's
   end-of-turn token `<|im_end|>` form a token trie built from
   `functions_definition.json` at startup. Only tokens that continue a path
   of the trie are allowed, so the model can only write a defined name, and
   the end-of-turn token tells `fn_a` apart from a longer name `fn_ab`.
   The choice among names is the model's: no keyword rule is involved.
2. JSON record. The selected definition gives the layout
   `{"prompt":...,"name":...,"parameters":{"<key>": <value>,...}}`.
   Braces, keys, the copied prompt and the chosen name are fixed tokens;
   only the parameter values and the delimiter after each value (`,` or `}`)
   are generated. The prompt and the name are written with `json.dumps` and
   `<` escaped as `\u003c`, so their tokens never form a special token.

Value masks are computed from the vocabulary file: each token ID is mapped
back to its raw bytes, and a token is allowed only if the bytes written so
far plus that token's bytes remain a prefix of a valid value of the declared
type followed by its delimiter, with nothing after the delimiter.

| Type | Allowed value | Complete when |
| --- | --- | --- |
| `number` | `-?(0\|[1-9][0-9]*)(\.[0-9]+)?` (no exponent, `+` or digit grouping) | a whole number and the delimiter |
| `string` | a JSON string: escapes, UTF-8, no raw control characters | the closing quote and the delimiter |
| `boolean` | `true` or `false` | the word and the delimiter |

String masks work on bytes, so a UTF-8 character split across tokens is
kept. The completed record is parsed as JSON and checked against a Pydantic
model built from the definition, before it is saved.

## Design decisions

The function is chosen first, with its own prompt listing every function's
name, description and argument types; the arguments are generated next,
with a prompt that shows only the chosen definition. Both prompts use
Qwen3's chat format with an empty thinking block, and the same instructions
are used for every input and function set.

Every output token, fixed JSON included, is preceded by a forward pass on
the full context. Skipping the forward pass where only one token is allowed
would be faster; the cost of this choice is shown in Performance analysis.

The input prompt is copied, and function names and parameter keys are escaped
with `json.dumps`. Parameters follow definition order. The SDK is wrapped by
one concrete class; application code only receives plain token-ID lists.

Numbers are saved as floats, as in the Subject's example (`2.0`). The
grammar lets the model end a number right after its integer digits,
because forbidding that made it keep writing digits (`1000…`) until the
limit. The saved record then appends `.0` to the generated text (`10`
becomes `10.0`): a string edit, so no digit is lost to a float conversion,
and the model's context keeps the tokens it actually chose. Before saving,
every number argument must load as a finite float with Python's `json`
module; otherwise the prompt fails.

When the program validates JSON (the input files and the generated record),
JSON integers stay Python integers, including values beyond `2**53`.
Decimal and exponent literals are first parsed exactly with `Decimal`.
A float is accepted only if its final JSON decimal value equals the source;
`0.1` is supported. Integral decimal values may instead be stored as exact
integers when the float representation would round them. Values that would
lose precision, underflow to zero or overflow the finite float range are
rejected with a reason, rather than rounded or converted into JSON strings.

Very large integer literals are subject to Python's runtime integer/string
conversion limit. Decimal/exponent literals must fit the finite float range;
a large integer literal can therefore be accepted where its exponent form
is rejected. Long fractional decimals that cannot survive standard JSON
float output are unsupported. Generation currently permits signed integers
and decimal fractions, without exponent notation or grouping commas.
These are implementation limits, not additional Subject requirements.

A number value, including its delimiter, is limited to 32 tokens and any
other value to 256 tokens. Failure to finish reports the parameter and
limit, skips that prompt and returns a nonzero exit status; the output then
has no record for that prompt. A generated precision/type error also skips
that prompt; invalid input files stop startup before loading the model.
Successful results are saved after each prompt by atomic replacement of the
output file.

If the final component of the output path is a symbolic link, saving
replaces the link itself with the new regular output file. This is
intentional and permitted; the link target's contents remain unchanged.

Durability across power loss or an OS crash is not implemented, and there
are no plans to add it. Atomic replacement alone does not guarantee that
saved results survive either event.

Time-of-check to time-of-use (TOCTOU) races are not fully eliminated.
Filesystem paths or permissions can change between startup checks and a
later save. A successful preflight does not guarantee that saving will
succeed; save-time I/O errors are still reported.

## Performance analysis

Measured on a development machine (AMD Ryzen AI MAX+ 395), not on a campus
computer. Accuracy and time come from separate runs.

| Measure | Set | Result |
| --- | --- | --- |
| Valid, schema-compliant records | 99 prompts (9 sets) | 99/99 |
| Correct function | 99 prompts | 99/99 |
| Correct call (function and every argument) | Subject's 11 examples | 10/11 (90.9%) |
| Correct call | 8 further sets, 88 prompts | 71/88 (80.7%) |
| Correct call | all 99 prompts | 81/99 (81.8%) |
| Total time, `python -m src` on the 11 examples, CPU only | 1 run | 221.8 s |

The wrong calls are argument errors, not function errors: 17 of the 18 are
regular expressions for "every number" or "vowels" (for example `aeiou`
instead of `[aeiouAEIOU]`; one of them also writes `asterisks` instead of
`*`), and one drops the sign of `-14`. Accuracy therefore stays below 90% on the
harder sets and depends on how many prompts need such patterns.

The time run used CPU cores 0–15 with 6 threads, at most 16 GiB of memory, no
swap and the model already cached; the 11 prompts took 11.0–38.9 s each. The
SDK recomputes the whole prefix for every output token and has no cache, so
each token costs one forward pass whose cost grows with the prompt and the
output so far. Copying the prompt into the record therefore costs one
forward pass per token, and a longer prompt or more prompts cost
proportionally more. Whether all test prompts finish within five minutes on
the evaluation computer has not been measured. Accuracy runs used the GPU
build of the same model in float32; their answers on the 11 examples match
the CPU run.

String masks parse the shared prefix once, then inspect only each
candidate's new bytes. The dashboard reuses the logits already computed and
performs no additional inference.

## Challenges faced

- The supplied SDK depends on torch and transformers internally, while the
  project's own code must not import them. The application therefore only
  uses the public methods of the SDK (`encode`, `decode`,
  `get_logits_from_input_ids`, `get_path_to_vocab_file`), wrapped in one
  class that returns plain lists.
- Tokens do not line up with JSON values: one token can hold a digit and a
  comma, or half of a UTF-8 character. The masks therefore test the bytes of
  the whole candidate, not single characters.
- Text such as `<|im_end|>` inside a prompt is encoded as one special token,
  which `decode` drops. Writing `<` as `\u003c` keeps the same JSON value in
  ordinary tokens. The prompts given to the model escape the request and
  the definitions the same way, so such text cannot end a chat turn early.
- Numbers did not end. With a mandatory fraction, the model wanted to stop
  after `10` and, unable to, wrote `1000…` until the limit. Allowing the
  delimiter after an integer and appending `.0` when saving fixed it. Without
  the space in `": "` before a value, the model also kept writing `2.000…`.
- Speed. The SDK recomputes the whole prefix for every token, so the
  parameter instructions were cut from nine sentences to four and the fixed
  JSON was compacted, measuring accuracy after each change.
- Regular expressions. The 0.6B model copies literal values well but often
  fails to turn "every number" or "vowels" into `\d+` or `[aeiouAEIOU]`.
  An instruction sentence about patterns fixed the Subject's number example;
  the other cases remain the main source of wrong arguments (see
  Performance analysis). Scoring the wrong and the expected values showed
  that the model itself prefers the wrong one in all 18 cases, so a better
  search under the same model (beam search, or decoding faithful to the
  model's distribution) would not fix them. Two other attempts were too
  costly for their gain and were not kept:
  - Two worked examples before the request (a character class, and a negative
    number copied as written) raised the 99 prompts from 81 to 89 correct
    when patterns are judged by their effect, but not the Subject's 11
    examples (10/11). They nearly doubled the CPU time of the 11 examples
    (221.8 s to 417.0 s, one run), because every forward pass reads the
    longer prompt.
  - Letting the model think before the JSON (Qwen3's think block, capped at
    160 tokens) lowered the 11 examples to 9/11, as the thought was cut off,
    and adds a forward pass for every thought token.

## Testing strategy

Unit tests check tries, masks, forced structure, validation and file handling
directly. Fixed logits exercise the decode loop, and boundary tests mock only
the external SDK. They do not download or load the model. `make lint` runs flake8 and mypy
with the flags required by the Subject.

Real-model checks run the application on sets of prompts with expected
answers written by hand: the Subject's examples and eight further sets of
eleven prompts and five functions. One set reuses the example functions
with new prompts; the others change the functions, and three of them add
boolean parameters. A call is counted correct
only if the function name and every argument equal the expected value
(numbers compared by decimal value, strings exactly). Valid JSON alone is
not counted as correct. Accuracy and timing are measured in separate runs.
Malformed and missing input files were run through the command line to
check that the program stops with one message, exit status 1 and no output
file.

## Example usage

```sh
uv sync
uv run python -m src
uv run python -m src --input my_prompts.json --output /tmp/calls.json
```

An excerpt of `data/output/function_calls.json` from the 11 examples (the
spaces inside `parameters` were written by the model and are valid JSON):

```json
[
  {"prompt":"What is the sum of 2 and 3?","name":"fn_add_numbers","parameters":{"a": 2.0,"b": 3.0}},
  {"prompt":"Greet shrek","name":"fn_greet","parameters":{"name":  "shrek"  }},
  {"prompt":"Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS","name":"fn_substitute_string_with_regex","parameters":{"source_string":  "Hello 34 I'm 233 years old","regex":  "\\d+","replacement":  "NUMBERS"}}
]
```

Each record has exactly `prompt`, `name` and `parameters`. When stderr is
redirected, the run ends with one line per prompt and a total, for example
`[1/11] done    fn_add_numbers  16.4 s` and
`11 prompts: 11 done, 0 failed, 216.4 s`. Generated files in `data/output/`
are not tracked.

## Resources

- [RFC 8259: The JavaScript Object Notation (JSON) Data Interchange Format](https://www.rfc-editor.org/rfc/rfc8259)
- [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- Willard and Louf, [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) (Outlines; masking logits with a grammar)
- Beurer-Kellner et al., [Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation](https://arxiv.org/abs/2403.06988) (DOMINO; token misalignment)
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) (jump-forward decoding of fixed text)
- [Python unittest](https://docs.python.org/3/library/unittest.html)
- [Pydantic](https://docs.pydantic.dev/latest/)
- [uv](https://docs.astral.sh/uv/)

AI assistance (Claude Code and Codex) was used for design discussion,
implementation, tests, build configuration and documentation. The human owner
makes architectural decisions and reviews the changes.
