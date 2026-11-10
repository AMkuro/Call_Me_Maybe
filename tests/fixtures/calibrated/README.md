# Calibrated function-calling test suites

Eight additional evaluation suites, each with 5 functions and 11 English
prompts. They are based on the difficulty and input structure of the
distributed Call Me Maybe example. The goal is to evaluate **unseen input**
and **unseen function definitions**, not to maximize adversarial difficulty.

These are *structurally calibrated candidates*: identical counts, arities
and task categories do not guarantee identical semantic difficulty.
No real-model accuracy or runtime is claimed.

## Coverage matrix

| Suite | Cases per function (in definition order) | Parameter variation | Notes |
| --- | --- | --- | --- |
| `original/` | 4/1/1/2/3 | Baseline number/string | Same function definitions; new prompts |
| `transfer/` | 1/3/2/2/3 | Baseline number/string | Changed function definitions; one negative number |
| `boolean/` | 2/1/1/4/3 | number+boolean and boolean | Boolean extension |
| `difference/` | 3/1/2/2/3 | Baseline number/string | Subtraction order |
| `magnitude/` | 1/2/3/2/3 | Baseline number/string | Maximum and length selection |
| `word_pairs/` | 2/3/1/2/3 | string+string | Two-string argument extraction |
| `timer_flags/` | 2/1/2/3/3 | number+boolean and boolean | Timer repeat and audio enable |
| `feature_flags/` | 3/2/1/2/3 | number+boolean and boolean | Light enable and autosave |

All suites retain the distributed example's function count (5), case count
(11), function arities (2/1/1/1/3), and regex-case count (3). Their
per-function case distribution is intentionally **uneven**. Short,
single-action commands, small numeric values, ordinary short strings and
natural-language regex descriptions match the overall sample style.

The original distributed data contains no boolean parameters. Therefore,
`boolean/`, `timer_flags/` and `feature_flags/` form a separate
**boolean extension** tier. `word_pairs/` also changes the first function
from two numbers to two strings. These are deliberate variations and should
not be presented as perfectly identical difficulty.

No suite requires executing a function to obtain its output: the expected
`parameters` always represent the **original input arguments**.

## Files per suite

- `functions_definition.json`: native CLI function definitions.
- `function_calling_tests.json`: native CLI prompt-only input.
- `measurement_fixture.json`: combined definitions and gold labels.
- `README.md` is also provided in the five newly added suites.

Keep the gold labels separate from the model's input. Do not inject expected
answers into the model's prompt.

## Run

From repository root, run each suite independently:

```sh
for suite in original transfer boolean difference magnitude word_pairs timer_flags feature_flags; do
  uv run python -m tools.measure_application \
    --cases "tests/fixtures/calibrated/$suite/measurement_fixture.json" \
    --output "/tmp/calibrated-$suite-results.json" \
    --repeat 1
done
```

Or pass the native files to the application:

```sh
uv run python -m src \
  --functions_definition tests/fixtures/calibrated/feature_flags/functions_definition.json \
  --input tests/fixtures/calibrated/feature_flags/function_calling_tests.json \
  --output /tmp/calibrated-feature_flags-calls.json
```

## Scoring

Report separately: function-selection accuracy, argument extraction and type
accuracy, full-call accuracy, output completion/error counts, and runtime.
Because the distribution of functions differs, include **per-function**
scores, not only aggregate accuracy.

The existing `tools.measure_application.assess_call` evaluates regex
strings by **exact text**. Equivalent regex patterns can have different
spelling; each fixture therefore includes `regex_checks` containing
input/output substitution pairs to aid independent semantic evaluation.
Verify the exact source text and replacement text, then apply the proposed
regex to all test inputs. Finite examples do not prove equivalence for all
possible strings. Review borderline cases manually.

Keep stressors (very long strings, malformed input files, missing arguments,
no matching function, multiple calls or adversarial content) in separate
tests; they are not difficulty-calibrated to the distributed sample.

An 11-case suite changes accuracy by about 9.1 percentage points for a
single error. Compare all suites using the same model, SDK, hardware,
inference settings, and scoring rules, and do not conflate runtime over 88
test requests with the Subject's five-minute performance target for an
individual evaluation workload.

## Static checks

```sh
uv run python -m unittest discover -s tests -p test_calibrated_fixtures.py
```

Static tests ensure files and labels align, types are correct, regex checks
agree with the gold patterns, boolean true/false coverage exists in each
boolean-extension suite, and each documented case distribution is retained.
They do **not** measure model accuracy.
