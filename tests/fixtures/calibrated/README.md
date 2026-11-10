# Calibrated function-calling test suites

Three supplementary test suites for the 42 Call Me Maybe project, based on
the distributed examples in `data/input/` (11 prompts, 5 functions).
The purpose is to test unseen requests and changed function definitions
without silently raising every difficulty dimension at once.

These are **candidate difficulty-matched suites**, not empirical proof of
equal model difficulty. All expected calls are supplied independently of
model predictions.

## Structural comparison

| Property | Distributed | `original/` | `transfer/` | `boolean/` |
| --- | --- | --- | --- | --- |
| Function definitions | 5 | 5 | 5 | 5 |
| Prompts | 11 | 11 | 11 | 11 |
| Cases per function (definition order) | 2/2/2/2/3 | 4/1/1/2/3 | 1/3/2/2/3 | 2/1/1/4/3 |
| Argument counts (definition order) | 2/1/1/1/3 | Same | Same | Same |
| Supported parameter types | number, string | Same | Same | number, string, boolean |
| Natural language | Short English requests | Short English requests | Short English requests | Short English requests |
| Regex tasks | 3 | 3 | 3 | 3 |

The cases per function are deliberately **uneven** within every suite.
The total (11) and the share of regex cases (3/11) remain constant to
reduce difficulty drift. The function-frequency difference is intentional:
do not assume the resulting *aggregate* accuracy is strictly comparable
without reporting accuracy by function and task type.

- `original/`: exact distributed function definitions with unseen
  requests/values. Tests generalization to natural-language paraphrases.
- `transfer/`: different functions with matching parameter arities/types,
  analogous actions and new request wording. Tests unseen function definitions.
- `boolean/`: analogous basic arithmetic-like, greeting, text and regex
  requests, but replaces part of the parameter schema with `boolean`.
  The `fn_set_alarm` cases combine one number and one boolean; the
  `fn_set_notifications` cases extract a boolean from short positive and
  negative commands. Both `true` and `false` are represented.

## Difficulty boundaries

The first two suites do not add booleans. The third is an **explicit
type-extension comparison**, not an equal-type baseline. This added feature
may increase or decrease model difficulty; its accuracy should be reported
separately and not used to claim a calibrated baseline score. The original
distributed sample has no boolean parameters.

Keep out-of-scope stressors (very long strings, missing required arguments,
unknown actions, multiple calls per prompt, adversarial instructions,
non-finite numbers and deeply nested JSON) in independent stress suites.
`transfer-07` includes a negative number that was absent from the
distributed sample; disclose it when discussing comparability.

The prompts provide the argument values; do **not** compute the numerical
result of a function. For the alarm, `hour` is explicitly specified as an
integer and `repeat` is explicitly indicated (no time-zone interpretation).

## Files and score semantics

Each suite contains three files:

- `functions_definition.json`: native CLI function definitions.
- `function_calling_tests.json`: native CLI input (array with `prompt`).
- `measurement_fixture.json`: both definitions and prompts plus gold
  `expected_name` and `expected_parameters`.

The `regex_checks` array contains sample inputs/outputs for grading
functionally equivalent regular expressions. The existing runner
`tools.measure_application.assess_call` compares regex **text** directly.
A differently written but functionally correct pattern may be marked
incorrect by that runner. For semantic assessment, compare the function,
keys, typed arguments, the exact `source_string` and `replacement`, and
apply the candidate regex using global substitution to all the
`regex_checks` inputs. Passing finitely many checks does not establish
regex equivalence for every possible string.

## Run from repository root

```sh
for suite in original transfer boolean; do
  uv run python -m tools.measure_application \
    --cases "tests/fixtures/calibrated/$suite/measurement_fixture.json" \
    --output "/tmp/calibrated-$suite-results.json" \
    --repeat 1
done
```

For native CLI input, for example:

```sh
uv run python -m src \
  --functions_definition tests/fixtures/calibrated/boolean/functions_definition.json \
  --input tests/fixtures/calibrated/boolean/function_calling_tests.json \
  --output /tmp/calibrated-boolean-calls.json
```

Measure and report selection accuracy, per-function full-call accuracy,
argument type/value accuracy, regex exact and semantic accuracy, completion
rate and errors separately. Note the small sample sizes: one error changes
an 11-case suite accuracy by about 9.1 percentage points. Compare timings
per suite rather than judging all 33 prompts against the distributed
11-prompt runtime. Hold model, SDK, machine and inference settings fixed.

Static fixture checks can be run without the model via:

```sh
uv run python -m unittest discover -s tests -p test_calibrated_fixtures.py
```

No real-model correctness or performance measurements are claimed here.
