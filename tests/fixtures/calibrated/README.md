# Calibrated function-calling test sets

These two *additional*, held-out suites are calibrated against the 11 prompts
and 5 function definitions in `data/input/` on `feat/implementation`.
They test model generalization at approximately the sample's difficulty.
They are **not** a proof that future evaluation prompts have the same difficulty.

## Calibration contract

| Property | Distributed sample | Each additional suite |
| --- | --- | --- |
| Function count | 5 | 5 |
| Prompt count | 11 | 11 |
| Prompts per function (definition order) | 2, 2, 2, 2, 3 | 2, 2, 2, 2, 3 |
| Parameter arities (definition order) | 2, 1, 1, 1, 3 | 2, 1, 1, 1, 3 |
| Parameter types (definition order) | number+number, string, string, number, string×3 | Identical |
| Actions | binary arithmetic, greeting, text transform, unary math, regex replacement | Analogous |
| Language | English short requests | English short requests |
| Regex | natural-language descriptions of digit/vowel/literal replacement | Analogous |

- `original/` retains the distributed function set, but uses 11 unseen
  phrasings and values. This isolates lexical/input generalization.
- `transfer/` replaces all five functions with new names/descriptions and
  analogous signatures. This measures generalization to unseen definitions.
- These tests intentionally exclude boolean, missing arguments, multiple
  requested calls, long inputs and adversarial instructions: their difficulty
  is not comparable to the distributed sample. Test them in a separate stress
  tier instead.
- `transfer-07` has one negative argument, a limited additional variation.
  The original sample contains only nonnegative number arguments; disclose
  this difference when comparing accuracy. Number *values* are inputs, not
  function return values: the system should not compute the answer.

## Files

For each suite:
- `functions_definition.json`: native CLI function definitions.
- `function_calling_tests.json`: native CLI prompt array, with only `prompt`.
- `measurement_fixture.json`: same functions/prompts with gold labels for the
  existing `tools.measure_application` runner.

`baseline_slot` is the 1-based position of the analogous distributed sample
case. The fixture's `regex_checks` provide additional input/output pairs
for evaluating equivalent regular expressions.

## Run (from repository root)

```sh
uv run python -m tools.measure_application \
  --cases tests/fixtures/calibrated/original/measurement_fixture.json \
  --output /tmp/calibrated-original-results.json --repeat 1

uv run python -m tools.measure_application \
  --cases tests/fixtures/calibrated/transfer/measurement_fixture.json \
  --output /tmp/calibrated-transfer-results.json --repeat 1
```

Or use the native CLI input files, for example:

```sh
uv run python -m src \
  --functions_definition tests/fixtures/calibrated/transfer/functions_definition.json \
  --input tests/fixtures/calibrated/transfer/function_calling_tests.json \
  --output /tmp/calibrated-transfer-calls.json
```

## Scoring caveat: regex equivalence

The existing `tools.measure_application.assess_call` compares the expected
`regex` string literally. A different regular expression can be equally
correct, so its `correct` field is **exact-match accuracy, not necessarily
semantic accuracy**, for the three regex cases in each suite.

For semantic grading of such a case:
1. Require the expected function name, parameter keys and types.
2. Require `source_string` and `replacement` to equal the gold values.
3. Compile the proposed `regex` and apply global replacement to every
   `regex_checks` input; all corresponding outputs must match. Treat
   invalid patterns as incorrect.
4. The finite checks only establish agreement on the specified examples,
   not equivalence on all strings. Review borderline patterns manually.

Report function-selection accuracy, non-regex exact-match accuracy,
regex semantic accuracy, full-call accuracy and missing/error counts
separately. Use a consistent model/SDK/hardware/configuration when comparing
the distributed baseline with these additional suites. Run suites separately
for the five-minute performance criterion; increasing total case count
would otherwise make the timing comparison misleading.

No real-model accuracy measurements are claimed by these fixtures.
