# feature_flags calibrated fixture

This set contains five functions and eleven short English requests, following
the distributed Call Me Maybe task's approximate scale.

- Cases per function (definition order): 3, 2, 1, 2, 3.
- Number of parameters per function: 2, 1, 1, 1, 3.
- Parameter kinds: number, string and boolean. Both boolean values are tested;
  the first function uses one number and one boolean.
- Three regex-replacement cases include semantic input/output checks.
- Numerical arguments are copied, not calculated; the return types do not
  determine the argument values.
- This is a boolean-extension tier, not a directly equal-type comparison with
  the distributed sample (which did not provide boolean arguments).

Use the native `functions_definition.json` and
`function_calling_tests.json` with the CLI, or the combined
`measurement_fixture.json` with the measurement runner.

```sh
uv run python -m tools.measure_application \
  --cases tests/fixtures/calibrated/feature_flags/measurement_fixture.json \
  --output /tmp/feature_flags-results.json --repeat 1
```

The expected `regex` text is an example; use the `regex_checks` input/output
pairs for an additional semantic check of alternative regex patterns. Passing
finite tests is not a proof of complete regex equivalence.

This fixture has not been evaluated using the real Qwen3 model.
