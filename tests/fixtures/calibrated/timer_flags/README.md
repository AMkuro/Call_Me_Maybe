# timer_flags calibrated fixture

This suite has five functions and eleven short English prompts. It changes a single core input dimension against the distributed 42 Call Me Maybe examples: boolean-extension with number/boolean mixed arguments.

- Function-frequency distribution (definition order): 2, 1, 2, 3, 3. Unlike an even split, this deliberately tests an uneven workload.
- Arity distribution: 2, 1, 1, 1, 3. Three natural-language regex-substitution cases have input/output semantic checks.
- Parameter values, not function execution results, are the evaluation target. The regex patterns in the fixture are example correct representations, not necessarily the only correct patterns.
- This fixture avoids long text, missing arguments, multiple function calls and adversarial inputs, keeping the non-type difficulty close to the distributed sample.
- Boolean-extension tier: number + boolean arguments in the first function, and a single boolean argument in the fourth. Both true and false are tested. The distributed sample does not include boolean, so compare this tier separately.

The `measurement_fixture.json` works with the existing model runner; `functions_definition.json` and `function_calling_tests.json` work with the native CLI.

```sh
uv run python -m tools.measure_application \\
  --cases tests/fixtures/calibrated/timer_flags/measurement_fixture.json \\
  --output /tmp/timer_flags-results.json --repeat 1
```

No real-model execution or accuracy claim is included with this test data.
