# difference calibrated fixture

Basic subtraction, short strings and natural-language regex replacement.

- Five functions, eleven short English prompts, with cases per function in definition order: 3, 1, 2, 2, 3.
- Parameter arities: 2, 1, 1, 1, 3; three regex-replacement problems with output checks. No nested arguments, missing data or long/adversarial prompts.
- Calibration tier: number/string; subtraction direction slightly increases argument-order sensitivity.
- The model outputs the function name and **input arguments**, not the calculated result. Regex ground truth is an example pattern; alternative equivalent patterns require semantic assessment.
- Native CLI files: `functions_definition.json` and `function_calling_tests.json`. Accuracy fixture: `measurement_fixture.json`.

Run from repository root:

```sh
uv run python -m tools.measure_application \\
  --cases tests/fixtures/calibrated/difference/measurement_fixture.json \\
  --output /tmp/difference-results.json --repeat 1
```

No Qwen3 model-based accuracy or timing measurements have been run as part of creating this fixture.
