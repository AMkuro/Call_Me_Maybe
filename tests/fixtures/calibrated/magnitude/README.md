# magnitude calibrated fixture

Maximum, greeting, text length, squaring and natural-language regex.

- Five functions, eleven short English prompts, with cases per function in definition order: 1, 2, 3, 2, 3.
- Parameter arities: 2, 1, 1, 1, 3; three regex-replacement problems with output checks. No nested arguments, missing data or long/adversarial prompts.
- Calibration tier: number/string; counting task still requires original string, not its computed length.
- The model outputs the function name and **input arguments**, not the calculated result. Regex ground truth is an example pattern; alternative equivalent patterns require semantic assessment.
- Native CLI files: `functions_definition.json` and `function_calling_tests.json`. Accuracy fixture: `measurement_fixture.json`.

Run from repository root:

```sh
uv run python -m tools.measure_application \\
  --cases tests/fixtures/calibrated/magnitude/measurement_fixture.json \\
  --output /tmp/magnitude-results.json --repeat 1
```

No Qwen3 model-based accuracy or timing measurements have been run as part of creating this fixture.
