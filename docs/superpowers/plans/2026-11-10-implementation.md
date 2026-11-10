# Approved skeleton implementation

Goal: implement the leaf functions of the skeleton approved and merged in PR #43.
Spec: docs/subject-v1.7.md, docs/architecture.md, ADR-0001 through ADR-0013.
ADR-0013 is approved; the owner additionally requested a live TUI (ADR-0014).

## Constraints and data flow

- Python >=3.10; numpy/json/Pydantic plus stdlib; only model_port imports SDK.
- Keep the existing flat modules and concrete Model; no test-only abstraction.
- CLI -> validated files -> Model/vocabulary/name trie -> name -> forced skeleton
  and constrained values -> dynamic validation -> atomic cumulative output.
- Never edit the distributed SDK. No CI or build-environment changes in this task.
- Complete one implementation batch, then required checks and one independent review.

## Review focus

- Dynamic parameter names that collide with Pydantic methods or start with underscores.
- Multi-byte UTF-8 split across tokens, JSON escapes and delimiter-crossing tokens.
- Prefix-related names, JSON-escaped names, empty parameters and changed function sets.
- Non-finite logits/numbers, empty masks, generation exhaustion and SDK exceptions.
- Malformed files and failed atomic replacement preserving the last good output.

## Steps

1. Write direct tests in tests/test_application.py for parse_args, both loaders,
   build_parameters_model, skeleton construction, trie traversal, vocabulary byte
   mapping, mask/completion and choose_token. Use fixed logits for decoder.generate.
   Test file failures using temporary paths. Observe failures before implementing.
2. Implement the existing leaf functions in src/cli.py, loader.py, schema.py,
   skeleton.py, name_trie.py, vocab.py, values.py, decoder.py, prompt.py,
   model_port.py and writer.py. Update pipeline contracts only where ADR-0013
   requires it. Keep delimiters owned by ValueSlot (ADR-0012).
3. Run real-model CLI/property checks on the supplied inputs and changed definitions.
   Record schema compliance and semantic accuracy separately from serial timing.
   Run make install, make lint, make test, then one fresh whole-branch review.
   Fix important findings with regression tests. Update architecture/development
   notes and diagram labels if interfaces change. Commit only task files.

## Progress

Current state (supersedes the historical checkpoints below): all leaves and
token-level TUI are implemented. Owner approvals are complete for ADR-0013/0014.
make lint and 91 tests pass. One independent review completed; its two Important
findings have regression fixes. The owner clarified that PR #44 approves the
foundation independently of semantic-accuracy tuning. It is ready for human
review of component boundaries and implementation contracts; the measured
accuracy is reference evidence for follow-up tuning, not an approval gate.

- Skeleton PR #43: MERGED, merge commit ffff6d7caf95b8e1a92d16782e2ce8fd7095afa2.
- Baseline: make install/lint/test passed; 53 tests, no real-model claim.
- Name terminator probe: five names round-trip and match contextual encoding.
- ADR-0013: awaiting owner confirmation; independent input/decoder tests can proceed.
- Input/schema/skeleton/decoder/writer and SDK-wrapper tests were observed failing
  on unimplemented leaves, then passing after implementation. Full suite: 67 tests
  passed; make lint passed (26 source files). CLI --help runs without model load.
- A real repository model cache exposed an existing cache-detection test's
  dependence on the checkout. Its REPO_ROOT is now isolated to the temporary
  fixture; no campus-survey production behavior changed.
- Remaining: ADR-0013 response, dependent implementations, real-model checks,
  final whole-batch independent review and commit. No implementation push/merge.
- Owner feedback: use provisional caps and test extremes repeatedly; fail with
  an explicit parameter-specific reason instead of silently shortening values.
  The 256-token cap and contextual PromptError messages are implemented.
- Limit tests: 1/255/256-token completion and 257/1024-token unfinished sequences,
  each repeated three times using fixed logits. New tests failed before the
  error-message change, then passed. Full suite: 69 tests passed; make lint passed.
  Prepared nine real-model stress cases in tests/fixtures/generation_limit_cases.json;
  actual model limit measurements still await the incomplete generator.
- SDK probe: im_end encodes to one ID (151645), is omitted by decode and can end
  each name-trie path. Prefix-related names were checked; accuracy was not.
- Official Qwen sources include Japanese training/support; the 0.6B Japanese
  MMMLU scores are 45.1 (thinking) and 37.9 (non-thinking), a different task from
  our required extraction accuracy. No Japanese accuracy guarantee or translation
  stage is proposed. Byte-level validation avoids character corruption even for
  accented Latin text and does not alter model training or its token IDs.
- Pending owner choices: EOS versus quoted-name termination, and byte handling
  with English-focused evaluation. Warning diagnostic fields are specified in
  ADR-0013 but not yet implemented; thresholds remain measurement-dependent.
- Owner approved EOS provisionally, byte-preserving grammar, provisional caps and
  diagnostic warnings. Add live TUI showing what is selected (ADR-0014).
- Execution: finish core and terminal display in this same branch, with direct
  tests first. One final independent whole-batch review follows real-model checks.
- Final review: fresh independent reviewer, no Critical, two Important findings.
  Diagnostic decode failures now show unavailable text without changing choices;
  actual value decode errors still propagate. The numeric measurement oracle
  now uses Decimal to distinguish exact agreement from float normalization.
  Both regressions were observed failing and then passing. Malformed logits
  dimensions/conversion also have a failing-then-passing regression.
- Review rulings: retain approved float normalization, but record precision loss;
  additional types/scientific generation remain outside approved scope; the
  256-token bounded-error policy stands; universal accuracy/runtime are not
  certified; pseudo-terminal tests do not establish every terminal emulator.
- Real-model accuracy: sample10/11 after tuning, changed definitions3/5. Every
  completed case matched its output schema. Precise cases and limitations are
  recorded in docs/implementation-verification.md and its JSON data.
- Stress: 32/128/512-character cases repeated in three parallel workers; 512 hit
  the 256-token cap with explicit errors each time. The costly 2048-character
  case was intentionally interrupted, not counted as completed or PromptError.
  The 17-digit decimal repeated three times and exposed precision loss. Other
  original large cases were not executed; no full-nine-case claim.
- Serial timing: no concurrent own model jobs, CPU float32, four threads, cached
  model. All11 samples completed from CLI startup to exit in98.94 seconds.
- User requested EOS name and queried external dictionaries versus tolist.
  Constant renamed EOS; startup vocabulary mapping already comes from the public
  vocabulary file. Tensor-to-list conversion is still needed; redundant Python
  int conversion/list copying was removed. No unmeasured cache layer added.
- Owner scope correction: PR #44 is the foundation implementation review.
  Do not block its approval on a 90% accuracy score. Preserve the measurements
  for later parameter/component tuning. Infrastructure defects revealed by any
  measurement remain review findings. Human implementation approval is pending.
- Owner input-contract correction: an empty prompt array is insufficient input
  and must fail before model startup, preserving existing output. Recorded as
  the 2026-11-10 update to ADR-0012. Two regression tests failed on the old
  implementation, then passed. Full make install/lint/test passed (91 tests).
  Real CLI checks covered both existing and absent output files with offline,
  absent model caches. Nonempty all-failed inputs still save `[]` and exit1.
