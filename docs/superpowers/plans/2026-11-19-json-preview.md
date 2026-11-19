# JSON preview implementation plan

**Goal:** Show the committed JSON prefix, current work and durable outcomes.
**Architecture:** Synchronous output-token observations append vocabulary bytes in
Dashboard. Rendering remains throttled; partial JSON is formatted lexically,
without parsing, synthesizing values, or adding closing punctuation.
**Tech stack:** Existing Python/ANSI display, no dependencies.
**Spec:** Owner-approved proposal in the conversation, 2026-11-19.

## Constraints and review focus

Work on local develop; preserve unrelated edits; no remote publication.
Keep model contexts, token choices, forwards, prompts and validation unchanged.
Distinguish actual output tokens from validated float-normalized saved text.
Check split UTF-8, controls, long content, small/resized terminals, stream failure,
exceptions and Ctrl+C. Redirected logs retain their existing contract.

## Tasks

- [x] Add failing preview/event tests, including fixed JSON and no-argument calls.
- [x] Add `output_token(data: bytes, *, token: int, mode: SelectionMode)` to
  GenerationProgress; notify after fixed/value commitment without extra forward.
- [x] Accumulate bytes and token count in Dashboard; format only committed text,
  wrap to terminal cells, retain cursor in a bounded viewport, throttle drawing.
- [x] Pass validated saved text and output location through existing lifecycle;
  preserve summaries and bounded final/failed previews after alternate screen.
- [x] Verify model inputs/results with observations on/off, lint, full tests,
  short rendering benchmark and actual pseudo-terminal example.
- [x] Update architecture with adopted rationale, evidence and limits; review diff
  and commit only authorized files locally.

Native execution: the approved implementation is carried out in this session.

Verification: make install succeeded; make lint passed (57 files); make test
passed (278 tests). Independent review fixes have regression tests. GPU TTY
and redirected saved JSON are byte-identical on two inputs. Raw functional
records and separate display timings are in docs/measurements/2026-11-19-json-preview/.
