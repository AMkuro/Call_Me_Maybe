# Local prompt improvement with an OSS harness

Use installed **OpenCode 1.15.10** for the agent loop and **Transformers Serve**
for the local GPU model. No custom token-generation or agent conversation loop.
The submission target remains Qwen3-0.6B through its unchanged SDK.

`opencode.json`, `improver.txt` and `tools/` are copied into isolated `/tmp`
workspaces. Each workspace can save one prompt candidate and run its fixed
11-case evaluation. All built-in OpenCode tools are disabled; only
`write_prompt` and `run` are allowed. Neither accepts a command, path, dataset
or expected answer. Native session state and raw tool logs remain available.

The trusted experiment workflow reuses existing validation/scoring and CPU
measurement code. It prepares two workspaces, runs OpenCode concurrently,
selects by fixed development gold, and repeats within the remaining budget.
Later proposals receive previous prompts and actual development scores, including
rejected candidates. Model-written score summaries never control promotion.
Once a candidate improves without regression, it stops both model services,
runs the native CLI at CPU0-15/6threads/16G/no swap, records raw and raw+8s,
then evaluates the frozen baseline/candidate on the eight calibrated suites.
The additional 88 cases are not accessible to the editing tools.

## Development environment

Use the existing `.venv-gpu`; serving dependencies stay outside submission
requirements and the CPU environment:

```sh
make prompt-server-install GPU_ENV=/absolute/path/to/.venv-gpu
make prompt-improve GPU_ENV=/absolute/path/to/.venv-gpu \
  FLAGS='--output .cache/prompt-experiments/pilot-1 --cache /existing/hf-cache --optimizer Qwen/Qwen3.5-9B --rounds 4 --workers 2'
```

The launcher uses transient user systemd units, loopback ports8011/8012, the
existing cached revision and BF16. It stops its own units on exit and before
CPU timing. OpenCode and its tool SDK are the OSS harness; the Python code
only supplies the two operations and the task-specific evaluation gates.
No push, PR or application prompt adoption occurs.

For this run, two earlier SDK-only attempts were interrupted before scoring.
Their artifacts and the two consumed slots are retained; six slots remain.
The backend, package versions, actual model snapshots, code and data hashes
are recorded with each evaluation. The eight-case-set audit is under
`docs/research/calibrated-fixture-review.md`.
