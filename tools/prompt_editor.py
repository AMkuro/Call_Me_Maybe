"""Local model with exactly prompt-save and fixed-development-run tools."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from src.json_io import loads
from src.schema import validate_number
from tools.prompt_candidates import dispatch, write_json
from tools.prompt_runtime import evidence_fingerprint


SYSTEM = (
    "You improve the English instructions of a SMALL function-calling model. "
    "You can only call write_prompt and run. Output ONE JSON tool "
    "call, with no markdown or explanation. write_prompt schema: "
    '{"tool":"write_prompt","prompt":{"name_instructions":"...",'
    '"parameter_instructions":"..."}}. run schema: '
    '{"tool":"run","candidate_id":"the ID returned by write_prompt"}. '
    "Only rewrite instructions, never the dataset, code or gold answers. "
    "Keep both instructions concise and generally applicable to unseen "
    "functions/requests. The target selects an exact function "
    "name first, then fills a JSON skeleton with typed parameter values. It "
    "cannot emit explanations or reasoning. It must extract input values, "
    "NOT compute the function result. Correct literal strings, signs and "
    "whitespace matter. You may teach a general rule with a short example, "
    "but do not copy these test requests/answers into the instructions. "
    "Keep correct existing behavior. Available tool: write_prompt first, run "
    "the returned candidate next. No other tools or file/command arguments."
)


def generate_action(
    model: Any, conversation: str, prefix: str,
) -> tuple[str, int]:
    """Complete one bounded JSON action using only the SDK's public methods."""
    tokens = model.encode(conversation + prefix).tolist()[0]
    eos = model.encode("<|im_end|>").tolist()[0]
    if len(eos) != 1:
        raise ValueError("optimizer tokenizer lacks the expected Qwen EOS")
    generated: list[int] = []
    for step in range(512):
        logits = model.get_logits_from_input_ids(tokens + generated)
        token = max(range(len(logits)), key=logits.__getitem__)
        if token == eos[0]:
            break
        generated.append(token)
        text = prefix + model.decode(generated)
        try:
            loads(text)
        except ValueError:
            continue
        return text, step + 1
    return prefix + model.decode(generated), len(generated)


def main() -> int:
    """Execute one editor session; the model controls only validated data."""
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("context", "store", "fixture", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--fingerprint", required=True)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "fingerprint": args.fingerprint, "actions": [], "finished": False,
        "started_monotonic_ns": time.monotonic_ns(),
    }
    try:
        import torch
        from llm_sdk import Small_LLM_Model

        model = Small_LLM_Model(args.optimizer, device="cuda",
                                dtype=torch.bfloat16, trust_remote_code=False)
        context = loads(args.context.read_text())
        provenance = loads(Path(os.environ[
            'CMM_PROMPT_PROVENANCE']).read_text())
        result['optimizer'] = args.optimizer
        result['input_context_tokens'] = int(model.encode(
            json.dumps(context, default=validate_number)).shape[-1])
        result['optimizer_peak_metric'] = 'torch max_memory_allocated bytes'
        torch.cuda.reset_peak_memory_stats()
        # Observations are data inside a user turn, not instructions or tools.
        conversation = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            "<|im_start|>user\n"
            + json.dumps(context, default=validate_number) + "\n"
            +
            "Propose one revised pair of instructions now.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        candidate_id: str | None = None

        def evaluate(path: Path) -> dict[str, Any]:
            """Run this saved candidate on the fixed development data."""
            trial = args.output.with_name("trial.json")
            command = [sys.executable, "-m", "tools.prompt_trial",
                       "--candidate", str(path),
                       "--fixture", str(args.fixture), "--report", str(trial),
                       "--fingerprint", evidence_fingerprint(
                           provenance['code_sha'], path, args.fixture,
                           provenance['runtime'])]
            code = subprocess.run(command, timeout=900).returncode
            if not trial.exists():
                raise ValueError(f"evaluation failed: exit {code}")
            report = loads(trial.read_text())
            result['trial_attempt'] = report
            if code or not report.get("finished"):
                raise ValueError(f"evaluation incomplete: exit {code}; "
                                 + str(report.get('error', '')))
            result["trial"] = report
            return {"score": report["score"]}

        for tool in ("write_prompt", "run"):
            prefix = ('{"tool":"write_prompt","prompt":' if tool ==
                      "write_prompt" else '{"tool":"run","candidate_id":')
            for attempt in range(2):
                input_tokens = int(model.encode(
                    conversation + prefix).shape[-1])
                action_text, token_count = generate_action(model, conversation,
                                                           prefix)
                observation: dict[str, Any] = {
                    "tool": tool, "attempt": attempt, "text": action_text,
                    "generated_tokens": token_count,
                    "input_tokens": input_tokens,
                }
                result["actions"].append(observation)
                try:
                    action = loads(action_text)
                    if not isinstance(action, dict):
                        raise ValueError('tool call must be a JSON object')
                    if action.get("tool") != tool:
                        raise ValueError(f"expected {tool}")
                    if tool == "run" and action.get("candidate_id") != (
                            candidate_id):
                        raise ValueError("run only the ID just saved")
                    reply = dispatch(args.store, action_text, evaluate)
                    observation["result"] = reply
                    if tool == "write_prompt":
                        candidate_id = reply["candidate_id"]
                        result["candidate_id"] = candidate_id
                    conversation += (
                        action_text
                        + "<|im_end|>\n<|im_start|>user\nTool result: "
                        + json.dumps(reply, default=validate_number)
                        + "\nNow call run with that exact candidate_id."
                        "<|im_end|>\n<|im_start|>assistant\n"
                        "<think>\n\n</think>\n\n"
                    )
                    write_json(args.output, result)
                    break
                except ValueError as error:
                    observation["error"] = str(error)
                    write_json(args.output, result)
                    if attempt:
                        raise
                    conversation += (
                        action_text + "<|im_end|>\n<|im_start|>user\n"
                        f"Invalid tool call: {error}. Correct it once."
                        "<|im_end|>\n<|im_start|>assistant\n"
                        "<think>\n\n</think>\n\n"
                    )
        result["finished"] = True
        return 0
    except (OSError, ValueError, RuntimeError,
            subprocess.SubprocessError) as e:
        result["error"] = str(e)
        return 1
    finally:
        if 'torch' in locals() and torch.cuda.is_available():
            result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        result["ended_monotonic_ns"] = time.monotonic_ns()
        write_json(args.output, result)


if __name__ == "__main__":
    raise SystemExit(main())
