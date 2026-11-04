"""Observe greedy token generation through the supplied SDK's public API."""

from __future__ import annotations

import argparse
import math
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def generate(
    model: Small_LLM_Model, prompt: str, max_new_tokens: int
) -> list[int]:
    """Print generation observations and return only the generated token IDs.

    The caller owns the model. SDK failures retain their cause and identify
    the operation (and generation step) that failed.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be blank")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    # Single user turn from Qwen3's enable_thinking=False chat template.
    text = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    stage = "encode/input conversion"
    generated_ids: list[int] = []
    try:
        input_ids: list[int] = model.encode(text).tolist()[0]
        if not input_ids:
            raise ValueError("encode returned no input tokens")
        print(f"input_text={text!r}")
        print(f"input_ids={input_ids}")
        for step in range(1, max_new_tokens + 1):
            stage = f"logits/selection at step {step}"
            logits: list[float] = model.get_logits_from_input_ids(input_ids)
            if not logits or not all(math.isfinite(x) for x in logits):
                raise ValueError("expected nonempty, finite logits")
            next_id = max(range(len(logits)), key=logits.__getitem__)
            input_ids.append(next_id)
            generated_ids.append(next_id)
            print(
                f"step={step} token_id={next_id} logit={logits[next_id]}"
            )
        print(f"generated_ids={generated_ids}")
        stage = "decode"
        print(f"generated_text={model.decode(generated_ids)!r}")
    except Exception as exc:
        raise RuntimeError(
            f"{stage}: {type(exc).__name__}: {exc}"
        ) from exc
    return generated_ids


def main(argv: list[str] | None = None) -> int:
    """Load Qwen3 once, process each prompt, and report failures on stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", nargs="+", help="one or more user messages")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0 or any(
        not prompt.strip() for prompt in args.prompts
    ):
        parser.error(
            "prompts must not be blank; max-new-tokens must be positive"
        )

    stage = "SDK import/initialization"
    try:
        from llm_sdk import Small_LLM_Model

        model = Small_LLM_Model(
            "Qwen/Qwen3-0.6B", trust_remote_code=False
        )
        for index, prompt in enumerate(args.prompts, start=1):
            stage = f"prompt {index} generation"
            print(f"prompt_index={index}")
            generate(model, prompt, args.max_new_tokens)
    except Exception as exc:
        print(f"error: {stage}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"interrupted: {stage}", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
