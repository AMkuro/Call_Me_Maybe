"""Read Qwen's byte-level vocabulary without losing partial UTF-8."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from .errors import StartupError


def load_vocab(path: Path) -> dict[int, bytes]:
    """Map each vocabulary ID to its original byte sequence.

    Effect: reads a file.

    Judgment: the SDK's Qwen tokenizer uses byte-level BPE. Its visible
    alphabet keeps printable bytes and maps the others above U+00FF.
    Reverse that alphabet, without decoding incomplete UTF-8 (ADR-0013).
    """
    visible = list(range(33, 127)) + list(range(161, 173))
    visible += list(range(174, 256))
    missing = [byte for byte in range(256) if byte not in visible]
    alphabet = dict(zip(
        (chr(point) for point in
         visible + list(range(256, 256 + len(missing)))),
        visible + missing,
    ))
    try:
        raw = TypeAdapter(dict[str, int]).validate_python(
            json.loads(path.read_text(encoding="utf-8")), strict=True,
        )
        if not raw or any(token < 0 for token in raw.values()):
            raise ValueError("vocabulary must have nonnegative token IDs")
        if len(set(raw.values())) != len(raw):
            raise ValueError("duplicate vocabulary token IDs")
        return {
            token: bytes(alphabet[char] for char in text)
            for text, token in raw.items()
        }
    except (OSError, ValueError, KeyError) as e:
        raise StartupError(
            f"cannot read byte vocabulary from {path}: {e}"
        ) from e
