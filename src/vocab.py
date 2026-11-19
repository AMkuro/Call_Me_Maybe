"""Read Qwen's byte-level vocabulary without losing partial UTF-8."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from pydantic import TypeAdapter

from .errors import StartupError
from .json_io import loads


def load_vocab(path: Path) -> Mapping[int, bytes]:
    """Read the vocabulary file; map each token ID to the bytes it stands for.

    The file is the SDK tokenizer's byte-level BPE vocabulary. It writes
    each byte as one character: printable bytes as themselves, the others
    from U+0100 upwards. Reversing that gives the exact bytes, also for a
    token that holds only part of a UTF-8 character.

    Args:
        path: The vocabulary JSON file, from token text to ID.

    Returns:
        A read-only mapping from each token ID to its bytes.

    Raises:
        StartupError: The file is unreadable, invalid or empty, has a
            negative or repeated ID, or a character outside the alphabet.
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
            loads(path.read_text(encoding="utf-8")), strict=True,
        )
        if not raw or any(token < 0 for token in raw.values()):
            raise ValueError("vocabulary must have nonnegative token IDs")
        if len(set(raw.values())) != len(raw):
            raise ValueError("duplicate vocabulary token IDs")
        return MappingProxyType({
            token: bytes(alphabet[char] for char in text)
            for text, token in raw.items()
        })
    except (OSError, ValueError, KeyError) as e:
        raise StartupError(
            f"cannot read byte vocabulary from {path}: {e}"
        ) from e
