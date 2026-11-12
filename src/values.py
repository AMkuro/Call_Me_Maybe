"""Recognize prefixes of one JSON value and its slot delimiter."""

import codecs
import re

from .schema import ParamType

_WHITESPACE = b" \t\r\n"
_NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_NUMBER_PREFIX = re.compile(rb"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]*)?)?\Z")


def _utf8_prefix(data: bytes) -> bool:
    """Accept UTF-8 or a sequence ending inside its final character."""
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError as e:
        return e.reason == "unexpected end of data" and e.end == len(data)


def _advance_string(phase: int, data: bytes, terminator: int) -> int:
    """Consume new bytes from a JSON-string state; -1 means invalid.

    States: 0 leading space, 1 string, 2 escape, 3..6 Unicode escape,
    7 after the closing quote, 8 after the slot delimiter.
    """
    for char in data:
        if phase == 0:
            if char == 34:
                phase = 1
            elif char not in _WHITESPACE:
                return -1
        elif 3 <= phase <= 6:
            if char not in b"0123456789abcdefABCDEF":
                return -1
            phase = 1 if phase == 6 else phase + 1
        elif phase == 2:
            if char == 117:  # u: four hexadecimal digits follow
                phase = 3
            elif char not in b'"\\/bfnrt':
                return -1
            else:
                phase = 1
        elif phase == 1:
            if char == 92:
                phase = 2
            elif char == 34:
                phase = 7
            elif char < 32:
                return -1
        elif phase == 7:
            if char == terminator:
                phase = 8
            elif char not in _WHITESPACE:
                return -1
        else:
            return -1
    return phase


def _value_state(data: bytes, kind: ParamType, terminator: bytes) -> int:
    """Recognize an entire candidate, including bytes within merged tokens."""
    data = data.lstrip(_WHITESPACE)
    if kind == "string":
        phase = _advance_string(0, data, terminator[0])
        if phase < 0 or not _utf8_prefix(data):
            return -1
        return 1 if phase == 8 else 0
    if data.endswith(terminator):
        value = data[:-1].rstrip(_WHITESPACE)
        if kind == "number":
            return 1 if _NUMBER.fullmatch(value) else -1
        return 1 if value in (b"true", b"false") else -1
    value = data.rstrip(_WHITESPACE)
    if kind == "number":
        pattern = _NUMBER if value != data and value else _NUMBER_PREFIX
        return 0 if pattern.fullmatch(value) else -1
    return 0 if any(word.startswith(data) or value == word
                    for word in (b"true", b"false")) else -1


def allowed_value_tokens(
    vocab: dict[int, bytes], param_type: ParamType, written: bytes,
    terminator: str,
) -> list[int]:
    """Allow prefixes of a value followed by its JSON ``terminator``.

    Judgment: takes the bytes written so far (``written``), because the
    next token of a number depends on the digits already written. The
    bytes of one character can span tokens (ADR-0013).

    A complete number may continue or take the terminator. No token may
    write anything after the terminator (ADR-0012).
    """
    delimiter = terminator.encode("ascii")
    if param_type == "string":
        phase = _advance_string(0, written, delimiter[0])
        if phase < 0 or not _utf8_prefix(written):
            return []
        decoder = codecs.getincrementaldecoder("utf-8")()
        decoder.decode(written, final=False)
        pending = decoder.getstate()[0]
        # Parse the shared prefix once. Each vocabulary candidate only
        # advances its own bytes, even for long values.
        return [token for token, value in vocab.items()
                if value and _advance_string(phase, value, delimiter[0]) >= 0
                and _utf8_prefix(pending + value)]
    return [token for token, value in vocab.items()
            if value and _value_state(written + value, param_type,
                                      delimiter) >= 0]


def is_value_complete(
    written: bytes, param_type: ParamType, terminator: str,
) -> bool:
    """Return True only after a whole value and its JSON ``terminator``.

    A valid number without its terminator is not finished (ADR-0012).
    """
    return _value_state(written, param_type, terminator.encode("ascii")) == 1
