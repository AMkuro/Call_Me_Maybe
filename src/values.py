"""Recognize prefixes of one JSON value and its slot terminator."""

import codecs
import re

from .schema import ParamType

_WHITESPACE = b" \t\r\n"
# A whole number as the LLM must write it: decimal point required.
_NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)\.[0-9]+\Z")
# Any prefix of _NUMBER, including the empty one.
_NUMBER_PREFIX = re.compile(rb"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]*)?)?\Z")

_QUOTE = ord('"')
_BACKSLASH = ord("\\")
_UNICODE_ESCAPE = ord("u")  # four hexadecimal digits follow

# States of _advance_string. The four Unicode-escape digit states are
# numbered so that the next digit state is the current one plus 1.
_INVALID = -1
_BEFORE_STRING = 0
_IN_STRING = 1
_AFTER_BACKSLASH = 2
_FIRST_HEX_DIGIT = 3
_LAST_HEX_DIGIT = 6
_AFTER_STRING = 7
_AFTER_TERMINATOR = 8

# Results of _slot_state. _INVALID above is the third result.
_INCOMPLETE = 0
_COMPLETE = 1


def _utf8_prefix(data: bytes) -> bool:
    """Accept UTF-8 or a sequence ending inside its final character."""
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError as e:
        return e.reason == "unexpected end of data" and e.end == len(data)


def _advance_string(phase: int, data: bytes, terminator: int) -> int:
    """Consume new bytes from a JSON-string state; return the new state.

    Nothing may follow the slot terminator, not even whitespace.
    """
    for char in data:
        if phase == _BEFORE_STRING:
            if char == _QUOTE:
                phase = _IN_STRING
            elif char not in _WHITESPACE:
                return _INVALID
        elif _FIRST_HEX_DIGIT <= phase <= _LAST_HEX_DIGIT:
            if char not in b"0123456789abcdefABCDEF":
                return _INVALID
            if phase == _LAST_HEX_DIGIT:
                phase = _IN_STRING
            else:
                phase += 1
        elif phase == _AFTER_BACKSLASH:
            if char == _UNICODE_ESCAPE:
                phase = _FIRST_HEX_DIGIT
            elif char not in b'"\\/bfnrt':
                return _INVALID
            else:
                phase = _IN_STRING
        elif phase == _IN_STRING:
            if char == _BACKSLASH:
                phase = _AFTER_BACKSLASH
            elif char == _QUOTE:
                phase = _AFTER_STRING
            elif char < 32:  # JSON forbids raw control characters
                return _INVALID
        elif phase == _AFTER_STRING:
            if char == terminator:
                phase = _AFTER_TERMINATOR
            elif char not in _WHITESPACE:
                return _INVALID
        else:
            return _INVALID
    return phase


def _slot_state(
    data: bytes, kind: ParamType, terminator: bytes,
) -> int:
    """Recognize an entire candidate, including bytes within merged tokens.

    Return _COMPLETE after a whole value and its terminator, _INCOMPLETE
    while it can still become one, and _INVALID otherwise. Nothing may
    follow the terminator, not even whitespace.
    """
    data = data.lstrip(_WHITESPACE)
    if kind == "string":
        phase = _advance_string(_BEFORE_STRING, data, terminator[0])
        if phase == _INVALID or not _utf8_prefix(data):
            return _INVALID
        if phase == _AFTER_TERMINATOR:
            return _COMPLETE
        return _INCOMPLETE
    if data.endswith(terminator):
        value = data[:-1].rstrip(_WHITESPACE)
        if kind == "number":
            whole = _NUMBER.fullmatch(value) is not None
        else:
            whole = value in (b"true", b"false")
        return _COMPLETE if whole else _INVALID
    value = data.rstrip(_WHITESPACE)
    if kind == "number":
        if value and value != data:
            # Whitespace ended the number, so no digit may follow it.
            pattern = _NUMBER
        else:
            pattern = _NUMBER_PREFIX
        return _INCOMPLETE if pattern.fullmatch(value) else _INVALID
    for word in (b"true", b"false"):
        if word.startswith(data) or value == word:
            return _INCOMPLETE
    return _INVALID


def allowed_slot_tokens(
    vocab: dict[int, bytes], param_type: ParamType, written: bytes,
    terminator: str,
) -> list[int]:
    """Allow prefixes of a value followed by its JSON ``terminator``.

    Judgment: takes the bytes written so far (``written``), because the
    next token of a number depends on the digits already written. The
    bytes of one character can span tokens. See docs/architecture.md §4.

    A complete number may continue or take the terminator. No token may
    write anything after the terminator.
    """
    terminator_bytes = terminator.encode("ascii")
    if param_type == "string":
        phase = _advance_string(_BEFORE_STRING, written,
                                terminator_bytes[0])
        if phase == _INVALID or not _utf8_prefix(written):
            return []
        decoder = codecs.getincrementaldecoder("utf-8")()
        decoder.decode(written, final=False)
        pending = decoder.getstate()[0]
        # Parse the shared prefix once. Each vocabulary candidate only
        # advances its own bytes, even for long values.
        end = terminator_bytes[0]
        return [token for token, value in vocab.items()
                if value and _advance_string(phase, value, end) != _INVALID
                and _utf8_prefix(pending + value)]
    return [token for token, value in vocab.items()
            if value and _slot_state(written + value, param_type,
                                     terminator_bytes) != _INVALID]


def is_slot_complete(
    written: bytes, param_type: ParamType, terminator: str,
) -> bool:
    """Return True only after a whole value and its JSON ``terminator``.

    A valid number without its terminator is not finished.
    """
    terminator_bytes = terminator.encode("ascii")
    state = _slot_state(written, param_type, terminator_bytes)
    return state == _COMPLETE
