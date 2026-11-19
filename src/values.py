"""Recognize prefixes of one JSON value and its slot terminator."""

import codecs
import re
from collections.abc import Mapping

from .schema import ParamType

_WHITESPACE = b" \t\r\n"
# A whole number as the LLM may write it: the fraction is optional, because
# forbidding the terminator after an integer made the model write "1000..."
# to the cap.
_NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
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


_INTEGER_SLOT = re.compile(r"\A(\s*-?[0-9]+)(\s*[,}])\Z")


def float_text(fragment: str) -> str:
    """Add ".0" to a number slot written as an integer: "10," -> "10.0,".

    The model may end a number at an integer, but the saved output holds
    floats. The text is edited, so no digit is lost to a float conversion.

    Args:
        fragment: The decoded text of a completed number slot.

    Returns:
        ``fragment`` with ".0" after an integer, otherwise unchanged.
    """
    return _INTEGER_SLOT.sub(r"\1.0\2", fragment)


def _utf8_prefix(data: bytes) -> bool:
    """Tell whether ``data`` is UTF-8, maybe cut in its last character.

    Args:
        data: Bytes written so far.

    Returns:
        True when ``data`` is UTF-8 or ends inside its final character.
    """
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError as e:
        return e.reason == "unexpected end of data" and e.end == len(data)


def _advance_string(phase: int, data: bytes, terminator: int) -> int:
    """Consume new bytes from a JSON-string state; return the new state.

    Nothing may follow the slot terminator, not even whitespace.

    Args:
        phase: The state before ``data``.
        data: The new bytes.
        terminator: The terminator byte.

    Returns:
        The state after ``data``, or _INVALID.
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
    """Classify ``data``, all the bytes written for one slot so far.

    Nothing may follow the terminator, not even whitespace.

    Args:
        data: The slot's bytes.
        kind: The value type.
        terminator: The terminator, one byte.

    Returns:
        _COMPLETE after a whole value and its terminator, _INCOMPLETE while
        it can still become one, and _INVALID otherwise.
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
    vocab: Mapping[int, bytes], param_type: ParamType, written: bytes,
    terminator: str,
) -> list[int]:
    """Return the IDs of the tokens that may follow ``written`` in a slot.

    A token is allowed when ``written`` plus its bytes is still a prefix of
    a ``param_type`` value followed by ``terminator``, or exactly that. A
    complete number may continue or take the terminator. No token may
    write anything after the terminator, and an empty token is never
    allowed.

    Args:
        vocab: The bytes of each token.
        param_type: The value type.
        written: All the bytes of the slot so far. They decide what may
            come next: the digits of a number, an open escape, or a UTF-8
            character split across tokens.
        terminator: "," or "}".

    Returns:
        The allowed token IDs, in vocabulary order.
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

    Args:
        written: All the bytes of the slot so far.
        param_type: The value type.
        terminator: "," or "}".

    Returns:
        Whether the slot is complete.
    """
    terminator_bytes = terminator.encode("ascii")
    state = _slot_state(written, param_type, terminator_bytes)
    return state == _COMPLETE
