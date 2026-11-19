"""Recognize JSON value prefixes and normalize generated number slots."""

import codecs
import re
from collections.abc import Mapping, Sequence

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

# Syntax states of _scan_json_string_fragment. Unicode escape states
# count down from four remaining hexadecimal digits to one.
_INVALID = -1
_BEFORE_STRING = 0
_IN_STRING = 1
_AFTER_BACKSLASH = 2
_ONE_HEX_DIGIT_LEFT = 3
_FOUR_HEX_DIGITS_LEFT = 6
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


def normalize_call_json(
    raw_json: str, number_ranges: Sequence[tuple[int, int]],
) -> str:
    """Add missing fractions only within generated number slots.

    Args:
        raw_json: The verified, unmodified text returned by generation.
        number_ranges: Ordered, nonoverlapping character ranges from
            generation, each including the number slot's terminator.
            Starts are inclusive and ends exclusive.

    Returns:
        The same JSON text, with integer number slots changed by float_text.
        No parsing or numeric conversion changes the original digits.
    """
    parts: list[str] = []
    offset = 0
    for start, end in number_ranges:
        parts.append(raw_json[offset:start])
        parts.append(float_text(raw_json[start:end]))
        offset = end
    parts.append(raw_json[offset:])
    return "".join(parts)


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


def _scan_json_string_fragment(
    state: int, fragment: bytes, terminator: int,
) -> int:
    """Scan JSON string syntax from ``state`` through a byte fragment.

    This checks syntax only; UTF-8 validity is checked separately.
    Nothing may follow the slot terminator, not even whitespace.

    Args:
        state: The syntax state reached before this fragment.
        fragment: The next bytes, possibly ending inside an escape.
        terminator: The slot terminator byte.

    Returns:
        The reached syntax state, or _INVALID for irreparable syntax.
        _AFTER_TERMINATOR means complete; all other valid states need
        more bytes. An empty fragment preserves the starting state.
    """
    for byte in fragment:
        if state == _BEFORE_STRING:
            if byte == _QUOTE:
                state = _IN_STRING
            elif byte not in _WHITESPACE:
                return _INVALID
        elif state == _IN_STRING:
            if byte == _BACKSLASH:
                state = _AFTER_BACKSLASH
            elif byte == _QUOTE:
                state = _AFTER_STRING
            elif byte < 32:  # JSON forbids raw control characters
                return _INVALID
        elif state == _AFTER_BACKSLASH:
            if byte == _UNICODE_ESCAPE:
                state = _FOUR_HEX_DIGITS_LEFT
            elif byte in b'"\\/bfnrt':
                state = _IN_STRING
            else:
                return _INVALID
        elif _ONE_HEX_DIGIT_LEFT <= state <= _FOUR_HEX_DIGITS_LEFT:
            if byte not in b"0123456789abcdefABCDEF":
                return _INVALID
            if state == _ONE_HEX_DIGIT_LEFT:
                state = _IN_STRING
            else:
                state -= 1  # One of the four required digits was consumed.
        elif state == _AFTER_STRING:
            if byte == terminator:
                state = _AFTER_TERMINATOR
            elif byte not in _WHITESPACE:
                return _INVALID
        else:
            return _INVALID
    return state


def _string_slot_state(
    state: int, fragment: bytes, terminator: int, pending_utf8: bytes = b"",
) -> int:
    """Classify a string slot after scanning its next fragment.

    Args:
        state: Syntax state before the fragment.
        fragment: The bytes to scan.
        terminator: The slot terminator byte.
        pending_utf8: An unfinished UTF-8 character from the valid prefix.

    Returns:
        _INVALID for invalid syntax or UTF-8, _COMPLETE after the
        terminator, otherwise _INCOMPLETE. Starting at _BEFORE_STRING
        with no pending UTF-8 classifies a whole slot.
    """
    state = _scan_json_string_fragment(state, fragment, terminator)
    if state == _INVALID or not _utf8_prefix(pending_utf8 + fragment):
        return _INVALID
    return _COMPLETE if state == _AFTER_TERMINATOR else _INCOMPLETE


def _slot_state(
    data: bytes,
    kind: ParamType,
    terminator: bytes,
) -> int:
    """Classify ``data``, all the bytes written for one slot so far.

    Nothing may follow the terminator, not even whitespace.

    Args:
        data: The slot's bytes.
        kind: The value type.
        terminator: The terminator, one byte.

    Returns:
        _COMPLETE after a whole value and its terminator, _INCOMPLETE while
        it can still become one, and _INVALID otherwise. This is the
        reference classification for completion and candidate filtering.
    """
    data = data.lstrip(_WHITESPACE)
    if kind == "string":
        return _string_slot_state(_BEFORE_STRING, data, terminator[0])
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
    vocab: Mapping[int, bytes],
    param_type: ParamType,
    written: bytes,
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
        state = _scan_json_string_fragment(
            _BEFORE_STRING, written, terminator_bytes[0],
        )
        if state == _INVALID or not _utf8_prefix(written):
            return []
        decoder = codecs.getincrementaldecoder("utf-8")()
        decoder.decode(written, final=False)
        pending = decoder.getstate()[0]
        # Parse the shared prefix once. Each vocabulary candidate only
        # advances its own bytes, even for long values.
        end = terminator_bytes[0]
        return [
            token
            for token, value in vocab.items()
            if value
            and _string_slot_state(state, value, end, pending) != _INVALID
        ]
    return [
        token
        for token, value in vocab.items()
        if value
        and _slot_state(written + value, param_type, terminator_bytes)
        != _INVALID
    ]


def is_slot_complete(
    written: bytes,
    param_type: ParamType,
    terminator: str,
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
