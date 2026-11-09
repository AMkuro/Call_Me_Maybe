"""Which tokens may be written in a value slot, per parameter type."""

from .schema import ParamType


def allowed_value_tokens(
    vocab: dict[int, str], param_type: ParamType, written: str,
    terminator: str,
) -> list[int]:
    """Allow prefixes of a value followed by its JSON ``terminator``.

    Judgment: takes the text written so far (``written``), because the
    next token of a number depends on the digits already written. The
    details for strings are open (ADR-0003).

    A complete number may continue or take the terminator. No token may
    write anything after the terminator (ADR-0012).
    """
    raise NotImplementedError


def is_value_complete(
    written: str, param_type: ParamType, terminator: str,
) -> bool:
    """Return True only after a whole value and its JSON ``terminator``.

    A valid number without its terminator is not finished (ADR-0012).
    """
    raise NotImplementedError
