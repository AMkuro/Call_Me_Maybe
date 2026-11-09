"""Which tokens may be written in a value slot, per parameter type."""

from .schema import ParamType


def allowed_value_tokens(
    vocab: dict[int, str], param_type: ParamType, written: str
) -> list[int]:
    """Return the token ids that keep ``written`` a valid value prefix."""
    raise NotImplementedError


def is_value_complete(written: str, param_type: ParamType) -> bool:
    """Return True when ``written`` is a whole value of ``param_type``."""
    raise NotImplementedError
