"""What generation reports while it runs, for a display to show."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol, TypeGuard

if TYPE_CHECKING:
    from .decoder import SelectionMode
    from .schema import ParamType

# Each stage is named after the function it runs, so the display
# points at the code that is working.
Stage = Literal["select_function", "generate_call_json", "validate_call"]


class GenerationProgress(Protocol):
    """Receives generation events to display; it cannot steer generation.

    Calls are synchronous. The ``tokens`` given to name_token belong to the
    generator: read them during the call, and do not change or keep them.
    """

    @property
    def available(self) -> bool:
        """False once the output stream has failed; reporting then stops."""
        ...

    def stage(self, stage: Stage) -> None:
        """Report that a stage of the prompt starts.

        Args:
            stage: The stage.
        """
        ...

    def function_selected(self, name: str) -> None:
        """Report the chosen function name.

        Args:
            name: The name.
        """
        ...

    def parameter_started(self, name: str, param_type: ParamType) -> None:
        """Report that the value of one parameter starts.

        Args:
            name: The parameter name.
            param_type: Its type.
        """
        ...

    def name_token(
        self, tokens: Sequence[int], *, mode: SelectionMode,
        raw_top: int | None,
    ) -> None:
        """Report one more name token.

        Args:
            tokens: The name's token IDs so far, borrowed for this call.
            mode: How the last token was chosen.
            raw_top: The unconstrained best token in greedy mode when it
                was computed, else None.
        """
        ...

    def output_token(
        self, data: bytes, *, token: int, mode: SelectionMode,
        fixed: bool = False,
    ) -> None:
        """Observe one committed JSON token's bytes, including fixed text.

        Args:
            data: Actual vocabulary bytes, before float-text normalization.
            token: The committed token ID.
            mode: How that token was selected.
            fixed: Whether this token belongs to fixed JSON text.
        """
        ...

    def value_token(
        self, written: bytes, step: int, *, token: int, mode: SelectionMode,
    ) -> None:
        """Report one more value token.

        Args:
            written: The value's bytes so far.
            step: The number of its tokens.
            token: The last token ID.
            mode: How the last token was chosen.
        """
        ...


def observing(
    progress: GenerationProgress | None,
) -> TypeGuard[GenerationProgress]:
    """Tell whether to notify ``progress`` and build what only it uses.

    Checked at every notification, because ``available`` turns False when
    the stream fails during a prompt.

    Args:
        progress: The receiver, or None.

    Returns:
        True when ``progress`` is given and available.
    """
    return progress is not None and progress.available
