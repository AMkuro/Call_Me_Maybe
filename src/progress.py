"""Static contract for synchronous, non-controlling generation observations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from .decoder import SelectionMode
    from .schema import ParamType

Stage = Literal["select_name", "generate_parameters", "validate_call"]


class GenerationProgress(Protocol):
    """Observe generation without changing its data or deciding its next step.

    Name IDs are borrowed for this call only: do not mutate or retain them.
    Availability describes the diagnostic stream, independently of isatty.
    """

    @property
    def available(self) -> bool:
        """Whether the diagnostic stream is usable."""
        ...

    def stage(self, stage: Stage) -> None:
        """Enter one generation or validation stage."""
        ...

    def function_selected(self, name: str) -> None:
        """Record the completed name selection."""
        ...

    def parameter_started(self, name: str, param_type: ParamType) -> None:
        """Begin one value, clearing its previous progress."""
        ...

    def name_token(
        self, tokens: Sequence[int], *, mode: SelectionMode,
        raw_top: int | None,
    ) -> None:
        """Synchronously consume the borrowed name prefix and its decision."""
        ...

    def value_token(
        self, written: bytes, step: int, *, token: int, mode: SelectionMode,
    ) -> None:
        """Observe a value prefix and the ID/mode that produced this step."""
        ...
