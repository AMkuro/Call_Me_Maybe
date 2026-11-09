"""The model wrapper. Only this file imports llm_sdk."""


class Model:
    """The model: a thin wrapper around the llm_sdk model.

    The SDK returns tensors; this class returns plain lists, so the rest of
    the program never sees torch.

    Effect: every method uses the model, so a caller of any method has an
    effect. The model itself is not changed by the calls.

    Judgment: a plain class with no abstract interface in front of it.
    It holds the SDK handle and no data to validate, so it is outside the
    pydantic rule (ADR-0011). logits is a shorter name for the SDK's
    get_logits_from_input_ids. Open: should the names follow the SDK?
    """

    def __init__(self) -> None:
        """Load the model from llm_sdk (ADR-0007).

        Any load failure is raised as StartupError with a clear message.

        Judgment: takes no arguments (the model name is the SDK default).
        How to hold settings is open.
        """
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Return the token ids of ``text``."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Return the text of ``ids``."""
        raise NotImplementedError

    def logits(self, ids: list[int]) -> list[float]:
        """Return the next-token logits after ``ids`` (one per token)."""
        raise NotImplementedError

    def vocab_path(self) -> str:
        """Return the path of the vocabulary file."""
        raise NotImplementedError
