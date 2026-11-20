"""Error kinds. The message is the only data they carry."""


class CallMeMaybeError(Exception):
    """Base class of the errors this program raises on purpose.

    Each carries only its message. StartupError stops the run;
    PromptError skips one prompt.
    """


class StartupError(CallMeMaybeError):
    """The run cannot start; raised before any prompt is processed.

    Covers unreadable or invalid input files, an output path that cannot
    be prepared, and a model, vocabulary or name tree that cannot be
    loaded or built. The program prints the message and exits with 1.
    """


class PromptError(CallMeMaybeError):
    """One prompt failed. It is skipped and the rest are still processed."""
