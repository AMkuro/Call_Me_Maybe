"""Error kinds. The message is the only data they carry."""


class CallMeMaybeError(Exception):
    """Base class of every error this program raises on purpose."""


class StartupError(CallMeMaybeError):
    """Bad arguments, bad definitions or a model that fails its check.

    Raised before any prompt is processed; the program exits non-zero.
    """


class PromptError(CallMeMaybeError):
    """One prompt failed. It is skipped and the rest are still processed."""
