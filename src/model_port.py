"""The model wrapper. Only this file imports llm_sdk."""

from .errors import PromptError, StartupError

# Qwen's response-ending special token; it ends every function-name path.
# A provisional choice, to revisit if it causes failures.
EOS = "<|im_end|>"


class Model:
    """A thin wrapper around the llm_sdk model that returns plain lists.

    The SDK returns tensors; these methods return lists, so the rest of the
    program never sees torch. The calls do not change the model. ``logits``
    is the SDK's get_logits_from_input_ids. It is not a pydantic model: it
    holds only the SDK object and no data to validate.
    """

    def __init__(self) -> None:
        """Load the SDK default model (Qwen/Qwen3-0.6B).

        Remote model code is not trusted. The SDK reads the model from the
        Hugging Face cache and downloads it when it is missing.

        Raises:
            StartupError: The model cannot be loaded.
        """
        try:
            from llm_sdk import Small_LLM_Model

            self._sdk = Small_LLM_Model(trust_remote_code=False)
        except Exception as e:
            raise StartupError(
                f"cannot load model: {type(e).__name__}: {e}"
            ) from e

    def encode(self, text: str) -> list[int]:
        """Return the token IDs of ``text``.

        Args:
            text: The text to encode.

        Returns:
            Its token IDs.

        Raises:
            PromptError: The SDK fails.
        """
        try:
            encoded: list[list[int]] = self._sdk.encode(text).tolist()
            return encoded[0]
        except Exception as e:
            raise PromptError(f"cannot encode: {type(e).__name__}: {e}") from e

    def decode(self, ids: list[int]) -> str:
        """Return the text of ``ids``.

        Args:
            ids: Token IDs.

        Returns:
            Their text.

        Raises:
            PromptError: The SDK fails.
        """
        try:
            return self._sdk.decode(ids)
        except Exception as e:
            raise PromptError(f"cannot decode: {type(e).__name__}: {e}") from e

    def logits(self, ids: list[int]) -> list[float]:
        """Return the next-token logits after ``ids``.

        Runs one forward pass on the whole of ``ids``.

        Args:
            ids: The context.

        Returns:
            One score per vocabulary ID.

        Raises:
            PromptError: The SDK fails.
        """
        try:
            return self._sdk.get_logits_from_input_ids(ids)
        except Exception as e:
            raise PromptError(
                f"model inference failed: {type(e).__name__}: {e}"
            ) from e

    def vocab_path(self) -> str:
        """Return the path of the vocabulary file.

        Returns:
            The path of the tokenizer's vocabulary JSON file.

        Raises:
            StartupError: The SDK fails.
        """
        try:
            return self._sdk.get_path_to_vocab_file()
        except Exception as e:
            raise StartupError(
                f"cannot locate vocabulary: {type(e).__name__}: {e}"
            ) from e
