"""Load and hold the validated generation assets for one application run."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from pydantic import (
    BaseModel,
    ConfigDict,
    InstanceOf,
    ValidationError,
    field_validator,
)

from .errors import StartupError
from .model_port import Model
from .name_trie import NameTrie
from .schema import FunctionDef
from .vocab import load_vocab


class GenerationAssets(BaseModel):
    """The model, its vocabulary, the definitions and their name tree.

    Built once per run and shared by every prompt.

    Attributes:
        model: The model.
        vocab: The bytes of each token ID, read-only.
        functions: A read-only copy of the definitions, keyed by name.
        trie: The name tree of ``functions``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: InstanceOf[Model]
    vocab: InstanceOf[Mapping[int, bytes]]
    functions: Mapping[str, InstanceOf[FunctionDef]]
    trie: InstanceOf[NameTrie]

    def __init__(self, functions: Mapping[str, FunctionDef]) -> None:
        """Load the model, read its vocabulary and build the name tree.

        Args:
            functions: The definitions, each keyed by its own name, as
                load_functions returns them.

        Raises:
            StartupError: The model, the vocabulary or the name tree cannot
                be loaded or built.
        """
        model = Model()
        vocab = load_vocab(Path(model.vocab_path()))
        trie = NameTrie(functions, model)
        try:
            super().__init__(
                model=model, vocab=vocab, functions=functions, trie=trie
            )
        except ValidationError as e:
            raise StartupError(f"invalid generation resources: {e}") from e

    @field_validator("functions")
    @classmethod
    def _read_only(
        cls, functions: Mapping[str, FunctionDef]
    ) -> Mapping[str, FunctionDef]:
        """Wrap the new dict Pydantic built, so no outside alias exists.

        Args:
            functions: The validated definitions.

        Returns:
            A read-only view of ``functions``.
        """
        return MappingProxyType(functions)
