"""Load and hold the validated generation assets for one application run."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from pydantic import (
    BaseModel, ConfigDict, InstanceOf, ValidationError, field_validator,
)

from .errors import StartupError
from .model_port import Model
from .name_trie import NameTrie
from .schema import FunctionDef
from .vocab import load_vocab


class GenerationAssets(BaseModel):
    """Combine validated resources for one application run.

    The input loader indexes definitions by name. This constructor builds
    the trie from those keys with the same model, and NameTrie validates
    its paths. functions is a read-only copy; vocab and SDK state remain
    shared and must not be changed after startup.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: InstanceOf[Model]
    vocab: InstanceOf[dict[int, bytes]]
    functions: Mapping[str, InstanceOf[FunctionDef]]
    trie: InstanceOf[NameTrie]

    def __init__(self, functions: Mapping[str, FunctionDef]) -> None:
        """Load the model, then build its vocabulary and the name tree.

        Shared by the application and the measurement tool, so both generate
        with the same resources. Raises StartupError.

        Effect: loads the model and reads its vocabulary file.
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
        """Wrap the new dict Pydantic built, so no outside alias exists."""
        return MappingProxyType(functions)
