"""Construct real validated tries with fixed tokenizer results."""

from collections.abc import Mapping, Sequence
from unittest.mock import patch

from src.model_port import EOS, Model
from src.name_trie import NameTrie


def trie_from_paths(paths: Mapping[str, Sequence[int]]) -> NameTrie:
    """Mock only encoding; keep constructor and all validators active."""
    with patch.object(
        Model,
        "encode",
        side_effect=lambda text: list(paths[text.removesuffix(EOS)]),
    ):
        return NameTrie(paths, object.__new__(Model))
