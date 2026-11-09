"""A token-level tree of the function names (ADR-0011)."""

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict


class NameTrie(BaseModel):
    """A node. ``name`` is set on a leaf: the function this path spells.

    Judgment: a leaf (a node with no children) carries the name, and
    leaf_name tells whether a path reached a leaf. It is a pydantic model
    because it holds data and the Subject says classes use pydantic for
    validation.
    """

    model_config = ConfigDict(frozen=True)

    children: dict[int, "NameTrie"] = {}
    name: str | None = None


def build_name_trie(
    names: Sequence[str], encode: Callable[[str], list[int]]
) -> NameTrie:
    """Build the tree from the token ids of each name plus a terminator.

    The terminator ends every path, so a name that is a prefix of another
    name is not a collision: the model chooses to end or to continue.
    Names are unique and non-empty (checked by the loader).
    Open: which terminator to use (ADR-0011).
    """
    raise NotImplementedError


def next_tokens(trie: NameTrie, prefix: Sequence[int]) -> list[int]:
    """Return the token ids that may follow ``prefix`` (empty at a leaf)."""
    raise NotImplementedError


def leaf_name(trie: NameTrie, prefix: Sequence[int]) -> str | None:
    """Return the function name if ``prefix`` reaches a leaf, else None."""
    raise NotImplementedError
