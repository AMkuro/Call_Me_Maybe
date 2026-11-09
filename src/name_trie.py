"""A token-level tree of the function names (ADR-0011)."""

from collections.abc import Callable, Sequence

from pydantic import BaseModel


class NameTrie(BaseModel):
    """A node. ``name`` is set on a leaf: the function this path spells."""

    children: dict[int, "NameTrie"] = {}
    name: str | None = None


def build_name_trie(
    names: Sequence[str], encode: Callable[[str], list[int]]
) -> NameTrie:
    """Build the tree from the token ids of each name.

    Raises StartupError when one name's tokens are a prefix of another's,
    or when a name's tokens are not a prefix of the tokens of the name
    followed by the closing quote and comma.
    """
    raise NotImplementedError


def next_tokens(trie: NameTrie, prefix: Sequence[int]) -> list[int]:
    """Return the token ids that may follow ``prefix`` (empty at a leaf)."""
    raise NotImplementedError


def leaf_name(trie: NameTrie, prefix: Sequence[int]) -> str | None:
    """Return the function name if ``prefix`` reaches a leaf, else None."""
    raise NotImplementedError
