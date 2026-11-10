"""A token-level tree of the function names (ADR-0011)."""

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from .errors import PromptError, StartupError

EOS = "<|im_end|>"


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
    Judgment: Qwen's response-ending token is provisional (ADR-0013).
    """
    root = NameTrie()
    try:
        for name in names:
            ids = encode(name + EOS)
            if not ids:
                raise ValueError(f"empty token path for {name!r}")
            node = root
            for token in ids[:-1]:
                if node.name is not None:
                    raise ValueError("function names have colliding EOS paths")
                node = node.children.setdefault(token, NameTrie())
            if node.name is not None or ids[-1] in node.children:
                raise ValueError("function names have colliding EOS paths")
            node.children[ids[-1]] = NameTrie(name=name)
    except (PromptError, ValueError) as e:
        raise StartupError(f"cannot build name tree: {e}") from e
    return root


def next_tokens(trie: NameTrie, prefix: Sequence[int]) -> list[int]:
    """Return the token ids that may follow ``prefix`` (empty at a leaf)."""
    node = trie
    try:
        for token in prefix:
            node = node.children[token]
    except KeyError as e:
        raise PromptError("invalid token path in name tree") from e
    return list(node.children)


def leaf_name(trie: NameTrie, prefix: Sequence[int]) -> str | None:
    """Return the function name if ``prefix`` reaches a leaf, else None."""
    node = trie
    for token in prefix:
        if token not in node.children:
            return None
        node = node.children[token]
    return node.name


def max_name_steps(trie: NameTrie) -> int:
    """Return the longest terminated path, without a fixed name-length cap."""
    pending = [(trie, 0)]
    longest = 0
    while pending:
        node, depth = pending.pop()
        longest = max(longest, depth)
        pending.extend((child, depth + 1) for child in node.children.values())
    return longest
