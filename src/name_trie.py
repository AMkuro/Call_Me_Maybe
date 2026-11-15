"""A token-level tree of the function names (ADR-0011)."""

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from .errors import PromptError, StartupError

EOS = "<|im_end|>"


class NameTrie(BaseModel):
    """A node. ``name`` is set on a leaf: the function this path spells.

    Invariant (checked per node when it is constructed): a node is a leaf
    exactly when it has a name. Which token ends a path is a tokenizer
    fact, so build_name_trie checks that every leaf follows the EOS token.
    Frozen is shallow: ``children`` must not be changed after construction.
    """

    model_config = ConfigDict(frozen=True)

    children: dict[int, "NameTrie"] = {}
    name: str | None = None

    @model_validator(mode="after")
    def _named_exactly_at_leaves(self) -> "NameTrie":
        """Reject an unnamed leaf or a named node that continues."""
        if (self.name is None) == (not self.children):
            raise ValueError("a node must be either named or have children")
        return self


def build_name_trie(
    names: Sequence[str], encode: Callable[[str], list[int]]
) -> NameTrie:
    """Build the tree from the token ids of each name plus EOS.

    Every path is the name's tokens followed by the single EOS token, so a
    name that is a prefix of another is not a collision: the model chooses
    to end or to continue. Names are unique and non-empty (loader).
    The tree is built as plain dicts and validated once by NameTrie, so the
    node checks see the finished tree.
    Judgment: Qwen's response-ending token is provisional (ADR-0013).
    """
    try:
        eos = encode(EOS)
        if len(eos) != 1:
            raise ValueError(f"{EOS} is not one token: {eos}")
        root: dict[str, Any] = {"children": {}}
        for name in names:
            ids = encode(name + EOS)
            if len(ids) < 2 or ids[-1] != eos[0] or eos[0] in ids[:-1]:
                raise ValueError(
                    f"{name!r} is not name tokens followed by EOS: {ids}"
                )
            node = root
            for token in ids[:-1]:
                node = node["children"].setdefault(token, {"children": {}})
            if eos[0] in node["children"]:
                raise ValueError(f"{name!r} has the same token path as "
                                 "another function name")
            node["children"][eos[0]] = {"name": name}
        return NameTrie.model_validate(root)
    except (PromptError, ValueError) as e:  # ValidationError is a ValueError
        raise StartupError(f"cannot build name tree: {e}") from e


def _node(trie: NameTrie, prefix: Sequence[int]) -> NameTrie:
    """Follow ``prefix``; a token that is not a child is an error."""
    node = trie
    try:
        for token in prefix:
            node = node.children[token]
    except KeyError as e:
        raise PromptError("invalid token path in name tree") from e
    return node


def next_tokens(trie: NameTrie, prefix: Sequence[int]) -> list[int]:
    """Return the token ids that may follow ``prefix`` (empty at a leaf)."""
    return list(_node(trie, prefix).children)


def leaf_name(trie: NameTrie, prefix: Sequence[int]) -> str | None:
    """Return the name at a leaf, or None on a valid unfinished path."""
    return _node(trie, prefix).name


def max_name_steps(trie: NameTrie) -> int:
    """Return the longest terminated path, without a fixed name-length cap."""
    pending = [(trie, 0)]
    longest = 0
    while pending:
        node, depth = pending.pop()
        longest = max(longest, depth)
        pending.extend((child, depth + 1) for child in node.children.values())
    return longest
