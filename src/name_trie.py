"""A token-level tree of the function names (ADR-0011)."""

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from pydantic import (
    BaseModel, ConfigDict, PrivateAttr, field_serializer, field_validator,
    model_validator,
)

from .errors import PromptError, StartupError

TokenPath = tuple[int, ...]


class NameTrie(BaseModel):
    """The token paths of the function names and the index to follow them.

    ``paths`` maps each name to its token IDs, ending with the ``eos`` ID.
    Validation checks every path and only then builds the private index
    (each valid prefix -> the tokens that may follow it, and each full path
    -> its name), so a NameTrie exists only in its checked, finished form.
    Every stored container is a read-only copy owned by this object.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    eos: int
    paths: Mapping[str, TokenPath]
    _next: Mapping[TokenPath, tuple[int, ...]] = PrivateAttr()
    _names: Mapping[TokenPath, str] = PrivateAttr()

    @field_validator("paths")
    @classmethod
    def _read_only(
        cls, paths: Mapping[str, TokenPath],
    ) -> Mapping[str, TokenPath]:
        """Wrap the new dict Pydantic built, so no outside alias exists."""
        return MappingProxyType(paths)

    @field_serializer("paths")
    def _plain(self, paths: Mapping[str, TokenPath]) -> dict[str, TokenPath]:
        """Serialize the read-only view as an ordinary dict."""
        return dict(paths)

    @model_validator(mode="after")
    def _index_checked_paths(self) -> "NameTrie":
        """Reject bad paths, then index them in name order.

        A path is name tokens then EOS, with EOS nowhere else, so no full
        path is a prefix of another: every full path is a leaf with a name,
        every shorter prefix continues and has no name, and no prefix is a
        dead end. Distinct paths keep two names from sharing one leaf.
        """
        if not self.paths:
            raise ValueError("a name tree needs at least one name")
        following: dict[TokenPath, dict[int, None]] = {}
        names: dict[TokenPath, str] = {}
        for name, path in self.paths.items():
            if len(path) < 2 or path[-1] != self.eos or self.eos in path[:-1]:
                raise ValueError(f"{name!r}: {list(path)} is not name "
                                 f"tokens followed by EOS {self.eos}")
            if path in names:
                raise ValueError(f"{name!r} and {names[path]!r} have the "
                                 "same token path")
            names[path] = name
            for end, token in enumerate(path):
                following.setdefault(path[:end], {})[token] = None
            following[path] = {}
        self._next = MappingProxyType(
            {prefix: tuple(tokens) for prefix, tokens in following.items()})
        self._names = MappingProxyType(names)
        return self

    @property
    def names(self) -> frozenset[str]:
        """Return the names reachable through the checked index."""
        return frozenset(self._names.values())


def build_name_trie(
    names: Sequence[str], encode: Callable[[str], list[int]], eos: str,
) -> NameTrie:
    """Encode each name plus the caller's ``eos`` text into a NameTrie.

    Tokenizer facts are read here: ``eos`` must be one token, because the
    tree ends each name with one EOS edge and selection adds one token per
    step. NameTrie checks the paths. Names are unique (loader); a repeated
    name is still rejected rather than silently merged.
    """
    try:
        eos_ids = encode(eos)
        if len(eos_ids) != 1:
            raise ValueError(f"{eos!r} is not one token: {eos_ids}")
        paths = {name: tuple(encode(name + eos)) for name in names}
        if len(paths) != len(names):
            raise ValueError("function names are not unique")
        return NameTrie(eos=eos_ids[0], paths=paths)
    except (PromptError, ValueError) as e:  # ValidationError is a ValueError
        raise StartupError(f"cannot build name tree: {e}") from e


def _known_prefix(trie: NameTrie, prefix: Sequence[int]) -> TokenPath:
    """Return ``prefix`` as an index key; an unknown path is an error."""
    key = tuple(prefix)
    if key not in trie._next:
        raise PromptError("invalid token path in name tree")
    return key


def next_tokens(trie: NameTrie, prefix: Sequence[int]) -> list[int]:
    """Return the token ids that may follow ``prefix`` (empty at a leaf)."""
    return list(trie._next[_known_prefix(trie, prefix)])


def leaf_name(trie: NameTrie, prefix: Sequence[int]) -> str | None:
    """Return the name at a leaf, or None on a valid unfinished path."""
    return trie._names.get(_known_prefix(trie, prefix))


def max_name_steps(trie: NameTrie) -> int:
    """Return the longest terminated path, without a fixed name-length cap."""
    return max(map(len, trie._names))
