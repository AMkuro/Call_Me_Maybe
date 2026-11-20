"""A token-level tree of the function names."""

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType

from orderedsets import OrderedSet
from pydantic import (
    BaseModel, ConfigDict, PrivateAttr, field_serializer, field_validator,
    model_validator,
)

from .errors import PromptError, StartupError
from .model_port import EOS, Model

TokenPath = tuple[int, ...]


class NameTrie(BaseModel):
    """The full token paths of the function names and the index to follow.

    Reaching a full path completes that name. Validation checks the paths
    and only then builds the private index (each valid prefix -> the
    tokens that may follow it, and each full path -> its name), so a
    NameTrie exists only in its checked, finished form. Every stored
    container is a read-only copy owned by this object.

    Attributes:
        paths: Each name's full path, the tokens of name + EOS text,
            read-only, in name order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    paths: Mapping[str, TokenPath]
    _next: Mapping[TokenPath, tuple[int, ...]] = PrivateAttr()
    _names: Mapping[TokenPath, str] = PrivateAttr()

    def __init__(self, names: Collection[str], model: Model) -> None:
        """Build the tree of ``names`` with ``model``'s tokenizer.

        Each name is encoded with EOS appended, in the given order.

        Args:
            names: The function names.
            model: The loaded model, used only to encode.

        Raises:
            StartupError: A name is repeated, encoding fails, or the paths
                are ambiguous (see _index_checked_paths).
        """
        try:
            paths = {name: tuple(model.encode(name + EOS)) for name in names}
            if len(paths) != len(names):
                raise ValueError("function names are not unique")
            super().__init__(paths=paths)
        except (PromptError, ValueError) as e:
            raise StartupError(f"cannot build name tree: {e}") from e

    @field_validator("paths")
    @classmethod
    def _read_only(
        cls, paths: Mapping[str, TokenPath],
    ) -> Mapping[str, TokenPath]:
        """Wrap the new dict Pydantic built, so no outside alias exists.

        Args:
            paths: The validated paths.

        Returns:
            A read-only view of ``paths``.
        """
        return MappingProxyType(paths)

    @field_serializer("paths")
    def _plain(self, paths: Mapping[str, TokenPath]) -> dict[str, TokenPath]:
        """Serialize the read-only view as an ordinary dict.

        Args:
            paths: The stored paths.

        Returns:
            A dict with the same items.
        """
        return dict(paths)

    @model_validator(mode="after")
    def _index_checked_paths(self) -> "NameTrie":
        """Reject ambiguous paths, then index them in name order.

        Full paths must be non-empty, distinct, and never a proper prefix
        of another full path; otherwise reaching a path could not tell that
        a name is complete. Then every full path is a leaf with a name,
        every shorter prefix continues and has no name, and no prefix is a
        dead end.

        Returns:
            This tree, with its index built.

        Raises:
            ValueError: There is no name, or a path is empty, repeated or a
                proper prefix of another. Pydantic reports it as a
                ValidationError.
        """
        if not self.paths:
            raise ValueError("a name tree needs at least one name")
        following: defaultdict[TokenPath, OrderedSet[int]] = defaultdict(
            OrderedSet)
        names: dict[TokenPath, str] = {}
        for name, path in self.paths.items():
            if not path:
                raise ValueError(f"{name!r} has an empty token path")
            if path in names:
                raise ValueError(f"{name!r} and {names[path]!r} have the "
                                 "same token path")
            names[path] = name
            for prefix_length, next_token in enumerate(path):
                prefix = path[:prefix_length]
                following[prefix].add(next_token)
        for path, name in names.items():
            if path in following:  # membership does not create an entry
                raise ValueError(f"the token path of {name!r} is a prefix "
                                 "of another name's path")
            following[path] = OrderedSet()
        self._next = MappingProxyType(
            {prefix: tuple(tokens) for prefix, tokens in following.items()})
        self._names = MappingProxyType(names)
        return self

    @property
    def names(self) -> frozenset[str]:
        """The names reachable through the checked index."""
        return frozenset(self._names.values())

    def next_tokens(self, prefix: Sequence[int]) -> list[int]:
        """Return the token IDs that may follow ``prefix``; none at a leaf.

        Args:
            prefix: The name's token IDs so far.

        Returns:
            A new list of the allowed token IDs, in name order.

        Raises:
            PromptError: ``prefix`` is not a path of the tree.
        """
        return list(self._next[self._known_prefix(prefix)])

    def leaf_name(self, prefix: Sequence[int]) -> str | None:
        """Return the name at a leaf, or None on a valid unfinished path.

        Args:
            prefix: The name's token IDs so far.

        Returns:
            The completed name, or None.

        Raises:
            PromptError: ``prefix`` is not a path of the tree.
        """
        return self._names.get(self._known_prefix(prefix))

    def _known_prefix(self, prefix: Sequence[int]) -> TokenPath:
        """Return ``prefix`` as an index key; an unknown path is an error.

        Args:
            prefix: Token IDs.

        Returns:
            ``prefix`` as a tuple.

        Raises:
            PromptError: ``prefix`` is not a path of the tree.
        """
        key = tuple(prefix)
        if key not in self._next:
            raise PromptError("invalid token path in name tree")
        return key
