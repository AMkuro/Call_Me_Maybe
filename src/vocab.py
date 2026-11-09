"""The vocabulary: token id <-> token string."""

from pathlib import Path


def load_vocab(path: Path) -> dict[int, str]:
    """Read the vocabulary file into a map from token id to string.

    Effect: reads a file.
    """
    raise NotImplementedError
