"""Export the allowlisted files of one commit into a new directory."""

import argparse
import fnmatch
import subprocess
from pathlib import Path, PurePosixPath


class SubmissionExport:
    """Read the manifest and blobs from the same immutable Git commit."""

    def __init__(self, ref: str) -> None:
        """Pin the commit so branch updates cannot change the export."""
        self.commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options",
             f"{ref}^{{commit}}"],
            text=True,
        ).strip()

    @staticmethod
    def matches(path: str, pattern: str) -> bool:
        """Match nested files and files at the pattern's directory root."""
        return fnmatch.fnmatchcase(path, pattern) or fnmatch.fnmatchcase(
            path, pattern.replace("**/", "")
        )

    def patterns(self) -> list[str]:
        """Read the allowlist from this pinned commit."""
        manifest = subprocess.check_output(
            ["git", "show", f"{self.commit}:submission-files.txt"], text=True
        )
        return [
            line.strip() for line in manifest.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def select(self) -> list[tuple[str, str, str]]:
        """Return ``(path, blob, mode)`` of the allowed files of this commit.

        Symlinks, unsafe paths and manifest entries that match nothing are
        rejected, so the selection is the same for the export and the
        release candidate.
        """
        patterns = self.patterns()
        entries = subprocess.check_output(
            ["git", "ls-tree", "-rz", self.commit]
        ).split(b"\0")
        selected: list[tuple[str, str, str]] = []
        matched: set[str] = set()
        for entry in filter(None, entries):
            metadata, raw_path = entry.split(b"\t", 1)
            mode, kind, blob = metadata.decode().split()
            path = raw_path.decode()
            hits = {p for p in patterns if self.matches(path, p)}
            if not hits:
                continue
            if kind != "blob" or mode not in {"100644", "100755"}:
                raise ValueError(f"Not a regular submission file: {path}")
            if ".." in PurePosixPath(path).parts or path.startswith("/"):
                raise ValueError(f"Unsafe path: {path}")
            matched.update(hits)
            selected.append((path, blob, mode))
        if not patterns or set(patterns) != matched:
            missing = set(patterns) - matched
            raise ValueError(f"Unmatched manifest entries: {missing}")
        return selected

    def export(self, destination: Path) -> None:
        """Copy allowed blobs; refuse existing output and symlinks."""
        selected = self.select()
        destination.mkdir(parents=True, exist_ok=False)
        for path, blob, mode in selected:
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(
                ["git", "cat-file", "blob", blob]
            ))
            target.chmod(0o755 if mode == "100755" else 0o644)
        print(f"Exported {len(selected)} files from {self.commit}")
        print(f"Destination: {destination}")


def main() -> None:
    """Export a commit without changing the checkout, index, or branch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args()
    SubmissionExport(args.ref).export(args.destination)


if __name__ == "__main__":
    main()
