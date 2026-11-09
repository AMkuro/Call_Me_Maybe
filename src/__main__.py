"""Entry point: ``uv run python -m src``."""

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the flow of docs/architecture.md section 4; return the exit code.

    Non-zero when a startup step fails or any prompt was skipped.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
