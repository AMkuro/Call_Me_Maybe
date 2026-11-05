"""Publish each cumulative result array without exposing partial JSON."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, ConfigDict, JsonValue, PrivateAttr

from src.validation import FunctionCall


def save_results(path: Path, results: list[dict[str, JsonValue]]) -> None:
    """Atomically replace output with a complete UTF-8 JSON array.

    A failed write or replacement leaves the previously published file intact.
    The temporary file lives beside the output to keep replacement on the same
    filesystem. Cleanup also runs on interruption; removal is best effort if
    filesystem permissions change during a failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                results, stream, ensure_ascii=False, allow_nan=False, indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


class ResultStore(BaseModel):
    """Own an output path and the results successfully published there."""

    model_config = ConfigDict(strict=True, extra="forbid")

    path: Path
    _results: list[dict[str, JsonValue]] = PrivateAttr(default_factory=list)

    def initialize(self, inputs: tuple[Path, Path]) -> None:
        """Protect input paths, then reset both published and owned results."""
        stage = "output path validation"
        try:
            self._check_inputs(inputs)
            stage = f"initialize output {self.path}"
            save_results(self.path, [])
            self._results = []
        except KeyboardInterrupt as exc:
            raise KeyboardInterrupt(stage) from exc
        except Exception as exc:
            raise RuntimeError(
                f"{stage}: {type(exc).__name__}: {exc}"
            ) from exc

    def append(self, result: FunctionCall, index: int) -> None:
        """Publish a validated result, advancing state only after success."""
        stage = f"input {index} save output {self.path}"
        try:
            results = [*self._results, result.model_dump()]
            save_results(self.path, results)
            self._results = results
        except KeyboardInterrupt as exc:
            raise KeyboardInterrupt(stage) from exc
        except Exception as exc:
            raise RuntimeError(
                f"{stage}: {type(exc).__name__}: {exc}"
            ) from exc

    def _check_inputs(self, inputs: tuple[Path, Path]) -> None:
        """Reject direct, relative, symlink, and hardlink input aliases."""
        for source in inputs:
            if self.path.resolve() == source.resolve() or (
                self.path.exists() and source.exists()
                and self.path.samefile(source)
            ):
                raise ValueError(f"output {self.path} aliases input {source}")
