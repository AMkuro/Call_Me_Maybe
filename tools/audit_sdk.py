"""Expose exact known upstream diagnostics and fail if the baseline changes."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

FLAGS = [
    '--warn-return-any', '--warn-unused-ignores', '--ignore-missing-imports',
    '--disallow-untyped-defs', '--check-untyped-defs',
]


def snapshot() -> dict[str, object]:
    """Run both linters and fingerprint every distributed SDK file."""
    files = subprocess.check_output(
        ['git', 'ls-files', '-z', 'llm_sdk'], text=True,
    ).rstrip('\0').split('\0')
    result: dict[str, object] = {
        'files': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                  for name in files},
    }
    commands = {
        'flake8': ['flake8', '--isolated', 'llm_sdk'],
        'mypy': ['mypy', '--config-file', '.github/ci/sdk-mypy.ini',
                 'llm_sdk', *FLAGS],
    }
    for name, arguments in commands.items():
        completed = subprocess.run(
            [sys.executable, '-m', *arguments], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        print(f'Upstream SDK {name} exit={completed.returncode}:')
        print(completed.stdout, end='')
        result[name] = {'returncode': completed.returncode,
                        'output': completed.stdout}
    return result


def main() -> int:
    """Compare without suppressing new errors or rewriting the baseline."""
    expected = json.loads(Path('.github/ci/sdk-baseline.json').read_text(
        encoding='utf-8',
    ))
    actual = snapshot()
    if actual != expected:
        print('ERROR: SDK files or diagnostics differ from the reviewed '
              'baseline. Investigate; do not auto-refresh it.')
        return 1
    print('KNOWN UPSTREAM DEBT: 11 flake8 + 4 mypy errors match the exact '
          'baseline. Full Subject lint is still failing; SDK was not edited.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
