"""Check a documented development subset of Subject v1.7, without inference."""

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib

FORBIDDEN = {
    'torch', 'pytorch', 'transformers', 'huggingface_hub', 'dspy', 'outlines',
}
REQUIRED = (
    'pyproject.toml', 'uv.lock', 'README.md', 'Makefile', '.gitignore',
    'llm_sdk/llm_sdk/__init__.py',
)


def check_dependencies(dependencies: list[str]) -> list[str]:
    """Parse PEP 508 requirements and normalize PEP 503 project names."""
    errors: list[str] = []
    packages: set[str] = set()
    for item in dependencies:
        try:
            packages.add(canonicalize_name(Requirement(item).name))
        except InvalidRequirement as exc:
            errors.append(f'invalid runtime dependency: {exc}')
    for name in ('numpy', 'pydantic'):
        if name not in packages:
            errors.append(f'missing runtime dependency: {name}')
    forbidden = {canonicalize_name(name) for name in FORBIDDEN}
    for name in sorted(packages & forbidden):
        errors.append(f'forbidden direct runtime dependency: {name}')
    return errors


def check_source(text: str, filename: str) -> list[str]:
    """Reject forbidden direct imports and private SDK import paths.

    This is a static import check, not proof that all possible SDK accesses
    are public. Aliased/dynamic attribute accesses still require code review.
    """
    errors: list[str] = []
    try:
        tree = ast.parse(text, filename=filename)
    except SyntaxError as exc:
        return [f'{filename}: {exc}']
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or '']
            if names[0].split('.')[0] == 'llm_sdk':
                names += [f'{names[0]}.{alias.name}' for alias in node.names]
        else:
            continue
        for name in names:
            parts = name.split('.')
            if parts[0] in FORBIDDEN or (
                parts[0] == 'llm_sdk'
                and any(part.startswith('_') for part in parts[1:])
            ):
                errors.append(f'{filename}:{node.lineno}: forbidden {name}')
    return errors


def check_inputs(root: Path) -> list[str]:
    """Validate demonstration file structure, independent of model output."""
    errors: list[str] = []
    for filename in ('function_calling_tests.json',
                     'functions_definition.json'):
        path = root / 'data/input' / filename
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data, list):
                raise ValueError('expected a JSON array')
            names: set[str] = set()
            for record in data:
                if not isinstance(record, dict):
                    raise ValueError('expected object records')
                if filename == 'function_calling_tests.json':
                    if set(record) != {'prompt'} or not isinstance(
                        record['prompt'], str
                    ):
                        raise ValueError('expected exactly one string prompt')
                else:
                    if set(record) != {
                        'name', 'description', 'parameters', 'returns'
                    }:
                        raise ValueError('invalid function definition fields')
                    name = record['name']
                    if not isinstance(name, str) or not name:
                        raise ValueError('expected a nonempty function name')
                    if name in names:
                        raise ValueError(f'duplicate function name: {name}')
                    names.add(name)
                    if not isinstance(record['description'], str):
                        raise ValueError('description must be a string')
                    if not isinstance(record['parameters'], dict):
                        raise ValueError('parameters must be an object')
                    for schema in [record['returns'],
                                   *record['parameters'].values()]:
                        if not isinstance(schema, dict) or not isinstance(
                            schema.get('type'), str
                        ):
                            raise ValueError('each schema needs a string type')
        except (OSError, ValueError) as exc:
            errors.append(f'{path}: {exc}')
    return errors


def check_layout(
    root: Path, tracked: list[str], submission: bool,
) -> tuple[list[str], list[str]]:
    """Check common files and report the PoC entry-point gap."""
    errors = [
        f'missing {path}' for path in REQUIRED
        if not (root / path).is_file()
    ]
    pending: list[str] = []
    if not (root / 'src/__main__.py').is_file():
        message = 'missing src/__main__.py: PoC is not a submission'
        (errors if submission else pending).append(message)
    for filename in tracked:
        if 'output' in Path(filename).parts:
            errors.append(f'generated output is tracked: {filename}')
    return errors, pending


def main() -> int:
    """Run local checks and clearly distinguish them from submission review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', action='store_true')
    args = parser.parse_args()
    root = Path('.')
    tracked = subprocess.check_output(
        ['git', 'ls-files', '-z'], text=True,
    ).rstrip('\0').split('\0')
    errors, pending = check_layout(root, tracked, args.submission)
    errors += check_inputs(root)
    config = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))
    dependencies = config['project'].get('dependencies', [])
    errors += check_dependencies(dependencies)
    for filename in tracked:
        if filename.endswith('.py') and not filename.startswith('llm_sdk/'):
            errors += check_source(Path(filename).read_text(encoding='utf-8'),
                                   filename)
    readme = Path('README.md').read_text(encoding='utf-8')
    if not readme.startswith('*This project has been created as part of '
                             'the 42 curriculum by '):
        errors.append('README first line must use the Subject attribution')
    for heading in ('Description', 'Instructions', 'Resources'):
        if f'## {heading}' not in readme:
            errors.append(f'README missing {heading} section')
    for message in pending:
        print(f'PENDING SUBJECT REQUIREMENT: {message}')
    print('Scope: layout, declared runtime dependencies, sample inputs, '
          'static imports. No model or full Subject certification.')
    for error in errors:
        print(f'ERROR: {error}')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
