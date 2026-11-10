"""Use function names as keys without changing order or selection behavior."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.errors import StartupError
from src.loader import load_functions
from src.model_port import Model
from src.name_trie import build_name_trie
from src.pipeline import select_name
from src.prompt import build_name_prompt
from src.schema import FunctionDef
from tools import measure_application
from test_constraints import encode_name


def function(name: str, description: str) -> FunctionDef:
    """Create independently named function definitions."""
    return FunctionDef.model_validate({
        'name': name, 'description': description,
        'parameters': {'value': {'type': 'number'}},
        'returns': {'type': 'number'},
    })


class FunctionRegistryTests(unittest.TestCase):
    """Registry lookup and enumeration preserve the input's meaning."""

    def test_loader_indexes_by_name_and_preserves_input_order(self) -> None:
        """Out-of-alphabetical-order input stays in its original order."""
        first = function('zeta', 'First listed action.')
        second = function('alpha', 'Second listed action.')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'functions.json'
            path.write_text(json.dumps([first.model_dump(),
                                        second.model_dump()]),
                            encoding='utf-8')
            functions = load_functions(path)
        self.assertIsInstance(functions, dict)
        self.assertEqual(list(functions), ['zeta', 'alpha'])
        self.assertEqual(functions['alpha'], second)

    def test_prompt_enumerates_definitions_in_registry_order(self) -> None:
        """Dictionary iteration must present definitions rather than keys."""
        first = function('zeta', 'First listed action.')
        second = function('alpha', 'Second listed action.')
        prompt = build_name_prompt('choose an action',
                                   {'zeta': first, 'alpha': second})
        self.assertIn(first.description, prompt)
        self.assertIn(second.description, prompt)
        self.assertLess(prompt.index('zeta'), prompt.index('alpha'))

    def test_fixed_logits_select_the_original_definition(self) -> None:
        """The selected name directly retrieves its original schema object."""
        first = function('z', 'First action.')
        second = function('a', 'Second action.')
        functions = {'z': first, 'a': second}
        trie = build_name_trie(list(functions), encode_name)
        model = object.__new__(Model)
        scores = [0.0] * 256
        scores[ord('a')] = 1.0
        with patch.object(Model, 'encode', return_value=[255]), \
                patch.object(Model, 'logits', return_value=scores):
            selected = select_name('choose second', functions, model, trie)
        self.assertIs(selected, second)

    def test_measurement_rejects_duplicate_names_before_model(self) -> None:
        """Fixtures must not silently overwrite same-name definitions."""
        fn = function('same', 'First action.').model_dump()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = root / 'cases.json'
            cases.write_text(json.dumps({'functions': [fn, fn],
                                         'cases': []}), encoding='utf-8')
            output = root / 'result.json'
            stderr = io.StringIO()
            with patch('sys.argv', ['measure', '--cases', str(cases),
                                    '--output', str(output)]), \
                    patch.object(measure_application, 'Model',
                                 side_effect=StartupError('model sentinel')) \
                    as model, contextlib.redirect_stderr(stderr):
                self.assertEqual(measure_application.main(), 1)
            model.assert_not_called()
            self.assertIn('duplicate function names', stderr.getvalue())
            self.assertFalse(output.exists())
