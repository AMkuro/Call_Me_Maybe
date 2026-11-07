"""Exercise CI rejection paths without an SDK import or model inference."""

import json
from pathlib import Path
import tempfile
import unittest

from tools.check_subject import (
    check_dependencies, check_inputs, check_source, check_layout,
)


class SubjectChecksTests(unittest.TestCase):
    """Keep development preflight distinct from submission certification."""

    def test_invalid_input_and_duplicate_functions(self) -> None:
        """Reject malformed input shapes and ambiguous function names."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/input').mkdir(parents=True)
            prompts = root / 'data/input/function_calling_tests.json'
            definitions = root / 'data/input/functions_definition.json'
            prompts.write_text('[{"prompt": 7}]', encoding='utf-8')
            function = dict(name='f', description='d', parameters={},
                            returns={'type': 'string'})
            definitions.write_text(json.dumps([function, function]),
                                   encoding='utf-8')
            errors = check_inputs(root)
            self.assertTrue(any('prompt' in error for error in errors))
            self.assertTrue(any('duplicate' in error for error in errors))
            prompts.write_text('{', encoding='utf-8')
            self.assertTrue(check_inputs(root))

    def test_forbidden_imports_and_private_sdk_imports(self) -> None:
        """Owned code cannot directly import forbidden inference packages."""
        for source in ('import torch', 'from transformers import AutoModel',
                       'from llm_sdk._internal import secret',
                       'from llm_sdk import _private'):
            with self.subTest(source=source):
                self.assertTrue(check_source(source, 'src/example.py'))
        self.assertEqual(check_source(
            'from llm_sdk import Small_LLM_Model', 'src/example.py'), [])

    def test_missing_entrypoint_is_a_submission_failure(self) -> None:
        """Report the absent application without certifying the PoC."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            errors, pending = check_layout(root, [], submission=False)
            self.assertIn('src/__main__.py', ' '.join(pending))
            errors, _ = check_layout(root, [], submission=True)
            self.assertIn('src/__main__.py', ' '.join(errors))

    def test_generated_output_is_rejected(self) -> None:
        """An output committed anywhere in the tree blocks development CI."""
        errors, _ = check_layout(Path('.'),
                                 ['data/output/function_calls.json'], False)
        self.assertTrue(any('output' in error for error in errors))

    def test_requirement_references_and_normalized_names(self) -> None:
        """Recognize names in direct URLs, extras, and PEP 503 equivalents."""
        for dependency in (
            'torch @ https://example.invalid/torch.whl',
            'huggingface.hub>=1', 'huggingface__hub>=1',
            'transformers[torch]>=4',
        ):
            with self.subTest(dependency=dependency):
                self.assertTrue(check_dependencies(
                    ['numpy', 'pydantic', dependency],
                ))
        self.assertEqual(check_dependencies([
            'numpy @ https://example.invalid/numpy.whl',
            'pydantic[extra]>=2',
        ]), [])
        self.assertTrue(check_dependencies(['numpy', 'pydantic', '???']))

    def test_source_syntax_error_is_rejected(self) -> None:
        """Report syntax failures with a useful filename."""
        self.assertTrue(check_source('def broken(', 'src/broken.py'))


if __name__ == '__main__':
    unittest.main()
