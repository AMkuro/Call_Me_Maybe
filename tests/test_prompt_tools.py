"""Verify the two OpenCode tool boundaries without loading a model."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.prompt_tools import call_tool


class PromptToolTests(unittest.TestCase):
    """One workspace owns one immutable candidate and a fixed evaluator."""

    def prepare(self, root: Path) -> None:
        """Write trusted workflow settings, separate from the LLM arguments."""
        config = {'store': str(root / 'candidates'),
                  'fixture': str(Path('tests/fixtures/prompt_development.json')
                                 .resolve()),
                  'output': str(root / 'editor.json'),
                  'provenance': str(root / 'provenance.json'),
                  'fingerprint': 'a' * 64,
                  'code_sha': subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], text=True).strip()}
        (root / 'tool-config.json').write_text(json.dumps(config))

    def test_one_candidate_per_workspace_and_no_extra_fields(self) -> None:
        """A tool cannot turn prompt data into file or command authority."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.prepare(root)
            prompt = {'name_instructions': 'Choose exact name.',
                      'parameter_instructions': 'Extract input values.'}
            first = call_tool(root, 'write_prompt', prompt)
            self.assertEqual(first, call_tool(root, 'write_prompt', prompt))
            with self.assertRaises(ValueError):
                call_tool(root, 'write_prompt', dict(prompt, command='shell'))
            with self.assertRaises(ValueError):
                call_tool(root, 'write_prompt', dict(
                    prompt, name_instructions='Another candidate.'))
            with self.assertRaises(ValueError):
                call_tool(root, 'run', {'fixture': '/tmp/other.json'})
            result = json.loads((root / 'editor.json').read_text())
            self.assertFalse(result['finished'])
            self.assertEqual(result['candidate_id'], first['candidate_id'])

    def test_run_requires_a_saved_candidate(self) -> None:
        """Missing prompt data fails before model initialization."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.prepare(root)
            with self.assertRaises(ValueError):
                call_tool(root, 'run', {})


if __name__ == '__main__':
    unittest.main()
