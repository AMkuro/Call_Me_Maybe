"""Verify the two OpenCode tool boundaries without loading a model."""

import json
from pathlib import Path
import subprocess
import tempfile
import shutil
import os
from typing import Mapping, Sequence
from unittest.mock import patch
import unittest

from tools.prompt_tools import call_tool
from tools.prompt_candidates import Candidate, save_candidate
from tools.prompt_experiment import editor_session


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

    def test_harness_uses_the_servers_exact_pinned_model_id(self) -> None:
        """The native server rejects the unversioned name before inference."""
        workspaces: list[Path] = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store = root / 'candidates'
            candidate_id = save_candidate(store, Candidate(
                name_instructions='select', parameter_instructions='extract'))
            fixture = root / 'fixture.json'
            fixture.write_text('{}')
            revision = 'b' * 40
            provenance = root / 'provenance.json'
            provenance.write_text(json.dumps({
                'code_sha': 'a' * 40,
                'runtime': {'models': {'Qwen/Qwen3.5-9B': {
                    'revision': revision}}},
            }))
            context = root / 'context.json'
            context.write_text(json.dumps({'parent_id': candidate_id}))
            job = root / 'job'
            job.mkdir()

            def inspect_config(command: Sequence[str], log: Path,
                               env: Mapping[str, str],
                               timeout: int = 1200) -> int:
                config_path = Path(env['OPENCODE_CONFIG'])
                workspaces.append(config_path.parent)
                config = json.loads(config_path.read_text())
                model_id = 'Qwen/Qwen3.5-9B@' + revision
                self.assertEqual(config['model'], 'local/' + model_id)
                self.assertIn(model_id, config['provider']['local']['models'])
                return 1

            try:
                with patch('tools.prompt_experiment.run_logged',
                           inspect_config):
                    editor_session(job, context, store, fixture,
                                   'Qwen/Qwen3.5-9B', dict(
                                       os.environ,
                                       CMM_PROMPT_PROVENANCE=str(provenance)))
            finally:
                for workspace in workspaces:
                    shutil.rmtree(workspace)


if __name__ == '__main__':
    unittest.main()
