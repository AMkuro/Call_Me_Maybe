"""Regressions for evidence integrity, concurrent publication and stopping."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from collections.abc import Iterator
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from typing import Any

from tools.prompt_candidates import Candidate, save_candidate, score_records
from tools.prompt_runtime import (
    accept_editor_result, evidence_fingerprint, pin_model_cache, has_oom,
)
from tools.prompt_experiment import read_state, search_status
from tools.prompt_experiment import main as experiment_main
from tools.prompt_trial import evaluate
from src.errors import PromptError
from types import SimpleNamespace


class PromptRuntimeTests(unittest.TestCase):
    """Exercise real files and processes, without loading any model."""

    def test_failed_or_stale_editor_cannot_supply_a_trial(self) -> None:
        """A prior finished file never rescues a failing new process."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'editor.json'
            data = {'fingerprint': 'old', 'finished': True,
                    'candidate_id': 'a' * 64,
                    'trial': {'finished': True, 'score': {'total': 11}}}
            path.write_text(json.dumps(data))
            for stamp, code in [('new', 0), ('old', 1)]:
                result = accept_editor_result(path, stamp, code)
                self.assertNotIn('trial', result)
                self.assertEqual(result['status'], 'editor_failed')
            self.assertEqual(accept_editor_result(path, 'old', 0)['status'],
                             'evaluated')
            data['finished'] = False
            path.write_text(json.dumps(data))
            self.assertNotIn('trial', accept_editor_result(path, 'old', 0))

    def test_type_denominators_include_failed_results(self) -> None:
        """Every gold argument counts even if there is no valid output."""
        cases = [{'id': str(i), 'prompt': 'number', 'expected_name': 'num',
                  'expected_parameters': {'v': i}} for i in range(3)]
        rows = [{'id': '0', 'call': {
                    'prompt': 'number', 'name': 'num',
                    'parameters': {'v': 0}}},
                {'id': '1', 'call': {'unexpected': True}}]
        result = score_records(cases, rows)
        self.assertEqual(result['by_type']['number'],
                         {'total': 3, 'correct': 1})
        self.assertEqual(result['errors'], ['1', '2'])

    def test_identical_parallel_candidates_publish_complete_data(self) -> None:
        """Concurrent savers must never read a partial file."""
        candidate = Candidate(name_instructions='select ' * 300000,
                              parameter_instructions='extract ' * 300000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ThreadPoolExecutor(max_workers=8) as workers:
                ids = list(workers.map(
                    lambda _: save_candidate(root, candidate), range(16)))
            self.assertEqual(len(set(ids)), 1)
            self.assertEqual(len(list(root.iterdir())), 1)
            self.assertEqual(Candidate.model_validate_json(
                (root / f'{ids[0]}.json').read_text()), candidate)

    def test_isolated_cache_pins_main_and_checks_missing_weights(self) -> None:
        """A source refs/main update cannot change the experiment's model."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, view = root / 'source', root / 'view'
            repo = source / 'hub/models--Org--Model'
            sha = 'a' * 40
            snapshot = repo / 'snapshots' / sha
            snapshot.mkdir(parents=True)
            for name in ('config.json', 'tokenizer.json',
                         'tokenizer_config.json', 'vocab.json'):
                (snapshot / name).write_text('{}')
            (snapshot / 'model.safetensors').write_bytes(b'fake weight bytes')
            (repo / 'refs').mkdir()
            (repo / 'refs/main').write_text('b' * 40)
            pinned = pin_model_cache(source, view, {'Org/Model': sha})
            self.assertEqual((view / 'hub/models--Org--Model/refs/main')
                             .read_text(), sha)
            self.assertEqual(pinned['Org/Model']['revision'], sha)
            child_env = dict(os.environ, HF_HOME=str(view),
                             HF_HUB_CACHE=str(view / 'hub'),
                             HF_HUB_OFFLINE='1', HF_TOKEN='')
            code = ("from huggingface_hub import hf_hub_download;"
                    "print(hf_hub_download('Org/Model','config.json',"
                    "local_files_only=True))")
            located = subprocess.check_output([sys.executable, '-c', code],
                                              env=child_env, text=True).strip()
            self.assertEqual(Path(located).resolve(),
                             (snapshot / 'config.json').resolve())
            self.assertEqual((repo / 'refs/main').read_text(), 'b' * 40)
            (snapshot / 'model.safetensors').unlink()
            with self.assertRaises(ValueError):
                pin_model_cache(source, root / 'incomplete',
                                {'Org/Model': sha})

    def test_slow_exclusive_writer_does_not_publish_partial_candidate(
        self,
    ) -> None:
        """A second saver cannot mistake an in-progress file for corruption."""
        original = io.open
        entered, release = threading.Event(), threading.Event()

        def slow_open(path: Any, mode: str = 'r', *args: Any,
                      **kwargs: Any) -> Any:
            stream = original(path, mode, *args, **kwargs)
            if mode != 'x':
                return stream

            @contextmanager
            def held() -> Iterator[Any]:
                with stream:
                    entered.set()
                    release.wait(3)
                    yield stream

            return held()

        candidate = Candidate(name_instructions='select',
                              parameter_instructions='extract')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('io.open', slow_open):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    first = pool.submit(save_candidate, root, candidate)
                    entered.wait(0.1)
                    try:
                        second = save_candidate(root, candidate)
                    finally:
                        release.set()
                    self.assertEqual(first.result(), second)

    def test_versions_and_model_revisions_invalidate_evidence(self) -> None:
        """Changed weights and dependencies must invalidate evidence."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt, fixture = root / 'prompt.json', root / 'fixture.json'
            prompt.write_text('{}')
            fixture.write_text('{}')
            runtime = {'models': {'target': 'a'}, 'packages': {'torch': '1'}}
            first = evidence_fingerprint('code', prompt, fixture, runtime)
            for modified in (
                {'models': {'target': 'b'}, 'packages': {'torch': '1'}},
                {'models': {'target': 'a'}, 'packages': {'torch': '2'}},
            ):
                self.assertNotEqual(first, evidence_fingerprint(
                    'code', prompt, fixture, modified))

    def test_heldout_seal_prevents_returning_to_search(self) -> None:
        """Once held-out data are opened, resume keeps the fixed candidate."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            path.write_text(json.dumps({
                'identity': 'old', 'phase': 'heldout',
                'candidate_id': 'a' * 64, 'next_round': 2, 'trials': [],
            }))
            state = read_state(path, 'old')
            self.assertEqual(state['phase'], 'heldout')
            self.assertEqual(state['candidate_id'], 'a' * 64)
            with self.assertRaises(ValueError):
                read_state(path, 'new')
            self.assertEqual(json.loads(path.read_text())['phase'], 'heldout')

    def test_all_failed_editors_are_not_no_improvement(self) -> None:
        """Distinguish failed proposals from unimproved valid trials."""
        self.assertEqual(search_status([{'status': 'editor_failed'}]),
                         'search_failed')
        self.assertEqual(search_status([{'status': 'evaluated'}]),
                         'no_improvement')

    def test_interrupt_stops_both_worker_process_groups(self) -> None:
        """A signal to main must stop and reap worker children."""
        program = '''
import signal,sys
from pathlib import Path
from tools.prompt_runtime import parallel, run_logged, cancel_signal
signal.signal(signal.SIGTERM,cancel_signal)
root=Path(sys.argv[1])
def work(index):
    code=("import os,time;from pathlib import Path;"
          "Path(%r).write_text(str(os.getpid()));time.sleep(40)"
          %str(root/str(index)))
    return run_logged([sys.executable,'-c',code],
                      root/(str(index)+'.log'),dict(__import__('os').environ),50)
try:
    parallel(work,[0,1],2)
except KeyboardInterrupt:
    sys.exit(130)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = subprocess.Popen(
                [sys.executable, '-c', program, directory])
            try:
                deadline = time.monotonic() + 8
                while not all((root / str(i)).exists() for i in range(2)):
                    if (time.monotonic() > deadline
                            or parent.poll() is not None):
                        self.fail('worker fixture did not start')
                    time.sleep(0.02)
                children = [int((root / str(i)).read_text()) for i in range(2)]
                parent.send_signal(signal.SIGTERM)
                self.assertEqual(parent.wait(timeout=8), 130)
                for pid in children:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait()

    def test_native_inputs_equal_fixed_development_fixture(self) -> None:
        """Changed native inputs must not silently use stale gold."""
        fixture = json.loads(Path(
            'tests/fixtures/prompt_development.json').read_text())
        self.assertEqual(fixture['functions'], json.loads(Path(
            'data/input/functions_definition.json').read_text()))
        self.assertEqual([{'prompt': c['prompt']} for c in fixture['cases']],
                         json.loads(Path(
                             'data/input/function_calling_tests.json')
                                    .read_text()))

    def test_selected_candidate_survives_code_update(self) -> None:
        """Revalidate a frozen candidate on new code before opening holdout."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            path.write_text(json.dumps({
                'identity': 'old', 'phase': 'selected',
                'candidate_id': 'a' * 64, 'next_round': 2, 'trials': [],
            }))
            state = read_state(path, 'new')
            self.assertEqual(state['identity'], 'new')
            self.assertEqual(state['phase'], 'selected')
            self.assertEqual(state['candidate_id'], 'a' * 64)

    def test_code_update_does_not_reset_consumed_rounds(self) -> None:
        """A refresh after six proposals leaves only two slots."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            path.write_text(json.dumps({
                'identity': 'old', 'phase': 'search', 'next_round': 3,
                'candidate_id': None, 'trials': [],
            }))
            self.assertEqual(read_state(path, 'new')['next_round'], 3)

    def test_startup_merge_reloads_prompt_constants(self) -> None:
        """Reload imported prompt constants after merging the base branch."""
        with tempfile.TemporaryDirectory() as directory:
            arguments = ['experiment', '--output', directory, '--cache',
                         directory, '--optimizer', 'Qwen/Qwen3.5-9B']
            with patch('sys.argv', arguments), patch(
                'tools.prompt_experiment.synchronize', return_value=True,
            ), patch('tools.prompt_experiment.os.execv',
                     side_effect=SystemExit(75)), patch(
                         'tools.prompt_experiment.stop_children'), patch(
                             'tools.prompt_experiment.signal.signal'):
                with self.assertRaises(SystemExit) as restarted:
                    experiment_main()
                self.assertEqual(restarted.exception.code, 75)

    def test_target_oom_is_execution_failure_not_no_improvement(self) -> None:
        """Real per-case PromptError capture must retain a retryable OOM."""
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'report.json'
            fake = SimpleNamespace(vocab_path=lambda: 'unused',
                                   encode=lambda text: [1])
            error = PromptError('OutOfMemoryError: GPU out of memory')
            with (
                patch('tools.prompt_trial.Model', return_value=fake),
                patch('tools.prompt_trial.load_vocab', return_value={}),
                patch('tools.prompt_trial.build_name_trie', return_value=None),
                patch('tools.prompt_trial.process_prompt', side_effect=error),
            ):
                fixture = Path('tests/fixtures/prompt_development.json')
                trial = evaluate(fixture, report, 'stamp')
            self.assertEqual(trial.get('failure_kind'), 'gpu_oom')
            scored = {'status': 'evaluated', 'trial': trial}
            self.assertEqual(search_status([scored]), 'search_failed')

    def test_optimizer_oom_is_retryable_without_candidate(self) -> None:
        """An editor that could not load still reports a resource failure."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'editor.json'
            path.write_text(json.dumps({
                'fingerprint': 'stamp', 'finished': False,
                'error': 'OutOfMemoryError: GPU out of memory',
            }))
            failed = accept_editor_result(path, 'stamp', 1)
            self.assertTrue(has_oom(failed))
            self.assertNotIn('candidate_id', failed)
            self.assertEqual(search_status([failed]), 'search_failed')


if __name__ == '__main__':
    unittest.main()
