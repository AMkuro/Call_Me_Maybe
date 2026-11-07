"""校舎PC調査スクリプトの検査。SDK・モデル・ネットワークは使わない。"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools import campus_survey as survey

TOKENS = {'secret-host', 'alice'}


def bench_args(**overrides: object) -> argparse.Namespace:
    """benchmark 用の既定引数を作る。"""
    values: dict[str, object] = dict(
        lengths=[16, 64], repeats=2, gen_tokens=3, budget=60, threads=None,
        allow_download=False, hf_home=None, debug=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class AnonymousIdTests(unittest.TestCase):
    """匿名IDの再現性と、識別情報そのものの拒否。"""

    def test_hmac_is_stable_and_key_dependent(self) -> None:
        """同じ鍵・hostname なら同じID、鍵が違えば別ID。"""
        key = 'k' * 16
        first = survey.hmac_pc_id(key, 'c1r2p3.42tokyo.jp')
        self.assertEqual(first, survey.hmac_pc_id(key, 'C1R2P3'))
        self.assertNotEqual(first, survey.hmac_pc_id('j' * 16, 'c1r2p3'))
        self.assertNotEqual(first, survey.hmac_pc_id(key, 'c1r2p4'))
        self.assertRegex(first, r'^h[0-9a-f]{12}$')
        self.assertTrue(survey.PC_ID.fullmatch(first))
        # 鍵なしの単純ハッシュとは一致しない。
        plain = hashlib.sha256(b'c1r2p3').hexdigest()[:12]
        self.assertNotEqual(first[1:], plain)

    def test_hmac_rejects_short_key_and_empty_host(self) -> None:
        """推測しやすい短い鍵と空の hostname は拒否する。"""
        with self.assertRaises(ValueError):
            survey.hmac_pc_id('short', 'c1r2p3')
        with self.assertRaises(ValueError):
            survey.hmac_pc_id('k' * 16, ' ')

    def test_explicit_id_rejects_identifiers(self) -> None:
        """hostname・username を含むIDや不正な文字種を拒否する。"""
        with patch.object(survey, 'sensitive_tokens', return_value=TOKENS):
            self.assertEqual(survey.validate_pc_id('pc-07'), 'pc-07')
            for bad in ('secret-host', 'x-alice-1', 'PC07', 'a/b', 'ab',
                        '../x', 'a' * 33):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    survey.validate_pc_id(bad)

    def test_hmac_key_is_not_stored(self) -> None:
        """鍵は環境変数か端末入力から読む。非対話で鍵が無ければ失敗する。"""
        with patch.dict('os.environ', {survey.HMAC_ENV: 'k' * 16}):
            self.assertEqual(survey.read_hmac_key(), 'k' * 16)
        with patch.dict('os.environ', {}, clear=True), \
                patch.object(sys, 'stdin', io.StringIO()), \
                self.assertRaises(ValueError):
            survey.read_hmac_key()

    def test_meta_validation(self) -> None:
        """メタデータはキー・値とも安全な形式だけ受け付ける。"""
        self.assertEqual(survey.parse_meta(['floor=2f', 'row=r3']),
                         {'meta.floor': '2f', 'meta.row': 'r3'})
        for bad in ('floor', 'Floor=2', 'floor=', 'floor=/home/x',
                    'floor=a"b', 'x=' + 'a' * 65):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                survey.parse_meta([bad])


class PublicReportTests(unittest.TestCase):
    """許可リスト方式の出力と個人情報の混入防止。"""

    def setUp(self) -> None:
        """端末固有の文字列をテスト用の値に固定する。"""
        patcher = patch.object(survey, 'sensitive_tokens',
                               return_value=TOKENS)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_unknown_keys_are_refused(self) -> None:
        """許可されていないキーは出力しない。"""
        self.assertEqual(survey.render({'cpu.model': 'X'}),
                         'cpu.model = X\n')
        for key in ('hostname', 'env.home', 'user', 'cpu', 'meta.Bad'):
            with self.subTest(key=key), self.assertRaises(survey.LeakError):
                survey.render({key: 'x'})

    def test_unsafe_values_are_replaced(self) -> None:
        """パスや引用符を含む値は値ごと未取得扱いにする。"""
        for value in ('/home/alice/x', '~/x', 'a"b', 'x' * 121, "it's"):
            with self.subTest(value=value):
                self.assertEqual(
                    survey.render({'os.name': value}),
                    'os.name = unavailable (unsafe_value)\n')

    def test_identifiers_and_addresses_are_refused(self) -> None:
        """hostname・username・IP・MAC・メールの混入を拒否する。"""
        for value in ('SECRET-HOST', 'by alice', '10.0.0.1',
                      'aa-bb-cc-dd-ee-ff', 'a@b.jp'):
            with self.subTest(value=value), \
                    self.assertRaises(survey.LeakError):
                survey.render({'os.name': value})
        # 部分一致（malice など）は誤検出しない。
        self.assertIn('malice', survey.render({'os.name': 'malice'}))
        self.assertIn('2.9.1', survey.render({'deps.torch': '2.9.1+cpu'}))

    def test_real_inspect_has_no_identifiers(self) -> None:
        """この端末での実際の inspect 出力に識別情報が含まれない。"""
        patch.stopall()
        text = survey.render(survey.inspect())
        for token in survey.sensitive_tokens():
            self.assertNotIn(token, text.lower())
        self.assertNotIn(str(Path.home()), text)
        self.assertNotIn('/', text)

    def test_main_refuses_leaking_report(self) -> None:
        """混入を検出したら何も書かずに終了コード 2 を返す。"""
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(survey, 'inspect',
                             return_value={'os.name': 'alice'}), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = survey.main(['--pc-id', 'pc-01', '--write',
                                '--output-dir', directory])
            self.assertEqual(code, 2)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_write_appends_runs(self) -> None:
        """同じ端末の再測定は同じファイルに run_id 付きで追記される。"""
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(survey, 'inspect',
                             return_value={'os.name': 'Ubuntu'}), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            for _ in range(2):
                self.assertEqual(survey.main(
                    ['--pc-id', 'pc-01', '--meta', 'floor=2f', '--write',
                     '--output-dir', directory]), 0)
            text = (Path(directory) / 'pc-pc-01.txt').read_text()
        self.assertEqual(text.count('=== run '), 2)
        self.assertEqual(text.count('meta.floor = 2f'), 2)
        self.assertEqual(text.count('# Call Me Maybe'), 1)


class InspectRobustnessTests(unittest.TestCase):
    """情報源が欠けていても安全に未取得として扱う。"""

    def test_reason_vocabulary(self) -> None:
        """例外は固定の理由語に変換され、メッセージは残らない。"""
        cases = ((FileNotFoundError('/home/alice'), 'not_found'),
                 (PermissionError('x'), 'permission_denied'),
                 (ValueError('x'), 'parse_error'),
                 (ImportError('x'), 'not_installed'),
                 (NotImplementedError(), 'not_supported'),
                 (RuntimeError('/home/alice'), 'error'))
        for exc, reason in cases:
            with self.subTest(exc=exc):
                def fail(error: Exception = exc) -> str:
                    raise error
                self.assertEqual(survey.safe(fail),
                                 f'unavailable ({reason})')

    def test_all_sources_missing(self) -> None:
        """ファイルも外部コマンドも無い環境でも inspect が完走する。"""
        def missing(path: object) -> str:
            raise FileNotFoundError(path)

        with patch.object(survey, 'read', side_effect=missing), \
                patch.object(shutil, 'which', return_value=None), \
                patch.object(os, 'sched_getaffinity',
                             side_effect=PermissionError, create=True), \
                patch.object(os, 'getloadavg', side_effect=OSError):
            report = survey.inspect()
        self.assertEqual(report['cpu.model'], 'unavailable (not_found)')
        self.assertEqual(report['cpu.affinity_cpus'],
                         'unavailable (permission_denied)')
        self.assertEqual(report['uv.version'], 'unavailable (not_installed)')
        self.assertEqual(report['memory.total_gib'],
                         'unavailable (not_found)')
        survey.render(report)

    def test_cgroup_limits(self) -> None:
        """cgroup v2 の階層から最も厳しい制約を選ぶ。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / 'user.slice' / 'app.scope'
            leaf.mkdir(parents=True)
            (root / 'user.slice' / 'cpu.max').write_text('200000 100000\n')
            (leaf / 'cpu.max').write_text('max 100000\n')
            (leaf / 'memory.max').write_text(str(4 * 2**30))
            (root / 'user.slice' / 'memory.max').write_text('max')
            dirs = [leaf, root / 'user.slice', root]
            with patch.object(survey, 'cgroup_dirs', return_value=dirs):
                self.assertEqual(survey.cgroup_cpu_quota(), '2.00')
                self.assertEqual(survey.cgroup_memory('memory.max'), '4.0')
                self.assertEqual(survey.safe(
                    lambda: survey.cgroup_memory('memory.high')),
                    'unavailable (not_found)')

    def test_cgroup_v1_is_not_supported(self) -> None:
        """cgroup v2 でなければ not_supported とする。"""
        with patch.object(survey, 'read', return_value='4:cpu:/x'):
            self.assertEqual(survey.safe(survey.cgroup_cpu_quota),
                             'unavailable (not_supported)')

    def test_model_cache_detection(self) -> None:
        """完全なスナップショットだけを found とし、場所はラベルで返す。"""
        with tempfile.TemporaryDirectory() as directory:
            hf_home = Path(directory)
            snapshot = hf_home / 'hub' / survey.MODEL_DIR / 'snapshots/abc'
            snapshot.mkdir(parents=True)
            (snapshot / 'config.json').write_text('{}')
            with patch.dict('os.environ', {}, clear=True), \
                    patch.object(Path, 'home',
                                 return_value=hf_home / 'nohome'):
                report = survey.model_cache(hf_home)
                self.assertEqual(report['model.cache'], 'partial')
                (snapshot / 'tokenizer.json').write_text('{}')
                (snapshot / 'model.safetensors').write_text('w')
                report = survey.model_cache(hf_home)
        self.assertEqual(report['model.cache'], 'found')
        self.assertEqual(report['model.cache_location'], 'explicit')


class BenchmarkTests(unittest.TestCase):
    """benchmark は明示指定時のみ、ダウンロードなしで動く。"""

    def setUp(self) -> None:
        """SDK を模したモジュールを用意する（本物は import しない）。"""
        self.model = MagicMock()
        self.model.encode.return_value.tolist.return_value = [
            list(range(400))]
        self.model.get_logits_from_input_ids.return_value = [0.0, 2.0, 1.0]
        self.constructor = MagicMock(return_value=self.model)
        self.sdk = types.ModuleType('llm_sdk')
        setattr(self.sdk, 'Small_LLM_Model', self.constructor)
        self.found = ('explicit', Path('hub'), Path('snapshot'))

    def run_bench(self, **overrides: object) -> tuple[survey.Report, int]:
        """依存が import 可能な状態で benchmark を実行する。"""
        with patch.dict(sys.modules, {'llm_sdk': self.sdk}), \
                patch.dict('os.environ', {}, clear=True), \
                patch.object(importlib.util, 'find_spec',
                             return_value=object()), \
                contextlib.redirect_stderr(io.StringIO()):
            result = survey.benchmark(bench_args(**overrides))
            self.environ = dict(os.environ)
        return result

    def test_inspect_never_loads_sdk(self) -> None:
        """既定の inspect では SDK を import もロードもしない。"""
        with patch.dict(sys.modules, {'llm_sdk': self.sdk}), \
                patch.object(survey, 'benchmark') as bench, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(survey.main(['--pc-id', 'pc-01']), 0)
        bench.assert_not_called()
        self.constructor.assert_not_called()

    def test_skip_without_cached_model(self) -> None:
        """モデル未キャッシュならロード前に中止し、終了コード 3。"""
        with patch.object(survey, 'find_model', return_value=None):
            report, code = self.run_bench()
        self.assertEqual(code, 3)
        self.assertEqual(report['bench.status.reason'], 'model_not_cached')
        self.constructor.assert_not_called()

    def test_skip_without_dependencies(self) -> None:
        """torch 等が無ければロード前に中止する。"""
        with patch.object(survey, 'find_model', return_value=self.found), \
                patch.object(importlib.util, 'find_spec',
                             return_value=None), \
                contextlib.redirect_stderr(io.StringIO()):
            report, code = survey.benchmark(bench_args())
        self.assertEqual(code, 3)
        self.assertEqual(report['bench.status.reason'],
                         'sdk_dependencies_missing')

    def test_measurements_offline_on_cpu(self) -> None:
        """オフライン・CPU 指定で測定し、実測と推定を区別して出す。"""
        with patch.object(survey, 'find_model', return_value=self.found):
            report, code = self.run_bench(threads=2)
        self.assertEqual(code, 0)
        self.assertEqual(report['bench.status.result'], 'completed')
        self.constructor.assert_called_once_with(
            survey.MODEL_NAME, device='cpu', trust_remote_code=False)
        self.assertEqual(self.environ['HF_HUB_OFFLINE'], '1')
        self.assertEqual(self.environ['TRANSFORMERS_OFFLINE'], '1')
        self.assertEqual(self.environ['HF_HUB_CACHE'], 'hub')
        self.assertEqual(self.environ['OMP_NUM_THREADS'], '2')
        # 1 + repeats + lengths*repeats + gen_tokens 回の forward
        self.assertEqual(self.model.get_logits_from_input_ids.call_count,
                         1 + 2 + 2 * 2 + 3)
        for key in ('model_load_s', 'first_logits_len_64_s',
                    'repeat_logits_len_64_median_s',
                    'logits_len_16_median_s', 'generate_3_tokens_s'):
            self.assertIn(f'bench.measured.{key}', report)
        self.assertIn('bench.estimated.sample_total_s', report)
        self.assertIn('bench.estimated.assumed_gen_tokens', report)
        self.assertFalse(any(
            key.startswith('bench.measured.') and 'estimated' in key
            for key in report))
        with patch.object(survey, 'sensitive_tokens', return_value=TOKENS):
            survey.render(report)

    def test_download_needs_explicit_flag(self) -> None:
        """--allow-download 時だけオフライン指定を外す。"""
        with patch.object(survey, 'find_model', return_value=None):
            report, code = self.run_bench(allow_download=True,
                                          hf_home=Path('cache'))
        self.assertEqual(code, 0)
        self.assertNotIn('HF_HUB_OFFLINE', self.environ)
        self.assertEqual(self.environ['HF_HOME'], 'cache')
        self.assertEqual(report['bench.config.download_allowed'], 'yes')
        self.assertEqual(report['bench.config.model_cached_before_run'], 'no')

    def test_failure_keeps_only_type(self) -> None:
        """失敗時は段階名と例外型だけを残し、メッセージは保存しない。"""
        self.constructor.side_effect = OSError('/home/alice/.cache secret')
        with patch.object(survey, 'find_model', return_value=self.found):
            report, code = self.run_bench()
        self.assertEqual(code, 1)
        self.assertEqual(report['bench.status.stage'], 'model load')
        self.assertEqual(report['bench.status.error_type'], 'OSError')
        self.assertNotIn('secret', ' '.join(report.values()))

    def test_budget_stops_measurement(self) -> None:
        """上限時間を超えたら partial として停止段階を記録する。"""
        self.model.get_logits_from_input_ids.side_effect = [
            [0.0, 1.0], survey.BudgetExceeded()]
        with patch.object(survey, 'find_model', return_value=self.found):
            report, code = self.run_bench()
        self.assertEqual(code, 1)
        self.assertEqual(report['bench.status.result'], 'partial')
        self.assertEqual(report['bench.status.stage'], 'repeated logits')

    def test_linear_fit(self) -> None:
        """推定に使う一次近似が正しい。"""
        intercept, slope = survey.linear_fit([(10, 2.0), (20, 3.0),
                                              (30, 4.0)])
        self.assertAlmostEqual(intercept, 1.0)
        self.assertAlmostEqual(slope, 0.1)
        with self.assertRaises(ValueError):
            survey.linear_fit([(10, 1.0), (10, 2.0)])


if __name__ == '__main__':
    unittest.main()
