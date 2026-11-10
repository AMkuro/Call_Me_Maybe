# EVO-X3 検証環境（校舎 i7-14700 のSDK負荷に合わせた近似）

このディレクトリは実験用。`develop` から切った `exp/cpu-calibration` に置き、`main` には統合しない。
2 系統を**混ぜない**: CPU 時間評価（元の配布 SDK）と GPU 精度探索（別 venv）。

## 1. CPU 時間評価（元の配布 SDK・CPU のみ）

- 基準: `research/campus/pc-hdad92b5bf4f3.txt`（Dell OptiPlex AIO 7420 / i7-14700 / 15.3GiB）。
- 環境: `~/cmm-cpu-lab/venv`（Git 管理外）。版は `requirements-campus.txt` に固定
  （torch 2.15.0, transformers 5.22.2 ほか。校舎の benchmark レポートと同じ）。
  Python は 3.12.3（校舎は 3.12.10）。モデルは `~/cmm-cpu-lab/hf`（Git 管理外）。
- 方式: KVM ではなく `taskset` + `systemd-run --user`（`MemoryMax=16G`、swap 禁止）。
  理由: ユーザー cgroup に `cpuset` が委譲されておらず、VM はこの目的に対して利点がない。
  GPU は使わない（`CUDA_VISIBLE_DEVICES` を空にし、`device='cpu'`）。
- 採用プロファイル **`phys16-t8`**: CPU 0–15（物理コア 16、SMT 兄弟を除く）、`OMP_NUM_THREADS=8`。

### 校正結果（`runs/` の各 3 回の中央値、秒）

| 項目 | 校舎 | phys16-t8 | 差 |
|---|---|---|---|
| logits 入力16 | 0.1002 | 0.1234 | +23% |
| logits 入力64 | 0.1989 | 0.1819 | -9% |
| logits 入力256 | 0.4864 | 0.4785 | -1.6% |
| 8 トークン逐次生成 | 1.5603 | 1.5478 | -0.8% |
| ピーク RSS (MiB) | 5123 | 5132 | +0.2% |

比較した 6 条件（all32 既定/t20/t16/t12、phys16 既定/t8）のうち、phys16-t8 が入力256・生成の一致と
再現性（3 回のばらつき最小）で最良。ただし **入力16 は全条件で約 +23〜28% 遅い**（短い入力の固定
オーバーヘッドの差と推測。未検証）。`phys16-default`（affinity だけ絞りスレッド数は既定）は
過剰購読で +45〜150% 遅くなるため使わない。

### 限界

- 候補は 6 条件の粗い格子。1 点に合わせ込んだ調整はしていないが、入力16 の差は残る。
- アーキテクチャが違う（Zen 5 / AVX-512 対 Raptor Lake / AVX2）。**VM/cgroup による時間予測を実機性能と
  同一視しない。** 課題の全評価データで300秒以内かは未確認で、達成済みとは言えない。
- メモリ容量を16GiBに制限しても、帯域・キャッシュ・命令実装は変わらない。短いSDK負荷の校正から、
  実アプリの入力長・生成回数での一致は保証しない。同じベンチの反復はばらつきを調べるもので、
  異なる負荷への代表性を増やすものではない。
- 校舎測定は最高 97°C・スロットリング 7788 回。EVO-X3 は最高 85°C（phys16-t8）で、スロットリングの
  検出手段がない（AMD は `throttle_events` が `not_supported`）。実機はこれより遅い可能性がある。
- 校舎 benchmark は SDK 呼び出しのみで、Constrained Decoding を含まない。

### 再現

```sh
uv venv --python 3.12 ~/cmm-cpu-lab/venv
uv pip install --python ~/cmm-cpu-lab/venv/bin/python -r research/evo-x3/requirements-campus.txt ./llm_sdk
# 初回のみモデル取得（この回は比較に使わない）
CUDA_VISIBLE_DEVICES= ~/cmm-cpu-lab/venv/bin/python -m tools.campus_survey benchmark \
  --pc-id evo-x3-dl --hf-home ~/cmm-cpu-lab/hf --allow-download
bash research/evo-x3/run_calibration.sh      # 6 条件 x 3 回
```

### PoC の実行時間測定

```text
bash research/evo-x3/measure.sh <repo か worktree> <label> [src への引数]
```

`uv run python -m src` の壁時間・ピーク RSS・終了コード・温度を `measurements/<label>.txt` に残す
（方式の実装には何も要求しない）。注意: `uv run` は対象リポジトリの `uv.lock`（レビュー時と同じ版）を使う。
校舎 benchmark の版（上記固定）とは異なる可能性があり、版の差による時間の違いは未評価。
2026-11-10に実アプリの配布11件を`measure.sh`で実行し、55.59秒・最大RSS約4.66GiB・終了コード0を確認した。
[原記録](measurements/cpu-phys16-t8-ta_kry5i-post-review.txt)と[条件・限界](../../docs/implementation-verification.md#時間)を参照する。
この回はリポジトリのCPU用`.venv`とモデルキャッシュを使用した。校舎実機のアプリ実測は未実施である。

### 再現可能性を調べるCPU・メモリ調査（モデル不要）

アプリ実測と環境調査は別である。校舎と開発機で下記を実行し、JSONを比較する。
モデル・SDK・torchの取得や実行は行わない。既存のNumPy/Pydantic環境を使う。

```sh
.venv/bin/python -m tools.hardware_probe measure --label campus > /tmp/campus-hardware.json
```

開発機側の測定・比較と、測る負荷・制限は[調査スクリプトの説明](../../docs/research/hardware-probe.md)を参照する。
結果はCPU計算・メモリ処理・スレッド数による差であり、環境の完全再現を自動判定するものではない。

### 参考: 校舎PCでのアプリ実測

測るのは配布11件のアプリ起動からモデル読込・関数選択・引数生成・検証・JSON保存までの壁時間と最大RSS。
モデル単体のforward、メモリ帯域、開発機との性能一致度を測るコマンドではない。
セットアップ済みのCPU環境で、既存のモデル・語彙キャッシュだけを使う。
校舎PCの容量を圧迫するため、リポジトリ配下へのモデル取得は案内しない。
キャッシュが見つからなければ停止し、ダウンロードは行わない。実機では再現用のaffinity・容量制限を掛けず、
スレッド数も既定値にする。結果は`/tmp`の専用ディレクトリに保存する。

```sh
git switch feat/implementation
git pull --ff-only
uv sync --locked

cmm_hub=$(
  uv run --no-sync python - <<'PY'
from tools.campus_survey import find_model
found = find_model(None)
if found is None or not (found[2] / 'vocab.json').is_file():
    raise SystemExit('既存のモデル・語彙が見つからないため測定しません')
print(found[1])
PY
) && {
  cmm_run_dir=$(mktemp -d /tmp/cmm-campus-run-XXXXXX) &&
  env -u OMP_NUM_THREADS -u MKL_NUM_THREADS -u OPENBLAS_NUM_THREADS \
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= HF_HUB_CACHE="$cmm_hub" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /usr/bin/time -f 'wall_s=%e max_rss_kib=%M exit_code=%x' \
    -o "$cmm_run_dir/time.txt" \
    uv run --no-sync python -m src \
    --output "$cmm_run_dir/function_calls.json" \
    >"$cmm_run_dir/stdout.log" 2>"$cmm_run_dir/stderr.log"
  cat "$cmm_run_dir/time.txt"
  printf 'results: %s\n' "$cmm_run_dir"
}
```

## 2. GPU 精度探索（別 venv・CPU 時間評価には使わない）

- 環境: `~/cmm-cpu-lab/venv-rocm`。公式 ROCm 7.2 版 `torch 2.14.1+rocm7.2`（`gfx1151` 対応を確認）、
  他は CPU 環境と同じ固定版、加えて `accelerate==1.15.0`。
- **配布 SDK は無改造のまま動く。パッチは不要。** 唯一の障害は SDK のコードではなく、venv に
  `accelerate` が無かったこと（SDK の `device_map="auto"` が要求する）。`accelerate` を入れて解消した。
- SDK は `device=None` で `cuda`（ROCm でも同名）を自動選択し、既定 dtype は `float16`。

### 検証（4 プロンプトの next-token logits、CPU float32 を基準）

| 構成 | 最大絶対差 | 平均絶対差 | argmax | top5 | ビット一致 |
|---|---|---|---|---|---|
| GPU float32 | 0.0001 | 0.00001 | 一致 | 一致 | **否** |
| GPU float16（SDK 既定） | 0.0394 | 0.00473 | 一致 | 一致 | **否** |

logits の幅は約 35.5。**ビット単位の一致は保証しない**（実際に一致していない）。4 プロンプトでの
argmax/top5 一致は、全入力での一致を意味しない。GPU 時間の測定は行っていない。

- SDK の SHA-256 は git 上の配布版・CPU venv・GPU venv で同一（`8c497ae2…`）。`llm_sdk/` に差分なし。
- 戻し方: `~/cmm-cpu-lab/venv-rocm` を削除するだけ。リポジトリ側に変更はない。
- GPU 探索で見つけた構成は、**CPU 環境の元の SDK で別途確認**してから採用する。
