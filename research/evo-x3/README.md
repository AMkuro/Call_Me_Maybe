# EVO-X3 検証環境（校舎 i7-14700 のSDK負荷に合わせた近似）

[thermalの連続負荷調査](../../docs/research/safe-thermal-survey.md)は、手動で冷間・温間を作り分けずに温度とforward時間を測る。
このAMD開発機ではthermal counterがなく、現在の安全条件では開始しない。VMも設定管理の利点を含めて候補とする。

2026-11-11の[追加診断と調査案](../../docs/research/campus-runtime-feasibility.md)では、
校舎と開発機のCPU／メモリの時間比が負荷ごとに異なることを確認した。
以下のSDK負荷の校正は、実アプリの全入力や校舎の熱挙動を再現する根拠にはならない。

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
  同一視しない。** 300 秒以内は未測定で、達成済みとは言えない。
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

```sh
bash research/evo-x3/measure.sh <repo か worktree> <label> [src への引数]
```

`uv run python -m src` の壁時間・ピーク RSS・終了コード・温度を `measurements/<label>.txt` に残す
（方式の実装には何も要求しない）。注意: `uv run` は対象リポジトリの `uv.lock`（レビュー時と同じ版）を使う。
校舎 benchmark の版（上記固定）とは異なる可能性があり、版の差による時間の違いは未評価。
`measure.sh` は未実行（PoC がまだない）。

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
