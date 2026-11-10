# 校舎PCに近い負荷条件を探すための調査

`tools/hardware_probe.py`は、校舎PCと開発機で同じCPU・メモリ負荷を測り、違いを比較する。
アプリの11件処理時間ではなく、環境の再現可能性を調べるための材料を集めるスクリプトである。
モデル・SDK・torchを使わず、モデル取得や自動インストールも行わない。

## 実行するコマンド

Linuxのリポジトリ直下で、NumPy・Pydanticが入っている既存のPython環境から実行する。
既存の`.venv`を使う場合は以下のとおり。結果ファイルだけを`/tmp`へ書く。
校舎側にはCPU割当やスレッド数の制限を事前に掛けない。

```sh
# 校舎PC
.venv/bin/python -m tools.hardware_probe measure --label campus > /tmp/campus-hardware.json

# 開発機
.venv/bin/python -m tools.hardware_probe measure --label development > /tmp/development-hardware.json

# 校舎側のJSONを開発機へ持ってきて比較
.venv/bin/python -m tools.hardware_probe compare /tmp/campus-hardware.json /tmp/development-hardware.json
```

測定中は、何を測っているか、中央値・最小～最大・実効スループットをstderrに表示する。
JSONには生のナノ秒値と反復回数を残す。hostname・username・ホームパスは収集しない。

既存の`phys16-t8`に近いCPU割当も調べるなら、**開発機側で**次のように実行する。
スクリプト自体はCPU割当・電源設定・メモリ制限を変更しない。

```sh
taskset -c 0-15 .venv/bin/python -m tools.hardware_probe measure --label development-phys16 > /tmp/development-phys16.json
.venv/bin/python -m tools.hardware_probe compare /tmp/campus-hardware.json /tmp/development-phys16.json
```

## 何を測るか

既定では、スレッド数既定・1・4・8を別プロセスで順番に測る。利用可能なCPU数を超える既定候補は省く。
BLAS環境変数はNumPyのimport前に設定する。メモリ処理も、その条件のスレッド数で配列を分割する。
各処理はウォームアップ後、約50ms以上の反復数を決め、5サンプル測る。外れ値は除外しない。

| 出力の名前 | 実際の処理 | 比較する性質 |
| --- | --- | --- |
| `memory_read` | float32配列を分割して読み、float64へ足し合わせる | 連続読込＋加算の実効速度 |
| `memory_copy` | 別のfloat32配列へ分割コピーする | 連続読込・書込の実効速度 |
| `matmul_1` | `(1,1024) × (1024,3072)`のfloat32行列積 | 小さい行数の計算 |
| `matmul_64` | 同じ積を64行で実行 | 中程度の行数の計算 |
| `matmul_256` | 同じ積を256行で実行 | 大きい行数の計算 |

CPU名・命令の特徴・割当CPU・物理コア数・メモリ容量・共有L3の合計・NumPy/BLAS版も記録する。
`native_threads`はLinuxで見えるプロセス全体のスレッド数で、BLASだけのスレッド数ではない。

メモリのGB/sは、読込なら配列サイズ、コピーなら読込＋書込の2倍を処理時間で割った**論理バイト量**である。
実際のバス転送量を計測した値ではなく、読込時の加算やスレッドへの仕事の受け渡しも時間に含む。
行列計算のGFLOP/sも、行列の大きさから計算した名目の演算数による。

## 結果の読み方

`compare`は校舎側の`default`を基準に、開発機の各条件を処理ごとに比較する。
比は「開発機の秒数 ÷ 校舎PCの秒数」で、1なら同じ、0.5なら開発機の所要時間が半分。
`worst_deviation`は、その条件で1から最も離れた比の差である。

例えば行列計算は1に近くても、コピーが0.3なら、その条件は両方の負荷を再現できていない。
平均1個やアプリ全体の時間だけを合わせて「再現済み」とはしない。
採用条件や許容差は、人間が結果と目的を見て判断する。ツールは最適条件を自動採用しない。
アプリ総時間に対する便宜的な「6スレッド実測＋8秒」は、このNumPy負荷の時間や比率には適用しない。
NumPy・BLAS・Python・収集スクリプトが違う場合は比較結果に表示する。
不完全な測定、違う配列サイズ、欠けた処理・サンプルは比較を拒否する。

## メモリ・実行時間と限界

- 配列は既定256MiBを2つ、合計512MiB。一時領域と行列も見込んだ推定量が、
  ホストの空きメモリとcgroup v2の残容量の小さい方の1/4を超える場合は、配列確保前に停止する。
- `--array-mib 128`で小さくできるが、比較する両方のPCを同じサイズにする。
  各配列が割当CPU内の共有L3合計の4倍より小さい場合は、JSONの`cache_warning`に注意を残す。
- 既定の全体予算は60秒。子プロセスを親が監視し、超過は停止して`status=partial`・終了コード1にする。
  正常終了は`status=complete`・終了コード0。結果は端末出力か、明示したリダイレクト先にだけ保存する。
- これはNumPy/BLASの負荷比較であり、Qwenのforward、attention、マスク処理を再現するものではない。
  行列は繰り返し使われるためキャッシュの影響がある。メモリ処理は連続アクセスで、
  ランダムアクセスの遅延は測っていない。標準STREAMベンチマークでもない。
- 温度の開始・終了値と各サンプルの差を確認し、他の重い処理を止めて測る。
  測定中の最高温度や周波数、他プロセス負荷を常時監視しているわけではない。
- メモリ容量を制限しても帯域は変わらない。条件を変えても読込・コピー・計算の全部が近づかなければ、
  この方法ではその範囲の性能を再現できないと判断する材料になる。ハードウェアそのものの完全再現は保証しない。

**未確認:** このスクリプトによる校舎PCの結果はまだない。提供された
[校舎アプリの3回の時間](../measurements/2026-11-11-campus-application-reported.json)は別の測定である。
開発機での動作確認だけで、
校舎性能やアプリ実行時間を再現できたとは扱わない。

配列をキャッシュより大きくする考え方は[STREAMの説明](https://www.cs.virginia.edu/stream/ref.html)を参考にした。
BLASのスレッド制御は[NumPyの設定説明](https://numpy.org/doc/stable/reference/global_state.html)に基づく。
ソフトウェア情報には[show_config](https://numpy.org/doc/stable/reference/generated/numpy.show_config.html)の
公開APIを使い、コンパイラやライブラリのローカルパスはレポートへ入れない。

## 開発機での確認

[原記録](../measurements/2026-11-10-hardware-probe-development.json)は、256MiB配列・5サンプル・
既定/1/4/8スレッドで開発機を1回測った結果。全条件が完了した。
小さい配列でのCLI測定と自己比較、予算超過時の終了・不完全結果、メモリ不足拒否、
共有L3の重複計数防止、報告データの検証、単位・比較比率をテストで確認した。
この実行ではモデルを読み込まず、ダウンロードも行っていない。
校舎との比較に必要なのは、上のコマンドで得る校舎側のJSONである。

全146テストとflake8・Subject指定mypy（39ファイル）が成功し、独立レビューは指摘なしだった。
レビュアーも追加14テストを再実行した。校舎結果や環境の再現性を検証済みとするレビューではない。
