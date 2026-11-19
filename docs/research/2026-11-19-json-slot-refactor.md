# JSONスロット文法の責務整理と検証（2026-11-19）

対象はローカルdevelopの`src/values.py`と制約テスト。作業開始時のHEADは
`0400edbc07afd54c8a8ca6f8d3e45a75d389de45`。
既存のvalues.pyの未コミット差分は整形だけで、AST比較でもHEADと一致した。
README・development.md・提出監査の既存変更は今回のコミットに含めない。
SubjectはAMkuro/pdfのローカル原本`Call_Me_Maybe/en.subject.pdf`（v1.7）を全文確認した。

## 責務と設計判断

変更前は`_slot_state`と文字列候補の最適化が、それぞれ走査結果とUTF-8の検査を組み合わせていた。
変更後も`_slot_state`がスロット全体の3状態を定義し、`is_slot_complete`がその完成状態を使う。
文字列では共通の`_string_slot_state`が、構文状態・追加バイト・未完UTF-8から3状態を返す。
`_slot_state`は初期状態から、`allowed_slot_tokens`は共通接頭辞の走査済み状態からこれを呼ぶ。
候補列挙は空候補を除外し、語彙の順序を保つ。number・booleanの判定は変更していない。

構文の状態遷移は`_scan_json_string_fragment`にだけ置く。旧名の`_advance_string`よりも、
JSON文字列のバイト断片を指定状態から走査する役割が明確になるため、依頼の第一候補を採用した。
`phase/data/char`を`state/fragment/byte`へ変更し、開始・文字列内部・エスケープ・
Unicodeの各桁・閉じ引用符後の順に分岐を並べた。
Unicode状態は残り4桁から減らし、残り1桁を消費すると文字列内部へ戻す。
整数状態の範囲と残り桁数の関係を定数名とコメントで示し、クラスや遷移表は追加しない。
UTF-8検査は走査関数から分離したままである。

全候補で`_slot_state(written + value, ...)`を呼ぶ案は、長い接頭辞の再走査を語彙数だけ
繰り返すため採用しない。構文状態の引き継ぎと、UTF-8の未完文字だけの引き継ぎで増分評価を保つ。
現在の設計の説明は[architecture.md](../architecture.md)の「値の文法とバイト列の扱い」に追記した。
プロンプト、forward、logits、argmax、SDK、保存形式、公開関数の引数は変更していない。

## 正しさの検証

`tests/test_constraints.py`に以下の2テストを追加した。実装変更前に追加し、既存動作で通過することを確認した。

- 独立した期待値による3状態の検査。引用符、エスケープ、Unicodeの各桁、未完・不正UTF-8、
  終端後の余分なバイトを扱う。4桁未満でUnicodeエスケープを閉じる例は不正とする。
- `_slot_state`と候補リストの同値性。3型と両終端で、空状態から完成までの各バイト接頭辞を検査する。
  seed=1942の文字列30件ずつと固定例を使い、全256種の単一バイト、空候補、境界をまたぐ候補、
  乱数バイト256件など計528候補を比較する。非連続の降順IDにより順序とID集合も確認する。
  修復不可能な不正UTF-8の生成状態も比較対象に含む。

独立期待値の検出力は、プロセス内でUnicode状態を4桁から3桁へ一時変更して確認した。
3桁直後の閉じ引用符を受理してしまう変異で1件失敗し、元の状態では通過する。
この検査でソースファイルは変更していない。
独立レビューでは、変更前の作業ツリーのコピーとの追加比較も行い、
乱数接頭辞15,000件・両終端の分類と許可リストで差異は報告されなかった。重大な指摘はなかった。

実行コマンドと最終結果：

```sh
make install UV=/tmp/cmm-uv
UV_CACHE_DIR=/tmp/cmm-uv-cache make lint UV=/tmp/cmm-uv
UV_CACHE_DIR=/tmp/cmm-uv-cache make test UV=/tmp/cmm-uv
.venv/bin/python -m unittest discover -s tests -p test_constraints.py -v
UV_CACHE_DIR=/tmp/cmm-uv-cache make gpu-run UV=/tmp/cmm-uv \
  FLAGS='--output /tmp/cmm-slot-refactor-output.json'
git diff --check
```

installは成功。sandbox内の最初の試行は既存uvキャッシュの書き込み制限で失敗し、
ホスト実行で成功した。`/tmp/cmm-uv`には再作成前のvenvからuvをコピーした。
lint/testは書き込み可能な一時キャッシュを指定した。
lintはflake8とSubject指定のmypyフラグで成功（57ソースファイル）。
最初に追加テストの行長超過を4件検出し、整形後に通過した。
制約テスト26件、全体280件が成功。全体には固定logitsによる生成、骨組み、JSON出力の回帰テストを含む。
`git diff --check`も成功。

GPUはホスト実行の`.venv-gpu`で`torch.cuda.is_available() == True`と
`Radeon 8060S Graphics`を確認した。`make gpu-run`は既知の11入力を11件成功・0件失敗で終了。
その後、保存結果をstdlibの`json.loads`で読み、入力順・promptの一致と、
`FunctionCall`および関数ごとの`build_parameters_model`によるPydantic検証を別途実行した。
11/11件がJSONとして解析でき、型・必須引数の検証を通過した。
これは構造の確認であり、意味的な正解率の測定ではない。

## 候補列挙だけの性能測定

CPUはAMD RYZEN AI MAX+ 395、Python 3.12.3。
Qwen3-0.6Bの実語彙151,643件（snapshot `c1899de289a04d12100db370d81485cdf75e47ca`）を使った。
開始引用符にASCIIの`a`を0・128・8192個追加し、UTF-8の未完バイト`e3 81`の有無で6条件とした。
終端は`}`。各条件で変更前後の許可IDリストが一致することを先にassertする。
変更前後を交互に測り、順序も回ごとに反転した。各3回の平均を9組取り、その中央値を下表に示す。
外れ値は除外していない。モデル推論・品質チェックの終了後に直列で実施した。

| 接頭辞バイト数 | 末尾の未完UTF-8 | 変更前 ms | 変更後 ms |
| ---: | --- | ---: | ---: |
| 1 | なし | 48.64 | 42.81 |
| 3 | e3 81 | 99.74 | 93.25 |
| 129 | なし | 48.53 | 42.82 |
| 131 | e3 81 | 100.29 | 94.13 |
| 8193 | なし | 48.80 | 43.89 |
| 8195 | e3 81 | 100.00 | 93.26 |

測定範囲では長い文字列でも大幅な遅延は認めなかった。
共通関数の呼び出しは増えるが、通常文字列の分岐を先に判定する変更も含むため、各変更の寄与は分離していない。
この結果は候補列挙の所要時間であり、アプリケーション全体の速度や生成精度の改善を示さない。
キャンパス時間推定の8秒加算は適用しない。

再実行コマンド：

```sh
PYTHONPATH=. .venv/bin/python \
  docs/measurements/2026-11-19-json-slot-refactor/benchmark.txt
```

[測定スクリプト](../measurements/2026-11-19-json-slot-refactor/benchmark.txt)は
固定した変更前コミットからコードを読み込む。モデルキャッシュのパスはこのホストに依存する。
[生の測定値](../measurements/2026-11-19-json-slot-refactor/timing.json)に全サンプルを保存した。

## 制約とレビューの重点

同値性は有限個のテストと状態引き継ぎの論拠で確認したもので、全バイト列の形式証明ではない。
実モデルの生成結果について、変更前後の全文比較や未知入力の正解率測定は実施していない。
今回の調査範囲で、別途動作修正が必要な既存不具合は発見していない。

レビューでは、共通判定への依存、UTF-8未完部分の引き継ぎ、Unicodeの4桁からの減算、
空候補・終端後の拒否、ID順序の比較テストを重点的に確認する。
