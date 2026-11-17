# ADR-0026の実装と実SDK検証（2026-11-17 UTC）

ADR-0026に従い、固定JSON部分を1 IDずつ確定する実装をローカルdevelopへ追加した。
検証は未コミットの作業ツリーで行った。精度調整、SDK改変、名前EOS・値文法・256トークン上限の変更は行っていない。
後続の開発者指示により、実装・テストと設計・検証記録を意味のある単位に分けてローカルコミットする。公開は行わない。

## 条件と証跡

開始HEADは`176e958ee4b230284a01d36c50a103788b1ae9b1`。
`src/name_trie.py`と`src/schema.py`には別作業の未コミット変更があり、手を加えていない。
SDKとこの2ファイルは開始・終了時のSHA-256で照合する。
ローカルにADR-0026がないため、前回レビューで確認した
[GitHub developのADR-0026](https://github.com/AMkuro/Call_Me_Maybe/blob/ec78370c01bdfb8730ea22d91b408c89f111ac1d/docs/design/adr/0026-tokenwise-forced-json.md)
と開発者の今回の実装指示を用いた。ADRファイルは追加・変更していない。

実機はRadeon 8060S Graphics、Qwen/Qwen3-0.6Bの既存キャッシュ
`c1899de289a04d12100db370d81485cdf75e47ca`をオフラインで使用した。
サンドボックス内ではGPUが見えず、ホスト実行でGPU可用性を確認した。
既存`.venv-gpu`、ROCm torch、SDKの意図的なfloat32設定を使用した。
float32はCPUとの完全一致を保証しない。パッケージ版・ソースハッシュは
[environment.json](../measurements/2026-11-17-tokenwise-json/environment.json)に保存した。

原記録は[測定ディレクトリ](../measurements/2026-11-17-tokenwise-json/)。
`before.json` / `after.json`は精度・境界・forward回数で、時間測定ではない。
`after-initial.json`はlintの整形前の初回実装の記録で、最終ソースの結果と混同しない。
`timing-before.json` / `timing-after.json`は別実行の総アプリ時間、
`cli-before.json` / `cli-after.json`はそのとき保存したJSONである。
過去の測定ファイルは変更していない。

## アプリ変更前のSDK検証

SDKの公開encode、decode、logits、vocabパスだけで次を確認した。
推論・トークナイザーのprivate属性は使っていない。

| 試験 | 実測 |
| --- | --- |
| 通常のJSON先頭、キー、句読点 | SDK復号と語彙バイト列が元の文字列に一致 |
| `encode("hel") + encode("lo")` | `[49122, 385]`。全文encodeは`[14990]`。どちらも`hello`へ復号し、モデルも受理 |
| 上の分割でのlogits | 最大絶対差2.844806671142578。同じ文脈でargmaxは双方198 |
| `"日` / `本語"}`の独立encode | IDは全文encodeと異なるが復号一致。モデル受理。logits最大差1.410684585571289 |
| 数値→区切り→次キーの境界 | 断片連結が正しく復号し、forwardが成功 |
| `"<|im_` / `end|>"`の独立encode | 復号は`"<|im_end|>"`。全文encodeは特殊ID151645を含む。argmaxは分割4710、全文151645と異なる |
| 固定文字列内の`<|im_end|>` | 特殊ID151645はbyte vocabに存在せず、decodeから欠落。`"前<|im_end|>後"`が`"前後"`になる |
| `\u003c`によるJSON表現 | 通常の語彙IDだけになり、SDK復号はJSONテキストと一致。JSON解析後は元の`<`に戻る |
| UTF-8途中分割 | `🧑🏽‍💻`に含まれる結合文字のbytesがID378=`[226,128]`、ID235=`[141]`に分割。列全体では正しく復号 |
| 一意候補の後の分岐 | 一意候補ではlogits 0回。そのIDを含む文脈で次の分岐のlogitsを1回取得 |

したがって、ID列の正規トークン化との一致は不要だが、分割がモデルの点数や選択を変えないとはいえない。
この実装では通常ケースの分割を維持し、特殊表記の固定断片だけを意味の等しいJSON escapeへ変える。
UTF-8とescapeの補足試験は`utf8.json`にある。

## 実装

`src/pipeline.py`だけをアプリの変更対象にした。

- `_append_forced`は固定断片内の`<`を`\u003c`へ置換する。固定断片の`<`はJSON文字列内の文字であり、構文自体を変更しない。
- 固定断片をencodeし、語彙bytes・SDK復号・期待文字列を照合してから、各IDを1個ずつOへappendする。logitsは呼ばない。
- `generate_parameters`は指示のPと出力Oを分離し、既存の`generate_value`へP + Oを渡す。
- 値ループのVは従来どおり1 IDずつ増える。次の分岐の文脈はP + O + Vである。
- 戻ったVは完成値の照合後にOへ統合する。再encodeや再選択はしない。
- `_decode_output`で完成断片と全文の語彙bytes／SDK復号を照合する。全文は期待断片の連結とも照合して返す。
- `process_prompt`は既存のstrict検証後に、出力promptと入力の完全一致を検査する。名前の一致検査は既存どおり。

未知ID、不正UTF-8、encode/decode不一致はPromptErrorにする。
期待するテキストを別経路で保存して成功扱いにしない。
名前用EOS経路はJSON本体へコピーせず、選ばれた元の関数名をJSON化する。
既存のTUI通知・生成step数・エラー処理の境界は維持した。

## テストと不変条件

新規`tests/test_forced_json.py`は12件。
実装前には先行11件が11件ともFAIL、実装後に成功した。
最後の1件は、異なる固定ID分割を保持する追加の回帰確認である。

| 不変条件 | 対応する試験 |
| --- | --- |
| 固定IDを1個ずつ確定、固定forwardなし | `test_fixed_ids_commit_individually_without_forward` |
| prompt・name・キーの特殊表記を保持 | `test_special_prompt_name_keys_preserve_meaning` |
| 固定→値→固定の文脈と全型・UTF-8・JSON escape | `test_unicode_escapes_multiple_types_and_boundaries` |
| 引数0個・特殊表記でforwardなし | `test_zero_parameters_special_text_need_no_forward` |
| 同じbytesの別IDを再encodeせず継続 | `test_duplicate_byte_tokens_keep_the_selected_id` |
| encode不一致、未知ID、SDK復号欠落 | `test_encode_mismatch_fails_before_value_inference`、`test_unknown_fixed_id_is_a_prompt_error`、`test_sdk_decode_loss_is_a_prompt_error` |
| 完成値・全文のSDK復号を照合 | `test_completed_value_decode_must_match_grammar_bytes`、`test_full_decode_must_match_concatenated_fragments` |
| 入力promptの完全一致 | `test_process_rejects_changed_prompt` |
| 固定IDが非正規分割でも内容一致を許す | `test_fixed_noncanonical_ids_are_preserved` |
| 一意候補→分岐、上限・終了・空候補 | 既存`test_application.DecoderTests`、`test_token_selection`、`test_terminal_progress` |
| TUIなし・あり・利用不能・出力失敗で結果と入力が同じ | `test_complete_call_and_value_progress_with_observation` |

`test_skeleton_flow`には完全なASCII語彙を渡すよう修正した。
`test_terminal_progress`は空encodeを実際の固定IDへ変え、対応語彙とlogits長を補った。
生成結果、通知ID、step、モデル呼出回数、各表示モードの入力一致のassertionは維持した。
テスト削除や成功条件の緩和は行っていない。

独立レビューでも重大な指摘はなく、関連28テストを別途実行して成功した。
保存JSONは独立したキー集合・プリミティブ型・入力文面の検査でも確認した。
[集計](../measurements/2026-11-17-tokenwise-json/summary.json)と
[チェック記録](../measurements/2026-11-17-tokenwise-json/checks.json)も保存した。

今回のチェック結果:

- 変更前: `make test` 210件成功（10.287秒）、`make lint`成功。
- 変更後: `make install`成功。SDKのソースとGPU環境は保持。
- 再作成したCPU環境で`make test` 222件成功、`make lint`成功（Subjectフラグ、49ファイル）。
- `make subject`、`make subject-audit`成功。いずれも提出の完全な適合証明ではない。

## 実モデルの回帰（精度・forward回数）

変更前後とも同じ19入力を1回ずつ実行した。
既存のcalibrated/original（11件）とchanged_functions_cases（5件）を利用し、
特殊表記を含む名前・キー、全型、日本語、JSON escape、引数0個の3件を`boundary.json`へ追加した。
評価は既存`assess_call`の型・数値保持を含む完全一致で、語意上同等なregexを別採点する評価ではない。

| 集合 | 完了／JSON／スキーマ適合 | 関数名正答 | 全引数完全一致 | 呼出全体正答 | forward合計 |
| --- | --- | --- | --- | --- | --- |
| calibrated/original | 11/11 → 11/11 | 11/11 → 11/11 | 9/11 → 9/11 | 9/11 → 9/11 | 105 → 105 |
| changed_functions | 5/5 → 5/5 | 4/5 → 4/5 | 2/5 → 2/5 | 2/5 → 2/5 | 39 → 39 |
| 境界3件 | 3/3 → 3/3 | 3/3 → 3/3 | 2/3 → 2/3 | 2/3 → 2/3 | 14 → 14 |
| 合計 | 19/19 → 19/19 | 18/19 → 18/19 | 13/19 → 13/19 | 13/19 → 13/19 | 158 → 158 |

全引数完全一致は全入力を分母とし、関数名の正誤とは別に期待引数と比較した。
全件で出力promptを保持し、変更前後のFunctionCallは同一だった。
構文・スキーマ適合は生成した関数定義への適合であり、意味上の正答と区別する。
既存の意味的誤りは残る。境界例の日本語も適合・完了したが抽出の完全一致は失敗した。
この19件で未見入力の精度や90%目標の達成を主張しない。

## 別実行の時間

標準`data/input/function_calling_tests.json`の11件を、変更前後に`make gpu-run`で各1回、
他のモデル推論と重ねず実行した。精度の19件とは異なる入力集合である。
外側の`/usr/bin/time`でmakeの起動から正常終了までを測った。

| 条件 | 変更前 | 変更後 |
| --- | --- | --- |
| 総アプリ時間 | 25.25秒 | 25.16秒 |
| 終了コード | 0 | 0 |
| 保存JSON | 11件 | 11件、変更前と同一 |

1回ずつで分散・熱状態を統制していないため、差0.09秒を改善と判断しない。
固定部分のforward増加はなく、この測定で大きな時間悪化は観測しなかった。
GPUのraw時間であり、校舎CPU用の8秒補正は加えていない。

## 再現方法と制約

原記録には、その実行に使ったスクラッチ検証スクリプトを`verify.py.txt`、`utf8.py.txt`として保存した。
新たな常用ツールや依存パッケージは追加していない。
再実行するなら、リポジトリをcwdとして次を用いる。GPUが見える環境で実行する。

```sh
mkdir -p /tmp/cmm-0026
cp docs/measurements/2026-11-17-tokenwise-json/boundary.json /tmp/cmm-0026/
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" \
  .venv-gpu/bin/python \
  docs/measurements/2026-11-17-tokenwise-json/verify.py.txt \
  --phase review --output /tmp/cmm-0026/review.json
```

スクリプトは公開Model境界のlogits回数を測定し、実際のlogitsは置換せず元の実モデルへ渡す。
変更前の再現には元のpipelineソースが必要で、現在の作業ツリーでbaselineを作り直したとは扱わない。
特殊表記に対する処置は出力JSONの固定断片に限定する。
指示プロンプト中の特殊表記や意味理解を一般に解決したとは主張しない。
SDKが別モデル・別版で異なる復号や語彙を使う場合は、理由付き失敗になり得る。
