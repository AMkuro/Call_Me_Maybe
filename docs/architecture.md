# アーキテクチャ

状態: **スケルトン承認済み**(2026-11-10、PR #43)。本実装に着手した。
構成は `src/` の直下に平らに置く(サブパッケージは作らない)。
決定の根拠は `docs/design/adr/` にある。この文書は「何がどこにあり、どの順で呼ばれるか」を書く。

CLI、入力検証、骨組み、名前の木、バイト列による値の文法、動的な引数検証、
マスクによる貪欲選択、SDK ラッパー、原子的な保存を実装した。
[ADR-0013](design/adr/0013-implementation-details.md) の承認に従い、本実装を検証中。
[ADR-0014](design/adr/0014-terminal-progress.md) の TUI でトークン単位の選択を表示する。
構文・型の検証成功と、意味の正答・時間の要件を満たすことは別に評価する。

記号は `docs/subject-v1.7.md` と同じ(【明文】【解釈】【設計判断】【測定】)。

## 1. Subject のコード規則(実装が必ず守るもの)
`docs/subject-v1.7.md` の第1章から、コードに直接かかわるものを移した。出典の章番号は原文のもの。

- 【明文】Python 3.10 以上。flake8 準拠。変数・引数・戻り値に型ヒント。全関数で mypy エラーなし(IV.1)。
- 【明文】関数とクラスに PEP 257 の 英語のdocstring(IV.1)。
- 【明文】例外を適切に処理する。未処理の例外でクラッシュしたら「動作しない」扱い。資源はコンテキスト
  マネージャ等で確実に解放する(IV.1)。
- 【明文】**すべてのクラスは、検証に pydantic を使う**(IV.3.1、原文 "All classes must use pydantic for
  validation.")。【解釈】「pydantic のモデルであれ」とまでは書いていない。
- 【明文】使ってよいパッケージは `numpy` と `json`(IV.3.1)。
- 【明文】`pytorch`、`transformers`、`huggingface` 系、`outlines`、`dspy` 類は禁止(IV.3.1)。
  【解釈】プロジェクト自身のコードが直接使うことの禁止。SDK 経由は許容(`docs/subject-v1.7.md` 第1章)。
  **`src` から `import torch` しない。**(ユーザー、2026-11-09。当然のこととして守る)
- 【明文】関数の選択は LLM が行う。ヒューリスティクスは不可(IV.3.1)。
- 【明文】`llm_sdk` の private なメソッド・属性は使わない(IV.3.1)。使える公開 API は
  `encode`、`decode`、`get_logits_from_input_ids`、`get_path_to_vocab_file`(V.3.1)。
- 【明文】constrained decoding で 100% 有効な JSON を保証する。無効トークンの logits を負の無限大にして、
  残りから選ぶ。プロンプトだけに頼らない(V.1、V.3.3)。
- 【明文】出力は `data/output/function_calls.json`。各要素のキーはちょうど `prompt`、`name`、`parameters`
  (V.4)。入力不正とファイル欠落は適切に処理する(V.2、V.5)。
- 【明文】実行は `uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]`
  (IV.3.2)。
- 【明文】レビューでは別のプロンプトと別の関数セットが使われる(V.2)。

## 2. 今日決まったこと(ユーザー、2026-11-09)
- 【設計判断】**出力先の既定は `data/output/`**(既定ファイル名は `function_calls.json`)。Subject の記述が
  曖昧なので、最も明確な解釈を採ることに決定。
- 【設計判断】関数名の木は**トークン単位**で、`encode(名前 + 終端)` で作る(案 B)。根拠は ADR-0011。終端は EOS (`<|im_end|>`) を暫定採用する(ADR-0013)。
- 【設計判断】実装は**小さな単位ごとの監査をせず、まとめて進める**。理由: AI の文脈を保てる。1つずつ
  監査すると、前の変更を直したくなったときに手間が増える。(`AGENTS.md` の工程 5 を直した)
- 【設計判断】スケルトン(関数の形だけのコード)は別ブランチ・別 PR で作る。PR を境目にする。
- 【設計判断】モデルは、`llm_sdk` を包む具体的なクラス `Model`(`model_port.py`、§3)。根拠は ADR-0011。
- 出力先、名前の木、モデルのラッパー、`import torch` の禁止は、`docs/design/adr/0011-name-trie-and-output-path.md` に記録した。
- 値の終了・JSON の組み立て・保存に関する修正方針は、[ADR-0012](design/adr/0012-skeleton-review-corrections.md) に記録した。

### 【測定】トークン化(2026-11-09、SDK のトークナイザー、サンプルの関数 5 つ)
- 骨組み `{"prompt": "...", "name": "` の直後に名前を続けても、骨組み側のトークンは変わらなかった。
- 名前 + 閉じ引用符を、文脈の中でエンコードした結果と、単独でエンコードした結果は、5 つとも同じだった。
- **`",`(閉じ引用符 + カンマ)は 1 つのトークン**になる。また、名前だけをエンコードした列は、名前 + `"`、
  名前 + `",`、名前 + `", "parameters": {` のどれをエンコードした列の接頭辞にもなっていた(5 つとも。
  追加の測定)。
- 語彙 151643 のうち、引用符と他の文字を含むトークンは 1343 個。
- 限界: 5 つのサンプル名だけ。レビューの関数名では確かめていない。トークン列は 1 つの正規の分割だけを許す
  (別の分割は許さない)。

## 3. ファイル構成(暫定・平ら)
```
src/
  __main__.py   入口。引数を読み、実行し、終了コードを返す
  cli.py        引数の定義と検証(pydantic)
  loader.py     functions_definition.json と入力 JSON の読み込み・検証
  schema.py     pydantic のモデル: 関数定義、出力 1 件、型名。引数の事後検証用モデルを動的に作る
  model_port.py モデルのラッパー(`Model`)。SDK を import するのはこのファイルだけ
  vocab.py      語彙ファイルの読み込み。トークン ID とバイト列の対応
  name_trie.py  関数名のトークン列の木。接頭辞の衝突は EOS を木に含めて解く
  skeleton.py   関数定義から、強制する骨組み(キー、記号)と値の位置・終端を組み立てる
  values.py     値の型ごとの許可トークン(number、string、boolean)
  decoder.py    制約付きデコードの共通ループ(マスク、貪欲選択、強制トークン)
  prompt.py     プロンプトの組み立て(差し替え可能な設定)
  pipeline.py   1 件の処理: 関数名の選択 → 骨組み → 値の生成 → 組み立て → 検証
  writer.py     出力の保存(1 件ずつ原子的に)
  display.py    選択中のトークン、値、診断値を stderr の TUI に表示する
  errors.py     エラーの種類とメッセージ
tests/          純粋な部分はデータで、ループは固定 logits を返す関数で、全体は実モデルで確かめる(`AGENTS.md` の工程 4)
```
- 【設計判断】平らに置くのは、早い段階で大きな構成を決めないため。育ったら、まとまりごとに分ける。
- 【設計判断】SDK に触れるのは `model_port.py` だけ。
- 【設計判断】データの pydantic モデルは `frozen=True`(変更できない)。副作用のある関数は、docstring に `Effect:` の行を書く。
  `Effect:` がない関数は、引数だけで結果が決まる(純粋)。

### モデルのラッパー(`model_port.py`)
`Model` は、`llm_sdk` のモデルを包む**薄いクラス**。SDK は tensor を返すが、`Model` は普通のリストを返すので、
ほかのコードは `torch` を見ない。

```python
class Model:
    def __init__(self) -> None: ...                       # llm_sdk のモデルを読み込む(失敗は明確なメッセージで終了。ADR-0007)
    def encode(self, text: str) -> list[int]: ...         # 文章 → トークン ID
    def decode(self, ids: list[int]) -> str: ...          # トークン ID → 文章
    def logits(self, ids: list[int]) -> list[float]: ...  # 次のトークンの点数(語彙の数だけ)
    def vocab_path(self) -> str: ...                      # 語彙ファイルの場所
```

| `Model` の操作 | SDK の公開 API | 違い |
|---|---|---|
| `encode` | `encode` | SDK は `torch.Tensor` を返す。`Model` が `list[int]` にして、`torch` を `src` に出さない |
| `decode` | `decode` | 同じ |
| `logits` | `get_logits_from_input_ids` | 名前だけ短くした |
| `vocab_path` | `get_path_to_vocab_file` | 名前だけ短くした |

- **使う側**: `pipeline` の各関数は `Model` を受け取る。`decoder.generate` はループに要る `logits` の関数だけを受け取る。
- 抽象的な型を置かない理由、pydantic の扱い、テストのしかたは ADR-0011 と `AGENTS.md` の工程 4。

## 4. 実行順(どの関数がどの順で呼ばれるか)
1. `__main__` が `cli` で引数を検証する。
2. `loader` が関数定義と入力を読んで検証する(空、重複、未対応の型は起動時エラー)。
3. `model_port` の `Model()` がモデルを読み込む。読み込みの失敗は、明確なメッセージで終了する(ADR-0007)。
4. `vocab` が語彙を読み、`name_trie` が関数名の木を作る。
5. 入力 1 件ごとに `pipeline` が次を行う。
   1. `prompt` が関数選択用のプロンプトを作る。
   2. `decoder` と `name_trie` で関数名を選ぶ。候補が 1 つになったら残りのトークンを強制する。
   3. `skeleton` が、選んだ関数の骨組みを作る。
   4. `pipeline` が prompt と名前を JSON 文字列として組み立てる。骨組みを強制しながら、値の位置で
      `values` の許可トークンだけを残して生成する。値は `ValueSlot.terminator` の区切りまで含める。
   5. 組み立てた全文を JSON としてパースし、`schema` の動的モデルで検証する。検証・変換後の parameters を返す。
6. `writer` が成功した結果を 1 件ずつ保存する。失敗した件は飛ばし、標準エラーに理由を出し、
   最後に終了コードを非 0 にする。結果 0 件なら `[]` を保存する。保存の `OSError` は `main` が
   捕捉し、原因を表示して非 0 で終了する。

### 実行順の図(スケルトンの関数名)
`括弧内` はファイル。`src/` の関数と対応している。薄い青紫の枠は関数単位のまとまり、
内側の箱はその関数内の処理、ひし形は分岐を表す。実線は実行順、破線は例外の経路である。
図は本実装の主な生成フローを示す。TUI への観察用通知と、その整形は省略した。

![実行順の図(スケルトンの関数名)](diagrams/skeleton-flow.svg)

[全体図の D2 ソース](diagrams/skeleton-flow.d2)。

#### process_prompt の内部

1 件の入力は `select_name` → `generate_parameters` → `validate_call` の順に進む。
図の矢印に、関数定義 `fn`、JSON 全文 `text` という受け渡すデータを示した。

![process_prompt の内部の実行順](diagrams/process-prompt.svg)

[process_prompt の D2 ソース](diagrams/process-prompt.d2)。

#### select_name: プロンプトから関数を選ぶ

`build_name_prompt` はユーザーの入力と全関数の定義から関数選択用の文字列を作り、
`model.encode` がトークン列にする。全関数の説明と引数の型を列挙し、名前だけを応答するよう指示する。

`generate` は毎回 `allowed_next` を呼ぶ。この関数は、起動時に作った名前の木を使い、
`leaf_name` で完了を確認し、未完了なら `next_tokens` で続けられる ID を得る。
許可 ID が 1 個なら強制し、複数ならモデルの点数から選ぶ。木の葉は名前に続く EOS の後ろにあり、上限は木の最長経路から計算する(ADR-0013)。

![select_name の関数選択ループ](diagrams/select-name.svg)

[select_name の D2 ソース](diagrams/select-name.d2)。

#### generate_parameters: 骨組みと引数値を組み立てる

`Forced` はそのまま書き、`ValueSlot` は `generate_value` で値と区切りを生成する。
値の生成にも同じ `generate` を使うが、完了判定と許可 ID の取得は `values` の関数が担当する。
語彙はバイト列として保持し、UTF-8 の途中を壊さずに検査する。文字列は既存の接頭辞の
文法状態を一度計算し、各候補の追加分だけを検査する。

![generate_parameters の引数生成ループ](diagrams/generate-parameters.svg)

[引数生成図の D2 ソース](diagrams/generate-parameters.d2)。
配色は [workflow-style.d2](diagrams/workflow-style.d2) で共有する。
D2 v0.9.0 で描画を確認した。レイアウトエンジンと余白は各 D2 ソースの末尾に指定する。
詳細図は TALA の `top`・`left` で行と列を揃え、戻り線で枠が広がるのを防いでいる。
ラベルを変更したときは、箱と文字の重なりも確認する。再生成:

```sh
for diagram in skeleton-flow process-prompt select-name generate-parameters; do
  d2 "docs/diagrams/$diagram.d2" "docs/diagrams/$diagram.svg"
done
```

### TUI と診断値

`decoder.generate` は `TokenChoice` を任意の通知関数に渡す。`pipeline` が関数・引数の文脈と
表示用の文字列を付け、`main` と `display` が入力番号と画面更新を担当する。
候補が1つなら強制と表示し、未計算の確率は数値にしない。複数候補では同じ logits から
許可確率の合計・上位2候補の差・制約前の第1候補を求める。名前の第1候補が候補外なら警告する。
数値は正答率ではなく、プロンプト比較の補助情報である。

stderr が端末なら自動画面更新、リダイレクト時は選択結果・警告・結果のログになる。
画面を閉じる際は端末状態を戻し、完了結果を通常の行として残す。

## 5. 実測後に見直す項目

- EOS が出力の障害となる場合の終端の形式(ADR-0013)。
- 長い値の上限は暫定256トークン。理由付きのエラーにし、実測で見直す。
- 文字列の意味上の追加制約、数値の表記変更は未採用。
- 関数セット変更時の精度とプロンプト、警告の確率閾値。
