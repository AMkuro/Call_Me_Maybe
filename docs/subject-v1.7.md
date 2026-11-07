# Call Me Maybe Subject v1.7 — ローカル参照資料

エージェントがSubjectの要件を確認するためのローカル資料。**この文書を優先し、毎回GitHubから
原文を取得しない。** 原文の再確認が必要なのは次の場合だけ。

- Subjectのバージョンが 1.7 でなくなったことが判明した場合
- この文書だけでは解決できない、重要な解釈上の問題が生じた場合

## 出典とバージョン

- バージョン: **1.7**（原文表紙に `Version: 1.7`）
- 原文: <https://github.com/AMkuro/pdf>、ブランチ `milestone3`、`Call_Me_Maybe/` 配下
  （`Call-Me-Maybe.en.subject.txt`、`en.subject.pdf`）。コミットハッシュは記録しない。
- 章番号は原文のもの（例: V.5 = 第V章 第5節）。原文はテキスト抽出のため、一部の行順が崩れている。

## 記号の約束

| 記号 | 意味 |
|---|---|
| 【明文】 | Subjectに書かれている事項 |
| 【解釈】 | 明文から導いた読み。Subjectは明言していない |
| 【設計判断】 | このプロジェクトが決めたこと。Subjectの要求ではない |

## 1. 必須要件と禁止事項

### 言語・品質（IV.1）【明文】

- Python 3.10 以上。flake8 準拠。
- 例外を適切に処理し、クラッシュさせない。レビュー中に未処理例外でクラッシュしたら「動作しない」扱い。
- ファイル等の資源はコンテキストマネージャ等で確実に解放する。
- 変数・引数・戻り値に型ヒント。mypy を全関数でエラーなしにする。
- 関数・クラスに PEP 257 の docstring。

### Makefile（IV.2）【明文】

- 必須ルール: `install`、`run`、`debug`（pdb）、`clean`、`lint`。
- `lint` は `flake8 .` と
  `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`。
- `lint-strict`（任意）は `flake8 .` と `mypy . --strict`。

### 追加要件（IV.3.1）【明文】

- すべてのクラスは pydantic で検証する。
- 使ってよいパッケージ: `numpy`、`json`。
- **dspy およびそれに類するパッケージは全面禁止。pytorch、huggingface パッケージ、transformers、outlines も含む。**
- モデルは `Qwen/Qwen3-0.6B`（既定）。他モデルは「Qwen3-0.6B でも動く」場合に限り可。
- **関数の選択はLLMで行う。ヒューリスティクス等は不可。**
- **`llm_sdk` のprivateなメソッド・属性は使用禁止。**
- 仮想環境を作り、`uv` で `numpy` と `pydantic` を入れる。`llm_sdk` は `src` と同じ階層にコピーしてよい。
- **レビュアーとmoulinetteは `uv sync` だけを実行する。**
- すべてのエラーを適切に処理し、明確なエラーメッセージを出す。予期しないクラッシュは不可。

【解釈】pytorch等の禁止は、プロジェクト自身のコードが直接使うことの禁止と読む。
配布 `llm_sdk`（V.3.1）は内部でそれらを使うため、SDK経由の間接利用は許容される。
Subjectはこの境界を明言していない。

## 2. 実行方法と入出力契約

### 実行（IV.3.2）【明文】

```
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

- `src` はソースのフォルダ。既定の入力は `data/input/`、既定の出力は `data/output/`。
- 例: `--functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json`

### 入力（V.2）【明文】

- `function_calling_tests.json`: `{"prompt": "..."}` オブジェクトの配列。
- `functions_definition.json`: 関数の配列。各要素は `name`、`description`、
  `parameters`（引数名 → `{"type": ...}`）、`returns`（`{"type": ...}`）を持つ。
  例にある型は `number` と `string`。
- **レビュー時は別のプロンプトと別の関数セットでテストされる。**
- 入力JSONの不正・ファイル欠落を適切に処理する。

### 出力（V.4、V.4.2）【明文】

- 単一のJSONファイル `data/output/function_calls.json`。配列で、プロンプトごとに1オブジェクト。
- 各オブジェクトのキーは**ちょうど** `prompt`（string）、`name`（string）、`parameters`（object）。
- 検証規則: 有効なJSON（末尾カンマ・コメントなし）／キーと型が定義と完全一致／
  余計なキーや文章は禁止／必須引数はすべて存在／引数の型は定義どおり（number、string、boolean 等）。
- 例: `{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}`

## 3. 配布SDKの使用条件

### 提供API（V.3.1）【明文】

| メソッド | 内容 |
|---|---|
| `get_logits_from_input_ids(input_ids: List[int]) -> List[float]` | トークンID列から次トークンのlogitsを返す |
| `get_path_to_vocab_file() -> str` | トークンIDと文字列の対応を持つvocabファイルのパス |
| `encode(text: str) -> Tensor` | テキストをトークンIDのテンソルへ |
| `decode(token_ids: List[int]) -> str` | （任意）IDをテキストへ |

- `llm_sdk` は提出リポジトリに含める（VIII）。`src` と同じ階層にコピーしてよい（IV.3.1）。
- privateなメソッド・属性は使用禁止（IV.3.1）。
- 【設計判断】配布 `llm_sdk` は変更しない（`AGENTS.md`）。

## 4. 動的スキーマとConstrained Decoding

### 要件（V.1、V.3.3）【明文】

- Constrained decodingで**100%有効なJSON**を保証する（V.1）。
- 仕組み: モデルがlogitsを出す → スキーマ・構造を保つトークンを特定 →
  無効トークンのlogitsを**負の無限大**にする → 残りから選ぶ（V.3.3）。
- JSONの構文だけでなく**スキーマ**も強制する。例: 数値と定義された引数には、floatとして成立する
  トークンだけを許す（V.3.3）。
- **プロンプトだけでモデルが自発的に正しいJSONを出すことに頼ってはならない**（V.3.3）。
- vocabファイルでトークンと文字列を対応づけることが、有効トークンの判定に重要（V.3.3）。
- トークンごとの反復: 生成したトークンを入力に足し、完了まで繰り返す（V.3.2）。

### 解釈【解釈】

- 関数名は `functions_definition.json` に実在する名前だけを生成できる必要がある。
- 引数の型（number、string、boolean 等）は、関数定義ごとに実行時に決まる。
  関数セットはレビュー時に変わるため、スキーマは実行時の定義から作る。

## 5. 精度・JSON適合率・実行時間（V.5）

【明文】

- **90%以上**: 関数選択と引数抽出の正答（Near-perfect accuracy）。
- **100%**: 有効かつスキーマ適合なJSON。
- **合理的な速度: 標準的なハードウェアで、全テストプロンプトを5分未満で処理する。**
- 不正入力、ファイル欠落、境界ケースを適切に処理する。
- 本文に「Qwen3-0.6Bは5億パラメータ」とある（モデル名の0.6Bと不整合だが、要件には影響しない）。

## 6. セットアップ・事前計算・実行時間の区別

Subjectは以下をどれも定義していない。ここに書くのは解釈と設計判断だけ。

- 【明文】レビュアーとmoulinetteは `uv sync` だけを実行する（IV.3.1）。
- 【明文】実行は `uv run python -m src ...`（IV.3.2）。
- 【明文】5分は「全テストプロンプトの処理」に対する制限（V.5）。
- 【解釈】`uv sync` の時間、モデルのダウンロード・ロード時間が5分に含まれるかは**Subjectに書かれていない**。
- 【設計判断】`uv sync` は5分に含まれないとして作業を進める。ただしこれは確定仕様ではない。
- 【設計判断】性能評価は、安全側に `uv run python -m src` の起動から正常終了までを300秒以内とする。
- 【設計判断】開発中の汎用的な事前計算・キャッシュ・テーブル生成は許容する。
  ただし生成物はレビュー環境で再現・利用できなければならない（Git管理やビルド手順を含めて設計する）。
- 【明文】入力は変わり得る（V.2、V.4.2）。【解釈】提供サンプルの答えの事前計算・ハードコードは禁止
  （V.4.2に "Do not hardcode solutions based on the provided examples" 相当の記載。原文は行順が崩れている）。
- 【明文】関数選択は実行時にLLMが行う（IV.3.1）。

## 7. レビュー時に変更され得るもの

【明文】
- `function_calling_tests.json` と `functions_definition.json`（V.2、V.4.2）。
- 評価中に「小さな改修」を求められることがある（VIII）。数分で終わる範囲で、コードの理解確認が目的。
- `output/` ディレクトリはレビュー時に生成される（VIII）。

## 8. 提出物の構成（VIII）【明文】

- `src/`（実装）、`pyproject.toml` と `uv.lock`、`llm_sdk/`（配布物のコピー）、
  `data/input/`（デモ用テストファイル）、`README.md`、実行に必要なその他のファイル。
- `output/` は含めない。
- テストプログラムは提出・採点対象ではない（IV.3）。`.gitignore` でPython成果物を除外する。

### README（VI）【明文】

- 英語で書く。1行目は斜体で `This project has been created as part of the 42 curriculum by <login1>[, <login2>...]`。
- 必須節: Description、Instructions、Resources（AIの利用内容を含む）。
- 本プロジェクト固有: アルゴリズム説明（constrained decoding）、設計判断、性能分析、直面した課題、
  テスト戦略、使用例。

### ボーナス（VII）【明文】

必須部分が完璧な場合のみ評価。実装され動作していること。例: 複数モデル対応、トークナイザの自作
（`encode`/`decode` を使わず `get_logits_from_input_ids` と `get_path_to_vocab_file` を使う）、
高度なエラー回復、性能最適化（caching、batching）、包括的テスト、生成過程の可視化、入れ子引数、
`encode`/`decode` の公開実装、それらとconstrained decodingの統合の実演。

## 9. 現時点で未確定な解釈

Subjectに書かれておらず、確定仕様として扱ってはならない事項。

1. **5分の範囲**: `uv sync`、モデルのダウンロード・ロード、`uv run` 起動を含むか。
   「標準的なハードウェア」の定義。「全テストプロンプト」の件数（V.5）。
2. **出力パス**: V.4は `data/output/function_calls.json`、V.6は `output/function_calls.json`。
   どちらも既定の `--output` と整合させる必要があり、`data/output/` を既定とする（IV.3.2）のが最も明確。
3. **オプション表記**: IV.3.2に `-output`（ハイフン1つ）の表記があるが、原文の文字化けと読み、
   `--output` として扱う。V.6の `–` はダッシュの組版の問題。
4. **対応する型**: 例は `number` と `string`。V.4.2は「boolean等」と述べるだけで、
   `integer`、配列、入れ子の扱いは未定義（入れ子はボーナス）。
5. **正答の判定方法と分母**: 90%の計算単位（関数選択と引数抽出を別々に数えるか）は未定義。
6. **該当する関数がない／曖昧なプロンプト**: 出力に何を入れるか、スキップするかは未定義。
   "no extra keys" とクラッシュ禁止は両立させる必要がある。
7. **"private"の範囲**: 先頭アンダースコア付きを指すと読むが、定義はない。
8. **pytorch等の禁止範囲**: 上記「解釈」のとおり、SDK経由の間接利用の扱いは明言されていない。
9. **KV cache・並列化・バッチ処理**: Subjectは禁止も許可もしていない。ボーナス節に「caching、batching」とあるのみ。
10. **Numberの出力形式**: 例は `2.0`。整数相当値を `2` と出すか `2.0` と出すかの規則は未定義。
