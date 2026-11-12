# 基盤実装の設計

この文書は、自然言語の依頼を関数呼び出しの JSON に変換する要求を、
どのデータと処理で実現するか説明する。現行設計の正本は本書であり、要件は
[Subject v1.7 の要約](subject-v1.7.md)とユーザーの指示に従う。
現行実装は§1〜9、今回対象外の未実装の旧提案は§10に分ける。
実行結果は[検証記録](implementation-verification.md)を参照する。

開発用の[連続負荷・熱監視](superpowers/specs/2026-11-11-safe-thermal-survey.md)は、
CLIのthermal入口、親の安全監視、公開SDKだけを使う子の連続負荷、JSONL記録で構成する。
アプリ本体・SDKの実装は変更せず、温まる過程の調査とSubjectの実アプリ5分判定を分ける。

案Aは設計コミット`42a1f8b`に対するユーザーの実装承認を受け、ローカルdevelopで実装した（2026-11-13）。
本書に残っていた未実装のLF区切り方式は、現行のJSON区切り方式と区別して§10へ移した。
§9.1は変更前の構造を比較用に残し、現行の呼び出し順と所有関係は§9.8に記す。

## 1. 要求から処理上の条件へ

利用者は、自然言語で書いた依頼について、用意した関数のどれをどの引数で呼べばよいかを知りたい。
このプログラムは関数呼び出しの記録を作り、対象の関数自体は実行しない。
関数名や引数の構成は実行時の定義に従うため、特定の関数セットの正解をコードに埋め込まない。

| 要求・要件と出典 | 処理上の条件 | 実現する設計 |
| --- | --- | --- |
| 自然言語から関数と引数を選ぶ。選択は LLM が行う（IV.3.1、V.1） | 説明と依頼をモデルに渡し、定義された関数を選び、その引数値を生成する | 関数選択と引数生成を別の段階にする（§3、ADR-0001・0002） |
| 関数定義と入力はレビュー時に変わる（V.2） | 名前・引数名・型を入力から取得し、実行ごとに生成と検証の条件を作る | 定義から名前の木、JSON の骨組み、引数検証モデルを作る（§2～4） |
| 入力の空配列はモデル起動前に入力不足のエラーとする（ユーザー、2026-11-10） | プロンプト配列に1件以上あることを入力検証で確認する | `load_prompts` が空配列を StartupError にし、モデル起動と出力更新へ進ませない（§2・5） |
| 出力先へ書けない場合はモデル起動前に終了する（ユーザー、2026-11-10） | 保存先の準備と実際の書き込み確認を入力検証の直後に行う | `prepare_output`が親ディレクトリを作り、同じ場所で一時ファイルの作成・書き込みを試す（§2・5、ADR-0007） |
| 桁区切り数値・引用符内の実改行を扱う旧提案 | 現行実装には適用しない | 未実装事項として§10へ分離。今回の生成制御変更には含めない |
| 重複キー・型の暗黙変換・数値の丸めを黙認しない（ユーザー、ADR-0017） | 解析前の値を保持し、構造と型を厳密に検証する | 共通JSONフック→strict Pydantic→数値保持検査（§2・3.3） |
| JSON の構文とスキーマを生成中に制約する（V.3.3、V.4） | 定義外の名前、キー、型、余分な文章を出力に入れない | 名前の木で候補を制限し、構造を強制し、値の接頭辞でトークンを制限する（§3・4） |
| エラーを処理し、原因を明確にする（IV.1、IV.3.1） | 起動の失敗、1件の生成失敗、保存失敗を識別して終了・継続を決める | StartupError、PromptError、保存の OSError を別の境界で扱う（§5） |
| 途中の成功結果を残したい（ユーザー、ADR-0007・0012） | 成功済みの出力を失敗した次の入力や途中書き込みで壊さない | 成功ごとに一時ファイルへ全成功結果を書き、置換する（§5） |
| 何が選ばれたかを見て調整したい（ユーザー、ADR-0014） | 入力と選択内容、生成中の引数、警告値を結び付けて表示する | 推論時の観察データを TUI へ通知する（§6） |
| 精度はパラメーターや部品ごとに改善したい（ユーザー、PR #44 の範囲） | プロンプトや選択・生成の変更を、保存や表示の処理から分けて検証できる | 変更箇所と受け渡すデータの契約を定める（§7） |

Subject の最終目標には正答率90%以上、JSON・型適合率100%、全テスト5分未満もある（V.5）。
基盤は生成条件の制約、失敗の通知、調整用の観察手段を提供し、正答率と時間は別々に測定する。
正答率90%を今回の基盤 PR の承認条件にはしない。測定で基盤の不具合が分かればレビュー対象にする。
5分目標について本書が示すのは、推論回数とマスク検査の量を抑える処理と、単独で時間を測る方法である。
対象入力・機材を含む最終的な達成判定は、基盤の処理契約の確認とは分けて行う。

## 2. 入出力と、起動時に準備するデータ

入力不備が原因の失敗を推論の失敗と混同しないため、ファイルを先に読み、検証してからモデルを起動する。
`main` は CLI のパスを `parse_args` で受け取り、`load_functions` と `load_prompts` を呼ぶ。
全ての所有JSON読み込みは`json_io.loads`へ集める。標準`json.loads`の`object_pairs_hook`で
各オブジェクトのキー重複を辞書への上書き前に拒否し、`parse_constant`でNaN・Infinityを拒否する。
小数は`parse_float=Decimal`で正確に読み、整数はintで受け取る。Pydanticの`validate_json`には
重複拒否の設定がないため、解析1回の後に`TypeAdapter.validate_python(..., strict=True)`で検証する。
関数定義には、空の関数一覧、空白だけを含む空の関数名・重複した名前、未対応の型、
必須フィールド欠落、余分なフィールド、不正な型を認めない。
空・空白の説明と引数名はPydanticの検証後に関数名・引数位置付きでlogging警告をstderrへ出し、
値をそのまま保持する。空のparametersは引数0個として認める。
入力JSONの配列を検証した後、`index_functions`が関数名をキーにする`dict[str, FunctionDef]`へまとめる。
重複は代入前に拒否する。辞書の挿入順を入力順として保ち、定義の一覧には`values()`、名前の一覧にはキーを使う。
プロンプトの配列には1件以上が必要で、空配列 `[]` は `load_prompts` が入力不足の StartupError にする。
この検証はモデル起動前に行い、既存の出力は変更しない（ADR-0012 の2026-11-10の変更）。

`Args`の`Path`型はパスの型を確認するだけで、実在や読み書き可否は検査しない。
入力はloaderが実際に読み、正常なら`prepare_output(args.output)`へ進む。
`prepare_output`は必要な親ディレクトリを作り、既存の出力先が通常ファイルか確認したうえで、
同じディレクトリに一時ファイルを開いて書き込み・flush・削除する。
パスの不備や書き込み失敗はStartupErrorにし、`Model()`を呼ばずに終了する。
既存出力の内容は変えず、出力ファイルがなければ確認時には作成しない。
入力検証に失敗した場合は、この出力先の準備も行わない。

| 境界を越えるデータ | 内容・条件 | 次の処理での用途 |
| --- | --- | --- |
| 関数定義のJSON配列 → `dict[str, FunctionDef]` | 関数名から定義への対応。重複を拒否し入力順を保持する | 値を選択肢の提示に、キーを名前の木に使い、選択後は名前で定義を直接取得する |
| 入力ファイル → `list[str]` | 1件以上。各要素の `prompt` は文字列。入力順を保持する | 1件ずつ処理し、同じ文字列を出力の `prompt` にコピーする |
| SDK → `Model` | 公開 API の encode、decode、logits、語彙ファイルのパス | モデルを使用する処理だけがこのラッパーを受け取る |
| 語彙ファイル → `dict[int, bytes]` | トークン ID と、そのトークンが表すバイト列 | 候補の追加で中間表現の型別文法が保たれるかを調べる |
| 関数名 → `NameTrie` | `encode(名前 + EOS)` の経路と、葉にある元の関数名 | 定義に存在する名前の経路だけを生成時に許す |
| 成功した1件 → `FunctionCall` | ちょうど `prompt`、`name`、`parameters` を持つ | 検証済みの呼び出し記録として保存する |

現在の設計が扱う型は `number`、`string`、`boolean`。関数定義の引数・戻り値とも、この3型で検証する。
これは対応範囲についての設計判断であり、Subject に型が3種類と明記されているわけではない（ADR-0004）。
型や JSON の構造を変えずに、関数名、引数名、個数、説明を変えられる。

モデルは起動時に一度読み込む。語彙と名前の木も、その実行の全入力で共有する。
語彙は `load_vocab`、名前の木は `build_name_trie` が準備する。
SDK の encode は Tensor を返すため、`Model.encode` が `tolist()` で普通の ID リストにする。
語彙の辞書は ID の意味を調べるもの、`tolist()` は SDK の戻り値の型を変えるものであり、用途が異なる。

## 3. 1件の依頼を呼び出し記録へ変える

`process_prompt` は、関数を選ぶ `select_name`、引数を作る `generate_parameters`、
生成した記録を確認する `validate_call` の順に進む。それぞれが次の段階へ渡すものを決め、
表示や保存のためにモデルの選択結果を作り直さない。

![起動から保存までの実行順](diagrams/skeleton-flow.svg)

[全体図の D2](diagrams/skeleton-flow.d2)

1件の処理の現行の呼び出し順は§9.8に記す。既存の詳細図はLF方式の未実装案なので§10へ分離する。

### 3.1 関数の説明と依頼から、定義にある名前を選ぶ

「選択は LLM が行う」と「定義された名前だけを使う」を両立するため、
意味の選択にはモデルの点数を使い、名前として許す経路は入力された関数定義から制約する。

`build_name_prompt`は辞書の`values()`から全関数の名前・説明・引数名と型を入力順に列挙する。
利用者の依頼を添え、関数名だけを応答するよう指示する。`select_name`はこれをencodeし、自分の明示ループで木を照会して`select_next_token`を呼ぶ。
生成中は `next_tokens` で現在の経路の子を取得し、その ID だけを次の候補にする。
名前とEOSの後ろの葉に到達したら`leaf_name`で元の名前を取得し、`functions[name]`で定義を返す。
読み込み時に作った辞書を使い、選択のたびに関数の配列を再走査しない（ADR-0016）。

名前の末尾には Qwen の応答終了トークン `<|im_end|>` を含める。コード上の名前は `EOS` とする。
例えば `fn_add` と `fn_add_numbers` が共存するとき、`fn_add` の後で EOS と `_numbers` の経路が分かれる。
未完成の名前では EOS を許さない。SDK の decode は特殊トークンを表示しないため、
終了は表示文字列ではなく ID の経路で判断する。生成上限は名前の木の最長経路から計算する。

名前選択はここで完了し、引数生成は別の入力列で始める。名前選択で生成した EOS は
引数値の生成列に流用せず、選んだ名前を最終出力の JSON 文字列として組み立てる。
EOS はユーザーが承認した暫定方式で、障害となった場合に見直す（ADR-0013）。

変更前の共通ループの図は§9.1に比較資料として残す。現行の順序は§9.8を参照する。

### 3.2 決まったJSON構造と引数値を順に組み立てる（現行実装）

`generate_parameters`は引数用プロンプトをencodeし、promptと選択した名前を
`json.dumps`でエスケープしたJSON先頭を作る。この先頭をtextとモデル入力idsの両方へ追加する。
`build_skeleton(fn)`の要素を定義の順に処理する。

- `Forced`なら、そのtextを最終文字列へ足し、encodeしたIDもモデル入力へ足す。
- `ValueSlot`なら、引数名・型を通知して値表示とトークン数を初期化し、`generate_value`を呼ぶ。
  返るのは値と区切りを含むID列である。SDKの公開decodeで文字列へ変換してtextへ足し、ID列も入力へ足す。

`ValueSlot.terminator`は途中の引数なら`,`、最後の引数なら`}`である。
各値が自分の後ろの区切りを生成し、Forcedは同じ区切りを重ねない。最後にForcedで記録全体を閉じる。
引数が0個なら、空のparametersと閉じ括弧をForcedで書き、値の推論はしない。

例えばnumberのaとbなら、固定部分`"a": `、値`12,`、固定部分` "b": `、値`3}`、
記録を閉じる`}`の順になる。完成した値を別方式で正規化する処理は現在存在しない。
数値が単独で成立しても区切りが出るまでは続ける。`12`は`123`の途中かもしれないためである。

旧LF方式の図と説明は§10へ分離した。現行処理の説明にその図を使わない。

### 3.3 JSON と定義の両方で検証した値を保存へ渡す

生成後の検証は、構文制約と骨組みの実装に不整合があった場合の確認境界となる。
`validate_call` は全文を `json_io.loads` で読み、strictな`FunctionCall`で記録のキーと型を検証する。
`call.name`が選択済みの`fn.name`と完全一致することも確認し、不一致なら期待した名前と実際の名前を含むPromptErrorにする。
続いて `build_parameters_model` が選択済みの定義から作ったstrictなモデルで、必須引数、余分なキー、型を確認する。
内部のフィールド名とは別に元の引数名を保持し、`_private` や `model_dump` のような名前も出力に戻す。
numberはJSON整数・小数の両方を受け付けるが、bool・数値文字列・全角数字からの変換はしない。
booleanも整数・文字列から変換しない。これは構造化済みJSONの検査であり、依頼文の数値表現の解釈とは異なる。

JSON整数はintのまま保持する。小数・指数のDecimalには`validate_number`をPydanticの
BeforeValidatorとして適用し、有限のfloat候補を`json.dumps`→Decimalで再解釈して元の値と比較する。
一致ならfloatを使い、丸めがある整数値なら有限範囲内でintへ戻す。それ以外は精度損失エラーとする。
`0.1`はJSON再出力が一致するため受理し、二進数として厳密な表現は要求しない。
非有限値、オーバーフロー、非ゼロ値のゼロへのアンダーフローも拒否する。
戻り値には検証後の引数を入れ、標準JSON保存でnumber型を保つ。Decimalの文字列化や独自JSON出力器は使わない。

制限の根拠はPythonのint文字列変換の実行時上限、floatの有限範囲と再出力精度、既存の256トークン上限である。
小数・指数表記はfloat候補が有限である必要があり、巨大な整数表記と受理範囲が異なる。
長い小数は保存精度を保持できなければPromptErrorにし、正当な短い値に見せかけない。
これは実装上の制約であり、Subjectの追加要件とはしない。

語彙・測定fixture・環境調査・開発用入力チェックのJSONも同じ重複拒否・数値解析方針を使う。
配布SDK内部の解析は変更しない。LF方式を実装する場合にも、正規化後へこの値保持契約を適用する。

## 4. 候補トークンの判定と終了条件

`select_name`と`generate_value`が自分の生成列と上限を所有する。
名前は木、値は型の文法を直接照会し、共通の`select_next_token(model, ids, allowed)`で1トークンを選ぶ。
制約判定・終了判定・採用後の処理をコールバックへ渡さない。

1. 完了なら名前は元のFunctionDef、値は生成ID列を返す。上限ちょうどの完了も成功する。
2. 未完了で生成上限に達していたら、終端未到達を示す PromptError にする。
3. 続けられる候補がなければ PromptError にする。
4. 候補が1個ならその ID を強制する。複数なら logits を一度取得する。
5. `choose_token` が候補外を負の無限大にした点数列から最大の ID を選び、生成列へ追加する。

元の logits は変更しない。点数列の型・次元、NaN、正の無限大、許可候補がすべて非有限の場合を確認する。
値の上限は暫定256トークンで、未完了の値を切り詰めて成功扱いにはしない。
`generate_value` は失敗に引数名と型を付け、例えば「text の string 値が256トークンでも終端に達しない」と通知する。

### 値の文法とバイト列の扱い（現行実装）

`generate_value`のループ先頭で、語彙から生成済みのバイト列を連結する。
`is_value_complete(written, slot.type, slot.terminator)`が真なら完了し、
偽なら`allowed_value_tokens`で次のIDを求める。トークン全体を加えた接頭辞が正当か確認し、
区切りを越えて次の値や余分な文章を含むトークンを拒否する。

| 型 | 許す値の形 | 完了条件 |
| --- | --- | --- |
| number | 符号付き整数・小数。桁区切り・先頭プラス・指数は生成しない | 有効な数値とスロットの`,`または`}` |
| string | JSONの二重引用符・エスケープ・UTF-8。引用符内の生の制御文字は拒否 | 閉じ引用符の後にスロットの区切り |
| boolean | 小文字のtrueまたはfalse | 完成した単語の後にスロットの区切り |

値の前後に文法が認める空白は許すが、区切り後の追加文字は許さない。
UTF-8の文字が複数トークンへ分かれるため、文法は文字列のdecode結果ではなくバイト列で判定する。
stringでは共通接頭辞の状態を一度求め、各語彙候補の追加分を検査する。
未完成UTF-8を保持するマスク処理と、表示用decodeで未完成文字を省く処理は別である。

## 5. 失敗と、保存する結果の扱い

Subject はエラー処理を求めるが、1件失敗した後の出力内容までは定めていない。
ここではユーザー承認済みの方針に従い、起動時エラーでは終了し、1件の失敗は記録して次へ進む。
成功済みの呼び出しを入力順に保持し、`save_results` は出力先と同じディレクトリの一時ファイルへ書いてから置換する。

| 発生した条件 | 継続と終了コード | 出力ファイル |
| --- | --- | --- |
| 不正な CLI 引数 | 推論前に終了、非0 | 書き換えない |
| 入力のプロンプト配列が空 `[]` | モデル起動前に入力不足の StartupError を表示して終了、1 | 書き換えない。出力ファイルがなければ作成しない |
| 出力先の準備・書き込み確認の失敗 | モデル起動前に出力先と原因を表示して終了、1 | 既存出力は保持し、確認用の一時ファイルを削除する。準備途中の親ディレクトリは残り得る |
| 入力・定義・モデル・語彙・名前の木の起動失敗 | StartupError を表示して終了、1 | 書き換えない |
| 1件の選択・値生成・検証の失敗 | PromptError と入力位置を表示して次へ進み、最終的に1 | 失敗した件は保存しない。成功があればその都度更新する |
| 1件以上の入力を処理したが全件失敗した | 1 | 今回の成功件数を表す `[]` を保存する |
| 保存失敗 | 原因と出力先を表示して直ちに終了、1 | 置換前の最後のファイルを保つ。後続の推論をしない |

要求上は各入力に1記録を作ることを目指すが、上限や失敗処理を設けるだけで全件の生成成功を
保証できるわけではない。成功した記録の構文・型の条件と、全入力が成功したかは分けて確認する。
SDK 呼び出しの例外は `Model` が起動時または1件の失敗へ変換する。
理由と検証範囲は ADR-0007・0012、および検証記録にある。
出力の事前確認は、その時点の一時ファイル作成・書き込みを確かめるもの。
後の状態変化や最終置換の失敗もあり得るため、保存時の例外処理は維持する。
出力先パスの最終要素がシンボリックリンクなら、その名前に新しい通常ファイルを保存し、
リンク自体を置き換えてよい。リンク先の内容は更新しない（ユーザー決定、ADR-0007）。

## 6. 選択を観察し、調整へつなげる

以下は現在の実装の説明である。生成制御・表示境界の採用理由と検証は§9に記す。
過去の未承認案は採用根拠にせず、現行コードから比較した。

ユーザーがプロンプトや部品を調整するとき、どの入力のどの段階で何が選ばれたかを見られるようにする。
`decoder.select_next_token`は採用ID・選択方式・要求時だけraw_topを返す。
`pipeline`の名前/値ループはID追加後にGenerationProgressへ同期通知し、表示から制御判断を受け取らない。
共有資源はGenerationResourcesが保持し、mainが具体的なDashboardを別引数で接続する。
通知先の後付け代入、Run.say、TokenChoice、生成制御コールバックは廃止した。

入力番号、段階、依頼、関数、引数と型、途中の値・件数に加え、名前/値とも現在の採用IDとforced/greedyを表示する。
名前のID列はSequenceとして借り、表示は変更・保持せず同期的に整形する。
トークンごとに通知するが、描画は既存の間隔で最新状態を表示する。全ステップの履歴保存は行わない。
候補ランキング、候補表専用のlogit・allowed_mass・gap、値生成のraw_top計算は存在しない。

名前選択で制約前のargmaxが採用IDと異なる場合は警告する。これは制約前の最大値がマスクされたことを示す。
該当関数なしとは判定せず、許可候補から最大のものを選ぶ。
警告に必要なraw_topのIDだけを同じlogitsから取得し、上位k件の取得やソートは行わない。
1候補の位置では追加推論をせず、forcedとして途中経過を通知する。
これらはプロンプト比較の補助情報であり、関数の正答率そのものではない（ADR-0011・0014）。

stderr が端末なら同じ領域を更新し、リダイレクト時は制御文字のない通常ログ(警告と要約)にする。
どちらも最後に、プロンプトごとに完了か失敗かと秒数を 1 行ずつ、最後に全体の行を残す(ADR-0021)。
端末のときは代替画面を出た後に書く。全体の時間はモデルの読み込みを含まない。
入力中の制御文字をエスケープし、終了・例外・中断時に端末状態を戻す。
診断表示用の decode が失敗したときは ID と取得不能の表示を使い、選択した結果を捨てない。
実際の値を組み立てる decode の失敗は、1件の生成失敗として扱う。
stderr への書き込みで OSError が起きたら表示を止め、生成と保存は続ける。

## 7. 部品の境界と、変更時に保つ条件

精度調整のためにモデルの呼び出し、文法、保存、表示を一つの処理へ集めず、
以下の受け渡しを保つ。実装は `src/` の直下に置き、モデルを差し替えるための抽象インターフェースは設けない。

| 調整・変更するもの | 主な担当 | 他の処理へ渡す条件 |
| --- | --- | --- |
| モデルへ与える説明・形式 | `prompt.py` の設定値と組み立て関数 | 名前段階・引数段階それぞれの文字列を返す。名前の応答形式は EOS の木と整合させる |
| 関数を選ぶ処理 | `pipeline.select_name`、`name_trie.py` | 入力された定義の `FunctionDef` を返す |
| トークンを選ぶ方法 | `decoder.py` | 許可IDから1つ選び、ID・方式・任意診断を返す。列や上限は管理せず、観察で選択を変えない |
| 引数値の許可文法・終了条件 | `values.py`、`generate_value` | 値とスロットのJSON区切りのID列を返す。文字列の中の区切り文字では終了しない |
| JSON全文の組み立て | `generate_parameters`、`skeleton.py` | 値のdecode結果とForcedを最終文字列へ追加し、両方のID列をモデル入力へ継続する |
| 値の生成上限 | `pipeline.MAX_VALUE_STEPS` | 超過を値の短縮で隠さず、理由付きの PromptError にする |
| 引数の型と数値の扱い | `schema.py`、値の文法、保存 | 定義のキーを保持し、検証後の値を出力する。変更時は生成と検証の双方を確認する |
| SDK との接続 | `model_port.py` の具体的な `Model` | 公開 API だけを使い、encode と logits は普通のリスト、decode は文字列を返す |
| 表示とログの形式 | `display.py`の同期受信・整形 | 選択結果と保存データを変えない |

実装上の制約は、Python 3.10以上、型注釈・docstring、Subject の flake8/mypy の条件、
許可された依存関係、SDK の公開 API の利用である（Subject 要約 §1・3）。
検証するデータにはPydanticを使い、GenerationResourcesはInstanceOfで参照型を検証する。
Model・Dashboardは既存の通常クラス、GenerationProgressは静的なProtocolである。
Subjectの全クラス要件の解釈上の留保と、検証境界は§9.9に記す。
SDKの既存のfloat32変更と復元手順は開発文書を参照する。

## 8. 要件を実現できているかの確認

以下は現行実装の確認条件である。§10の旧提案を検証済みという意味には使わない。

| 設計で確かめる条件 | 確認方法 |
| --- | --- |
| 入力変更から生成条件を作る | 入力・定義を変えた loader の直接テストと、変更関数セットの実モデル評価 |
| 名前から定義を取り出し、入力順と重複拒否を保つ | `test_function_registry.py`で辞書の順序、プロンプトへの掲載、固定logitsの選択、測定fixtureの重複拒否を確認 |
| 名前の接頭辞が衝突せず定義外へ進まない | `test_prefix_names_choose_end_or_continue` と木の直接テスト |
| 値の文法、UTF-8、区切りの境界を保つ | 現行JSON文法の直接テストで、値・スロットの区切り・UTF-8・終端越えを検査する |
| マスクと強制が指定どおりに働く | 固定 logits で候補外の最大値を排除し、1候補の推論省略と上限ちょうどの終了を確認 |
| 任意の引数名と検証後の型を保持する | `test_arbitrary_parameter_names_survive_validation` と欠落・余分なキー・非有限値の直接テスト |
| 失敗時の出力契約を保つ | 一時ファイル、置換失敗、部分成功、空配列の起動時エラー、全件失敗のテスト |
| 出力先の不備で不要なモデル起動をしない | `test_output_preflight.py`でディレクトリ・権限・パス・flush失敗を調べ、モデル未起動と出力保持を確認 |
| 観察で結果を変えず、端末を戻す | 通知あり・なしの比較、診断 decode 失敗の回帰テスト、疑似端末の実行 |

テストの所在は [アプリケーション](../tests/test_application.py)、[文法と木](../tests/test_constraints.py)、
[端末表示](../tests/test_terminal_progress.py)、[処理全体の制御](../tests/test_skeleton_flow.py)。
出力先の早期確認は[起動前の出力先検査](../tests/test_output_preflight.py)にある。
辞書の受け渡しは[関数名と定義の対応](../tests/test_function_registry.py)にある。
実モデルの精度、時間、測定しなかった条件は検証記録に分けて残す。
この表は確認すべき条件と方法を定めるもので、各検査を実行した結果は検証記録を参照する。

本実装のレビューでは、各処理がこの条件を守り、部品の調整を観察・検証できるかを確認する。
EOS、暫定上限、数値の精度、未対応の型などを変更するときは、要件と本書の該当箇所を見直す。

D2図はTUIへの通知と整形を省略している。§9.1の名前図は変更前の共通ループ、
§10のLF方式の図は未実装の旧提案を示す。現行案Aの具体的な呼び出し順は§9.8を参照する。
配色は [workflow-style.d2](diagrams/workflow-style.d2)、生成方法と余白の指定は各 D2 ソースにある。

## 9. 実装済み：明示的な生成ループと1トークン選択（案A）

設計コミット42a1f8bの案A・共有資源の案C・トークン観測契約をユーザーが承認し、実装した。
この節は採用理由・代替案・検証境界を含む現行設計である。§9.1だけは変更前の比較記録である。
選択関数に上限を持たせず、資源に表示先や生成途中の状態を含めない。検証済みの辞書や木の再走査も行わない。
過去の未承認案・履歴文書を根拠に採用せず、現行コード・Subject要約・既存テストから比較した。
ランキング、LF方式、プロンプト調整、Dashboard全体の状態モデル再設計は対象外とする。

### 9.1 変更前の呼び出し順と隠れた依存（42a1f8bまで）

```text
main → _start → 入力検証・出力準備 → Model・語彙・名前の木・Runの構築
     → Dashboardをwithで開始 → run.report = dashboard.report
     → _process_all → 入力とpromptをRun.sayで通知
       → process_prompt
         → select_name → プロンプトencode → decoder.generate
           → allowed_next → leaf_name / next_tokens
           → 上限・空候補確認 → forced または next_logits → choose_token
           → generatedへ追加 → observed → Run.say → Dashboard.report
         → generate_parameters → プロンプトとJSON先頭をencode
           → build_skeletonの順にForced / ValueSlotを処理
           → generate_value → decoder.generate
             → allowed_next → bytes連結 → is_value_complete / allowed_value_tokens
             → 共通の選択・追加 → observed → Run.say → Dashboard.report
           → 値をdecodeし、textとidsへ追加
         → validate_call → JSON解析・FunctionCall・引数モデルの検証
       → 成功ごとの保存 → 成功または失敗を通知
     → Dashboard終了 → 端末復帰・要約 → 終了コード
```

両方の`allowed_next`はNoneで完了、listで継続を表す。decoderはどの文法を処理しているか知らない。
コールバックは許可IDを返すものと、選択後の副作用を持つものに分かれ、役割は同じではない。

| 箇所 | 役割・入力・出力 | 外側から暗黙に読むもの・副作用 |
| --- | --- | --- |
| select_name.allowed_next | 生成ID列→完了または許可ID | ローカルtrie。木の照会のみ |
| select_name.observed | TokenChoice→戻り値なし | runのmodel・vocab・report。表示用decode、警告整形、stderr通知 |
| generate_value.allowed_next | 生成ID列→完了または許可ID | ローカルvocabとslotの型・区切り。バイト列を再構成して文法検査 |
| generate_value.observed | TokenChoice→戻り値なし | vocabとrun。UTF-8表示文字列と件数の整形、stderr通知 |
| decoder.next_logits | 入力ID列→logits | 通常はModel.logitsの束縛メソッド。クロージャではないが推論先の間接化である |
| Run.report / say | label・text→戻り値なし | mainが後から代入した通知先。生成制御ではなく表示資源への間接経路 |

変更前のループは、`generate`から判定関数へ戻り、また`generate`から観測関数へ戻る。
呼び出し元を読むだけでは、完了・上限・推論・追加・通知の実行順が並んで見えない。
TokenChoiceは変更前には観測時だけPydantic検証され、生成ID列のcopyも観測時だけ作られる。
名前警告でしか使わないraw_topは、変更前は観測付きの値生成でも計算されている。

変更前の名前生成図（現行インターフェースを表す図ではない）:

![名前の候補を制約する処理](diagrams/select-name.svg)

[select_name の D2](diagrams/select-name.d2)

### 9.2 共通処理と段階固有の処理

| 処理 | 名前の生成 | 値の生成 | 現行の所有者 |
| --- | --- | --- | --- |
| プロンプト・初期ID | 関数一覧から名前用入力 | 選択関数と固定JSONを含む入力 | 各pipeline関数 |
| 終了判定 | leaf_nameでEOS後の葉を確認 | バイト列とslot.type・terminatorでis_value_complete | 各ループに直接記載 |
| 許可ID | next_tokens | allowed_value_tokens | 各ループに直接記載 |
| 生成上限 | 木の最長経路 | MAX_VALUE_STEPS | 各ループ。終了確認後、選択前に確認 |
| 空候補・forced・モデル呼び出し | 同じ順序 | 同じ順序 | 共通の1トークン選択関数 |
| 負の無限大マスク・argmax・不正logits拒否 | 同じ | 同じ | 既存choose_tokenを維持 |
| 生成列への追加 | 名前のローカル列 | 値のローカル列 | 各ループに直接記載 |
| 結果の解釈 | 選択FunctionDefを返す | 値とJSON区切りのID列を返す | 各pipeline関数 |
| エラーの文脈 | 名前生成のエラー | 引数名・型を付けたPromptError | 共通エラーを各段階で扱う |
| 採用IDと選択方式 | 名前の各ステップ | 値の各ステップ | 共通の1選択関数がIDとmodeを返し、各ループが表示へ通知 |
| 診断値の取得 | 表示が利用可能なときだけraw_topを要求 | 診断を要求しない | 共通の1選択関数内で要求時だけ計算。modeとは独立 |
| 文字列整形・描画 | 名前の途中と制約警告 | 値の途中とトークン数 | display。生成側はIDと既存の型付きデータだけを渡す |

### 9.3 案Aの基本構造（承認済み）

名前と値の各ループが、完了・許可集合・上限・追加・通知を通常の呼び出し順に管理し、
1トークンのforced/推論/制約付きargmaxだけを共通化する。短い上限チェックの重複は許容する。
比較した案Bは、共通ループへ木またはslot/vocabを渡して分岐する方式である。
ループ自体は1箇所にできるが、2つの文法・終了・表示の文脈を共通制御へ集める必要があるため、
今回の順読みを優先して案Aを選ぶ基本方針が承認された。
以下は基本構造を再選択するものではなく、案Cの具体化と、ID/方式を両段階で観測する契約の承認前レビューである。

### 9.4 観測情報と、確定する場所

変更前のdecoder.TokenChoiceにはtokenとforcedがあった。一方、acff8d9の提案では選択結果が
(token, raw_top)だけになり、値の表示にはbytesと件数しか渡していなかった。
名前のIDはID列の末尾から読めるが、値の末尾トークンIDはbytesから一意に復元できない。
またraw_top=Noneはforcedだけでなく「greedyだが診断未要求」でも起こる。選択方式の代用にはならない。

| 情報 | 確定する場所・根拠 | 観測する処理・用途 |
| --- | --- | --- |
| 採用ID | select_next_tokenの1候補分岐、またはchoose_tokenの戻り値 | 名前/値の両方。画面の採用ID |
| forced / greedy | select_next_tokenで実際に通った分岐 | 名前/値の両方。推論を省略した確定か、モデルの制約付きargmaxか |
| raw_top | greedyかつ要求時だけ、同じlogitsのargmax | 名前だけ。制約警告に使い、選択方式の表示には使わない |
| 名前の途中・ステップ数 | 名前ループのgenerated | 名前の公開decode、末尾ID、lenによる件数と警告のstep |
| 値の途中・ステップ数 | 値ループのgeneratedとvocab | 表示するbytesとlen。IDはbytesから逆引きせず選択結果を渡す |

選択関数の戻り値は次の3値である。新しい結果クラスは作らない。

```python
# decoder.pyに置く、選択方式そのものの型別名
SelectionMode = Literal["forced", "greedy"]

def select_next_token(
    model: Model,
    ids: list[int],
    allowed: list[int],
    *,
    include_raw_top: bool = False,
) -> tuple[int, SelectionMode, int | None]: ...
```

1候補なら(token, "forced", None)を返し、推論・診断・配列変換はしない。
複数候補なら1回のModel.logitsと既存choose_tokenを使い、(token, "greedy", raw_topまたはNone)を返す。
同点や有限の許可候補が1つしかない分布でも、複数候補からモデルを呼んだならgreedyである。
空候補・不正logitsのエラー、マスク・argmax・同点規則は現行を維持する。

modeは常に返し、表示の有無によって計算方法を変えない。raw_topだけを要求時に取得する。
forcedならinclude_raw_top=Trueでもraw_topはNone、greedyならFalseの場合もNoneになる。
名前ループは進捗が利用可能なときだけTrue、値ループは常にFalseで呼ぶ。
警告はgreedy・raw_topあり・採用IDと異なる場合に現在と同じ意味で出す。

生成途中のID列・step・上限は戻り値へ入れない。modeを別の可変状態として保存せず、
各ループで`token, mode, raw_top = select_next_token(...)`と受け取り、そのステップの通知までに使う。
全logits返却やModel.last_logits等の隠れた状態、追加forward、softmax、上位k件は採らない。

ここでのforcedは「許可IDが1つなので推論を省いた選択」である。
build_skeletonのForcedは固定JSON部品で、選択ループのステップではない。
固定部品をencodeした各IDについて、選択していないトークン観測や架空のstepを追加しない。

### 9.5 観測インターフェースの比較と推奨

| 方式 | 利点 | 欠点・判断 |
| --- | --- | --- |
| 段階別メソッドで必要なスカラーを渡す | 1回の通知で方式・採用ID・途中内容を同じステップとして扱える。クラス生成や動的なpayload判別が不要 | 共通のmodeは両メソッドにあるが、それぞれの利用先が明確。推奨 |
| 共通token_selectedと途中表示を別々に通知 | 共通メソッドにIDとmodeを集約できる | 2通知の途中で描画するとID/方式と途中内容が別ステップになり得る。同期のための追加状態や手順が必要なので採らない |
| TokenObservation等の不変Pydanticモデルへまとめる | 名前付きフィールドで共通情報を読め、不正な外部イベントを受けるなら検証に意味がある | 今回は内部の選択結果。各トークンでの構築・再検証か、表示用の二重包装が増えるため採らない |
| 段階と共通payloadを1つの汎用通知へ渡す | メソッド数は減る | bytes/名前ID列・任意のraw_top等の組合せ判定が必要で、labelによる意味の切替に戻りやすいので採らない |

単純な引数追加を出発点にせず、名前と値で必要な途中表現が違うことと、1ステップの情報を揃えて描くことから、
既存の段階別メソッドを見直す。name_tokenではID列の末尾を使い、同じ採用IDを別引数にも持たせない。
value_tokenではそのIDを復元できないため、採用IDを明示する。modeの意味は両方で同じである。

GenerationProgressは引き続き小さな構造的型契約とし、mainがDashboardを接続する。
具体的なDashboardを直接渡す案は型宣言を減らせるがpipelineが端末実装へ依存する。
共有バッファや別スレッドの描画は終了・復帰管理を増やすため、今回も採らない。

```python
Stage = Literal["select_name", "generate_parameters", "validate_call"]

class GenerationProgress(Protocol):
    @property
    def available(self) -> bool: ...

    def stage(self, stage: Stage) -> None: ...

    def function_selected(self, name: str) -> None: ...

    def parameter_started(self, name: str, param_type: ParamType) -> None: ...

    def name_token(
        self,
        tokens: Sequence[int],
        *,
        mode: SelectionMode,
        raw_top: int | None,
    ) -> None: ...

    def value_token(
        self,
        written: bytes,
        step: int,
        *,
        token: int,
        mode: SelectionMode,
    ) -> None: ...
```

SelectionModeはdecoderにある型別名を型検査用に参照する。progress.pyは将来アノテーションと
TYPE_CHECKINGを用い、実行時にdecoder・pipeline・display・SDKをimportしない。
StageとParamTypeも型別名で、Enumや新しい観測クラスは導入しない。
ProtocolのPythonクラスとしての性質とSubjectの留保は§9.9に記す。

| 通知 | 呼び出す場所 | データの利用先 |
| --- | --- | --- |
| stage | process_promptで各段階に入る直前 | 段階表示。引数生成開始時に現在トークン欄を初期化 |
| function_selected | select_nameで葉から名前が確定した時 | 確定した関数名の表示 |
| parameter_started | generate_parametersでValueSlotへ入る前 | name/type表示、値・件数・現在トークン欄の初期化。terminatorは不要 |
| name_token | 名前ループで採用ID追加後 | tokens[-1]を採用ID、modeを方式として表示。ID列を公開decodeし、raw_topで従来の警告を作る |
| value_token | 値ループで採用ID追加後 | tokenを採用ID、modeを方式、writtenを途中の値、stepをトークン数として表示 |
| available | 診断要求と各通知の前に読む | 診断ストリームが利用可能か。isattyや生成の継続フラグではない |

名前の通知は生成ループ所有のlistをSequence[int]として同期的に借用する。Sequenceは静的な読み取り専用契約であり、
実行時の不変性を保証しない。表示側は要素を変更せず、呼び出し終了後までID列への参照を保持しない。
必要なdecode・整形を呼び出し中に完了し、Dashboardには整形済みの文字列だけを保存する。
採用IDはtokens[-1]、件数はlen(tokens)から読み、mode/raw_topの引数構成は維持する。
値はbytes・整数・文字列Literalを渡す。全メソッドの戻り値はNoneであり、生成制御には使用しない。

Sequenceはcollections.abcからimportする。表示側のdecode補助関数もSequence[int]を受け取り、
既存Model.decodeのlist[int]境界でのみ`tokens if isinstance(tokens, list) else list(tokens)`とする。
通常の生成listにはコピーが発生せず、tuple等の場合だけこの境界でO(n)のlistコピーが1回発生する。
Model.decodeと配布SDKの公開decode実装を確認したところ、listはそのまま下位のtokenizerへ渡される。
下位実装内部のメモリ確保までゼロとは主張しない。入力listの非変更は実装後のSDK回帰試験でも確認する。
悪意ある受け手による変更を実行時に防ぐ契約ではなく、表示実装のレビュー・型検査・回帰試験で守る。

値のwrittenは、現在observedで行う語彙バイト列の連結をループ側へ移して作る。
bytesの取得は生成済み内容の観測であり、表示用decode・repr・文字列整形ではない。
名前と値の表示用decode、警告文、ID/方式の表記、制御文字処理はdisplayへ集約する。

表示は例えば`Token: 42 (forced)`または`Token: 42 (greedy)`という1つの現在トークン欄を追加する。
logitや確率の欄、ランキングは追加しない。name_token/value_tokenは各ステップの値・件数・ID/方式を
まとめてfieldsへ反映してから描画し、新しいIDと前ステップの途中内容を混ぜて描かない。
これは受信メソッド内の局所的な更新であり、新しい状態機械や通知のトランザクション機構は作らない。

全ステップで通知し、画面は既存の描画間隔で最新状態を表示する。高速なステップを含む全履歴の目視・保存は保証しない。
表示待ち・sleep・各トークンのログ保存を新設せず、既存の性能特性をまず維持する。
リダイレクト時もavailableは真になり得る。通知は受けるが、現在と同じ警告と要約を残し、ANSIや全トークンのログは出さない。
stderrのOSError後はavailableを偽にして表示と追加診断を省くが、選択・保存は続ける。

### 9.6 案C：上位で共有資源をまとめ、下位へ必要な依存を渡す

案Cはユーザーが採用した管理方針である。全依存を個別引数にする前案は上位処理について撤回する。
具体的な構造体をGenerationResourcesとし、pipeline.pyの上位処理入口の近くに定義する。
生成サービスやFactory階層は作らず、既存_startが組み立てる。

```python
class GenerationResources(BaseModel):
    """Validated references shared by one application run."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: InstanceOf[Model]
    vocab: InstanceOf[dict[int, bytes]]
    functions: InstanceOf[dict[str, FunctionDef]]
    trie: InstanceOf[NameTrie]
```

このPydanticモデルは通常の型付きdictフィールドを再検証するモデルではなく、検証済み参照の組立境界である。
Pythonオブジェクトから構築し、外部JSONから読み込んだり保存したりする用途を作らない。

| フィールド | 組立時に検証すること | そこで検証しないこと |
| --- | --- | --- |
| model | Modelのインスタンスであること | SDKの初期化済み状態や推論の正しさ |
| vocab | dictのインスタンスであること | ID/bytesの全走査。内容はload_vocabで検証・変換済み |
| functions | dictのインスタンスであること | 各FunctionDefや空/重複の再検証。load_functions/index_functionsが担当 |
| trie | NameTrieのインスタンスであること | ノードの再構築・経路の再走査。build_name_trieが担当 |

必須4フィールドの存在・取り違えた参照型・余分なreport等を拒否し、通常代入による参照の差し替えを拒否する。
InstanceOf[dict[...]]はPython入力でdict自身の型だけを調べ、型引数が示す中身を検証しない点を明記する。
構造体に入ったことを「辞書内容が検証済みである証明」とせず、生成元の順序と組に不変条件を持たせる。

- mainは1実行分のGenerationResourcesを所有し、_process_all/process_promptへ同じ参照を渡す。
  測定ツールも同じ関数セットの評価バッチで一度作り、プロンプトごとには再構築しない。
- _startは入力検証と出力準備の後にModel、vocab、同じfunctionsからのtrieを順に作って組み立てる。
  必ずこの作成順を使い、異なる関数セットのtrieを後から混ぜない。対応の正しさを型チェックだけで保証したとは扱わない。
- frozenは浅い不変性である。辞書・木の子辞書・Model内部を深く不変化するものではない。
  共有辞書・木は構築完了後に書き換えず、モデルの内部処理は既存SDKの資源管理に任せる。
- generated、入力の継続ids、JSON text、現在のslot、step、mode、raw_top、progressは含めない。
  report/sayや通知先の後付け代入も作らない。表示資源と生成途中の状態は別の所有者が持つ。
- 組立のValidationErrorは_startでStartupErrorへ変換する。既存の起動エラー処理へ接続し、
  失敗を隠すためmodel_constructや全フィールドのSkipValidationを使わない。

| 構造体の選択肢 | 判断 |
| --- | --- |
| 通常dataclassやNamedTuple | 参照は簡潔に束ねられるが、新しい実行時クラスに対するPydantic検証がなくなるため採らない |
| 全内容を通常のPydanticフィールドとして受け直す | vocab/functions等の走査やcopyが重複するため採らない |
| 参照型を検証するfrozen BaseModel | 組立契約を検証し、loaderの仕事と生成状態を持たない。推奨 |

上位は資源セットを扱い、下位の関数には必要な依存だけをキーワードで渡す。

| 関数 | 受け取る資源と利用 |
| --- | --- |
| _process_all / process_prompt | resourcesを受け取り、必要なフィールドを下位へ展開。progressは別引数 |
| select_name | model・functions・trie。vocabや資源構造体全体は渡さない |
| generate_parameters | model・vocab。fnは選択済みの処理対象。progressも別に渡す |
| generate_value | model・vocabと処理対象slot/ids。functions・trieは渡さない |
| select_next_token | model・ids・allowedと診断要求だけ。共有構造体・表示先・上限は渡さない |

DashboardもGenerationResourcesを丸ごと受け取らず、mainから必要なmodel/vocab参照と件数だけを受け取る。
vocabはmainでMappingProxyType(resources.vocab)という読み取り専用viewにして渡し、辞書内容はcopyしない。
名前の公開decodeと警告表記だけに使い、表示側から共有語彙への書き込み経路を作らない。
Modelの所有権をDashboardへ移すわけではなく、表示メソッドでは公開decodeのみを使う。

### 9.7 更新した型付きシグネチャ

以下は実装したインターフェースである。GenerationResourcesの具体定義は§9.6、GenerationProgressは§9.5を使う。

```python
def process_prompt(
    user_prompt: str,
    resources: GenerationResources,
    *,
    progress: GenerationProgress | None = None,
) -> FunctionCall: ...

def select_name(
    user_prompt: str,
    *,
    model: Model,
    functions: dict[str, FunctionDef],
    trie: NameTrie,
    progress: GenerationProgress | None = None,
) -> FunctionDef: ...

def generate_parameters(
    user_prompt: str,
    fn: FunctionDef,
    *,
    model: Model,
    vocab: dict[int, bytes],
    progress: GenerationProgress | None = None,
) -> str: ...

def generate_value(
    slot: ValueSlot,
    ids: list[int],
    *,
    model: Model,
    vocab: dict[int, bytes],
    progress: GenerationProgress | None = None,
) -> list[int]: ...

def select_next_token(
    model: Model,
    ids: list[int],
    allowed: list[int],
    *,
    include_raw_top: bool = False,
) -> tuple[int, SelectionMode, int | None]: ...

def choose_token(
    logits: Sequence[float], allowed: Collection[int],
) -> int: ...
```

_startは`tuple[Args, list[str], GenerationResources]`を返す。Dashboardの構築は
`Dashboard(total: int, model: Model, vocab: Mapping[int, bytes])`で、__init__はNoneを返す。
具体的なDashboardはmainが持ち、生成へ渡すのは別引数のGenerationProgressだけである。

### 9.8 データフローと表示の責務

```text
_start：入力/出力先検査 → Model → vocab → 同じfunctionsからtrie → GenerationResources
main：resourcesから必要な参照を取り出してDashboardをwithで開始
  → _process_all(prompts, resources, ...) → process_prompt(prompt, resources, progress=...)
    → stage(select_name) → select_name(model, functions, trie, progress)
    → stage(generate_parameters) → generate_parameters(model, vocab, fn, progress)
      → Forcedの追加 / parameter_started(name, type) → generate_value(model, vocab, slot, ids, progress)
    → stage(validate_call) → validate_call → FunctionCall
  → 成功ごとに保存 → 完了/失敗通知 → 次の入力
with終了：端末復帰 → 要約 → mainの終了コード
```

各名前/値ループは、終了確認→許可ID計算→上限確認→1選択→ID追加→1ステップの通知の順に書く。
0/1候補ではprefix_idsをそのまま選択関数へ渡し、複数候補のときだけprefix_ids + generatedを作る。
上限ちょうどの終了は成功し、未完成で上限と空候補が同時なら上限エラーを優先する。

- 名前: `token, mode, raw_top`を受け、ID追加後にgeneratedをSequence[int]としてmode/raw_topとともにname_tokenへ同期的に渡す。
  採用IDは列の末尾、件数と警告のstepはそのlenから分かる。
- 値: 同じ3値を受け、ID追加後に語彙からwrittenを連結し、written・len(generated)・token・modeをvalue_tokenへ渡す。
  raw_topは要求していないのでNoneであり、表示にも渡さない。
- どちらもprogressがない、またはavailableが偽なら観測用bytesの構築・decodeを行わない。
  診断だけを省いてもmodeと採用ID、許可集合、終了条件、モデル入力は同じである。

表示はこのデータから文字列とToken欄を作り、生成には何も返さない。
診断decodeのPromptErrorは取得不能表記、stderrのOSErrorは表示停止、最終値decodeの失敗は生成失敗として扱う。
中断は握りつぶさずwithで端末を戻す。順序・エラー・保存・公開SDKの条件は維持する。

```text
main ──→ pipeline（GenerationResources・上位調停・各生成ループ） ──→ decoder ──→ model_port
  │          └──→ progress（表示の型契約。SelectionMode等は型参照だけ）
  └──→ display ──→ model_port（公開decodeのみ）
```

pipelineから具体的なdisplayをimportしない。displayはpipelineの資源構造体をimportしない。
生のModel/vocabは表示処理に必要な参照としてmainで接続し、Progressのトークン通知には含めない。
共有資源・生成中の可変状態・表示状態の3つの所有を分け、資源構造体を制御フローやサービスロケータにはしない。

### 9.9 検証境界と性能費用

GenerationResourcesはPydanticを実際に使う実行時クラスであり、4参照の型・必須フィールド・余分なフィールドを組立時に検証する。
SDKハンドルのInstanceOfはModelの型一致だけで、初期化済み・ロード成功・正しい推論の保証ではない。
ローカルのスキーマ確認では、未初期化のModelインスタンスでもこの型検査を通ることも確認した。
動作確認は既存Modelの例外処理とモデル回帰試験が担当する。

| データ | 検証場所・追加費用 |
| --- | --- |
| 外部のprompts/functions/vocab | 既存のloaderのstrict検証、空/重複・変換の検査を維持。資源構造体で内容を重ねて走査しない |
| 名前の木・ValueSlot | 既存のPydanticモデル構築と木の経路検査を維持。資源の組立や表示で再構築しない |
| 資源の4参照 | InstanceOfによる固定個数の検査。辞書内容・ノード数に比例するcopy/検証はしない |
| モデルのlogits | choose_tokenの次元・数値・非有限値・許可ID検査を維持。診断のargmax以外の配列処理を加えない |
| 採用ID・mode・raw_top | 内部の選択分岐から得るプリミティブ。新たなPydanticモデルへ毎ステップ包み直さない |
| 完成したFunctionCall | 既存のJSON・strict型・数値保持・スキーマ検証を維持。生成制約とは異なる境界 |
| Dashboardのtotal/model/vocab view | 件数の導出値、同じModel、検証済みdictのview。内容の再検証やcopyをしない |

#### 参照保持の実測（2026-11-13、プロジェクト.venvのPydantic 2.14.3）

以下はPythonオブジェクトからの通常コンストラクタ呼び出しの結果であり、JSON入力の保証ではない。
ModelはSDKをロードせずobject.__new__で作り、型一致が初期化済み状態を保証しないことも確認した。

| 構成 | model | vocab辞書 | functions辞書 | 内部FunctionDef / そのparameters・returns | trie / 子ノード |
| --- | --- | --- | --- | --- | --- |
| 通常のdictフィールド（ModelだけInstanceOf） | 同一 | 別辞書 | 別辞書 | 同一 | 同一 |
| 推奨InstanceOfフィールド | 同一 | 同一 | 同一 | 同一 | 同一 |
| 辞書・TrieをSkipValidation | 同一 | 同一 | 同一 | 同一 | 同一 |

通常のdictは全項目を処理して新しい辞書を作り、不正なキー・bytes値・FunctionDef値を拒否した。
既存モデルの同一性は、その内部の全検証が省略されることを意味しない。FunctionDefにはafter validatorがあり、
既存インスタンスでも実行され得るため、資源組立でそれを再走査しないInstanceOfを選ぶ。
InstanceOfはdictの型だけを検査し、不正な中身の辞書もそのまま受け入れた。
SkipValidationはフィールド全体に適用するとlistを辞書欄へ、整数をTrie欄へ渡しても拒否しなかった。
いずれもBaseModelの必須・extra検査は別であり、省略されない。
model_constructも不正型を受け入れるため採らない。arbitrary_types_allowedを全体へ追加する必要はない。

frozenは通常のフィールド代入と削除を拒否したが、元の辞書の更新、FunctionDef.parametersの変更、
Trie.childrenの更新、Model内部属性の変更は可能だった。参照の共有は変更も共有する。
したがって構築後に共有辞書・木を変更しないという所有規約は必要であり、frozenによる深い不変性とは区別する。
既に検証済みの資源だけを_startで一度組み立てる範囲なら、4つのInstanceOfは明示引数よりわずかな宣言で
資源セットの責務を表せる。未検証の外部データをこの構造体へ直接入れる用途には使わない。

公式資料: [属性コピーとモデル再検証](https://docs.pydantic.dev/latest/concepts/models/#attribute-copies)、
[浅い不変性](https://docs.pydantic.dev/latest/concepts/models/#faux-immutability)、
[InstanceOf / SkipValidation](https://docs.pydantic.dev/latest/concepts/validators/#special-types)、
[model_construct](https://docs.pydantic.dev/latest/concepts/models/#creating-models-without-validation)。
一般的説明を公式資料で確認し、このプロジェクトの辞書・モデルについては以下の実測で裏付けた。

<details>
<summary>実行した一時検証コード（恒久的なアプリ・テストファイルには追加しない）</summary>

プロジェクトルートから`PYTHONPATH=. .venv/bin/python /tmp/cmm-resource-probe.py`で実行。
初回は検証データのreturnsに未定義descriptionを含めて拒否されたため、既存スキーマに合わせて除いた。
以下は修正後に全assertが成功したコードである。

```python
import pydantic
from pydantic import BaseModel, ConfigDict, InstanceOf, SkipValidation, ValidationError
from src.model_port import Model
from src.name_trie import NameTrie
from src.schema import FunctionDef

class Plain(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)
    model: InstanceOf[Model]
    vocab: dict[int, bytes]
    functions: dict[str, FunctionDef]
    trie: NameTrie

class References(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)
    model: InstanceOf[Model]
    vocab: InstanceOf[dict[int, bytes]]
    functions: InstanceOf[dict[str, FunctionDef]]
    trie: InstanceOf[NameTrie]

class Skipped(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)
    model: InstanceOf[Model]
    vocab: SkipValidation[dict[int, bytes]]
    functions: SkipValidation[dict[str, FunctionDef]]
    trie: SkipValidation[NameTrie]

model = object.__new__(Model)  # SDK未初期化。型検査だけの限界を確認する。
fn = FunctionDef(name='f', description='test', parameters={},
                 returns={'type': 'string'})
vocab = {1: b'a'}
functions = {'f': fn}
trie = NameTrie(children={1: NameTrie(name='f')})
inputs = dict(model=model, vocab=vocab, functions=functions, trie=trie)

def rejected(operation):
    try:
        operation()
    except ValidationError:
        return
    raise AssertionError('expected ValidationError')

print('Pydantic', pydantic.__version__)
for cls in (Plain, References, Skipped):
    r = cls(**inputs)
    identities = [getattr(r, key) is value for key, value in inputs.items()]
    assert identities == ([True, False, False, True] if cls is Plain else [True]*4)
    assert r.functions['f'] is fn
    assert r.functions['f'].parameters is fn.parameters
    assert r.functions['f'].returns is fn.returns
    assert r.trie.children[1] is trie.children[1]
    print(cls.__name__, 'model/vocab/functions/trie:', identities, 'nested: same')
    rejected(lambda: cls(**{k: v for k, v in inputs.items() if k != 'model'}))
    rejected(lambda: cls(**inputs, report=None))
    rejected(lambda: setattr(r, 'vocab', {}))
    rejected(lambda: delattr(r, 'trie'))

for field, wrong in [('model', object()), ('vocab', []),
                     ('functions', []), ('trie', {})]:
    rejected(lambda: References(**(inputs | {field: wrong})))
for field, wrong in [('vocab', {'bad': 'bad'}), ('functions', {1: fn}),
                     ('functions', {'f': object()})]:
    rejected(lambda: Plain(**(inputs | {field: wrong})))
    assert getattr(References(**(inputs | {field: wrong})), field) is wrong
for field, wrong in [('vocab', []), ('functions', []), ('trie', 123)]:
    assert getattr(Skipped(**(inputs | {field: wrong})), field) is wrong
r = References(**inputs)
assert r.model is model and r.vocab is vocab
assert r.functions is functions and r.trie is trie
vocab[2] = b'b'
fn.parameters['late'] = fn.returns
trie.children[2] = NameTrie()
model.probe = True
assert r.vocab[2] == b'b' and r.functions['f'].parameters['late'] is fn.returns
assert r.trie.children[2] is trie.children[2] and r.model.probe
assert References.model_construct(**(inputs | {'vocab': []})).vocab == []
print('PASS: invalid roots/members, required/extra, frozen assignment/deletion, shallow mutability, construct bypass')
```

出力: Pydantic 2.14.3、Plainの4参照は`[True, False, False, True]`、
References/Skippedはともに`[True, True, True, True]`、内部モデルは同一。
不正型・不正な中身・必須/extra・代入/削除拒否・浅い可変性・construct迂回の全assert成功。
このスキーマ確認はLLMのロード、推論性能、提案APIの実装回帰の検証ではない。

</details>

| 観測の処理 | 費用と判断 |
| --- | --- |
| modeの確定と返却 | 既存分岐で固定文字列を返すだけ。戻り値tupleが2値から3値になるが、語彙走査・推論・Pydantic検証は増えない |
| 名前のSequence借用 | 生成listをそのまま同期的に渡し、履歴コンテナをコピーしない。SDK境界ではlist以外の入力に限りO(n)のlist化。decode自体の費用は残る |
| 値のbytes/件数/ID | bytes連結は旧observedと同じ。token・mode・lenの通知は定数個の値で、ID列のcopyは不要 |
| トークン欄の整形 | 小さな文字列と表示フィールドを追加する。モデル推論も全語彙の追加処理も行わない |
| 名前のraw_top | greedyで要求時だけ既存と同じargmax。forced・値・観測なしでは0回 |
| 観測なし・利用不能 | 観測データの構築・変換を省く。資源構造体は全実行で共有し、プロンプトやステップごとに作らない |

Sequence借用では旧TokenChoiceによる履歴コピーとtupleスナップショットを追加しない。
呼び出し終了後に生成listが伸びても、保存済みの表示文字列は変化しない。
非変更・非保持は実装の契約であり、型注釈だけで安全性を保証したとは扱わない。
描画間隔を保つ案についても、新しい欄が5分制約へ無影響と実装前に保証しない。

新しい実行時データクラスのGenerationResourcesはBaseModelとして検証を行う。
GenerationProgressは静的な契約として使うが、Protocol宣言もPythonのクラスでありPydantic検証ではない。
既存Model/Dashboard/例外を含むSubjectの「All classes」の解釈上の留保は残す。
資源の型確認がSDKの動作を保証した、または新しい構造体により全クラス要件を解決したとは記録しない。
無意味なvalidate_callやBaseModelとの多重継承をProtocolへ足して、形式的に解決したことにもしない。

### 9.10 実装対象・回帰検証・制約

| 対象ファイル・関数 | 実装した変更 |
| --- | --- |
| pipeline.py | Run/report/sayと4クロージャを廃止。GenerationResources、上位だけの資源受け取り、下位の明示依存、各ループと同期借用の観測契約を設ける |
| decoder.py | generate/TokenChoiceを廃止。SelectionMode型別名と、選択ID・方式・任意診断IDを返す1選択関数を設ける。choose_tokenを維持 |
| progress.py | 上記の5メソッド・availableだけの型契約。コールバック・配送・実行状態は持たない |
| display.py | name_token/value_tokenで採用IDと方式を整形し、途中内容と一緒に現在トークン欄を更新。元の描画・集計・復帰は維持 |
| __main__.py | _startで資源を一度組み立て、_process_allへ渡す。Dashboardへは必要な参照と語彙の読み取り専用viewだけを接続 |
| tools/measure_application.py | 評価バッチで資源を一度作る。MeasurementProgress(BaseModel)がケースごとの引数名と件数を保持し、従来のレポートキーと計測方式を維持。クロージャobserve/nonlocalを廃止 |
| 関連テスト | 共通選択と名前/値のループへ既存契約を移し、両段階のID/方式・非変更/非保持の通知・浅い資源検証を追加 |

以下の回帰条件を維持する。実行済みの範囲と未検証条件は末尾に分ける。

1. 名前と値の両方で、1候補はforced・推論0回、複数候補はgreedy・推論1回となる。
   greedyでinclude_raw_top=Falseならraw_top=Noneだが方式はgreedy。forcedはTrue指定でも診断0回。
2. modeが実際の分岐と一致し、表示あり/なし・リダイレクトで採用ID、モデル入力、forward回数、JSONが一致する。
   同点・raw最大がマスクされる場合・有限候補が1つだけの場合・不正logitsを既存規則で確認する。
3. 名前と値の通知が、各ステップの採用ID/方式と一致する。値の同じbytesに複数IDが対応しても逆引きせず正しいIDを渡す。
   名前のlistが同一参照で渡され、表示が変更・保持せず同期的に整形することを確認する。
   次ステップの追加でも保存済み表示文字列は変わらないこと、list以外のSDK境界変換と入力非変更も検査する。
4. トークン欄・途中内容が同じステップで描画されること、引数/段階開始時にトークン欄が初期化されること、
   UTF-8途中・特殊トークン・取得不能表記・制御文字・画面幅・警告・要約・端末復帰を確認する。
   availableとisattyを分け、リダイレクトの警告、stderr故障後の生成継続も確認する。
5. 0・1・255・256・上限超過、EOS、終了/上限/空候補の優先順位、引数順、ForcedとJSON区切り、
   生成後のPydantic検証、途中保存と保存失敗の契約を維持する。
6. 資源の必須/余分/不正な参照型・frozen代入拒否、参照の同一性、構築回数、functionsとtrieの対応を検査する。
   root型の検査と内容検査を混同せず、語彙の読み取り専用viewからの更新も拒否されることを確認する。
7. GPU環境を確認し、make gpu-runで既知入力の前後JSONをcmp・スキーマ検証する。
   精度試験と計時を分け、資源組立の費用、Sequence借用/必要時のlist変換、表示あり/なし、全時間とばらつきを測る。
   forward回数不変だけを性能・5分制約への適合とみなさない。
8. make install・make lint（Subjectのflake8/mypy）・make test、構造的な型適合と差分レビューを行う。

今回の実装では、選択APIを先に独立させ、次に生成ループ・表示・起動処理・測定呼び出し元を移行した。
測定ツールのMeasurementProgressは既存の件数収集に必要な状態を持つPydanticモデルであり、
新しい生成サービス・イベントバスではない。countsの辞書とparameter名を初期構築時に検証し、
内部の同期通知でプリミティブを更新する。名前のID列は変更・保持しない。

検証結果（2026-11-13）:

- make install成功。変更前は必須lint/mypyと188テスト、選択API追加時は192テスト成功。
- 旧generateのテストは名前/値の実ループと1選択関数へ移行した。境界0/1/255/256/超過、
  forced/greedy、固定logits・同点・不正値・マスク、名前警告、JSON検証、保存失敗の保証を維持する。
  TokenChoiceのsnapshot検査は借用Sequenceの非変更・非保持と整形済み文字列の独立性へ置き換えた。
- 観測なし・リダイレクト・利用不能・stderrのOSErrorで、固定入力の生成JSONとモデル入力・推論回数が一致。
  名前のforced→greedy、値のforced、同一bytesを持つ複数ID、UTF-8途中・特殊トークンを検査した。
- GenerationResourcesの4資源と内部FunctionDefの参照同一性、不正な外側型・必須/extra拒否、
  内部辞書の非再検証、frozenの代入/削除拒否と浅い可変性をテストした。
- 実GPU（Radeon 8060S、ROCm環境）でmake gpu-runの既知11入力を前後各1回、直列実行。
  両方11件完了・失敗0。JSONはcmpでバイト単位一致。11件すべての生成結果を既存validate_callで各関数スキーマへ再照合し成功した。

| 同じ既知11入力・stderrリダイレクト | 変更前42a1f8b | 変更後案A |
| --- | --- | --- |
| make gpu-run全体の実時間（秒） | 25.95 | 25.46 |
| Dashboardのアプリ時間（モデルロードを含まない、秒） | 19.7 | 19.3 |

生データは/tmp/cmm-a-before.json・after.json、各log・seconds.txtに保存した。
各1回で分散は測っていないため、高速化の根拠やSubjectの提出環境5分証明とはしない。
精度の一致と計時を分けて扱い、GPU時間へCPU向け暫定8秒を加算しない。
実GPUでの全forward数の計装比較、全ステップの実端末録画、追加fixtureの精度評価は未実施。
通常のテストでは実際の疑似端末で例外とKeyboardInterrupt後の復帰を確認する。
最終make lintはflake8・Subject必須mypyとも成功（48ソース）、make testは197テスト成功。
独立した差分レビューでも新規の重大不具合・設計逸脱・重大なテスト保証喪失は見つからなかった。
生成2モジュールにネストした関数・lambdaがないことと、設計書のPython例の構文も確認した。
既存の留保として、Dashboard.writeが停止扱いにするのはOSErrorであり、閉じたPython streamの
ValueError等は対象外である。今回の変更で新たに発生した回帰ではなく、全例外の吸収へは広げていない。

Subjectの全クラス要件は未解決の解釈リスクとして残る。資源の型検証を、SDK動作保証や
既存Model/Dashboard/Protocolを含む全クラス要件の解決と偽らない。

## 開発専用のプロンプト改善

[ADR-0018](design/adr/0018-local-prompt-experiments.md)の実験ツールは、提出アプリの外側で
指示文2つだけを差し替えて、既存の関数選択・生成・検証・CLIを評価する。
モデルの改稿と採点・Git同期・CPU測定・追加集合の開封を分け、制御側が後者を固定する。
SDK・生成方式は変更せず、実験ツールと開発環境は提出対象に含めない。


## 10. 今回対象外の旧提案：LFで区切る引数値（未実装）

この節は、本書に混在していた旧提案を現行設計から分離して保存したものである。
今回のクロージャ・コールバック見直しに含めず、採用・実装・性能を承認済みとは扱わない。
コードは現在も§3.2・4のJSON区切り方式である。以下の未来形・設計判断は旧提案の記述である。
関連する履歴ファイルは今回参照・変更していない。

![旧LF提案の処理図（現行実装ではない）](diagrams/process-prompt.svg)

[旧process_prompt図のD2](diagrams/process-prompt.d2)

### 10.1 旧提案の組み立て方

この節は[ADR-0015](design/adr/0015-value-records.md)の改行区切りを反映した設計で、コードは未変更。
モデルが読む値の中間表現と、保存するJSONの構造を分ける。

出力のキーと引数の名前・順序は既に分かっているため、`generate_parameters` はモデルにその構造を
選ばせる必要がない。入力の `prompt` と選択済みの名前を `json.dumps` でエスケープし、記録の先頭を作る。
`build_parameters_prompt`には選んだ1関数の定義と依頼、引数の順序を入れ、
関数を実行せず、値だけを引用符外のLFで区切って出すよう指示する。
引数生成用の入力は一度だけencodeし、複数の引数を同じ応答の続きとして生成する。
`PARAMETER_EXAMPLES`（既定は空）に例があれば、systemと依頼の間に会話の組（user: 定義と依頼、
assistant: 骨組みと同じ書き方の答え）として置く（ADR-0020）。空なら従来と同じ文字列になる。

その後の構造は `build_skeleton` が定義から作る。固定文字列を `Forced`、値を生成する場所を
`ValueSlot(name, type)`で表す。`Forced`は最終JSONの文字列だけに追加し、モデルの入力ID列には追加しない。
`ValueSlot`では`generate_value`が型に合う値と終端LFを生成し、生のID列を次の引数の入力へ継続する。
追加する`normalize_value`が、LFまでの中間表現をint/検査済みfloat・string・booleanの値へ正規化する（ADR-0017）。
正規化後の値を`json.dumps`でJSON化し、最終出力に差し込む。生成時の生の表記はモデルの文脈に残す。
引数がない場合は空の `parameters` と記録の閉じ括弧を強制する。

例えば `fn_add_numbers` が number の `a` と `b` を持つ場合、名前まで作った後は次の順で進む。
以下の`⏎`は実際のLFを示す。JSONの引数区切りカンマや構造の括弧はForcedが書く。
数値の桁区切りや文字列の内容としてのカンマは、中間表現に含められる。

| 部分 | モデルが生成する中間表現 | 最終JSONへ追加する内容 |
| --- | --- | --- |
| `Forced` | なし | `, "parameters": {"a": ` |
| a の `ValueSlot`（number） | `2,000,000⏎` | 正規化した`2000000.0` |
| `Forced` | なし | `, "b": ` |
| b の `ValueSlot`（number） | `3⏎` | 正規化した`3.0` |
| `Forced` | なし | `}}`（引数と記録を閉じる） |

`2`は単独で有効な数でも`265`の途中かもしれないため、値として成立しただけでは止めない。
モデルが続きを出すかLFで終えるかを選ぶ。桁区切りのカンマは終端にしない。
stringでは引用符内の実改行を保持し、引用符を閉じた後のLFだけで次の引数へ進む。
したがって1引数が複数の物理行にまたがることがあり、単純な行分割では処理しない。
booleanも単語が完成した後のLFで終了する。

![骨組みと値を組み立てる処理](diagrams/generate-parameters.svg)

[generate_parameters の D2](diagrams/generate-parameters.d2)

### 10.2 旧提案の値文法

`allowed_value_tokens` は、生成済みのバイト列に候補トークンの全バイトを足した結果が、
「指定型の中間表現と、その後の引用符外LF」の有効な接頭辞かを検査する。
候補の途中までが正しくても、終端LFを越えて次の値や文章を含むトークンは許さない。
完了の確認は `is_value_complete` が担当し、`generate_value` はその判定をしてから次の候補を求める。

| 型 | 許す内容 | 完了の条件 |
| --- | --- | --- |
| number | 符号、整数部、任意の小数部、英語形式の3桁区切り。先頭ゼロ・先頭プラス・指数表記は従来どおり範囲外 | 数値として成立した後のLF。`2,00`など不完全な桁区切りでは終了しない |
| string | 二重引用符で囲み、正しいエスケープ・UTF-8と引用符内の実LFを許す | エスケープされていない引用符で文字列を閉じた後のLF |
| boolean | 小文字の`true`または`false`の接頭辞 | 単語が完成した後のLF |

`12`の後では次の数字・桁区切り・LFを候補にできるが、`12.`の後では小数部が必要になる。
`2,`の後は3桁の数字をそろえてから次の桁区切り・小数点・LFへ進める。
文字列内のカンマ・`}`・実LFは内容として保持する。終端LFの後に余分な空白があるトークンも拒否する。
数値のカンマは形式を確認してから除く。stringの`"2,000,000"`は変更しない。
引用符内の実LFはJSONのエスケープに直して読み取り、最終出力では`json.dumps`で表現する。

Qwen のトークンは UTF-8 の1文字の途中で分かれることがあるため、完成前のトークンを個別に
文字列へ変換して文法判定しない。語彙からバイト列を連結して検査し、文字の途中を保持する。
文字列のマスクでは共通の接頭辞の状態を一度求め、各候補の追加分だけを検査する。
完成した中間表現を型別に正規化し、JSON化した値だけを最終出力へ追加する。これは文字の保存に関する設計であり、
日本語の指示の意味理解を改善する処理ではない（ADR-0013）。
