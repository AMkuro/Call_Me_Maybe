# 校舎PC調査（Call Me Maybe 実行環境）

42 校舎の複数PCで実行環境を同じ方法で調べて比較し、レビュー基準PCの選定と、
「CPU で 5 分以内」という実行時間要件の実現可能性の評価に使う。
スクリプト: [`tools/campus_survey.py`](../../tools/campus_survey.py)（標準ライブラリのみ、Python 3.10+）。

このディレクトリは公開リポジトリの `develop` に置く。`main` には入れない。

## 次のPCでの実行手順（Claude Code 不要）

```sh
git clone -b develop https://github.com/AMkuro/Call_Me_Maybe.git
cd Call_Me_Maybe
# 1. 内容の確認（標準出力のみ。ファイルは書かない）
python3 -m tools.campus_survey --pc-id cmm-002 --meta floor=2f
# 2. 問題なければ research/campus/pc-cmm-002.txt に追記する
python3 -m tools.campus_survey --pc-id cmm-002 --meta floor=2f --write
```

- 管理者権限・追加パッケージ・ネットワークは使わない。1 秒未満で終わる。
- `deps.*` と `sdk.importable` は、スクリプトを実行したインタプリタでの有無を示す
  （`python3` で実行すれば、`.venv` や uv 環境の状態ではない）。
- 同じPCで再測定するときは同じ `--pc-id` を使う。同じファイルに
  `=== run <UTC時刻>-<mode> ===` のブロックとして追記される。
- `--meta` は任意の公開メタデータ（`floor=2f`、`row=r3` など）。何回でも指定できる。
  値に使えるのは英数字と `._+-(),=%@#` と空白だけ（`/` は使えない）。

### 匿名ID

`--pc-id` は**利用者が明示的に決める**匿名ID（`[a-z0-9][a-z0-9-]{2,31}`）。
`cmm-001` のような連番を推奨する。hostname・username を含むIDは拒否される。
「ID ↔ 実際の席」の対応表はリポジトリにも共有PCにも置かず、手元で管理すること。

対応表を持ちたくない場合は、hostname の秘密鍵付き HMAC-SHA256 からIDを作れる。

```sh
python3 -m tools.campus_survey --pc-id-from-hostname   # 鍵を端末で入力（エコーなし）
```

同じ鍵なら同じPCは常に同じID（`h` + 12桁16進）になる。鍵は16文字以上で、
Git にも共有PCにも保存しないこと（パスワードマネージャ等で保管する）。
鍵なしの単純ハッシュは hostname 一覧から総当たりで戻せるため使わない。

## benchmark（CPU 推論測定・明示指定時のみ）

配布 `llm_sdk` の公開 API（`Small_LLM_Model(device='cpu')`、`encode`、
`get_logits_from_input_ids`）と `Qwen/Qwen3-0.6B` を使う。既定値は profile `v1`：

| 測定 | 内容 |
|---|---|
| `sdk_import_s` / `model_load_s` | SDK の import とモデルのロード |
| `first_logits_len_64_s` | ロード直後の初回 logits（64 トークン） |
| `repeat_logits_len_64_{min,median,max}_s` | 同じ入力での繰り返し（`--repeats`、既定 3 回） |
| `logits_len_<N>_median_s` | 入力長ごとの logits 時間（`--lengths`、既定 16,64,256） |
| `generate_<K>_tokens_s` | 64 トークンの入力からの greedy 逐次生成（`--gen-tokens`、既定 8） |
| `rss_*_mib` / `peak_rss_mib` | メモリ使用量（`/proc/self/status`） |

- `bench.measured.*` は実測値、`bench.estimated.*` は実測値から計算した**推定値**。
  推定は入力長に対する一次近似で、前提（サンプル 11 件、入力 300 トークン、
  生成 40 トークン、KV キャッシュなし）を同じブロックに記録する。
  constrained decoding の処理時間は含まない。
- モデルがキャッシュになければ**ロード前に中止**する（終了コード 3）。
  ダウンロードは `--allow-download` を付けた場合だけ行う。
  通常は `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` を設定して実行する。
- 全体の上限時間は `--budget`（既定 180 秒、最大 900 秒）。超えたら
  `bench.status.result = partial` として、その時点までの実測値を残す。
- モデルキャッシュは `--hf-home`、`HF_HUB_CACHE`、`HF_HOME`、リポジトリの `.cache/`、
  `~/goinfre`・`~/sgoinfre` の浅い階層、`~/.cache/huggingface` の順に探す。

torch・transformers が必要なので、SDK 環境で実行する（初回は uv が依存を取得する）。

```sh
# 初回のみ: モデルを goinfre に取得する（このときの load 時間は比較に使えない）
uv run --no-project --with ./llm_sdk python -m tools.campus_survey benchmark \
  --pc-id cmm-001 --hf-home ~/goinfre/huggingface --allow-download
# 測定（キャッシュ済みモデルのみ使用）
uv run --no-project --with ./llm_sdk python -m tools.campus_survey benchmark \
  --pc-id cmm-001 --hf-home ~/goinfre/huggingface --write
```

PC 間で条件をそろえるため、`--lengths` などの既定値は変えずに実行する。
変えた場合は `bench.config.*` に記録される。`bench.config.model_cached_before_run = no`
の回はダウンロード時間を含むので、比較から除く。
失敗時の詳細は `--debug` で端末に表示する（レポートには例外型と段階名だけを残す）。

## 公開レポートの安全対策

- 出力は許可リストにあるキーだけ。値は安全な文字種（`/`・`~`・引用符・`:` を含まない）
  の 120 字以内に限り、それ以外は `unavailable (unsafe_value)` に置き換える。
- 書き込み前に全文を検査し、hostname・username・ホームディレクトリ名、
  IPv4/IPv6/MAC/メールアドレス形式を検出したら何も書かずに終了する（終了コード 2）。
- コマンドの生出力・環境変数・例外メッセージは保存しない（スレッド数の環境変数は
  数値かどうかだけ記録する）。シリアル番号・ネットワーク情報・他ユーザーの情報は読まない。
- 取得できない項目は `unavailable (<理由>)` とする。理由は固定の語彙
  （`not_found`、`permission_denied`、`not_supported`、`parse_error`、
  `not_installed`、`timeout`、`error`）。
- 公開に適さない詳細を残す必要があるときは、Git 管理外の `.cache/` 配下に置く
  （現状のスクリプトはそうした詳細を収集しない）。

`--write` の前に必ず標準出力の内容を目で確認し、push 前にもレビューすること。
