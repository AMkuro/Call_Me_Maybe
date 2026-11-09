# ADR-0008: 依存関係と配布 SDK

状態: 決定(ユーザー、2026-11-08)。リスクあり。**2026-11-09 に更新あり(末尾の「2026-11-09 の更新」。
置き換わった記述は、その節が優先する)。**

## 背景
- 【明文】レビュアーと moulinette は `uv sync` だけを実行する(IV.3.1)。
- 【測定】`pyproject.toml` の依存は `numpy` と `pydantic` だけ。`uv.lock` に torch、transformers、`llm_sdk` は
  入っていない。CI は `make install` でこれだけを入れ、モデルを動かす検証はしていない。
- 【測定】リポジトリの `llm_sdk/llm_sdk/__init__.py` は、配布 zip の中身と一致しない。履歴では
  `dc9e9e0`(「fix the layout」)で、このファイルだけが変わった(+28/-20)。差は行の折り返し、
  import の並べ替え、空行の削除だけに見える。lint のための整形と推測する(未確認)。
  `pyproject.toml` と `uv.lock` は zip と一致する。
- 【測定】CI の基準 `.github/ci/sdk-baseline.json` は、整形後の `__init__.py` のハッシュを持っている。

## 決定
- 【設計判断】`llm_sdk` はパス依存として `pyproject.toml` に書く。提出物の構成(Subject VIII)の制限に沿う。
  - 【測定】クリーンなクローンでの検証(2026-11-08、uv 0.11.21、Python 3.12.3): 今の構成で
    `uv sync --locked` は通るが、`numpy`・`pydantic`・開発ツールしか入らない。リポジトリ直下から
    `import llm_sdk` すると、外側のディレクトリ(名前空間パッケージ)に当たり、`Small_LLM_Model` がない。
  - 【測定】`llm_sdk` をパス依存にすると、パッケージとしてインストールされ、`Small_LLM_Model` を import
    できた(インストールされた通常のパッケージが、名前空間パッケージより優先された)。
  - 【測定】`torch` を直接の依存に書かないと、`tool.uv.sources` の CPU 専用インデックスは効かず、PyPI の
    CUDA 版が選ばれる(98 パッケージ、`nvidia-*` を含む)。`torch` を直接の依存に書くと CPU 版(`+cpu`)に
    なり、65 パッケージになった。
  - 【測定】CPU 版の環境の大きさは約1.1GB。モデルのキャッシュは約1.5GB。モデルの読み込みは約16秒
    (ダウンロードを含む)。1トークンの入力での logits の呼び出しは、約0.1秒。
- 【設計判断】**配布 SDK の正は、`llm_sdk.zip` の展開結果とする。** 3ファイルの sha256:
  - `llm_sdk/llm_sdk/__init__.py`: `1638cfe1b23c7166a026e668aecb5be21b79a4d3c18bbf1e40dc09d2a36f2311`
  - `llm_sdk/pyproject.toml`: `5eaf932b8c808b2626cd375c3191d44c97b7b49ea8cb846099eafd763f9356a1`
  - `llm_sdk/uv.lock`: `571e86020172c9c2e6e9c9ed7bffa0c396593c654acbedd3e6bbdecce0d8980c`
- 【設計判断】`make sdk-install` で zip を展開し、sha256 を照合して、評価用に入れ直せるようにする。
  このターゲットで入れた SDK のパスのときだけ、読み込む。**(2026-11-09: sha256 の照合はやめた。更新節を参照)**
- 【設計判断】SDK は外部のものとして、flake8 と mypy の対象から外す。外部のものなので、その違反は
  このプロジェクトの違反にならない、と決める。
- 【解釈】Subject は、配布 SDK が flake8/mypy を通る必要があるとは書いていない。除外してよいとも書いていない。
  一方、Subject の lint コマンドは、リポジトリ直下の `flake8 .` と `mypy . …` なので、文字どおりには
  `llm_sdk/` も対象に入る。
- 【設計判断】防御のため、Makefile のコマンドは Subject のまま残し、**設定ファイル**(`.flake8` の除外、
  `pyproject.toml` の mypy 設定)で SDK を除外する。

## 帰結(既存物への影響)
- `.github/ci/sdk-baseline.json`、`tools/audit_sdk.py`、`make lint-sdk`、`.github/ci/sdk-mypy.ini` は、
  整形後の SDK を前提にしている。基準の更新と、役割の見直しが要る。
- AGENTS.md の「Never edit the supplied llm_sdk」と、リポジトリの現状(整形済み)は食い違っている。

## 見直しの条件
- 評価が、SDK を含めた lint を求めるとき。
- クリーンな環境で、`uv sync` の後に `llm_sdk` を import できないと分かったとき
  (リポジトリ直下の `llm_sdk/` ディレクトリが、内側のパッケージより先に拾われる疑い。未検証)。

## 2026-11-09 の更新(ユーザーの決定。#41 で実施)
- 【設計判断】**依存の書き方**: `dependencies = numpy, pydantic, llm-sdk`(パス依存)。torch は
  `project.dependencies` に書かず、依存グループ `cpu`(既定のグループ)に書き、`tool.uv.sources` で
  PyTorch の CPU 専用インデックス(`explicit`)に向ける。`uv sync` の既定が CPU 版になる。背景の
  「依存は `numpy` と `pydantic` だけ」は、当時の測定。
- 【設計判断】**sha256 の照合をやめた。** `make sdk-install` は、zip を展開するだけ(`SDK_ZIP=<zip>` で別の
  zip も渡せる。既定は `llm_sdk.zip`)。展開した結果を、その場で作ったハッシュと照合しても何も確かめられない。
  zip の破損は `unzip` が検出し、改ざんは git の差分に出る。`llm_sdk.sha256` は削除した。
- 【設計判断】**開発用の局所変更を 1 つ入れた**: リポジトリの `llm_sdk/llm_sdk/__init__.py` で、全デバイスの
  dtype を float32 にした(元は cuda/mps で float16)。GPU でも CPU(レビュー機)と同じ数値にするため。
  コメントで明記した。CPU では元から float32 なので、CPU の動作は変わらない。
- 【設計判断】**最終提出の前に、`make sdk-install` で配布の版へ戻し、コミットしてから書き出す。**
  Subject VIII は "llm_sdk/ directory (copied from the provided package)" で、zip は提出物の一覧にない。
  `llm_sdk.zip` は提出しない(`submission-files.txt` に入れない)。戻し忘れの確認は、仕組みにしない(ユーザー、
  2026-11-09。手間に見合わない)。`make sdk-install` は提出する `Makefile` に残す(レビュアーが `SDK_ZIP=<zip>` で
  自分の zip を渡して入れ直せる)。
- 【設計判断】**`make lint-sdk`、`tools/audit_sdk.py`、`.github/ci/sdk-mypy.ini` は削除した。** SDK は無視する
  (`.flake8` と mypy の設定で除外。決定は変えない)。`.github/ci/sdk-baseline.json` は
  `tools/campus_survey.py` が読むので残したが、SDK の変更後は一致しない。
- 【設計判断】AGENTS.md の「Never edit the supplied llm_sdk」は、上の局所変更の例外つきに直した。
- 【測定】(2026-11-09)「リスクあり」の根拠を確かめた。配布 zip の `llm_sdk/llm_sdk/__init__.py` は、設定を使わずに
  `flake8` を実行すると 23 件(E303 が 6、E501 が 6、F401 が 2 など)、Subject の mypy のフラグで 3 件のエラー
  (`--strict` では 5 件)が出る。Subject の `flake8 .` と `mypy .` は、文字どおりには `llm_sdk/` も対象に入るので、
  設定ファイルによる除外が外れると、lint が通らない。リスクは、除外を認めるかの解釈に残る。
- 【測定】(2026-11-09)背景の「`dc9e9e0` の差は整形だけ」は、差分で確認した(import の並べ替え、行の折り返し、空行の
  削除。コミットは「fix the layout」)。整形の動機が lint だったかは、推測のまま。#41 で、`__init__.py` を zip から
  戻したので、この整形版は今のリポジトリにない(今の差は float32 の局所変更だけ)。
- 【測定】(2026-11-09)見直しの条件にある「`uv sync` の後に `llm_sdk` を import できないかもしれない」は、起きなかった。
  リポジトリ直下から `import llm_sdk` すると、インストール済みの通常のパッケージが読まれ、`Small_LLM_Model` が
  使えた(`site-packages` の版)。
- 【測定】uv は、ローカルの SDK のソース変更を拾わず、古いキャッシュのビルドを使い回す(float16 が残った)。
  `uv sync --locked --reinstall-package llm-sdk` で回避した(`make install`)。
- 【測定】`uv.lock` の macOS 用 torch(2.14.1 と 2.15.0)の wheel は `macosx_14_0_arm64` だけ。Intel Mac
  では `uv sync` が通らない(Codex の指摘、ロックで確認)。ロックは校舎 PC で作り直す方針。手元の Mac の
  アーキテクチャは未確認。
