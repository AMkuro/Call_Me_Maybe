# ADR-0008: 依存関係と配布 SDK

状態: 決定(ユーザー、2026-11-08)。リスクあり。

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
  このターゲットで入れた SDK のパスのときだけ、読み込む。
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
