# ADR-0023: 上位k件のトークン候補ランキングを削除する

状態: ユーザーの削除指示に基づく実装方針（2026-11-12）。実装・検証結果は末尾に記録する。

## 背景と決定

対象はfeat/token-viewer-v2の4e210b8572cd275292c6edade2fcc10722323309。
今回の明示対象を優先し、このHEADからrefactor/remove-token-rankingへ分岐する。
originはfetch済み。前回の責務分離提案は今回実装しない。

ユーザーは、可視化の目的を「制約によってトークンが順次確定する過程の確認」とし、
上位k件の候補取得・ランキング表示を削除するよう指示した。
Subject v1.7の制約付きargmaxにはランキングは必要ない。
候補数を減らす代案もランキング計算を残すため採用しない。

## 変更範囲とデータの流れ

- decoder: TOP_CANDIDATES、_top、TokenChoice.top、候補表だけが使うlogit・allowed_mass・marginと
  exp・sort・argpartitionを削除する。TokenChoiceの既存Pydantic設定は維持する。
- 名前選択の制約警告は既存ADR-0011/0013の要件として残す。
  同じlogitsの制約前argmaxのIDだけをTokenChoice.raw_topとして通知する。
  許可集合内の採用IDと異なる場合だけ警告する。forced位置には推論もraw_topもない。
  これは候補ランキングではなく、警告に必要な最大値の位置の取得である。
- pipeline: _show_choiceを削除し、途中の値・トークン数と警告を通知する。
  _token_textは警告で使うため残す。表示用decode失敗の代替表記を維持する。
- display: candidatesフィールド・専用複数行描画・候補による強制再描画を削除する。
  進捗描画の更新間隔、要約、端末復帰は維持する。
- 関連テスト・architecture・development・ADR-0014/0021・設計一覧を更新する。
  ADRの過去の判断と測定は履歴として残す。

allowed_next、forced、モデル呼び出し、choose_token、生成列と終了判定、JSON検証と保存は変更しない。
Run・Model・Dashboardの構成や通知契約全体を変更せず、依存追加とSDK修正は行わない。
生成の制御と進捗通知はpipeline、描画と要約・端末資源はdisplayという現行境界を保つ。

## 検証

固定logitsで観測あり・なしのID列とforward回数を比較し、forced・制約警告・途中経過を確認する。
ランキング専用テストは削除・置換し、トークン表記、要約、安全な描画のテストは残す。
同点・負の無限大を含むケースも確認する。制約前argmaxの同点は小さいIDになる。
旧top-10の境界同点にはargpartitionによる曖昧さがあったが、生成側のargmax規則は変わらない。
make install・make lint・make test、可能ならGPUで同一11入力の出力JSONを前後比較する。
精度・出力一致の検証と性能計時を区別する。未実施の検証は成功扱いしない。

## 実装・検証結果（2026-11-12）

上記の削除を実装した。choose_token、allowed_nextの実装、終了判定、JSON検証・保存処理は
変更していない。新しいクラス・依存・CLIオプションは追加せず、Pydantic設定も維持した。

- make install: 成功。CPU用.venvのみ再構築し、GPU環境とSDKソースは未変更。
- 変更前: make lint成功（必須mypy44ファイル）、188テスト成功。
- 変更後: make lint成功（必須mypy44ファイル）、187テスト成功。
  uvキャッシュはsandbox内で書けるようUV_CACHE_DIR=/tmp/call-me-maybe-uvを指定した。
- テストの差分: 旧8テスト中、候補ランキング・表の5テストを削除し、警告用表記と
  要約を維持、表の描画テストを値の描画へ置換。ランキング計算をしないこと、固定logitsでの
  観測あり・なしとforward回数、制約警告、全生成フローのJSON・値進捗の4テストを追加。
  合計は188 - 5 + 4 = 187。test_dashboard_candidates.pyはtest_dashboard_progress.pyへ置換した。
- 新しい「ランキング・確率計算なし」のテストが変更前にはnp.expで失敗し、削除後に成功した。
  forcedの無推論、同点・負の無限大、観測中のJSONと値・トークン数も確認した。
- 既存の疑似端末による例外時カーソル復帰、制御文字のエスケープ、リダイレクト時の
  警告・要約、保存失敗と途中成功の保持のテストは成功した。
- ホストの.venv-gpuでGPU利用可を確認（Radeon 8060S Graphics）。make gpu-runで同じ既知11入力を
  前後各1回実行し、双方終了コード0、11件完了・0件失敗。stderrはリダイレクトした。
  出力先は/tmp/cmm-ranking-before.json、/tmp/cmm-ranking-after.json。
  cmpで完全一致し、両方の全11件を再解析して関数定義のスキーマ検証にも成功した。
  両JSONのSHA-256: `c17884267e3b6e46026edc6f34d30458a26b5cad6ce08007a66a2925667f521e`。
- ソース検索でランキング取得・候補フィールド・専用診断計算の残存なし。履歴ADRと研究の
  過去測定は残した。git diff --check成功。別レビュアーも重大な問題を指摘しなかった。

実モデルの実行は出力一致の確認であり、独立した性能ベンチマークや正答率の測定ではない。
実モデルでの端末画面の目視確認はしていない。端末復帰・描画は固定データと疑似端末で検証した。
既知11入力の一致は、未知入力すべての正答やSubjectの5分制約への適合を証明しない。
