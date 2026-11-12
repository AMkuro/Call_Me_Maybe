# 非実験ブランチのローカルdevelop統合（2026-11-12）

## 作業前の状態と統合方針

ユーザーは、新しい作業ブランチを作らず既存の非実験作業をローカルdevelopへ集約することを指示した。
AGENTS.mdのブランチ作成・PR運用より、この明示指示を優先する。push・PR・main変更・ブランチ削除はしない。

- 開始時の主作業ツリー: refactor/remove-token-ranking、`08705b1f0323535f89cb590064a44bb753977105`。未コミット変更なし。
- **統合前のローカルdevelop: `20c956cd1d272082befaffb3cfe471ddcb5d4da3`**。
- fetch後のorigin/develop: `e97dd73deb66e6dd9144a1c1fb9defdf11dd55e4`。
- main/origin/main: `f5f8ece7dd14bb6f7e83f49b8096f322f77b5f2e`。developの祖先であり変更しない。
- developは既存campus-runtime-design worktreeで開かれていた。未変更を確認して同じ20c956cの
  detached HEADに切り替え、主作業ツリーでdevelopを開いた。他worktreeの内容を破棄・stashしていない。
- 新しいブランチ・worktreeは作らず、すべて既存ブランチを維持した。

git log、merge-base、rev-list、git cherryだけでなく、共通祖先からの変更パスと現行ファイルのblob・treeを比較した。
古いブランチと現行developの全体diffには後続追加ファイルの見かけ上の削除も含まれるため、そのまま採用しない。
以下の共通祖先は、作業前のorigin/develop（e97dd73）とのmerge-baseである。
origin/HEADはorigin/mainへのシンボリック参照であり、独立したブランチとして扱わない。

## 全ブランチの分類

ローカル・リモートの30参照を列挙する。同じ名前でもHEADが異なる場合は別に確認した。

| ブランチ | 調査時HEAD | 共通祖先 | 分類 | 統合方法 | 根拠・結果 |
| --- | --- | --- | --- | --- | --- |
| `develop` | `20c956cd1d27` | `20c956cd1d27` | 統合先 | FF後に選別統合 | 20c956cから開始。origin/developのe97dd73へFF |
| `docs/poc-results` | `ff781e58870e` | `f5f8ece7dd14` | 実験資料として除外 | なし | ff781e5はPoC実装・プロンプト・実験結果の追加。名前がdocsでも実験 |
| `exp/local-prompt-improvement` | `097a6318fe64` | `ffff6d7caf95` | 実験として除外 | なし | プロンプト探索ハーネス・9Bモデル等。既採用srcと結果はorigin/developに残す |
| `exp/poc-a-single-stage-json` | `68bdd64d02d7` | `b3ae261cf663` | 実験として除外 | なし | 単段JSON生成方式のPoC |
| `exp/poc-b-numeric-select` | `f0ac60228a47` | `b3ae261cf663` | 実験として除外 | なし | 数値関数選択方式のPoC |
| `exp/poc-c-name-select` | `da35f2447859` | `b3ae261cf663` | 実験として除外 | なし | 関数名選択方式のPoC |
| `feat/cpu-watch` | `83c54af2cf95` | `ffff6d7caf95` | 部分採用 | ファイル単位で再適用 | uv固定とADR履歴のみ採用。監視ツールは子孫監視と時間集計の不具合で保留。旧AGENTS運用は不採用 |
| `feat/implementation` | `8dd88fa712d5` | `20c956cd1d27` | 有効部分は統合済み | 再取込なし | srcはorigin/developと一致。残差は探索ツール・探索用CMM_MODULE・関連文書/テスト。実験資産は除外 |
| `feat/token-viewer-v2` | `4e210b8572cd` | `e97dd73deb66` | 後続に包含 | 08705b1までFF | ランキング削除後のブランチから履歴を含め統合。旧候補表は復活させない |
| `integrate/implementation` | `7b57b831dbd8` | `20c956cd1d27` | 実質統合済み | 再取込なし | 7b57b83とe97dd73のtreeが完全一致。squash#45による履歴差のみ |
| `main` | `f5f8ece7dd14` | `f5f8ece7dd14` | 対象外・祖先 | 変更なし | f5f8eceはdevelopの祖先。固有の未採用変更なし |
| `refactor/remove-token-ranking` | `08705b1f0323` | `e97dd73deb66` | 統合 | FF | 08705b1まで採用。全srcとREADMEを統合後も維持 |
| `refactor/token-viewer-boundaries` | `a57a3cb0b5c5` | `e97dd73deb66` | 文書のみ統合 | no-ff merge | a57a3cbの提案をADR0024へ改番。未承認・未実装、ランキング部分は失効と明記 |
| `origin/ci/agent-auto-integration` | `52f29a554901` | `e3dd4ec4bbd5` | 統合済み・後続で置換 | 再取込なし | squash#30/#35。契約テストと設定は一致、後続#41の手動CI・SDK lint除外を維持 |
| `origin/develop` | `e97dd73deb66` | `e97dd73deb66` | 統合先 | FF後に選別統合 | 20c956cから開始。origin/developのe97dd73へFF |
| `origin/docs/architecture` | `850003d1edec` | `3d970c31db9b` | 統合済み・後続で置換 | 再取込なし | squash#42。ADR0008等は一致。名前木・Modelの後続決定を維持 |
| `origin/docs/ci-verification-results` | `8deaf0871416` | `2b4f9c8e74bb` | 実質統合済み | 再取込なし | 変更対象2ファイルがorigin/developと完全一致。squash#35 |
| `origin/docs/design-process` | `1b613b56d5b6` | `b3ae261cf663` | 統合済み・後続で置換 | 再取込なし | squash#40。ADR0001/0006一致、他ADRは承認済みの後続更新を維持 |
| `origin/docs/literature-survey` | `8878dcaa2273` | `f5f8ece7dd14` | 統合 | cherry-pick -x | 8878dcaの独立した文献調査5ファイルを採用 |
| `origin/docs/subject-v1.7` | `54625074a773` | `b3ae261cf663` | 実質統合済み | 再取込なし | squash#38。分岐HEADのtreeは共通祖先と同一。現行ローカル要約を維持 |
| `origin/exp/cpu-calibration` | `f7a069454c3c` | `b3ae261cf663` | 実験として除外・一部既採用 | なし | CPU校正の原記録。既存squash#39の内容を維持し、実験ブランチ自体はマージしない |
| `origin/feat/campus-pc-survey` | `9c0ca4e56da7` | `f5f8ece7dd14` | 統合済み・後続で拡張 | 再取込なし | squash#36。原票一致、surveyとテストの差分はthermal・strict JSON・キャッシュ分離の後続拡張 |
| `origin/feat/implementation` | `097a6318fe64` | `ffff6d7caf95` | 有効部分は統合済み | 再取込なし | srcはorigin/developと一致。残差は探索ツール・探索用CMM_MODULE・関連文書/テスト。実験資産は除外 |
| `origin/feat/make-env` | `fb180ef87e8b` | `b3ae261cf663` | 統合済み・後続で置換 | 再取込なし | squash#41。dev.mk・SDK・export・lock等一致。現在のMakefileと運用を維持 |
| `origin/feat/skeleton` | `38b9e8b06adb` | `7c2686eaf344` | 統合済み・後続で置換 | 再取込なし | squash#43。現行実装はskeletonの未実装葉を完成。旧stubへ戻さない |
| `origin/feat/token-trace-tui` | `0634b3db99e5` | `ffff6d7caf95` | 旧案・実験を含むため除外 | なし | Rich/別端末/JSONL/top-kの旧ビューア。現行ADR0021/0023で置換。探索ツールも混在 |
| `origin/feat/token-viewer-v2` | `4e210b8572cd` | `e97dd73deb66` | 後続に包含 | 08705b1までFF | ランキング削除後のブランチから履歴を含め統合。旧候補表は復活させない |
| `origin/integrate/implementation` | `7b57b831dbd8` | `20c956cd1d27` | 実質統合済み | 再取込なし | 7b57b83とe97dd73のtreeが完全一致。squash#45による履歴差のみ |
| `origin/main` | `f5f8ece7dd14` | `f5f8ece7dd14` | 対象外・祖先 | 変更なし | f5f8eceはdevelopの祖先。固有の未採用変更なし |
| `origin/test/calibrated-function-calling-cases` | `8d63548988a3` | `ffff6d7caf95` | 部分採用 | ファイル単位で再適用 | 8d63548の8スイートと整合性テストのみ採用。古いアプリは採らず、既存lint不備2件を修正 |

## 重複・競合と統合方法

1. developをorigin/developへFFし、続いてrefactor/remove-token-rankingの08705b1へFFした。
   token-viewer-v2の履歴と削除コミットがともに含まれ、候補表が最終treeに復活しない。
2. refactor/token-viewer-boundariesはno-ff merge。唯一のテキスト競合はdocs/design/README.mdの
   ADR0023の行だった。削除決定を0023として残し、未承認提案を0024へ改番して両方を記載した。
   architectureのリンクも追従。提案内の候補表・診断3段階は旧案で、今後は再設計が必要と追記した。
3. 文献調査は8878dcaをcherry-pick -x。アプリと独立した5ファイルだけで競合なし。
4. 校正テストは8d63548からtests/fixtures/calibrated/とtest_calibrated_fixtures.pyだけを採用した。
   元のブランチの古いsrcと文書差分は取り込まない。
5. CPU監視は83c54afからtools/cpu_watch.py・テスト・ADRとuvのdev固定を採用し、
   ignoreと説明を現行文書に再適用。旧運用059ae8dは採らない。ADR0021の別の番号重複も0025へ解消した。
   CPU精度探索・ブランチ指定は過去の決定として残し、現在のGPU開発とdevelop運用による失効を明記した。
   その後のレビューで、終了した親の子孫が監視・停止から漏れる不具合と、最後の停止時間の集計漏れが判明した。
   d8e3f57の履歴は残して7ef90f9で本体・専用テスト・記録ignore・実行手順を外した。
   uv固定とADR履歴だけを最終採用とし、ツールの再採用は別途の修理・回帰検証まで保留する。
6. 新規採用テストに元からあったflake8の行長違反とmypyのobject型推論エラーだけを最小修正した。
   データ検証の内容は変更していない。

## 記録したコミット

- `08705b1`まで: 削除済みビューアの既存14コミットをFFで採用。
- `aa49fad`: 未承認の責務分離案のmergeとADR0024への改番。
- `838171b`: 文献調査のcherry-pick（原コミット8878dca）。
- `6d41d72`: 校正済み8スイートの部分採用（原HEAD8d63548）。
- `d8e3f57`: CPU監視・既存uv固定の部分採用とADR0025への改番（原HEAD83c54af）。
- `1bd115b`: 採用元テストの既存lint不備の修正。
- `7ef90f9`: CPU監視本体の採用を保留。uv固定と設計履歴は維持。
- 本記録のコミット: 統合結果・検証証拠・残課題のみ。最終HEADはこの記録を含むコミットとしてgit logで確認できる。

部分採用のコミットメッセージに、元ブランチの完全SHAと対象パスを記録した。
実験ブランチをマージしていない。既にdevelopに入っていた実験由来の採用結果は消していない。

## 検証と限界

| 検証 | 結果 | 条件・限界 |
| --- | --- | --- |
| 統合前developのlint/test | 成功、27ソース・81テスト | 20c956cをgit archiveで/tmpへ展開。統合後と同じCPU用venvで比較 |
| make install | 成功 | 採用したuv固定を含むlockでCPU用.venvを再構築。GPU環境・SDKソースは未変更 |
| 最終make lint | 成功、45ソース | flake8とSubjectの必須mypyフラグすべて |
| 最終make test | 成功、188テスト | 10.230秒は単体テスト全体の時間で、LLMの性能計時ではない |
| 固定logits・forced・JSON検証・保存失敗・進捗・端末復帰 | 全既存テスト成功 | 観測あり/なしの選択一致、forced無推論、疑似端末、制御文字、要約を含む |
| 校正済み8スイート | 静的整合性テスト成功 | 88件の定義・gold・型・regex等の確認。88件の実モデル正答率は測っていない |
| GPUの既知11入力 | 終了コード0、11件完了・0件失敗 | .venv-gpu、Radeon 8060S Graphics、make gpu-run、stderrリダイレクト |
| GPU出力の前後比較 | cmpで完全一致、全11件が関数スキーマ検証成功 | 基準08705b1の/tmp/cmm-ranking-after.jsonと統合後/tmp/cmm-integration-after.json |
| アプリ・README・SDKの意図しない差分 | なし | git diff 08705b1 -- src README.md llm_sdk が空 |
| ADR番号・見出しと文書リンク | 成功 | 25件の番号が一意、主要統合文書の相対リンクが実在 |
| Git状態 | develop以外の全ブランチHEADは調査時と一致 | main・リモート参照を含む。ブランチ追加・削除・push・PR操作なし |
| 統合レビュー | CPU監視の既存不具合2件を発見し採用保留 | 子孫停止漏れは最小プロセスで再現、時間欠落は制御フローで確認 |

基準developを最初にarchiveで展開した際、git check-ignoreを使うテストだけが.git不在で失敗した。
GIT_DIRとGIT_WORK_TREEを明示して同じスナップショットを再実行し、81件すべて成功した。
これは検証環境の不足であり、developの既存不具合とは判定していない。
校正テスト採用時のlint不合格は元ブランチにも存在する2件を1bd115bで修正した。
CPU監視採用直後は208件が通ったが、レビューで検出した不具合をテストが網羅していなかったため、
それを安全性の根拠にせず本体と専用20テストを保留し、最終188件を再実行した。

全チェックにはUV_CACHE_DIR=/tmp/call-me-maybe-uvを指定した。記録したローカルログは
/tmp/cmm-integration-install.log、/tmp/cmm-integration-final-checks.log、
/tmp/cmm-integration-original-develop-checks.log、/tmp/cmm-integration-gpu.log。
出力JSONのSHA-256は基準・統合後とも
`c17884267e3b6e46026edc6f34d30458a26b5cad6ce08007a66a2925667f521e`。

実モデル実行は出力の回帰検証であり、独立した性能ベンチマークではない。
統合前20c956cは旧スケルトン段階なので、現在の実装に対するJSON回帰基準には採用済み08705b1を使った。
5分制約・未知入力の正答率・実モデルのTUI目視・Python3.10実行は今回未検証。
手動CIの起動やリモートの保護設定検証も行っていない。

## 残課題と旧ブランチの扱い

- AGENTS.mdの新規ブランチ作成・最新integrationブランチ起点・PRごとのauto-merge手順は、
  今後のローカルdevelop集約運用に合わない。ファイル全体の改訂は別作業とし、今回は変更していない。
- docs/ci-auto-integration.mdは冒頭に旧設計と明記した履歴。CIは手動のままにし、再有効化や保護設定変更はしていない。
- ADR0024の責務分離、ADR0015のLF方式、ADR0020の未決項目は未承認/未実装の状態を維持する。
- 非実験の安全に採用できる差分は集約済み。未統合で残る通常開発の変更は、feat/cpu-watchの
  ツールと専用テスト（子孫監視・停止中の時間集計に既知の不具合）である。
  修理を別途行うか、既存thermalツールとの用途整理を先に行うかは今後の判断とする。
  他の残差は、実験資産、置換済み実装、旧運用か未承認の実装案である。
- feat/token-viewer-v2・refactor/remove-token-ranking・refactor/token-viewer-boundaries、
  integrate/implementation、旧docs/build/skeleton/campusブランチは将来の整理候補。
  feat/implementationとtoken-trace-tuiには実験資産が混在するので、ブランチ名だけで削除可とは判断しない。
  今回はすべて維持した。
- exp/local-prompt-improvement、exp/poc-a-single-stage-json、exp/poc-b-numeric-select、
  exp/poc-c-name-select、origin/exp/cpu-calibration、docs/poc-resultsは実験として残す。
- 採用を保留したCPU監視の実温度・一時停止を伴う本番負荷、sandbox内外の時間同等性は未確認。
  固定CPU0–15・8スレッド×2という旧機材前提も持つため、通常のGPU開発から自動起動しない。
- ブランチ削除・push・PR・mainへのリリースは別途の明示指示が必要である。
