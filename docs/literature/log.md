# log: 小さなLLMの制約付き関数呼び出し

## 0周目（検索役は使わず、親が WebSearch で実施）
- 検索語: constrained decoding token boundary、BFCL、selection bias、function calling small language models、token healing、regex generation、LLMLingua など
- 媒体の偏り: 挙がったのは arXiv 論文と会議録が大半。個人ブログは Guidance の文書と LMSYS ブログのみ
- 検索が空で返った回: 2バッチ目の8本（同時に投げすぎた可能性）。後で数本ずつ再実行した

## 1周目
- 33本を arXiv から保存（BFCL のみ取得失敗: PMLR 404、OpenReview 403）
- 動き: 言葉で引き直す、後ろへ（関連論文）、前へ（Zheng ら、DOMINO の周辺）
- 論点2・3に原文が付いた。論点4〜6は一部のみ

## 2周目（ユーザーの「あるはずだから諦めずに探して」を受けて）
- 検索語: number of tools degrade function calling、prefill latency CPU prompt length、prompt length sensitivity、TinyLLM、llguidance overhead、Less is More edge
- chain.py cites: Small Models Big Tasks、Less is More、DOMINO（OpenAlex）
- 新しく保存: Less-is-More、CONFETTI、Tool-DC、MCPVerse、ProSA、POSIX、TinyLLM、aPSF、mllm-NPU、MobileLLM-Flash、Parser Stack Classification、LLM Inference at the Edge、AWS 小型モデルのツール呼び出し、How Many Tools、AgentFloor、SynCode、StructuredRAG、CarbonCall（arXiv 2402.12819 は無関係と分かり削除）
- 論点4に近い原文: AgentFloor が qwen3:0.6b を直接測っていた（道具1回の呼び出し段 44%）。TinyLLM が1B未満と1〜3Bを比較
- 論点6に近い原文: mllm-NPU・LLM Inference at the Edge が prefill の遅延とプロンプト長の比例を述べる
- まだ見つからない: 0.6B の関数呼び出しで few-shot の数・プロンプトの長さを測った統制実験、KV キャッシュ無しの CPU で文法制約つきデコードの時間を測った研究

## ユーザーの感想（原文のまま）
- 「いや、あるはずだから諦めずに探して」

## 3周目（言葉を隣の分野に広げた）
- 検索語: semantic parsing few-shot constrained decoding、multiple choice symbol binding、generative retrieval document identifiers、dialogue API argument filling、ToolGen、tool schema compression
- 新しく保存: Shin ら 2104.08768、Robinson 2210.12353、Xue 2406.01026、GR-IDs 2305.11841、API arg filling 2407.12016、NL-ToD 2407.15055、ToolGen 2410.03439 ほか
- 抜き書きに追加: Robinson（記号結びつけ）。照合 25 件 OK

## 4周目（ユーザーの「見つかっていない2つを出すまで帰ってこないで」を受けて）
- 検索役は使わず、親が WebSearch で実施。標準 9本・extended 4本（計13本）
- 保存: 2609.23742（0.6B〜4B の制約付きデコード）、2605.26128（Constraint Tax）、2402.10466（対話状態追跡で in-context 0〜5 個）、2509.22947（0.13B/0.5B/1B で shot 数）、2510.17388（Atomic Instruction Gap・Qwen 0.5B〜72B、0 と 3-shot）、2509.20677、2510.01251（KV キャッシュ無しの計算量）
- (i) few-shot の数: 近い研究を2本抜き書き（2509.22947、2510.17388）。どちらも制約付きデコードではなく、関数呼び出しそのものでもない。2402.10466 は 0〜5 個で一貫して改善とあるが、7〜13B 級
- (ii) CPU・KV キャッシュ無し: 実測した研究は見つからない。理論式のみ（2510.01251）。llguidance のマスク計算（約 50 µs/token）は公式の主張
- 2609.23742 と 2605.26128 は shot 数・時間の記述が本文に無いことを grep で確認（まだ全文は読んでいない）

## ユーザーの発言（原文のまま）
- 「未了のものをなしにすることと、まだ見つかっていないものに対してはあるはずなので、、調べ方が悪いか？って感じです。」
- 「まだ見つかっていないものの二つを出してくるまで帰ってこないでくれ。200とか調べきってみて」

## 5周目（ユーザーの選択「1」＝検索を続ける）
- 検索: 標準 3本・extended 7本（累計 約 23本）
- 保存: 2604.14862（スキーマのキー名）、heyneo ブログ（Qwen3-0.6B の CPU 関数呼び出し 50 クエリ）、2410.09407（CAMPHOR）
- 抜き書き追加: 2609.23742 のオーバーヘッド率（XGrammar 1.6〜3.7%、Outlines 9〜13%）、heyneo の CPU 遅延、CAMPHOR の 95% 削減。照合 37 件 OK
- 結果: (i)(ii) とも「0.6B・制約付き・KV キャッシュ無し CPU」を直接測った研究は見つからない
- ユーザー発言（原文）: 「自分たちで測るのはバックグラウンドでできるでしょ」
- 対応: 手元で A の 4 種プロンプト（v6c/v6b/v6a/v5）を dev 30 件、CPU 0-15 固定・8 スレッドで順に計測（scratchpad の cpu_sweep.sh、バックグラウンド）

## 6周目（ユーザー「constrained で探してみなよ」）
- 検索語: constrained decoding + in-context examples / shots / prompt sensitivity / KV cache / 小モデル（extended 4本）
- 保存: ACL 2025 産業 GCD 論理式変換、RANLP 2025 The Hidden Cost of Structure、2602.17475 Small LLMs for Medical NLP、2603.03305 Draft-Conditioned CD、ToolDec、trie 論文、BenchCLAMP
- (i) に近い測定が出た: 約1Bで 0→4-shot +9.4 / 制約付き +3.8 / 併用 +13.2、4-shot で推論時間 +53%、制約付きは実行時間に有意影響なし。GCD の 0-shot が制約なし 5-shot を上回る（2B〜14B超）。RANLP は逆向き（制約付きは例を足すと急に改善、4〜8B）→ disagreements に並べた
- 抜き書き 48 件 照合 OK
- ユーザー発言（原文）: 「検索を続けろ」「いや、もっと探してみな。constrainedで探してみなよ」

## 自分たちの測定
5周目の依頼で A のプロンプト 4 種を手元で測った。結果は PoC 結果側（ブランチ `docs/poc-results` の `docs/poc/README.md`）に置いた。
