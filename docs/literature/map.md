# map: 小さなLLMでの制約付き関数呼び出し

問い: Qwen3-0.6B で、自然言語の依頼から関数呼び出し JSON（関数名＋引数）を作るとき、(A) 1段で JSON 全体 / (B) 番号で選んでから引数 / (C) 名前で選んでから引数 のどれを、どう作ると精度と時間（KV キャッシュ無しの CPU で全入力300秒以内）が両立するか。方式の選定はしない。
題: 小さなLLMの関数呼び出し
読み: ちいさな えるえるえむ かんすう よびだし せいやくつき でこーど
最初の検索語: constrained decoding token boundary / function calling small language models

## 調査の論点
- [x] 1. この分野の知見はどこに集まっているか（媒体・用語）— 用語は constrained / guided / structured generation、token misalignment、prompt boundary problem、token healing、selection bias、function calling。媒体は arXiv（cs.CL）、ICLR・ICML・ACL・EMNLP、LMSYS ブログ、Guidance・Outlines の文書
- [x] 2. 文法で縛るデコードが、サブワードトークンの境界のずれや先頭空白で精度を落とす仕組みと対策 — arxiv-2403.06988-domino-guiding-llms.txt:90、arxiv-2502.14969-lost-in-space.txt:41、arxiv-2312.07104-sglang.txt:1098
- [x] 3. 番号（ID）で選ばせると偏る理由と、偏りを減らす方法 — arxiv-2309.03882-not-robust-mc-selectors.txt:125、arxiv-2510.00307-biasbusters.txt:166
- [ ] 4. 小さなモデルで、few-shot・プロンプト長・関数名の選び方（1段／2段／専用トークン）が関数呼び出しの精度にどう効くか — 一部のみ（arxiv-2605.00334-agentfloor-small-models-tool-use.txt:394 で qwen3:0.6b を直接測定、arxiv-2511.22138-tinyllm-edge-agentic.txt:38、arxiv-2210.12353-robinson-mcp-symbol-binding.txt:31 で記号方式の能力差、arxiv-2506.01859-confetti.txt:598 でツール数）。「0.6B・制約付きデコード下で few-shot の数とプロンプト長を測った統制実験」は見つからず
- [ ] 5. 正規表現・置換文字列など、引数の中身を作る失敗 — 一部のみ（arxiv-2403.18327-autoeval-formal-synthesis.txt:1244）。小さなモデルでの測定は見つからず
- [ ] 6. 文法制約つきデコードの時間（強制トークンの省略・KV キャッシュ無し）— 一部のみ（arxiv-2312.07104-sglang.txt:1098、arxiv-2407.05858-mllm-npu-prefill.txt:82 で prefill が遅延の大半、arxiv-2603.23640-edge-inference-prefill.txt:96 で長さに比例）。KV キャッシュ無しの CPU で制約付きデコードを測った研究は見つからず

## 用語
- constrained / guided / structured decoding（制約付き・誘導付き生成）— arxiv-2307.09702-outlines-guided-generation.txt
- token misalignment（トークンの不整合）— arxiv-2403.06988-domino-guiding-llms.txt
- prompt boundary problem（プロンプト境界問題）、token healing — arxiv-2412.03719-lms-over-characters.txt
- selection bias / token bias（選択肢の ID への偏り）— arxiv-2309.03882-not-robust-mc-selectors.txt
- functional token（関数をトークン1つで表す）— arxiv-2404.01744-octopus-v2.txt
- jump-forward decoding — arxiv-2312.07104-sglang.txt

## 人
- 横断的に名前が挙がる著者は未整理（次の手がかり）

## 媒体
- arXiv cs.CL — https://arxiv.org/list/cs.CL/recent
- ICLR / ICML / ACL / EMNLP の会議録
- LMSYS ブログ — https://www.lmsys.org/blog/2024-02-05-compressed-fsm （jump-forward decoding）
- Guidance の文書（token healing）— https://guidance.readthedocs.io/en/latest/example_notebooks/tutorials/token_healing.html
- BFCL — https://gorilla.cs.berkeley.edu/leaderboard.html

## 原文（sources/）
| 論文・資料 | ファイル | 元の URL | どこから来たか（周・動き） |
|---|---|---|---|
| 〔論文〕DOMINO: 制約付き生成のトークン不整合 | arxiv-2403.06988-domino-guiding-llms.txt | https://arxiv.org/pdf/2403.06988 | 1周・言葉で引き直す |
| 〔論文〕Lost in Space: 先頭空白と構造化出力 | arxiv-2502.14969-lost-in-space.txt | https://arxiv.org/pdf/2502.14969 | 1周・言葉で引き直す |
| 〔論文〕From LMs over Tokens to LMs over Characters | arxiv-2412.03719-lms-over-characters.txt | https://arxiv.org/pdf/2412.03719 | 1周・近くを眺める |
| 〔論文〕Grammar-Aligned Decoding | arxiv-2405.21047-grammar-aligned-decoding.txt | https://arxiv.org/pdf/2405.21047 | 1周・前へ |
| 〔論文〕Let Me Speak Freely?（形式の制約と性能） | arxiv-2408.02442-let-me-speak-freely.txt | https://arxiv.org/pdf/2408.02442 | 1周・言葉で引き直す |
| 〔論文〕Synchromesh | arxiv-2201.11227-synchromesh.txt | https://arxiv.org/pdf/2201.11227 | 1周・後ろへ（DOMINO の関連） |
| 〔論文〕PICARD | arxiv-2109.05093-picard.txt | https://arxiv.org/pdf/2109.05093 | 1周・後ろへ |
| 〔論文〕Outlines: Efficient Guided Generation | arxiv-2307.09702-outlines-guided-generation.txt | https://arxiv.org/pdf/2307.09702 | 1周・言葉で引き直す |
| 〔論文〕XGrammar | arxiv-2411.15100-xgrammar.txt | https://arxiv.org/pdf/2411.15100 | 1周・言葉で引き直す |
| 〔論文〕Flexible and Efficient Grammar-Constrained Decoding | arxiv-2502.05111-flexible-efficient-gcd.txt | https://arxiv.org/pdf/2502.05111 | 1周・前へ |
| 〔論文〕SGLang（jump-forward decoding） | arxiv-2312.07104-sglang.txt | https://arxiv.org/pdf/2312.07104 | 1周・言葉で引き直す |
| 〔論文〕Large Language Models Are Not Robust Multiple Choice Selectors | arxiv-2309.03882-not-robust-mc-selectors.txt | https://arxiv.org/pdf/2309.03882 | 1周・言葉で引き直す |
| 〔論文〕BiasBusters: ツール選択の偏り | arxiv-2510.00307-biasbusters.txt | https://arxiv.org/pdf/2510.00307 | 1周・前へ |
| 〔論文〕Large Language Models are not Fair Evaluators | arxiv-2305.17926-not-fair-evaluators.txt | https://arxiv.org/pdf/2305.17926 | 1周・前へ |
| 〔論文〕Permutation Self-Consistency | arxiv-2310.07712-permutation-self-consistency.txt | https://arxiv.org/pdf/2310.07712 | 1周・前へ |
| 〔論文〕Calibrate Before Use | arxiv-2102.09690-calibrate-before-use.txt | https://arxiv.org/pdf/2102.09690 | 1周・後ろへ |
| 〔論文〕Fantastically Ordered Prompts | arxiv-2104.08786-fantastically-ordered-prompts.txt | https://arxiv.org/pdf/2104.08786 | 1周・後ろへ |
| 〔論文〕Active Example Selection for ICL | arxiv-2211.04486-active-example-selection.txt | https://arxiv.org/pdf/2211.04486 | 1周・言葉で引き直す |
| 〔論文〕Small Models, Big Tasks（小モデルの関数呼び出し） | arxiv-2504.19277-small-models-big-tasks.txt | https://arxiv.org/pdf/2504.19277 | 1周・言葉で引き直す |
| 〔論文〕Octopus v2（関数トークン） | arxiv-2404.01744-octopus-v2.txt | https://arxiv.org/pdf/2404.01744 | 1周・言葉で引き直す |
| 〔論文〕Hammer（関数マスク） | arxiv-2410.04587-hammer.txt | https://arxiv.org/pdf/2410.04587 | 1周・言葉で引き直す |
| 〔論文〕TinyAgent（ツール検索でプロンプトを短くする） | arxiv-2409.00608-tinyagent.txt | https://arxiv.org/pdf/2409.00608 | 1周・言葉で引き直す |
| 〔論文〕xLAM | arxiv-2409.03215-xlam.txt | https://arxiv.org/pdf/2409.03215 | 1周・言葉で引き直す |
| 〔論文〕APIGen | arxiv-2406.18518-apigen.txt | https://arxiv.org/pdf/2406.18518 | 1周・前へ |
| 〔論文〕ToolACE | arxiv-2409.00920-toolace.txt | https://arxiv.org/pdf/2409.00920 | 1周・近くを眺める |
| 〔論文〕Gorilla | arxiv-2305.15334-gorilla.txt | https://arxiv.org/pdf/2305.15334 | 1周・後ろへ |
| 〔論文〕ToolLLM | arxiv-2307.16789-toolllm.txt | https://arxiv.org/pdf/2307.16789 | 1周・後ろへ |
| 〔論文〕CallNavi（ルーティングと呼び出し） | arxiv-2501.05255-callnavi.txt | https://arxiv.org/pdf/2501.05255 | 1周・近くを眺める |
| 〔論文〕LLMLingua（プロンプト圧縮） | arxiv-2310.05736-llmlingua.txt | https://arxiv.org/pdf/2310.05736 | 1周・言葉で引き直す |
| 〔論文〕Qwen3 技術報告 | arxiv-2505.09388-qwen3-technical-report.txt | https://arxiv.org/pdf/2505.09388 | 1周・言葉で引き直す |
| 〔論文〕∀uto∃val（正規表現の翻訳の誤り） | arxiv-2403.18327-autoeval-formal-synthesis.txt | https://arxiv.org/pdf/2403.18327 | 1周・言葉で引き直す |
| 〔論文〕Neural Generation of Regular Expressions | arxiv-1608.03000-nl-to-regex.txt | https://arxiv.org/pdf/1608.03000 | 1周・後ろへ |
| 〔論文〕TransRegex | arxiv-2012.15489-transregex.txt | https://arxiv.org/pdf/2012.15489 | 1周・前へ |
| 〔論文〕Less-is-More（エッジでのツール数の削減） | arxiv-2411.15399-less-is-more-edge-function-calling.txt | https://arxiv.org/pdf/2411.15399 | 2周・言葉で引き直す |
| 〔論文〕CONFETTI（ツール数と性能） | arxiv-2506.01859-confetti.txt | https://arxiv.org/pdf/2506.01859 | 2周・言葉で引き直す |
| 〔論文〕Tool-DC | arxiv-2603.11495-tool-dc.txt | https://arxiv.org/pdf/2603.11495 | 2周・言葉で引き直す（未読） |
| 〔論文〕MCPVerse | arxiv-2508.16260-mcpverse.txt | https://arxiv.org/pdf/2508.16260 | 2周・言葉で引き直す（未読） |
| 〔論文〕ProSA（プロンプトの感度） | arxiv-2410.12405-prosa-prompt-sensitivity.txt | https://arxiv.org/pdf/2410.12405 | 2周・言葉で引き直す（未読） |
| 〔論文〕POSIX（プロンプトの感度） | arxiv-2410.02185-posix-prompt-sensitivity.txt | https://arxiv.org/pdf/2410.02185 | 2周・言葉で引き直す（未読） |
| 〔論文〕TinyLLM（エッジの小型モデル） | arxiv-2511.22138-tinyllm-edge-agentic.txt | https://arxiv.org/pdf/2511.22138 | 2周・言葉で引き直す |
| 〔論文〕aPSF（プロンプト構造、付録に長さの影響） | arxiv-2604.06699-apsf-prompt-structure.txt | https://arxiv.org/pdf/2604.06699 | 2周・言葉で引き直す |
| 〔論文〕mllm-NPU（オンデバイスの prefill） | arxiv-2407.05858-mllm-npu-prefill.txt | https://arxiv.org/pdf/2407.05858 | 2周・言葉で引き直す |
| 〔論文〕MobileLLM-Flash | arxiv-2603.15954-mobilellm-flash.txt | https://arxiv.org/pdf/2603.15954 | 2周・言葉で引き直す（未読） |
| 〔論文〕Parser Stack Classification（マスク計算の時間） | arxiv-2608.03065-parser-stack-classification.txt | https://arxiv.org/pdf/2608.03065 | 2周・言葉で引き直す（未読） |
| 〔論文〕LLM Inference at the Edge | arxiv-2603.23640-edge-inference-prefill.txt | https://arxiv.org/pdf/2603.23640 | 2周・言葉で引き直す |
| 〔論文〕AWS 小型モデルのツール呼び出し | arxiv-2512.15943-aws-small-model-tool-calling.txt | https://arxiv.org/pdf/2512.15943 | 2周・言葉で引き直す（未読） |
| 〔論文〕How Many Tools Should an LLM Agent See? | arxiv-2605.24660-how-many-tools-should-an-agent-see.txt | https://arxiv.org/pdf/2605.24660 | 2周・前へ |
| 〔論文〕AgentFloor（qwen3:0.6b を含む測定） | arxiv-2605.00334-agentfloor-small-models-tool-use.txt | https://arxiv.org/pdf/2605.00334 | 2周・前へ |
| 〔論文〕SynCode | arxiv-2403.01632-syncode.txt | https://arxiv.org/pdf/2403.01632 | 2周・前へ（未読） |
| 〔論文〕StructuredRAG | arxiv-2408.11061-structuredrag.txt | https://arxiv.org/pdf/2408.11061 | 2周・前へ（未読） |
| 〔論文〕CarbonCall | arxiv-2504.20348-carboncall.txt | https://arxiv.org/pdf/2504.20348 | 2周・前へ（未読） |
| 〔論文〕Shin ら: 制約付きデコードで少数例の意味解析 | arxiv-2104.08768-shin-constrained-lm-semantic-parsers.txt | https://arxiv.org/pdf/2104.08768 | 3周・言葉の広げ直し（意味解析）（未読） |
| 〔論文〕Robinson ら: 記号による多肢選択と cloze | arxiv-2210.12353-robinson-mcp-symbol-binding.txt | https://arxiv.org/pdf/2210.12353 | 3周・言葉の広げ直し（多肢選択） |
| 〔論文〕Xue ら: 記号結びつけの強化 | arxiv-2406.01026-xue-strengthened-symbol-binding.txt | https://arxiv.org/pdf/2406.01026 | 3周・前へ（未読） |
| 〔論文〕How Does Generative Retrieval Scale（ID の設計の比較） | arxiv-2305.11841-generative-retrieval-ids.txt | https://arxiv.org/pdf/2305.11841 | 3周・言葉の広げ直し（生成型検索）（未読） |
| 〔論文〕生成型情報検索のサーベイ | arxiv-2404.14851-generative-ir-survey.txt | https://arxiv.org/pdf/2404.14851 | 3周・言葉の広げ直し（未読） |
| 〔論文〕API 引数の充填（対話システム） | arxiv-2407.12016-api-argument-filling.txt | https://arxiv.org/pdf/2407.12016 | 3周・言葉の広げ直し（対話）（未読） |
| 〔論文〕NL-ToD（スキーマ誘導の API 呼び出し生成） | arxiv-2407.15055-nl-tod-schema-api-call.txt | https://arxiv.org/pdf/2407.15055 | 3周・言葉の広げ直し（対話）（未読） |
| 〔論文〕ToolGen（ツールを1トークンにして生成） | arxiv-2410.03439-toolgen.txt | https://arxiv.org/pdf/2410.03439 | 3周・言葉の広げ直し（未読） |

## 次の手がかり
- 別の分野の言葉で探した結果、見えてきた領域: 意味解析（制約付きデコード+少数例）、生成型検索（文書 ID を数字か題名か）、多肢選択の記号結びつけ、対話システムの API 引数充填、ツールのスキーマ圧縮（遅延ロード）
- BFCL の論文（ICML 2025）— 取得失敗（PMLR は 404、OpenReview は 403）。ユーザーのブラウザで取れるか
- 小さなモデルで、プロンプトの長さ・few-shot の数と関数呼び出しの精度を直接測った研究
- KV キャッシュ無しで、CPU 上の文法制約つきデコードの時間を測った研究
- 「関数名を専用トークンにする」（Octopus v2）を、学習なしで近似できるかを扱う研究
