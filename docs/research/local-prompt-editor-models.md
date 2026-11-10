# ローカルのプロンプト改善用モデルの選定調査

調査日: 2026-11-11。ユーザーの指示で、2つの調査エージェントが
公式モデル資料と独立した評価・研究を分担して調べた。
**評価対象のQwen3-0.6Bと、指示文を改稿する改善用モデルを区別する。**

## 推奨と根拠

最初に試す改善用モデルは **Qwen/Qwen3.5-9B、GPU、BF16、non-thinking** とする。
当初の0.6B案はユーザーが不適切と指摘したため撤回した。
8Bを暫定案として挙げた後、調査で見つかった新しい9Bへ更新した。
これは最初の実験候補の選定であり、この課題で最良と実測した結論ではない。
本体のQwen3-0.6B・float32・greedy・Constrained Decodingは固定する。

| 候補 | 調べた根拠 | メモリと導入上の判断 |
| --- | --- | --- |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 提供元のIFEval 91.5、IFBench 64.5、BFCL-V4 66.1。独立したプロンプト改稿の研究もある | 公開weight総量は約19.3 GB。2プロセスの最初の試行候補にする |
| [Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12B-it) | system roleとtool calling。異なる系列の比較候補 | 公開weight約23.9 GB。実ロード、固有テンプレート、利用条件は実行前に確認する |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 新しい系列で指示追従・agent能力を提供元が報告 | 公開weight約55.6 GB。2コピーは可視100 GiBを重みだけで超えるため、必要なら1コピーの拡大候補 |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 同じ提供元のIFEval 89.8、BFCL-V4 50.3 | 約9.34 GB。安価な対照候補だが、今回は余裕のある環境で9Bを先に試す |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) / [14B](https://huggingface.co/Qwen/Qwen3-14B) | 従来のdense Transformer。既存SDKに近い構成 | Qwen3.5のロードやROCm実行に問題がある場合の代替候補 |

ダウンロード量、パラメータ数×2 bytesの概算と、推論時のピーク使用量は別である。
上表はピークVRAMの実測ではない。提供元のスコアはモデルカード内の条件によるもので、
今回の別モデル向けプロンプト編集能力や系列間の公平な順位を保証しない。

[LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)も調査した。
常にthinkingを出す設定と固有のツール形式は、今回の512トークン・JSON操作の制限に合わせる
追加作業があるため初回候補にしない。Ryzen AI Max+395での公表速度はllama.cppの量子化条件で、
今回のSDK/BF16へ移して使えない。

## プロンプト改稿の直接的な研究と限界

[Zhou・Chengの2026年10月のプレプリント](https://arxiv.org/html/2610.27367)では、
Qwen3.5-9B自身が解答・誤答分析・プロンプト改稿を行い、thinkingを無効にしている。
5課題の平均は初期41.38、通常GEPA 43.02、提案方式45.68だった。
ただし100候補の保持と20回のvalidation、4096出力トークンを使っており、
今回の最大8候補・512トークン、別の0.6B向け改稿の効果を示す実験ではない。
独立追試もこの調査では確認できなかった。

[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)は関数呼び出し、
[IFEval](https://arxiv.org/abs/2311.07911)は検証可能な指示追従を測る。
これらはツールの形式を守る能力の参考になるが、改稿の効果を直接測らない。
[IFBench](https://arxiv.org/html/2507.02833)は既知の指示形式への過適合にも注意を向ける。
[Maほかの研究](https://arxiv.org/html/2402.02101v1)には、もっともらしい誤答分析や改稿でも、
対象モデルの実際の誤りの改善に結びつかない例がある。
改善用モデルの説明ではなく、対象0.6Bの固定入力での結果で採否を決める。

## 既存環境との互換性の確認範囲

確認したホスト環境はRadeon 8060S、可視100 GiB、torch 2.14.1+rocm7.2、
Transformers 5.22.2。共有RAMは124 GiB、確認時のディスク空きは622 GBだった。

SDKの公開constructorはモデルIDとdtypeを受け取る。既存Transformersには
`qwen3_5`からテキスト用`Qwen3_5ForCausalLM`への対応と、通常のtorchによる
DeltaNet/convの実装がある。新しいランタイムが必須とまでは判断しなかった。
**実際のモデルロード・ROCm推論・並列動作は未検証である。**

公式repositoryのvocab.json、tokenizer.json、merges.txtと
[non-thinking用テンプレート](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/chat_template.jinja)を確認した。
通常のsystem/user ChatMLに2操作のJSON形式を書き、空のthinkブロックで完結形を生成する案である。
固有の`tools=`形式はXMLのtool callを要求するため、今回のJSON操作と混在させない。
SDKのlogits取得はprefix全体を毎回計算する。KV cacheを使う他ランタイムの公表速度は当てはまらない。

## 実行状況と次の比較

Qwen3.5-9Bの公式モデル情報では、revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`、safetensors総量19306310880 bytesを確認した。
共有キャッシュへの取得を開始したが、ユーザーの作業終了指示で中断した。
取得済み部分は残し、モデルの実行はしていない。

再開時はまず既知のツール実装レビュー指摘を直す。その後、9Bが有効な2操作と一般的な改稿を
出せるかを小さく確認し、0.6Bの完了・型適合・完全正答・既存正答の喪失を独立に測る。
別系列も比較するなら、8候補の予算を事前にモデル間で均等配分する案がある。
この比較案はまだ採用・実施していない。追加33件はモデルや候補の選定へ流用しない。
現状の制限と再開手順は[引継ぎ](../handoffs/2026-11-11-local-prompt-improvement.md)にある。
