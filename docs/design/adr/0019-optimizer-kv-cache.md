# ADR-0019: 改善側は既存OSSハーネスと推論サーバーを使う

状態: ユーザー指定を反映（2026-11-11）。

## 背景と指示

SDKで始めた9Bの2試行は、最初の改稿だけで数分かかった。SDKは生成ごとにprefix全体を再計算する。
ユーザーは改善用だけのKV cache利用を承認した後、「OSS harnessを使う。作る必要はなく、
tools・prompt・workspacesを設定してworkflowを考える」と範囲を明確にした。Ponytail fullを適用する。

## 構成

導入済みOpenCode 1.15.10をエージェント実行基盤として使い、既存Transformers Serveへ接続する。
これは実装側の選定で、ユーザーが個別の製品名を指定したものではない。
独自のトークン生成・エージェント会話ループは削除した。既存の採点・CPU計測・追加評価の連携は再利用する。

- 設定: [OpenCode設定](../../../research/prompt-improvement/opencode.json)、指示文、作業領域、2ツール。
- モデル: Qwen3.5-9B/BF16/GPU、native generateのKV cache。モデルrevisionを固定する。
  サーバーとクライアントのモデルIDを同じ`モデル名@revision`にする。
- 権限: 組み込みtoolsをすべて無効化し、`write_prompt`と`run`だけを許す。
- 作業領域: エージェントごとに分け、1領域につき1候補だけを受け付ける。
- `write_prompt`: 2つの指示文字列のみ。`run`: 引数なしで固定の11件を評価する。
  コマンド、パス、データ集合、goldはモデルに指定させない。
- 実行: 既存OSSが会話・tool callingを担当し、native sessionとtool logを保存する。
  こちらのコードはデータ操作と、固定された採点・昇格・環境同期だけを担当する。
  次の改稿には、未採用候補も含めた開発用の指示文と実スコアを渡し、同じ決定的入力の再送を避ける。
  モデルの文章による自己申告は採点・昇格に使わない。
- GPUサービス: loopbackの2つのtransient user unitで並列実行し、CPU測定前と終了時に停止する。

対象のQwen3-0.6BはSDK・float32・既存の生成方式を固定する。改善側の速さを対象の推論性能と混同しない。
SDKで開始した2試行は採点前に中断した。出力と消費済み2枠を残し、残り6枠で初回の計8枠を完了する。

## 確認・制限

OpenCodeの実効設定で、公開toolsがちょうど2つであることを確認した。
Native serverは42トークンのtool callを7.48秒で返した。小さな接続確認であり、改稿効果の証明ではない。
初回の接続ではクライアントのrevision欠落により6接続とも推論前に拒否された。
この設定失敗は品質試行に数えず記録を残し、モデルIDの一致を直接テストして修正した。
実際の改善は対象0.6Bで評価し、改善した1候補だけをCPU・固定した88件へ進める。
現時点で確認した構成はLinux/ROCmの開発環境であり、提出環境の保証ではない。

出典: [OpenCode V1 Custom Tools](https://opencode.ai/v1/docs/custom-tools/)、
[Transformers Serve](https://huggingface.co/docs/transformers/serve-cli/serving)。
