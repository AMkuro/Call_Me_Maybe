#!/usr/bin/env bash
# 任意のブランチ/worktree の `uv run python -m src` を測る（既定は校正済み phys16-t8）。
# 方式ごとの実装には何も要求しない。壁時間とピーク RSS だけを記録する。
# CPU 時間評価用: 元の配布 SDK・CPU のみ。GPU 版 SDK では使わない。
# 使い方: bash research/evo-x3/measure.sh <リポジトリのパス> <ラベル> [python -m src への引数...]
# 出力:   research/evo-x3/measurements/<ラベル>.txt
# CMM_OMP_THREADS=6 などで比較候補を指定できる。既存プロファイルの採用変更ではない。
set -u
LAB=${CMM_LAB:-$HOME/cmm-cpu-lab}  # venv・モデルの置き場 (Git 管理外)
THREADS=${CMM_OMP_THREADS:-8}
if [[ ! $THREADS =~ ^([1-9]|1[0-6])$ ]]; then
  echo 'CMM_OMP_THREADS must be an integer from 1 to 16.' >&2
  exit 2
fi
REPO=${1:?repo path}
LABEL=${2:?label}
shift 2
OUT=$(cd "$(dirname "$0")" && pwd)/measurements
mkdir -p "$OUT"
cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= HF_HOME=$LAB/hf \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=$THREADS

# hwmon の最高温度 (millidegree)。取れなければ空。
temp() { cat /sys/class/hwmon/hwmon*/temp*_input 2>/dev/null | sort -n | tail -1; }

T0=$(temp)
systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 \
  taskset -c 0-15 /usr/bin/time -f 'wall_s=%e max_rss_kib=%M' -o "$OUT/$LABEL.time" \
  uv run python -m src "$@"
RC=$?
{
  cat "$OUT/$LABEL.time"
  echo "exit_code=$RC"
  echo "temp_start_mc=${T0:-unavailable} temp_end_mc=$(temp)"
  echo "profile=phys16-t$THREADS mem_max=16G swap=0 cpus=0-15 omp=$THREADS"
} > "$OUT/$LABEL.txt"
rm -f "$OUT/$LABEL.time"
cat "$OUT/$LABEL.txt"
exit $RC
