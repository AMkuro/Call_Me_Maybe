#!/usr/bin/env bash
# 任意のブランチ/worktree の `uv run python -m src` を測る（既定は校正済み phys16-t8）。
# 方式ごとの実装には何も要求しない。壁時間・ピーク RSS・CPU 温度を記録する。
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
MODULE=${CMM_MODULE:-src}
case "$MODULE" in
  src|tools.prompt_trial) ;;
  *) echo 'CMM_MODULE must be src or tools.prompt_trial.' >&2; exit 2 ;;
esac
# 測定開始前に CPU を冷ます目標 (℃) と、下がらないと判断するまでの秒数。
COOL_TARGET=${CMM_COOL_TARGET_C:-50}
COOL_STALL=${CMM_COOL_STALL_S:-60}
if [[ ! $COOL_TARGET =~ ^[1-9][0-9]{0,2}$ || ! $COOL_STALL =~ ^[1-9][0-9]{0,3}$ ]]; then
  echo 'CMM_COOL_TARGET_C (1-999) and CMM_COOL_STALL_S (1-9999) must be integers.' >&2
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

# CPU パッケージ温度 (millidegree)。AMD は k10temp の Tctl、Intel は coretemp の Package。
# CPU のセンサーが無ければ空 (他のセンサーの値で代用しない)。
SENSOR=
for hwmon in /sys/class/hwmon/hwmon*; do
  case $(cat "$hwmon/name" 2>/dev/null) in
    coretemp|k10temp|zenpower) SENSOR=$hwmon/temp1_input; break ;;
  esac
done
tctl() { [ -n "$SENSOR" ] && cat "$SENSOR" 2>/dev/null; }

# 目標温度まで待つ。COOL_STALL 秒のあいだ 1℃ も下がらなければ、測定せずに止める。
WAIT_START=$SECONDS
T=$(tctl)
if [ -n "$T" ]; then
  BEST=$T
  LAST_DROP=$SECONDS
  while [ "$T" -gt $((COOL_TARGET * 1000)) ]; do
    if [ $((SECONDS - LAST_DROP)) -ge "$COOL_STALL" ]; then
      echo "CPU did not cool: ${T} mC after ${COOL_STALL}s without a 1C drop (target ${COOL_TARGET}C)." >&2
      exit 3
    fi
    sleep 1
    T=$(tctl)
    if [ "$T" -le $((BEST - 1000)) ]; then BEST=$T; LAST_DROP=$SECONDS; fi
  done
fi
COOL_WAIT=$((SECONDS - WAIT_START))

# 実行中の最高温度を 1 秒ごとに記録する。
T0=$T
TMAX_FILE="$OUT/$LABEL.tmax"
echo "${T0:-0}" > "$TMAX_FILE"
( MAX=${T0:-0}
  while sleep 1; do
    T=$(tctl)
    if [ -n "$T" ] && [ "$T" -gt "$MAX" ]; then MAX=$T; echo "$MAX" > "$TMAX_FILE"; fi
  done ) &
SAMPLER=$!

systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 \
  taskset -c 0-15 /usr/bin/time -f 'wall_s=%e max_rss_kib=%M' -o "$OUT/$LABEL.time" \
  uv run python -m "$MODULE" "$@"
RC=$?
kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
TMAX=$(cat "$TMAX_FILE")
# AMD の公表する最高動作温度 (Tjmax) は 100℃。
LIMIT=not_reached
[ -n "$T0" ] && [ "$TMAX" -ge 100000 ] && LIMIT=reached
{
  cat "$OUT/$LABEL.time"
  echo "exit_code=$RC"
  echo "tctl_start_mc=${T0:-unavailable} tctl_end_mc=$(tctl || echo unavailable)" \
    "tctl_max_mc=$([ -n "$T0" ] && echo "$TMAX" || echo unavailable)" \
    "cool_wait_s=$COOL_WAIT cool_target_c=$COOL_TARGET thermal_limit=$LIMIT"
  echo "profile=phys16-t$THREADS mem_max=16G swap=0 cpus=0-15 omp=$THREADS module=$MODULE"
} > "$OUT/$LABEL.txt"
rm -f "$OUT/$LABEL.time" "$TMAX_FILE"
cat "$OUT/$LABEL.txt"
exit $RC
