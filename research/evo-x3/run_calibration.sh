#!/usr/bin/env bash
# 校舎 benchmark (tools.campus_survey benchmark, profile v1) を EVO-X3 上で条件を変えて繰り返す。
# 各条件 3 回。結果は research/evo-x3/runs/<条件>-<n>.txt に bench.* 行だけ保存する。
# 使い方: bash research/evo-x3/run_calibration.sh
set -u
LAB=${CMM_LAB:-$HOME/cmm-cpu-lab}  # venv・モデルの置き場 (Git 管理外)
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=$LAB/venv/bin/python
OUT=$REPO/research/evo-x3/runs
mkdir -p "$OUT"
cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 条件名:affinity:OMP_NUM_THREADS(空=既定)
CONFIGS="
all32-default:0-31:
all32-t20:0-31:20
all32-t16:0-31:16
all32-t12:0-31:12
phys16-default:0-15:
phys16-t8:0-15:8
"

for n in 1 2 3; do
  for c in $CONFIGS; do
    IFS=: read -r name cpus threads <<<"$c"
    envs=()
    [ -n "$threads" ] && envs=(env OMP_NUM_THREADS="$threads")
    sleep 20  # 連続実行による熱の持ち越しを減らす
    # メモリ上限 16GiB、swap 禁止 (校舎 15.3GiB 相当)
    systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 \
      taskset -c "$cpus" "${envs[@]}" "$PY" -m tools.campus_survey benchmark \
      --pc-id "evo-x3-$name" --hf-home $LAB/hf 2>&1 \
      | grep -E '^bench\.' > "$OUT/$name-$n.txt"
    echo "done $name-$n"
  done
done
