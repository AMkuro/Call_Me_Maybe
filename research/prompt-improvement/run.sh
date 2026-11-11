#!/usr/bin/env bash
# Native OSS servers + OpenCode agents; Python retains only evaluation gates.
set -euo pipefail
GPU_ENV=$(realpath "${1:?GPU environment}")
shift
CACHE=${HF_HOME:-$PWD/.cache/huggingface}
MODEL=Qwen/Qwen3.5-9B
REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
CMM_ARGS=("$@")
for ((i=0; i<${#CMM_ARGS[@]}; i++)); do
  case ${CMM_ARGS[i]} in
    --cache) CACHE=${CMM_ARGS[i+1]} ;;
    --optimizer) MODEL=${CMM_ARGS[i+1]} ;;
    --optimizer-revision) REV=${CMM_ARGS[i+1]} ;;
  esac
done
CACHE=$(realpath "$CACHE")
"$GPU_ENV/bin/python" - <<'PY'
import socket
for port in (8011, 8012):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
PY
UNITS=("cmm-prompt-${BASHPID}-8011" "cmm-prompt-${BASHPID}-8012")
echo "optimizer_units=${UNITS[*]}"
cleanup() { systemctl --user stop "${UNITS[@]}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
for i in 0 1; do
  systemd-run --user --quiet --collect --unit="${UNITS[i]}" \
    -p "WorkingDirectory=$PWD" -p RuntimeMaxSec=1800 \
    --setenv="HF_HOME=$CACHE" --setenv="HF_HUB_CACHE=$CACHE/hub" \
    --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1 \
    "$GPU_ENV/bin/transformers" serve "$MODEL@$REV" --device cuda:0 \
    --dtype bfloat16 --reasoning off --host 127.0.0.1 --port "$((8011+i))"
done
export CMM_OPTIMIZER_UNITS="${UNITS[*]}"
"$GPU_ENV/bin/python" - <<'PY'
import time, urllib.request
remaining = {8011, 8012}
deadline = time.monotonic() + 120
while remaining and time.monotonic() < deadline:
    for port in list(remaining):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=2):
                remaining.remove(port)
        except OSError:
            pass
    if remaining:
        time.sleep(1)
if remaining:
    raise SystemExit(f'OSS model server startup failed: {sorted(remaining)}')
PY
"$GPU_ENV/bin/python" -m tools.prompt_experiment "$@"
