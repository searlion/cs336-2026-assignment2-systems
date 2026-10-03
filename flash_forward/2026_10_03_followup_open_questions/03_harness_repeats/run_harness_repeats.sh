#!/usr/bin/env bash
# run_harness_repeats.sh: final_bench.sh with 10 repetitions instead of 3.
# The unchanged harness on every version, interleaved, with a 20-second cool-down before
# each run, while nvidia-smi logs the SM clock and power every 200 ms (used by item 1).
#   PY=../../.venv/bin/python ./03_harness_repeats/run_harness_repeats.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FU=$(cd "$HERE/../../2026_10_03_followup" && pwd)
PY=${PY:-python}
OUT=$HERE/runs
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw.instant,temperature.gpu,clocks_event_reasons.active \
    --format=csv -lms 200 > "$OUT/nvidia_smi.csv" &
SMI=$!
trap 'kill $SMI' EXIT
stamp() { date +"%Y/%m/%d %H:%M:%S.%3N"; }
for r in $(seq 1 10); do
  for v in v0_baseline v1_tiles v2_autotune v3_causal_skip v4_exp2 v5_final; do
    sleep 20
    echo "$(stamp) start $v $r" >> "$OUT/timeline.log"
    TRITON_PRINT_AUTOTUNING=1 PYTHONPATH="$FU/kernels/$v" "$PY" "$FU/bench_flashattention.py" > "$OUT/${v}_run$r.txt"
    echo "$(stamp) end $v $r" >> "$OUT/timeline.log"
    echo "$v run $r done"
  done
done
