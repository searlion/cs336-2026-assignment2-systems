#!/usr/bin/env bash
# run_picks.sh: 20 fresh processes per setting, interleaved, each letting the autotuner choose
# for causal N = 4096 (the shape where v2's choice varied), then 10 for non-causal N = 512.
# cache_results runs get their own TRITON_CACHE_DIR, which starts empty, so the first process
# tunes and writes the timings, and the rest should reuse them.
#   PY=../../.venv/bin/python ./run_picks.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FU=$(cd "$HERE/../../2026_10_03_followup" && pwd)
PY=${PY:-python}
OUT=$HERE/results
mkdir -p "$OUT"
CACHE_RESULTS_DIR=$(mktemp -d)
pick() {  # setting kernel_dir N causal
  local extra=()
  [ "$1" = v2_cache_results ] && extra=(env TRITON_CACHE_DIR="$CACHE_RESULTS_DIR")
  "${extra[@]}" env PYTHONPATH="$2" "$PY" "$HERE/pick.py" --N "$3" --causal "$4" \
      | sed "s/^{/{\"setting\": \"$1\", /" >> "$OUT/picks.jsonl"
}
for i in $(seq 1 20); do
  pick v2_default "$FU/kernels/v2_autotune" 4096 1
  pick v2_long_bench "$HERE/kernels/v2_long_bench" 4096 1
  pick v2_short_list "$HERE/kernels/v2_short_list" 4096 1
  pick v2_cache_results "$HERE/kernels/v2_cache_results" 4096 1
  echo "round $i done"
done
for i in $(seq 1 10); do
  pick v2_default "$FU/kernels/v2_autotune" 512 0
  pick v2_long_bench "$HERE/kernels/v2_long_bench" 512 0
  echo "N=512 round $i done"
done
