#!/usr/bin/env bash
# run_picks_long_cold.sh: does the 5x longer do_bench (100 ms warm-up per candidate) also remove
# the cold-GPU bias? The same conditions as run_picks_warm.sh (20 s idle before every process,
# N = 512 non-causal, the harness's first shape), with v2_long_bench and no extra warm-up.
#   PY=../../.venv/bin/python ./run_picks_long_cold.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
for i in $(seq 1 10); do
  sleep 20
  PYTHONPATH="$HERE/kernels/v2_long_bench" "$PY" "$HERE/pick.py" --N 512 --causal 0 \
      | sed "s/^{/{\"setting\": \"v2_long_bench_cold\", /" >> "$HERE/results/picks_long_cold.jsonl"
  echo "round $i done"
done
