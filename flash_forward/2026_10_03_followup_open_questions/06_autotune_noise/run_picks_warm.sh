#!/usr/bin/env bash
# run_picks_warm.sh: the clock-ramp test. In the 10-run harness batch, v2's autotuner picked
# (64, 16, 4, 5) at N = 512 non-causal (the harness's first shape) in 9 of 10 runs, although
# (64, 32, 4, 4) is 6% faster; in back-to-back fresh processes (run_picks.sh) it picked
# (64, 32, 4, 4) 9 times in 10. The harness sleeps 20 s before each run. Here: 20 s of idle GPU
# before every process, as in the harness, then the autotuner with and without 1 s of GPU
# warm-up first, interleaved, 10 times each.
#   PY=../../.venv/bin/python ./run_picks_warm.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FU=$(cd "$HERE/../../2026_10_03_followup" && pwd)
PY=${PY:-python}
for i in $(seq 1 10); do
  for w in 0 1; do
    sleep 20
    PYTHONPATH="$FU/kernels/v2_autotune" "$PY" "$HERE/pick.py" --N 512 --causal 0 --warm $w \
        | sed "s/^{/{\"setting\": \"v2_default_warm$w\", /" >> "$HERE/results/picks_warm.jsonl"
  done
  echo "round $i done"
done
