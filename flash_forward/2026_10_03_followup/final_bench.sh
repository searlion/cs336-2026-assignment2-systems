#!/usr/bin/env bash
# final_bench.sh: the unchanged harness on every version, three interleaved runs with a
# cool-down before each, so that slow drift in clocks and temperature hits all versions alike.
set -euo pipefail
PY=${PY:-python}
mkdir -p results/final
for r in 1 2 3; do
  for v in v0_baseline v1_tiles v2_autotune v3_causal_skip v4_exp2 v5_final; do
    sleep 20
    PYTHONPATH=kernels/$v $PY bench_flashattention.py > results/final/${v}_run$r.txt
    echo "$v run $r done"
  done
done
