#!/usr/bin/env bash
# run_followups.sh: two follow-up experiments that the first results called for (run after
# run_experiments.sh, never at the same time).
#   1. Item 6: is the autotuner's bias at N = 512 (the harness's first shape) caused by the GPU
#      clock ramping up after the harness's 20 s cool-down?  (06_autotune_noise/run_picks_warm.sh)
#   2. Item 13: SDPA's very first call in a process inside a cold capture, the other difference
#      between the article's first Round 0 capture and cold_capture.py's default.
#   3. Item 1: the Nsight Compute capture with the default --clock-control boost.
#   4. Item 6: the same cold test with the 5x longer do_bench (06_autotune_noise/run_picks_long_cold.sh).
#   PY=../.venv/bin/python ./run_followups.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
NSYS=/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
cd "$HERE"
echo "$(date +%T) start picks_warm"
PY=$PY 06_autotune_noise/run_picks_warm.sh > logs/picks_warm.log 2>&1
echo "$(date +%T) end picks_warm (exit $?)"
cd 13_smaller
$NSYS profile --trace=cuda,nvtx --gpu-metrics-devices=0 --gpu-metrics-frequency=20000 \
    --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
    -o reports/cold_capture_first_sdpa_call $PY cold_capture.py --warmup 0 --precall-sdpa 0 > ../logs/cold_first_call.log 2>&1
$NSYS export --type sqlite --force-overwrite true -o reports/cold_capture_first_sdpa_call.sqlite \
    reports/cold_capture_first_sdpa_call.nsys-rep > /dev/null
echo "$(date +%T) end cold capture (first SDPA call inside)"
# 3. Item 1: the same Nsight Compute capture as run_experiments.sh's clock step, with the default
#    --clock-control boost (that step ran base = 1.98 GHz and none).
cd "$HERE/01_clock"
/opt/nvidia/nsight-compute/2026.2.1/ncu --clock-control boost --section SpeedOfLight \
    --section SpeedOfLight_HierarchicalTensorRooflineChart \
    --metrics sm__cycles_elapsed.avg.per_second,gpc__cycles_elapsed.avg.per_second \
    --nvtx --profile-from-start off -f -o reports/clock_control_boost \
    $PY "$HERE/../2026_10_03_followup/compare_driver.py" --N 4096 --causal 0 1 --iters 1 \
    "$HERE/../2026_10_03_followup/kernels/v5_final" sdpa > "$HERE/logs/clock_boost.log" 2>&1
echo "$(date +%T) end ncu boost clock"
# 4. Item 6: does the longer do_bench also survive a cold start?
cd "$HERE"
PY=$PY 06_autotune_noise/run_picks_long_cold.sh > logs/picks_long_cold.log 2>&1
echo "$(date +%T) end picks_long_cold"
