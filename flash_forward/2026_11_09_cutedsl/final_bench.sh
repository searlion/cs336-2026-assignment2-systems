#!/usr/bin/env bash
# final_bench.sh: the unchanged harness on every version, interleaved, with a cool-down before
# each run, so slow drift in clocks and temperature hits all versions alike. Three runs of every
# version (and of NVIDIA's example at 64 x 64), then seven more of the CuTe final kernel and the
# Triton kernel, for ten in all. Every harness run also times SDPA. bench_with_power.py runs the
# harness's bench() unchanged while sampling NVML (power, temperature, clocks, throttle reasons).
# Then the sustained "fair fight" at N = 4096, and an nsys capture with GPU metrics (SM clock).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
NV_EXAMPLE=${NV_EXAMPLE:?path to CUTLASS v4.8.1 examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py}
mkdir -p results/final/power results/nsys
run() {  # name, kernel dir, run index
  sleep 20
  TRITON_PRINT_AUTOTUNING=1 PYTHONPATH=$2 $PY bench_with_power.py results/final/power/$1_run$3.json \
      > results/final/$1_run$3.txt 2>/dev/null
  echo "$1 run $3 done"
}
for r in 1 2 3; do
  for v in v0_naive v1_swizzle v2_ldmatrix v3_p_regs v4_cpasync v5_tuned; do run $v kernels/$v $r; done
  run triton_v5 ../2026_10_03_followup/kernels/v5_final $r
  sleep 20
  POWER_JSON=results/final/power/nvidia_example_run$r.json $PY bench_nvidia_example.py "$NV_EXAMPLE" 64,64 \
      > results/final/nvidia_example_run$r.txt 2>/dev/null
  echo "nvidia_example run $r done"
done
for r in 4 5 6 7 8 9 10; do
  run v5_tuned kernels/v5_tuned $r
  run triton_v5 ../2026_10_03_followup/kernels/v5_final $r
done
sleep 30
$PY fair_fight.py results/fair_fight.json 3 30 > results/fair_fight.txt 2>/dev/null
echo "fair fight done"
sleep 30
/opt/nvidia/nsight-systems/2026.1.3/bin/nsys profile --trace=cuda,nvtx --gpu-metrics-devices=all \
    --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o results/nsys/clocks_final \
    $PY compare_driver.py --N 4096 --causal 0 1 --iters 10 kernels/v5_tuned ../2026_10_03_followup/kernels/v5_final sdpa
echo "nsys clocks done"
