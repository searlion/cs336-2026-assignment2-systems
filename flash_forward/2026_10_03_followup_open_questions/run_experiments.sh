#!/usr/bin/env bash
# run_experiments.sh: every GPU experiment for the open questions, one after another (they must
# not overlap: each one times or profiles the GPU). The 10-run harness batch is separate
# (03_harness_repeats/run_harness_repeats.sh) because it takes half an hour on its own.
#   PY=../.venv/bin/python ./run_experiments.sh [step ...]
# Binary profiler reports go to <item>/reports/ (git-ignored); CSV/JSON/text exports sit next to them.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FU=$(cd "$HERE/../2026_10_03_followup" && pwd)
PY=${PY:-python}
NCU=/opt/nvidia/nsight-compute/2026.2.1/ncu
NSYS=/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
mkdir -p "$HERE/logs"
cd "$HERE"

tensor_pipe() {  # item 2
  cd 02_tensor_pipe && mkdir -p reports
  $PY matmul_variants.py time > matmul_timing.txt
  local m="gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second"
  m+=",sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
  m+=",sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_elapsed"
  m+=",sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed"
  m+=",sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_elapsed"
  m+=",sm__inst_executed_pipe_tensor_op_hmma_v2.avg.pct_of_peak_sustained_elapsed"
  m+=",sm__inst_executed_pipe_tensor_op_hmma_v2.sum,sm__inst_executed_pipe_tensor_op_imma_v2.sum"
  m+=",sm__pipe_tensor_cycles_active.avg.peak_sustained,sm__pipe_tensor_cycles_active.sum"
  for p in bf16_dst_fp32 fp16_dst_fp32 fp16_dst_fp16 tf32_dst_fp32 int8 fp8; do
    m+=",sm__ops_path_tensor_src_${p}_sparsity_off.sum"
    m+=",sm__ops_path_tensor_src_${p}_sparsity_off.sum.pct_of_peak_sustained_elapsed"
    m+=",sm__ops_path_tensor_src_${p}_sparsity_off.sum.peak_sustained"
  done
  $NCU --metrics "$m" --nvtx --profile-from-start off -f -o reports/matmul_variants $PY matmul_variants.py profile
  # The same counters for the attention kernels, from the article's reports.
  $NCU --metrics "$m" --nvtx --profile-from-start off -f -o reports/attention \
      $PY "$FU/compare_driver.py" --N 4096 --causal 0 --iters 1 "$FU/kernels/v0_baseline" "$FU/kernels/v1_tiles" "$FU/kernels/v5_final" sdpa
  $PY pipe_table.py reports/matmul_variants.ncu-rep reports/attention.ncu-rep > pipe_table.txt
  cd "$HERE"
}

nonsquare() {  # item 4
  $PY 04_nonsquare_causal/check_nonsquare_causal.py 04_nonsquare_causal/results.csv > 04_nonsquare_causal/summary.txt
}

key_tile() {  # items 5 and 8
  cd 05_key_tile && mkdir -p reports
  $PY key_tile_driver.py time timing.csv
  TRITON_CACHE_DIR=$(mktemp -d) $NCU --set full --import-source yes --nvtx --profile-from-start off -f \
      -o reports/v4_key_tiles $PY key_tile_driver.py profile
  cd "$HERE"
}

autotune_noise() {  # item 6
  cd 06_autotune_noise && mkdir -p results
  $PY make_variants.py
  $PY ground_truth.py results/ground_truth.csv
  PY=$PY ./run_picks.sh
  cd "$HERE"
}

pcast() {  # item 9
  $PY 09_pcast_accuracy/pcast_accuracy.py 09_pcast_accuracy/results.json > 09_pcast_accuracy/log.txt
}

shortlist() {  # item 11
  $PY 11_shortlist_retiming/retime_shortlist.py 11_shortlist_retiming/retime.csv
}

smaller() {  # item 13
  cd 13_smaller && mkdir -p reports
  $PY stages_occupancy.py time stages_timing.csv
  $NCU --section LaunchStats --section Occupancy --section SpeedOfLight --section WarpStateStats \
      --section SchedulerStats --nvtx --profile-from-start off -f -o reports/stages $PY stages_occupancy.py profile
  $PY reverse_order.py reverse_order.csv
  for w in 0 1; do
    $NSYS profile --trace=cuda,nvtx --gpu-metrics-devices=0 --gpu-metrics-frequency=20000 \
        --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
        -o reports/cold_capture_warmup$w $PY cold_capture.py --warmup $w
    $NSYS export --type sqlite --force-overwrite true -o reports/cold_capture_warmup$w.sqlite \
        reports/cold_capture_warmup$w.nsys-rep > /dev/null
  done
  cd "$HERE"
}

clock() {  # item 1
  cd 01_clock && mkdir -p reports results
  $PY sustained_clock.py results/sustained.json
  for cc in base none; do
    $NCU --clock-control $cc --section SpeedOfLight --section SpeedOfLight_HierarchicalTensorRooflineChart \
        --metrics sm__cycles_elapsed.avg.per_second,gpc__cycles_elapsed.avg.per_second \
        --nvtx --profile-from-start off -f -o reports/clock_control_$cc \
        $PY "$FU/compare_driver.py" --N 4096 --causal 0 1 --iters 1 "$FU/kernels/v5_final" sdpa
  done
  cd "$HERE"
}

harness_clock_nsys() {  # item 1: the GPU clock during each kernel of one harness run, from GPU metrics
  cd 01_clock && mkdir -p reports
  PYTHONPATH="$FU/kernels/v5_final" $NSYS profile --trace=cuda --gpu-metrics-devices=0 --gpu-metrics-frequency=10000 \
      --force-overwrite=true -o reports/harness_v5 $PY "$FU/bench_flashattention.py" > harness_v5_under_nsys.txt
  $NSYS export --type sqlite --force-overwrite true -o reports/harness_v5.sqlite reports/harness_v5.nsys-rep > /dev/null
  $PY kernel_clock_from_nsys.py reports/harness_v5.sqlite > harness_kernel_clocks.txt
  cd "$HERE"
}

small_n() {  # item 10: why SDPA only reaches ~33 TFLOP/s at N = 512
  mkdir -p 10_small_n/reports
  $NCU --set full -f --profile-from-start off -o 10_small_n/reports/n512 \
      $PY "$FU/compare_driver.py" --N 512 --causal 0 --iters 1 "$FU/kernels/v5_final" sdpa
  $PY "$FU/ncu_table.py" 10_small_n/reports/n512.ncu-rep > 10_small_n/ncu_table.txt
  $PY 10_small_n/analyze.py 10_small_n/reports/n512.ncu-rep > 10_small_n/summary.txt
}

compiler() {  # item 12
  $PY 12_compiler_behaviour/check_compiler.py > 12_compiler_behaviour/check_compiler.txt
}

smem() {  # item 13: the extra 1,024 bytes for fp32 at D = 128
  $PY 13_smaller/smem_fp32_d128.py > 13_smaller/smem_fp32_d128.txt
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(tensor_pipe nonsquare key_tile autotune_noise pcast shortlist smaller clock
                                 harness_clock_nsys small_n compiler smem)
for s in "${steps[@]}"; do
  echo "$(date +%T) start $s"
  "$s" > "logs/$s.log" 2>&1
  rc=$?
  echo "$(date +%T) end $s (exit $rc)"
done
