#!/usr/bin/env bash
# round6.sh: every measurement behind the article's Round 6, in the order they were run.
#   ./round6.sh   (uses .venv/bin/python; about 1.5 hours, most of it the fair fight's cool-downs)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
NCU=/opt/nvidia/nsight-compute/2026.2.1/ncu
TRITON=../2026_10_03_followup/kernels/v5_final
export PATH=/usr/local/cuda-13.0/bin:$PATH
mkdir -p results/round6 results/triton_v5 results/gluon

# Step 0: headroom. The wave tail (B*H = 21 is whole waves for both kernels), locked clock and wall clock.
$NCU --set full --profile-from-start off --nvtx -f -o results/round6/wave_probe $PY wave_probe.py ncu
$PY ncu_table.py results/round6/wave_probe.ncu-rep > results/round6/wave_probe_ncu.txt
$PY wave_probe.py time > results/round6/wave_probe_time.txt
# What Triton's tl compiles to at the configuration it runs at N = 4096, and the warp_specialize probe.
for c in 0 1; do
  t=$([ $c = 1 ] && echo causal || echo noncausal)
  $PY ../2026_10_03_followup/inspect_kernel.py $TRITON 64,128,4,2 --causal $c --ttgir --sass \
      --dump-sass results/triton_v5/v5_64x128x4x2_$t.sass > results/triton_v5/inspect_64x128x4x2_$t.txt
  $PY results/triton_v5/loop_sync_counts.py results/triton_v5/v5_64x128x4x2_$t.sass
done > results/triton_v5/loop_sync_counts.txt
(cd experiments/triton_ws_probe && ../../$PY probe.py) > results/triton_v5/warp_specialize_probe.txt
# mbarriers from CuTe DSL on sm_89, and Gluon (results/round6/cute_mbarrier_probe.txt and
# results/gluon/gluon_probe.txt were assembled from these runs; see those files' headers).
(cd experiments/mbarrier_probe && USE_DSL_WRAPPERS=1 ../../$PY probe.py; ../../$PY probe.py;
 ../../$PY arrive_probe.py dsl; ../../$PY arrive_probe.py asm) || true
(cd experiments/gluon_probe && ../../$PY probe.py mbarrier; WS_REGS=48 ../../$PY probe.py ws) || true

# Step 2: each version. Unchanged tests, bitwise identity with v5, SASS, interleaved A/B, Nsight Compute.
for v in v6_mbarrier v7_swpipe t6_rotated; do
  echo "== $v"
  (cd kernels/$v && PYTHONPATH=. ../../$PY -m pytest -q -p no:cacheprovider ../../test_flashattention_triton.py | tail -1)
done >> results/tests.txt
$PY bitwise_check.py kernels/v6_mbarrier > results/round6/bitwise_v6.txt
$PY bitwise_check.py kernels/v7_swpipe > results/round6/bitwise_v7.txt
for v in v5_tuned v6_mbarrier; do for c in 0 1; do $PY sass_counts.py kernels/$v --causal $c; done; done > results/round6/sass_v5_v6.txt
for c in 0 1; do $PY sass_counts.py kernels/v7_swpipe --causal $c; done > results/round6/sass_v7.txt
$PY ab_compare.py kernels/v5_tuned kernels/v6_mbarrier --repeat 2 > results/round6/ab_v6_default.txt
for cfg in 64,64,4,2,1 64,64,4,2,2 64,64,4,1,2; do
  $PY ab_compare.py kernels/v5_tuned kernels/v6_mbarrier --cfgB $cfg --rounds 5 --shapes 4096:0,4096:1
done > results/round6/ab_v6_configs.txt
$PY ab_compare.py kernels/v5_tuned experiments/v6_nanosleep --rounds 5 --shapes 4096:0,4096:1 > results/round6/ab_v6_nanosleep.txt
$PY ab_compare.py kernels/v5_tuned kernels/v7_swpipe --repeat 2 > results/round6/ab_v7.txt
for cfg in 64,128,4,2 64,64,4,2; do
  $PY ab_compare.py $TRITON kernels/t6_rotated --cfgA $cfg --cfgB $cfg --rounds 7 --shapes 4096:0,4096:1 --repeat 2
done > results/round6/ab_t6_rotated.txt
$NCU --set full --import-source yes --profile-from-start off --nvtx -f -o results/round6/ncu_v5_v6 \
    $PY compare_driver.py --N 4096 --causal 0 1 --iters 1 kernels/v5_tuned kernels/v6_mbarrier
$PY ncu_table.py results/round6/ncu_v5_v6.ncu-rep > results/round6/ncu_v5_v6.txt
$NCU --set full --import-source yes --profile-from-start off --nvtx -f -o results/round6/ncu_v5_v7 \
    $PY compare_driver.py --N 4096 --causal 0 1 --iters 1 kernels/v5_tuned kernels/v7_swpipe
$PY ncu_table.py results/round6/ncu_v5_v7.ncu-rep > results/round6/ncu_v5_v7.txt

# Step 3: the sustained, interleaved comparison at N = 4096, with Triton v5 also pinned to 64 x 64
# tiles (the same diagonal work as CuTe's), and Nsight Compute's count of that work.
# What Triton's autotuner picked at N = 4096 causal in the 10 harness runs (results/final), and the time:
for f in results/final/triton_v5_run*.txt; do
  echo "$(basename $f) causal 4096 K_TILE=$(grep -A3 'with key as (4096, 4096, 64, True' $f | grep 'best config' \
      | sed 's|.*K_TILE_SIZE: \([0-9]*\).*|\1|') ours_ms=$(grep -E '^\s+4096\s+True' $f | awk '{print $3}')"
done > results/round6/triton_harness_causal4096_picks.txt
$PY fair_fight.py results/round6/fair_fight_round6.json 5 30 4096 cute_v5=kernels/v5_tuned \
    cute_v6=kernels/v6_mbarrier cute_v7=kernels/v7_swpipe triton_v5=$TRITON \
    triton_64x64=$TRITON@64,64,4,3 t6_rotated=kernels/t6_rotated sdpa=sdpa > results/round6/fair_fight_round6.txt
for cfg in 64,128,4,2 64,64,4,3 64,32,4,4; do
  $NCU --set full --profile-from-start off --nvtx -f -o results/round6/triton_${cfg//,/x}_causal \
      $PY ../2026_10_03_followup/profile_driver.py $TRITON --N 4096 --causal 1 --config $cfg --iters 1 --sdpa 0
done
$PY ncu_table.py results/round6/triton_64x128x4x2_causal.ncu-rep results/round6/triton_64x64x4x3_causal.ncu-rep results/round6/triton_64x32x4x4_causal.ncu-rep \
    > results/round6/triton_tiles_causal_ncu.txt
$PY summarize_round6.py > results/round6/summary.txt
# Why is CuTe v5 faster causal at equal tiles? Executed instructions and stall samples per region.
$PY ../2026_10_03_followup/inspect_kernel.py $TRITON 64,64,4,3 --causal 1 --sass \
    --dump-sass results/triton_v5/v5_64x64x4x3_causal.sass > results/triton_v5/inspect_64x64x4x3_causal.txt
$PY results/round6/causal_gap.py > results/round6/causal_gap.txt
