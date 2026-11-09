#!/usr/bin/env bash
# profile_all.sh: every Nsight Systems and Nsight Compute capture used in the article.
#   ./profile_all.sh            (uses .venv/bin/python)
set -euo pipefail
cd "$(dirname "$0")"
NSYS=/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
NCU=/opt/nvidia/nsight-compute/2026.2.1/ncu
PY=${PY:-.venv/bin/python}
# Line information in the CuTe kernels (for the Source page), and fresh caches, so every
# kernel is compiled with it and the Triton kernel's line info points at today's files.
export CUTE_DSL_LINEINFO=1
export CUTE_DSL_CACHE_DIR=${CUTE_DSL_CACHE_DIR:-$(mktemp -d)}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$(mktemp -d)}
mkdir -p results/nsys results/ncu
NSYS_ARGS=(profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true)
NCU_ARGS=(--set full --import-source yes --profile-from-start off --nvtx -f)
TRITON=../2026_10_03_followup/kernels/v5_final
CUTE=(kernels/v0_naive kernels/v1_swizzle kernels/v2_ldmatrix kernels/v3_p_regs kernels/v4_cpasync kernels/v5_tuned)

# Round 0: the straightforward port next to the Triton kernel and SDPA.
$NSYS "${NSYS_ARGS[@]}" -o results/nsys/round0 \
    $PY compare_driver.py --N 4096 --causal 0 --iters 3 kernels/v0_naive $TRITON sdpa

# Every version, non-causal then causal, on one timeline.
$NSYS "${NSYS_ARGS[@]}" -o results/nsys/all_versions \
    $PY compare_driver.py --N 4096 --causal 0 1 --iters 2 "${CUTE[@]}" $TRITON sdpa

# One launch of every version under Nsight Compute, one report per mask.
$NCU "${NCU_ARGS[@]}" -o results/ncu/rounds_noncausal \
    $PY compare_driver.py --N 4096 --causal 0 --iters 1 "${CUTE[@]}" $TRITON sdpa
$NCU "${NCU_ARGS[@]}" -o results/ncu/rounds_causal \
    $PY compare_driver.py --N 4096 --causal 1 --iters 1 "${CUTE[@]}" $TRITON sdpa

# Host overhead at the smallest shape: the timeline shows the CPU side of each call.
$NSYS "${NSYS_ARGS[@]}" -o results/nsys/n512 \
    $PY compare_driver.py --N 512 --causal 0 --iters 20 kernels/v5_tuned $TRITON sdpa

# The tables.
$PY ncu_table.py results/ncu/rounds_noncausal.ncu-rep > results/ncu/rounds_noncausal.txt
$PY ncu_table.py results/ncu/rounds_causal.ncu-rep > results/ncu/rounds_causal.txt
$NSYS stats --report cuda_gpu_kern_sum --format csv -o results/nsys/all_versions results/nsys/all_versions.nsys-rep > /dev/null
$NSYS stats --report nvtx_sum,cuda_api_sum --format csv -o results/nsys/n512 results/nsys/n512.nsys-rep > /dev/null

# SASS counts, every version, both masks (the configuration each version picks at N = 4096).
for v in "${CUTE[@]}"; do
  for c in 0 1; do $PY sass_counts.py "$v" --causal $c; done
done > results/sass_counts.txt 2>&1

# The Triton kernel with the configuration its autotuner picks in the harness at N = 4096
# (64 x 128 tiles, 2 stages), pinned with the Triton article's own driver.
for c in 0 1; do
  $NCU "${NCU_ARGS[@]}" -o results/ncu/triton_v5_64x128x4x2_causal$c \
      $PY ../2026_10_03_followup/profile_driver.py $TRITON --N 4096 --causal $c --config 64,128,4,2 --iters 1 --sdpa 0
done
$PY ncu_table.py results/ncu/triton_v5_64x128x4x2_causal0.ncu-rep results/ncu/triton_v5_64x128x4x2_causal1.ncu-rep > results/ncu/triton_v5_pinned.txt
