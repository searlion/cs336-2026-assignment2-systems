#!/usr/bin/env bash
# profile_all.sh: every Nsight Systems and Nsight Compute capture used in the article.
#   PY=../../.venv/bin/python ./profile_all.sh
set -euo pipefail
NSYS=/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
NCU=/opt/nvidia/nsight-compute/2026.2.1/ncu
PY=${PY:-python}
# A fresh cache, so the line information baked into each kernel points at these files.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$(mktemp -d)}
mkdir -p results/nsys results/ncu
NSYS_ARGS=(profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true)
NCU_ARGS=(--set full --import-source yes --profile-from-start off -f)
ALL=(kernels/v0_baseline kernels/v1_tiles kernels/v3_causal_skip kernels/v4_exp2 kernels/v5_final sdpa)

# Round 0: the baseline next to SDPA.
$NSYS "${NSYS_ARGS[@]}" -o results/nsys/round0 $PY profile_driver.py kernels/v0_baseline --N 4096 --causal 0

# Round 0's counters: the baseline on its own, and SDPA on its own for the warp-state comparison.
$NCU "${NCU_ARGS[@]}" -o results/ncu/v0_N4096_nc $PY profile_driver.py kernels/v0_baseline --N 4096 --causal 0 --iters 1 --sdpa 0
$NCU "${NCU_ARGS[@]}" --nvtx --nvtx-include "sdpa N=4096 causal=False/" -o results/ncu/sdpa_N4096_nc \
    $PY profile_driver.py kernels/v0_baseline --N 4096 --causal 0 --iters 1

# Every version except v2 (whose kernel is v1's), non-causal then causal, on one timeline.
$NSYS "${NSYS_ARGS[@]}" -o results/nsys/all_versions \
    $PY compare_driver.py --N 4096 --causal 0 1 --iters 2 "${ALL[@]}"

# One launch of every version under Nsight Compute, one report per mask.
$NCU "${NCU_ARGS[@]}" -o results/ncu/rounds_noncausal $PY compare_driver.py --N 4096 --causal 0 --iters 1 "${ALL[@]}"
$NCU "${NCU_ARGS[@]}" -o results/ncu/rounds_causal $PY compare_driver.py --N 4096 --causal 1 --iters 1 "${ALL[@]:1}"

# Too many warps for the tile: 64x32 with 8 warps instead of 4.
$NCU --set full -f -o results/ncu/v1_8warps --nvtx --nvtx-include "ours N=4096 causal=False/" --launch-count 1 \
    $PY profile_driver.py kernels/v1_tiles --config 64,32,8,3 --iters 1 --sdpa 0

# The final profile: the final kernel with the configurations the harness's autotuner picks at
# N = 4096 (64x128 tiles with 2 stages for both masks; 64x32 with 4 stages is its other choice
# for causal), next to SDPA.
$NCU "${NCU_ARGS[@]}" -o results/ncu/final_noncausal \
    $PY profile_driver.py kernels/v5_final --N 4096 --causal 0 --config 64,128,4,2 --iters 1
$NCU "${NCU_ARGS[@]}" -o results/ncu/final_causal \
    $PY profile_driver.py kernels/v5_final --N 4096 --causal 1 --config 64,128,4,2 --iters 1
$NCU "${NCU_ARGS[@]}" -o results/ncu/final_causal_64x32x4x4 \
    $PY profile_driver.py kernels/v5_final --N 4096 --causal 1 --config 64,32,4,4 --iters 1 --sdpa 0
