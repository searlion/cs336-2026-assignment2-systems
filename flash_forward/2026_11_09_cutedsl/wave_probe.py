# wave_probe.py
# How much of the gap to the tensor-core peak is the wave tail? At N = 4096, D = 64, B*H = 32 gives
# 2048 blocks of 64 query rows: 12.19 waves for CuTe v5 non-causal (3 blocks per SM x 56 SMs = 168
# per wave), 18.29 for Triton v5 at 64 x 128 (2 per SM). B*H = 21 gives 1344 blocks: exactly 8 and
# 12 waves, so no partly filled last wave. Same per-block work, so any rise in efficiency from 32 to
# 21 is the tail.
#
#   python wave_probe.py ncu   [kernel dirs...]  (one launch of each, in NVTX ranges, for ncu)
#   python wave_probe.py time  [kernel dirs...]  (interleaved do_bench, 7 rounds, TFLOP/s)
import statistics
import sys
import time

import torch
import triton

from kload import load

mode = sys.argv[1]
dirs = sys.argv[2:] or ["kernels/v5_tuned", "../2026_10_03_followup/kernels/v5_final"]
N, D = 4096, 64
kernels = {}
for d in dirs:
    # Triton v5 with the configuration its autotuner picks at N = 4096 (64 x 128, 2 stages),
    # pinned so the probe does not autotune per B*H.
    fa = load(d, (64, 128, 4, 2) if "2026_10_03" in d else None)
    kernels[d.rstrip("/").split("/")[-1]] = fa
data = {BH: [torch.randn(BH, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3)] for BH in (21, 32)}
for fa in kernels.values():
    for BH in (21, 32):
        for c in (False, True):
            fa.apply(*data[BH], c)
torch.cuda.synchronize()
t0 = time.perf_counter()
while time.perf_counter() - t0 < 1.0:
    for fa in kernels.values():
        fa.apply(*data[32], False)
    torch.cuda.synchronize()

if mode == "ncu":
    torch.cuda.cudart().cudaProfilerStart()
    for name, fa in kernels.items():
        for c in (False, True):
            for BH in (21, 32):
                torch.cuda.nvtx.range_push(f"{name} BH={BH} causal={c}")
                fa.apply(*data[BH], c)
                torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
else:
    for c in (False, True):
        for name, fa in kernels.items():
            t = {21: [], 32: []}
            for _ in range(7):
                for BH in (21, 32):
                    t[BH].append(triton.testing.do_bench(lambda: fa.apply(*data[BH], c), return_mode="median"))
            fl = {BH: 4 * BH * N * N * D / (2 if c else 1) for BH in (21, 32)}
            tf = {BH: fl[BH] / statistics.median(t[BH]) * 1e-9 for BH in (21, 32)}
            print(f"{name:10s} causal={c!s:5} BH=21 {statistics.median(t[21]):.4f} ms {tf[21]:.1f} TFLOP/s   "
                  f"BH=32 {statistics.median(t[32]):.4f} ms {tf[32]:.1f} TFLOP/s   "
                  f"BH=32 throughput / BH=21 = {tf[32] / tf[21]:.3f}", flush=True)
