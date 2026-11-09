# profile_driver_cute.py
# A copy of the Triton article's profile_driver.py for the CuTe versions: --config sets
# FlashAttentionCuTe.CONFIG instead of a Triton config.
#
#   python profile_driver_cute.py kernels/v0_naive --N 4096 --causal 0
#   python profile_driver_cute.py kernels/v5_tuned --N 4096 --config 64,128,4,2
import argparse
import sys
import time

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("kernel_dir")
parser.add_argument("--N", type=int, default=4096)
parser.add_argument("--causal", type=int, default=0)
parser.add_argument("--iters", type=int, default=3)
parser.add_argument("--sdpa", type=int, default=1, help="also run SDPA, for comparison")
parser.add_argument("--config", default=None, help="bM,bN,warps[,stages]")
args = parser.parse_args()

sys.path.insert(0, args.kernel_dir)
from flashattention_autograd_function_triton import FlashAttentionTriton  # noqa: E402

if args.config:
    # bM, bN, warps[, stages]: the same tuple FlashAttentionCuTe.config returns.
    FlashAttentionTriton.CONFIG = tuple(int(x) for x in args.config.split(","))

B, H, D = 4, 8, 64  # the harness shape
causal = bool(args.causal)
q, k, v = (torch.randn(B * H, args.N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
q4, k4, v4 = (t.view(B, H, args.N, D) for t in (q, k, v))

# Warm-up: compile (and autotune, if the version autotunes) outside the ranges, then keep
# the GPU busy for about a second so its clocks have ramped up before the capture starts.
t0 = time.perf_counter()
while time.perf_counter() - t0 < 1.0:
    FlashAttentionTriton.apply(q, k, v, causal)
    F.scaled_dot_product_attention(q4, k4, v4, is_causal=causal)
    torch.cuda.synchronize()

# nsys --capture-range=cudaProfilerApi records only what runs between start and stop,
# so the timeline is not dominated by Python start-up and Triton compilation.
torch.cuda.cudart().cudaProfilerStart()
for _ in range(args.iters):
    torch.cuda.nvtx.range_push(f"ours N={args.N} causal={causal}")
    FlashAttentionTriton.apply(q, k, v, causal)
    torch.cuda.nvtx.range_pop()
    if args.sdpa:
        torch.cuda.nvtx.range_push(f"sdpa N={args.N} causal={causal}")
        F.scaled_dot_product_attention(q4, k4, v4, is_causal=causal)
        torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
