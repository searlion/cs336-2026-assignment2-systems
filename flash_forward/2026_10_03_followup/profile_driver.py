# profile_driver.py
# Runs one kernel version at one benchmark shape a few times, inside NVTX ranges,
# so that nsys and ncu have something small and predictable to record.
#
#   python profile_driver.py kernels/v0_baseline --N 4096 --causal 0
#   python profile_driver.py kernels/v1_tiles --N 4096 --config 128,64,4,3
#   python profile_driver.py kernels/v5_final --N 4096 --config 64,128,4,2
import argparse
import sys
import time

import torch
import torch.nn.functional as F
import triton

parser = argparse.ArgumentParser()
parser.add_argument("kernel_dir")
parser.add_argument("--N", type=int, default=4096)
parser.add_argument("--causal", type=int, default=0)
parser.add_argument("--iters", type=int, default=3)
parser.add_argument("--sdpa", type=int, default=1, help="also run SDPA, for comparison")
parser.add_argument("--config", default=None, help="Q_TILE,K_TILE,num_warps,num_stages")
args = parser.parse_args()

sys.path.insert(0, args.kernel_dir)
import flashattention_autograd_function_triton as kernel_module  # noqa: E402
from flashattention_autograd_function_triton import FlashAttentionTriton  # noqa: E402

if args.config:
    bq, bk, nw, ns = (int(x) for x in args.config.split(","))
    if hasattr(kernel_module.flash_fwd_kernel, "configs"):
        # An autotuned version: an autotuner with a single configuration uses it without timing.
        kernel_module.flash_fwd_kernel.configs = [
            triton.Config({"Q_TILE_SIZE": bq, "K_TILE_SIZE": bk}, num_warps=nw, num_stages=ns)]
    else:
        FlashAttentionTriton.Q_TILE_SIZE, FlashAttentionTriton.K_TILE_SIZE = bq, bk
        FlashAttentionTriton.NUM_WARPS, FlashAttentionTriton.NUM_STAGES = nw, ns

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
