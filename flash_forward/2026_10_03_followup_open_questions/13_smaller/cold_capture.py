# cold_capture.py
# Item 13: was SDPA's 3.4 ms in the first Round 0 capture the GPU clock still ramping up?
# The same work as profile_driver.py (v0 and SDPA, alternating, N = 4096), captured once from a
# cold start (compile, then 3 s idle, then straight into the capture) and once after the
# driver's 1-second warm-up, under `nsys --gpu-metrics-devices=0`, which samples the GPU's
# clock frequencies alongside the kernels. --precall-sdpa 0 leaves SDPA's very first call in the
# process for inside the capture, the other difference from the article's first capture.
#   nsys profile --trace=cuda,nvtx --gpu-metrics-devices=0 --capture-range=cudaProfilerApi ... \
#       python cold_capture.py --warmup 0
import argparse
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import load_version  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--warmup", type=float, default=0.0, help="seconds of warm-up before the capture")
parser.add_argument("--iters", type=int, default=6)
parser.add_argument("--precall-sdpa", type=int, default=1,
                    help="0: SDPA's first call in the process happens inside the capture")
args = parser.parse_args()

FA = load_version("v0_baseline").FlashAttentionTriton
B, H, N, D = 4, 8, 4096, 64
q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
q4, k4, v4 = (t.view(B, H, N, D) for t in (q, k, v))
FA.apply(q, k, v, False)  # compile
if args.precall_sdpa:
    F.scaled_dot_product_attention(q4, k4, v4)
torch.cuda.synchronize()
time.sleep(3.0)  # let the GPU drop back to its idle clocks
t0 = time.perf_counter()
while time.perf_counter() - t0 < args.warmup:
    FA.apply(q, k, v, False)
    if args.precall_sdpa:
        F.scaled_dot_product_attention(q4, k4, v4)
    torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(args.iters):
    torch.cuda.nvtx.range_push("ours")
    FA.apply(q, k, v, False)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("sdpa")
    F.scaled_dot_product_attention(q4, k4, v4)
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
