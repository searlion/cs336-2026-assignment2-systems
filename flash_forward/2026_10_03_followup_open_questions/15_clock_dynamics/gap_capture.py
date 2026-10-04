# gap_capture.py
# One Nsight Systems capture (GPC clock at 10 kHz) of three things, to explain the shape of
# nsys_clock_cold_warm.png:
#   cold    from idle (P8 for 5 s), 1.5 s of the cold/warm capture's work (v0 and SDPA
#           alternating, N = 4096, non-causal): how long the clock stays at its first value
#   gaps    then idle gaps of 5, 20, 50, 100, 200 and 500 ms, each followed by 0.3 s of the same
#           work: what the clock does while the GPU waits (as it does while nsys starts a
#           capture) and how quickly it comes back down once the work resumes
#   spin    from idle again, 1.5 s of torch.cuda._sleep (one thread spinning on one SM, little
#           power): where the clock goes when power is not the limit
# NVTX ranges mark each part.
#   nsys profile --trace=cuda,nvtx --gpu-metrics-devices=0 --gpu-metrics-frequency=10000 \
#       --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite \
#       -o reports/gap_capture -f true python gap_capture.py
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from clock_dynamics import wait_idle  # noqa: E402
from common import load_version  # noqa: E402
from nvml_ctypes import Device  # noqa: E402

GAPS_MS = (5, 20, 50, 100, 200, 500)
FA = load_version("v0_baseline").FlashAttentionTriton
B, H, N, D = 4, 8, 4096, 64
q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
q4, k4, v4 = (t.view(B, H, N, D) for t in (q, k, v))


def work(seconds):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        torch.cuda.nvtx.range_push("ours")
        FA.apply(q, k, v, False)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("sdpa")
        F.scaled_dot_product_attention(q4, k4, v4)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()


def spin(seconds):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        torch.cuda._sleep(2_000_000)
        torch.cuda.synchronize()


work(0.01)
spin(0.01)
dev = Device()
wait_idle(dev, 5.0)
torch.cuda.cudart().cudaProfilerStart()
torch.cuda.nvtx.range_push("cold")
work(1.5)
torch.cuda.nvtx.range_pop()
for gap in GAPS_MS:
    time.sleep(gap / 1000)
    torch.cuda.nvtx.range_push(f"after {gap} ms gap")
    work(0.3)
    torch.cuda.nvtx.range_pop()
wait_idle(dev, 5.0)
torch.cuda.nvtx.range_push("spin")
spin(1.5)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
