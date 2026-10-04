# ramp_capture.py
# The GPC clock as Nsight Systems measures it (10 kHz) over the first LOAD seconds of work after
# idle: the two 50-ms windows of nsys_clock_cold_warm.png, joined up. Compiles, waits until the
# GPU has sat idle in P8 for 5 s (checked through NVML queries that leave it in P8), then starts
# the capture and runs the cold/warm capture's work: v0 and SDPA alternating, N = 4096, non-causal.
#   nsys profile --trace=cuda,nvtx --gpu-metrics-devices=0 --gpu-metrics-frequency=10000 \
#       --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite \
#       -o reports/ramp_capture -f true python ramp_capture.py
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

LOAD = 2.5
FA = load_version("v0_baseline").FlashAttentionTriton
B, H, N, D = 4, 8, 4096, 64
q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
q4, k4, v4 = (t.view(B, H, N, D) for t in (q, k, v))
FA.apply(q, k, v, False)
F.scaled_dot_product_attention(q4, k4, v4)
torch.cuda.synchronize()
wait_idle(Device(), 5.0)
torch.cuda.cudart().cudaProfilerStart()
t0 = time.monotonic()
while time.monotonic() - t0 < LOAD:
    torch.cuda.nvtx.range_push("ours")
    FA.apply(q, k, v, False)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("sdpa")
    F.scaled_dot_product_attention(q4, k4, v4)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
