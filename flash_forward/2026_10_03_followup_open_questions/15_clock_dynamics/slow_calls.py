# slow_calls.py
# How often an SDPA call (N = 4096, v0 and SDPA alternating, as in the Round 0 capture) takes
# over 3 ms, with no profiler or NVML polling attached: CUDA events around each call for SECONDS
# seconds after a 2 s warm-up. The rate depends on what the desktop, which this GPU also drives,
# is doing at the time. Appends one line per run to slow_calls.txt.
#   flock /tmp/claude-gpu.lock python slow_calls.py
import datetime
import pathlib
import statistics
import sys
import time

import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_version  # noqa: E402

SECONDS = 15.0
FA = load_version("v0_baseline").FlashAttentionTriton
B, H, N, D = 4, 8, 4096, 64
q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
q4, k4, v4 = (t.view(B, H, N, D) for t in (q, k, v))
t0 = time.monotonic()
while time.monotonic() - t0 < 2.0:
    FA.apply(q, k, v, False)
    F.scaled_dot_product_attention(q4, k4, v4)
    torch.cuda.synchronize()
times = []
t0 = time.monotonic()
while time.monotonic() - t0 < SECONDS:
    FA.apply(q, k, v, False)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    F.scaled_dot_product_attention(q4, k4, v4)
    e.record()
    e.synchronize()
    times.append(s.elapsed_time(e))
slow = sorted(x for x in times if x > 3.0)
line = (f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {len(times)} SDPA calls, median {statistics.median(times):.3f} ms; "
        f"over 3 ms: {len(slow)}" + (f" (1 in {len(times) / len(slow):.0f}), {slow[0]:.2f}-{slow[-1]:.2f} ms" if slow else ""))
print(line)
with open(HERE / "slow_calls.txt", "a") as f:
    print(line, file=f)
