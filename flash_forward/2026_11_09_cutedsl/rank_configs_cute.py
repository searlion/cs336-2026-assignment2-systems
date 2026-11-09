# rank_configs_cute.py
# One do_bench per configuration and shape is noisy. Re-time, for every harness shape, the
# sweep's 4 best configurations plus v4's (64, 64, 4, 1), interleaved over 5 rounds, and
# report the median of the rounds.
#
#   python rank_configs_cute.py kernels/v5_tuned results/sweep_v5.csv > results/rank_v5.txt
import collections
import csv
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, sys.argv[1])
from flashattention_autograd_function_triton import FlashAttentionCuTe  # noqa: E402

B, H, D = 4, 8, 64
rows = list(csv.DictReader(open(sys.argv[2])))
by_shape = collections.defaultdict(list)
for r in rows:
    cfg = (int(r["bM"]), int(r["bN"]), int(r["warps"]), int(r["stages"]))
    by_shape[(int(r["N"]), int(r["causal"]))].append((float(r["tflops"]), cfg))

t0 = time.perf_counter()
x = torch.randn(B * H, 4096, D, device="cuda", dtype=torch.bfloat16)
while time.perf_counter() - t0 < 2.0:
    F.scaled_dot_product_attention(x, x, x)
    torch.cuda.synchronize()

print("median of 5 interleaved rounds, TFLOP/s (sweep's single measurement in brackets)")
for (N, causal), cands in sorted(by_shape.items(), key=lambda s: (s[0][1], s[0][0])):
    cands.sort(reverse=True)
    pick = [c for _, c in cands[:4]]
    if (64, 64, 4, 1) not in pick:
        pick.append((64, 64, 4, 1))
    q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    flops = 4 * B * H * N * N * D / (2 if causal else 1)
    times = collections.defaultdict(list)
    for _ in range(5):
        for cfg in pick:
            FlashAttentionCuTe.CONFIG = cfg
            times[cfg].append(triton.testing.do_bench(
                lambda: FlashAttentionCuTe.apply(q, k, v, bool(causal)), return_mode="median"))
    sweep = {c: t for t, c in cands}
    med = sorted((statistics.median(t), c) for c, t in times.items())
    best = med[0][0]
    print(f"N={N:5d} causal={causal}: " + ";  ".join(
        f"{c} {flops / m * 1e-9:5.1f} [{sweep[c]:5.1f}] {100 * best / m:5.1f}%" for m, c in med))
