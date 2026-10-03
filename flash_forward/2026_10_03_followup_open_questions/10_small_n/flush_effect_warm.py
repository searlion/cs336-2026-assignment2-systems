# flush_effect_warm.py
# Item 10, done properly: how much of the lead over SDPA at N = 512 comes from do_bench's cache
# flush? flush_effect.py and flush_effect_v1.py time "without the flush" with CUDA events around
# back-to-back calls, but at N = 512 a call through FlashAttentionTriton.apply costs about as much
# CPU time (~40 us) as the kernel takes on the GPU, so the GPU waits for the CPU between calls and
# those timings include launch overhead. This script instead uses
#   A  triton.testing.do_bench, as the harness does (zeroes a 256 MB buffer before every call);
#   G  triton.testing.do_bench_cudagraph (calls replayed from a CUDA graph: no flush, no CPU gaps),
# on a warm GPU (2 s of work first), with the order of the jobs shuffled in each of 7 rounds.
# Our kernel is called exactly as the harness calls it (FlashAttentionTriton.apply), with the
# autotuner of v5 left a single configuration: v5's usual choice at N = 512, (64, 32, 4, 4), and
# Round 1's (64, 32, 4, 3).
#   flock /tmp/claude-gpu.lock python flush_effect_warm.py > flush_effect_warm.txt
import pathlib
import random
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import load_version  # noqa: E402

B, H, D, N = 4, 8, 64, 512
CONFIGS = {"(64, 32, 4, 4)": (64, 32, 4, 4), "(64, 32, 4, 3)": (64, 32, 4, 3)}
versions = {}
for name, (bq, bk, nw, ns) in CONFIGS.items():
    mod = load_version("v5_final")  # a fresh module (and autotuner) per configuration
    mod.flash_fwd_kernel.configs = [triton.Config({"Q_TILE_SIZE": bq, "K_TILE_SIZE": bk}, num_warps=nw, num_stages=ns)]
    versions[name] = mod.FlashAttentionTriton

jobs = {}
for causal in (False, True):
    q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    q4, k4, v4 = (t.view(B, H, N, D) for t in (q, k, v))
    for name, FA in versions.items():
        jobs[(causal, f"ours {name}")] = lambda FA=FA, q=q, k=k, v=v, c=causal: FA.apply(q, k, v, c)
    jobs[(causal, "sdpa")] = lambda q4=q4, k4=k4, v4=v4, c=causal: F.scaled_dot_product_attention(q4, k4, v4, is_causal=c)
for fn in jobs.values():
    fn()
torch.cuda.synchronize()

big = [torch.randn(B, H, 4096, D, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
t0 = time.perf_counter()
while time.perf_counter() - t0 < 2.0:
    F.scaled_dot_product_attention(*big)
    torch.cuda.synchronize()

res = {(key, m): [] for key in jobs for m in "AG"}
for r in range(7):
    order = list(jobs)
    random.Random(r).shuffle(order)
    for key in order:
        res[(key, "A")].append(triton.testing.do_bench(jobs[key], return_mode="median"))
        res[(key, "G")].append(triton.testing.do_bench_cudagraph(jobs[key], rep=50, return_mode="median"))

print("N = 512, B = 4, H = 8, D = 64, bf16; medians over 7 rounds (ms); lead = median over rounds of sdpa/ours - 1")
for causal in (False, True):
    print(f"causal = {causal}")
    for m, label in (("A", "do_bench (flush before every call)"), ("G", "do_bench_cudagraph (no flush)")):
        sd = res[((causal, "sdpa"), m)]
        line = f"  {label:36s} sdpa {statistics.median(sd):.4f}"
        for name in CONFIGS:
            ours = res[((causal, f"ours {name}"), m)]
            lead = statistics.median(s / o - 1 for s, o in zip(sd, ours))
            line += f"   ours {name} {statistics.median(ours):.4f} (lead {100 * lead:+.1f}%)"
        print(line)
    for name in CONFIGS:
        a, g = (statistics.median(res[((causal, f"ours {name}"), m)]) for m in "AG")
        print(f"  flush costs ours {name} {100 * (a / g - 1):+.0f}%", end="")
    a, g = (statistics.median(res[((causal, "sdpa"), m)]) for m in "AG")
    print(f"; flush costs sdpa {100 * (a / g - 1):+.0f}%")
