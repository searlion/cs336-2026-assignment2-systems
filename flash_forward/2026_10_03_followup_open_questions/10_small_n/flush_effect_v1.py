# flush_effect_v1.py
# The same three timings as flush_effect.py, at N = 512 only, for Round 1's v1 configuration
# (64, 32, 4, 3) next to v4's (64, 32, 4, 4) and SDPA, so Round 1's claim about SDPA and the
# flush has its own measurement.
#   python flush_effect_v1.py > flush_effect_v1.txt
#
# (The rest of this header is flush_effect.py's.)
# Item 10: SDPA's kernel takes 0.051 ms at N = 512 under Nsight Compute but 0.065 ms in the
# harness, while ours takes 0.045 ms in both. triton.testing.do_bench zeroes a 256 MB buffer
# (to flush L2) right before every timed call. This times both kernels three ways:
#   A. do_bench as the harness does (flush enqueued immediately before each call);
#   B. no flush: back-to-back calls, CUDA events around each;
#   C. flush, wait for it to finish (synchronize), then time the call.
#   python flush_effect.py > flush_effect.txt
import pathlib
import statistics
import sys

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import launch, load_version  # noqa: E402

B, H, D = 4, 8, 64
v4 = load_version("v4_exp2")
v1 = load_version("v1_tiles")
cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")


def events(fn, n=2000, flush=False, wait=False):
    times = []
    for _ in range(n):
        if flush:
            cache.zero_()
            if wait:
                torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        times.append((s, e))
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in times)


for N in (512,):
    for causal in (False, True):
        q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        q4, k4, v4_ = (t.view(B, H, N, D) for t in (q, k, v))
        fns = {"v1 (64, 32, 4, 3)": lambda: launch(v1, q, k, v, causal, (64, 32, 4, 3)),
               "v4 (64, 32, 4, 4)": lambda: launch(v4, q, k, v, causal, (64, 32, 4, 4)),
               "sdpa": lambda: F.scaled_dot_product_attention(q4, k4, v4_, is_causal=causal)}
        n = 2000 if N == 512 else 200
        for name, fn in fns.items():
            fn()
            torch.cuda.synchronize()
            a = triton.testing.do_bench(fn, return_mode="median")
            b = events(fn, n)
            c = events(fn, n, flush=True, wait=True)
            print(f"N={N:5d} causal={causal!s:5s} {name:22s} A do_bench {a:.4f} ms   B no flush {b:.4f} ms   "
                  f"C flush, wait, then time {c:.4f} ms", flush=True)
