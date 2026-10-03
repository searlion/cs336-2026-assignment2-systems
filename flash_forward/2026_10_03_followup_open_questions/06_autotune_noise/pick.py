# pick.py
# One fresh process: let a kernel version's autotuner choose a config for one shape, and print
# the choice and the autotuner's own timings as one JSON line.
#   PYTHONPATH=<kernel dir> python pick.py --N 4096 --causal 1
import argparse
import json

import torch
from flashattention_autograd_function_triton import FlashAttentionTriton, flash_fwd_kernel

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=4096)
parser.add_argument("--causal", type=int, default=1)
parser.add_argument("--warm", type=float, default=0.0, help="seconds of GPU work (matmuls) before tuning")
args = parser.parse_args()

q, k, v = (torch.randn(32, args.N, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
if args.warm:
    # Bring the GPU clock up before the autotuner starts timing, to test whether the candidates
    # timed first are penalised by a clock that is still ramping up.
    import time
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.warm:
        a @ a
        torch.cuda.synchronize()
FlashAttentionTriton.apply(q, k, v, bool(args.causal))
torch.cuda.synchronize()


def name(c):
    return f"({c.kwargs['Q_TILE_SIZE']}, {c.kwargs['K_TILE_SIZE']}, {c.num_warps}, {c.num_stages})"


timings = getattr(flash_fwd_kernel, "configs_timings", None) or {}
print(json.dumps(dict(
    N=args.N, causal=bool(args.causal), warm_s=args.warm, chosen=name(flash_fwd_kernel.best_config),
    timings_ms={name(c): t[0] if isinstance(t, (list, tuple)) else t for c, t in timings.items()},
    bench_time_s=round(getattr(flash_fwd_kernel, "bench_time", float("nan")), 3),
)))
