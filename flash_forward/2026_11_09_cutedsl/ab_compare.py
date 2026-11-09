# ab_compare.py: interleaved A/B timing of two kernel dirs (do_bench median, the order alternating
# every round), median over the rounds, at B*H = 32, D = 64.
#   python ab_compare.py kernels/v5_tuned experiments/q_in_registers 64,64,4,1     (as in round 5:
#                                                                 one config forced on both)
#   python ab_compare.py kernels/v5_tuned kernels/v6_mbarrier --cfgB 64,64,4,2,1 --rounds 7 --repeat 2
# A config of "auto" (the default) uses each version's own table; a Triton dir takes
# bq,bk,warps,stages and is pinned to it.
import argparse
import statistics
import time

import torch
import triton

from kload import load_module

p = argparse.ArgumentParser()
p.add_argument("A")
p.add_argument("B")
p.add_argument("config", nargs="?", default=None, help="one config for both (round-5 usage)")
p.add_argument("--cfgA", default="auto")
p.add_argument("--cfgB", default="auto")
p.add_argument("--rounds", type=int, default=7)
p.add_argument("--repeat", type=int, default=1, help="repeat the whole comparison")
p.add_argument("--shapes", default="512:0,4096:0,4096:1", help="N:causal,...")
args = p.parse_args()
if args.config:
    args.cfgA = args.cfgB = args.config


def load(d, cfg):
    m = load_module(d)
    if cfg != "auto":
        t = tuple(int(x) for x in cfg.split(","))
        if hasattr(m, "flash_fwd_kernel"):
            m.flash_fwd_kernel.configs = [triton.Config({"Q_TILE_SIZE": t[0], "K_TILE_SIZE": t[1]},
                                                        num_warps=t[2], num_stages=t[3])]
        else:
            m.FlashAttentionCuTe.CONFIG = t
    return m.FlashAttentionTriton


A, B = load(args.A, args.cfgA), load(args.B, args.cfgB)
print(f"A = {args.A} [{args.cfgA}]   B = {args.B} [{args.cfgB}]   {args.rounds} rounds")
for rep in range(args.repeat):
    for item in args.shapes.split(","):
        N, c = int(item.split(":")[0]), bool(int(item.split(":")[1]))
        q, k, v = (torch.randn(32, N, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        A.apply(q, k, v, c), B.apply(q, k, v, c)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1:
            A.apply(q, k, v, c); B.apply(q, k, v, c); torch.cuda.synchronize()
        ta, tb = [], []
        for r in range(args.rounds):
            pair = [(ta, A), (tb, B)] if r % 2 == 0 else [(tb, B), (ta, A)]
            for lst, fn in pair:
                lst.append(triton.testing.do_bench(lambda: fn.apply(q, k, v, c), return_mode="median"))
        a, b = statistics.median(ta), statistics.median(tb)
        print(f"[{rep}] N={N} causal={c!s:5}: A {a:.4f} ms  B {b:.4f} ms  B/A-1 = {100 * (b / a - 1):+.1f}%"
              f"   (A {min(ta):.4f}-{max(ta):.4f}, B {min(tb):.4f}-{max(tb):.4f})", flush=True)
