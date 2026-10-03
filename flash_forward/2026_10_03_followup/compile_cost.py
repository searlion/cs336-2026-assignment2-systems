# compile_cost.py
# What does @triton.autotune cost on the first call? Run in a fresh process, with
# TRITON_CACHE_DIR pointing at an empty directory for a cold start, or at the
# directory a previous run filled for a warm start.
#
#   TRITON_CACHE_DIR=$(mktemp -d) python compile_cost.py kernels/v1_tiles --configs 8
import argparse
import json
import math
import sys
import time

import torch
import triton

parser = argparse.ArgumentParser()
parser.add_argument("kernel_dir")
parser.add_argument("--configs", type=int, default=8, help="autotune over the top-K sweep configs")
parser.add_argument("--candidates", default="results/autotune_candidates.json")
parser.add_argument("--shapes", default="4096:0,2048:0,4096:1", help="N:causal, in call order")
parser.add_argument("--cache-results", type=int, default=0)
args = parser.parse_args()

t_start = time.perf_counter()
sys.path.insert(0, args.kernel_dir)
import flashattention_autograd_function_triton as mod  # noqa: E402

kernel = mod.flash_fwd_kernel
if hasattr(kernel, "configs"):  # already autotuned: take the plain JIT function underneath
    kernel = kernel.fn
ranked = json.load(open(args.candidates))
configs = [triton.Config({"Q_TILE_SIZE": c[0], "K_TILE_SIZE": c[1]}, num_warps=c[2], num_stages=c[3])
           for c in ranked[: args.configs]]
tuned = triton.autotune(configs=configs, key=["N_QUERIES", "N_KEYS", "D", "is_causal"],
                        cache_results=bool(args.cache_results))(kernel)

B, H, D = 4, 8, 64


def call(q, k, v, causal):
    o = torch.empty_like(q)
    L = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    N = q.shape[1]
    grid = lambda meta: (triton.cdiv(N, meta["Q_TILE_SIZE"]), q.shape[0])  # noqa: E731
    tuned[grid](q, k, v, o, L,
                *q.stride(), *k.stride(), *v.stride(), *o.stride(), *L.stride(),
                N, N, 1.0 / math.sqrt(D), D=D, is_causal=causal)


torch.cuda.init()
t_ready = time.perf_counter()
out = dict(configs=args.configs, cache_results=args.cache_results,
           import_and_init_s=round(t_ready - t_start, 3), calls=[])
for spec in args.shapes.split(","):
    N, causal = int(spec.split(":")[0]), bool(int(spec.split(":")[1]))
    q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    call(q, k, v, causal)
    torch.cuda.synchronize()
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    call(q, k, v, causal)
    torch.cuda.synchronize()
    second = time.perf_counter() - t0
    out["calls"].append(dict(N=N, causal=causal, first_call_s=round(first, 3),
                             second_call_ms=round(second * 1e3, 3), best=str(tuned.best_config)))
print(json.dumps(out))
