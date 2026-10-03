# reverse_order.py
# Item 13: does starting the longest causal query tiles first help? The article measured it once
# (1.079 ms against 1.072 ms on v4). Here: v4 against v4 with the query-tile order reversed (the
# article's variant), and against a "longest first" variant that swaps the grid axes so heads vary
# fastest and reverses the query tiles, so that every head's longest tiles start first. Three tile
# shapes, causal, 10 rounds with the order shuffled in each round.
#   python reverse_order.py results/reverse_order.csv
import csv
import math
import pathlib
import random
import sys

import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import KERNELS, launch, load_version  # noqa: E402

src = (KERNELS / "v4_exp2" / "flashattention_autograd_function_triton.py").read_text()
old = "    query_tile_index = tl.program_id(0)\n"
assert src.count(old) == 1
d = HERE / "variants" / "v4_reversed"
d.mkdir(parents=True, exist_ok=True)
(d / "flashattention_autograd_function_triton.py").write_text(src.replace(
    old, "    # Start the longest causal rows first so the last wave is short.\n"
         "    query_tile_index = tl.num_programs(0) - 1 - tl.program_id(0)\n"))
old_ids = "    query_tile_index = tl.program_id(0)\n    batch_index = tl.program_id(1)\n"
assert src.count(old_ids) == 1
d2 = HERE / "variants" / "v4_longest_first"
d2.mkdir(parents=True, exist_ok=True)
(d2 / "flashattention_autograd_function_triton.py").write_text(src.replace(
    old_ids, "    # Longest causal rows first across all heads: the grid is (heads, query tiles), heads vary\n"
             "    # fastest, and the query tiles run from the last (longest) to the first.\n"
             "    batch_index = tl.program_id(0)\n"
             "    query_tile_index = tl.num_programs(1) - 1 - tl.program_id(1)\n"))
mods = {"v4": load_version("v4_exp2"), "v4_reversed": load_version("v4_reversed", d / "flashattention_autograd_function_triton.py"),
        "v4_longest_first": load_version("v4_longest_first", d2 / "flashattention_autograd_function_triton.py")}


def run(name, q, k, v, cfg):
    if name != "v4_longest_first":
        return launch(mods[name], q, k, v, True, cfg)[0]
    bq, bk, nw, ns = cfg
    o = torch.empty_like(q)
    L = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    mods[name].flash_fwd_kernel[(q.shape[0], triton.cdiv(q.shape[1], bq))](
        q, k, v, o, L, *q.stride(), *k.stride(), *v.stride(), *o.stride(), *L.stride(),
        q.shape[1], k.shape[1], 1.0 / math.sqrt(q.shape[2]), D=q.shape[2], Q_TILE_SIZE=bq, K_TILE_SIZE=bk,
        is_causal=True, num_warps=nw, num_stages=ns)
    return o


CONFIGS = [(64, 32, 4, 3), (64, 64, 4, 2), (64, 128, 4, 2)]
B, H, D = 4, 8, 64
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["round", "N", "kernel", "q_tile", "k_tile", "num_warps", "num_stages", "ms"])
    for N in (4096, 2048):
        torch.manual_seed(0)
        q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        jobs = [(name, cfg) for name in mods for cfg in CONFIGS]
        for name, cfg in jobs:
            o = run(name, q, k, v, cfg)
            assert torch.allclose(o, ref, atol=2e-2, rtol=2e-2), (name, cfg)
        for r in range(10):
            random.Random(r).shuffle(jobs)
            for name, cfg in jobs:
                ms = triton.testing.do_bench(lambda: run(name, q, k, v, cfg), return_mode="median")
                w.writerow([r, N, name, *cfg, f"{ms:.4f}"])
            fh.flush()
        print(f"N={N} done", flush=True)
