# retime_shortlist.py
# Item 11: is v5's five-config list really "within 1% of the best on every shape"? Re-time the
# list together with the top five configurations of every shape from the v4 sweep (the only
# realistic contenders for "best"), with the sweep's own method (FlashAttentionTriton.apply
# under do_bench, median), in 5 rounds with the order shuffled in each round.
#
#   python retime_shortlist.py results/retime.csv
import csv
import pathlib
import random
import sys

import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import FU, load_version  # noqa: E402

SHORTLIST = [(64, 32, 4, 3), (64, 32, 4, 4), (64, 32, 4, 5), (64, 128, 4, 2), (64, 32, 4, 2)]
B, H, D = 4, 8, 64
rows = [r for r in csv.DictReader(open(FU / "results" / "sweep_v4.csv")) if r["status"] == "ok"]
shapes = sorted({(int(r["N"]), r["causal"] == "True") for r in rows})
contenders = set(SHORTLIST)
for N, causal in shapes:
    best = sorted((r for r in rows if int(r["N"]) == N and (r["causal"] == "True") == causal), key=lambda r: float(r["ms"]))
    contenders |= {tuple(int(r[c]) for c in ("q_tile", "k_tile", "num_warps", "num_stages")) for r in best[:5]}
contenders = sorted(contenders)
print(f"{len(contenders)} configurations x {len(shapes)} shapes")

mod = load_version("v4_exp2")
FA = mod.FlashAttentionTriton


def run(q, k, v, causal, cfg):
    FA.Q_TILE_SIZE, FA.K_TILE_SIZE, FA.NUM_WARPS, FA.NUM_STAGES = cfg
    return FA.apply(q, k, v, causal)


with open(sys.argv[1], "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["round", "N", "causal", "q_tile", "k_tile", "num_warps", "num_stages", "in_shortlist", "ms"])
    for r in range(5):
        for N, causal in shapes:
            torch.manual_seed(0)
            q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
            order = contenders[:]
            random.Random(1000 * r + N + causal).shuffle(order)
            for cfg in order:
                run(q, k, v, causal, cfg)  # compile outside the timing
                ms = triton.testing.do_bench(lambda: run(q, k, v, causal, cfg), return_mode="median")
                w.writerow([r, N, causal, *cfg, cfg in SHORTLIST, f"{ms:.4f}"])
            fh.flush()
        print(f"round {r} done", flush=True)
