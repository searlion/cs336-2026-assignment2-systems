# ground_truth.py
# Item 6: how far apart are v2's four candidates really? Each is timed with a 5x longer
# do_bench than the autotuner uses, in 10 interleaved rounds (shuffled order per round), at the
# shapes where the article saw the autotuner disagree with itself.
#   python ground_truth.py results/ground_truth.csv
import csv
import pathlib
import random
import sys

import torch
import triton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import launch, load_version  # noqa: E402

CANDIDATES = [(64, 32, 4, 3), (64, 32, 4, 4), (64, 32, 4, 2), (64, 16, 4, 5)]
SHAPES = [(4096, True), (4096, False), (512, False)]
mod = load_version("v2_autotune")
inputs = {N: [torch.randn(32, N, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)] for N in (512, 4096)}
jobs = [(N, causal, cfg) for N, causal in SHAPES for cfg in CANDIDATES]
for N, causal, cfg in jobs:
    launch(mod, *inputs[N], causal, cfg)
torch.cuda.synchronize()
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["round", "N", "causal", "q_tile", "k_tile", "num_warps", "num_stages", "ms"])
    for r in range(10):
        random.Random(r).shuffle(jobs)
        for N, causal, cfg in jobs:
            ms = triton.testing.do_bench(lambda: launch(mod, *inputs[N], causal, cfg),
                                         warmup=100, rep=500, return_mode="median")
            w.writerow([r, N, causal, *cfg, f"{ms:.4f}"])
        fh.flush()
        print(f"round {r} done", flush=True)
