# rank_configs.py
# Rank sweep configurations by how close they get to the best configuration on every
# benchmark shape (mean of tflops / best_tflops_for_that_shape), and list the per-shape winners.
#
#   python rank_configs.py results/sweep_v1.csv results/autotune_candidates.json
import collections
import csv
import json
import sys

rows = [r for r in csv.DictReader(open(sys.argv[1]))]
shapes = sorted({(int(r["N"]), r["causal"] == "True") for r in rows})
best = {}
score = collections.defaultdict(dict)
for r in rows:
    if r["status"] != "ok":
        continue
    shape = (int(r["N"]), r["causal"] == "True")
    cfg = (int(r["q_tile"]), int(r["k_tile"]), int(r["num_warps"]), int(r["num_stages"]))
    score[cfg][shape] = float(r["tflops"])
    if shape not in best or float(r["tflops"]) > best[shape][1]:
        best[shape] = (cfg, float(r["tflops"]))

ranked = sorted(score, key=lambda c: -sum(score[c].get(s, 0) / best[s][1] for s in shapes) / len(shapes))
print("per-shape winners:")
for s in shapes:
    print(f"  N={s[0]:5d} causal={s[1]!s:5}  {best[s][0]}  {best[s][1]:.1f} TFLOP/s")
print("top configs by mean fraction of the per-shape best:")
for c in ranked[:16]:
    frac = [score[c].get(s, 0) / best[s][1] for s in shapes]
    print(f"  {c}  mean {sum(frac) / len(frac):.3f}  worst {min(frac):.3f}")
json.dump([list(c) for c in ranked], open(sys.argv[2], "w"))
