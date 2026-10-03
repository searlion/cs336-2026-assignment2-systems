# analyze.py
# Item 11: is the best of v5's five configurations within 1% of the best contender on every shape?
# Per shape: (a) using each configuration's median over the 5 rounds, (b) round by round. Also how
# Round 1's single winner (64, 32, 4, 3) does, and which contender is fastest.
#   python analyze.py retime.csv > summary.txt
import collections
import csv
import statistics
import sys

SHORTLIST = {(64, 32, 4, 3), (64, 32, 4, 4), (64, 32, 4, 5), (64, 128, 4, 2), (64, 32, 4, 2)}
rows = list(csv.DictReader(open(sys.argv[1])))
t = collections.defaultdict(lambda: collections.defaultdict(dict))  # shape -> cfg -> round -> ms
for r in rows:
    cfg = tuple(int(r[k]) for k in ("q_tile", "k_tile", "num_warps", "num_stages"))
    t[(int(r["N"]), r["causal"] == "True")][cfg][int(r["round"])] = float(r["ms"])
print(f"{len({c for s in t.values() for c in s})} contenders; rounds per config: "
      f"{sorted({len(v) for s in t.values() for v in s.values()})}")
print(f"{'shape':22s} {'best contender (median)':28s} {'best of list':22s} {'list/best':>9s} "
      f"{'per round: list/best':>26s} {'64x32x4x3/best':>14s}")
worst = 0
for (N, causal), cfgs in sorted(t.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    med = {c: statistics.median(v.values()) for c, v in cfgs.items()}
    best = min(med, key=med.get)
    best_list = min((c for c in med if c in SHORTLIST), key=med.get)
    ratio = med[best_list] / med[best]
    rounds = sorted({r for v in cfgs.values() for r in v})
    per_round = [min(cfgs[c][r] for c in cfgs if c in SHORTLIST and r in cfgs[c]) /
                 min(cfgs[c][r] for c in cfgs if r in cfgs[c]) for r in rounds]
    worst = max(worst, ratio)
    print(f"N={N:5d} causal={causal!s:5s}  {str(best):18s} {med[best]:.4f}  {str(best_list):14s} {med[best_list]:.4f} "
          f"{100 * (ratio - 1):+8.1f}%  {', '.join(f'{100 * (x - 1):+.1f}' for x in per_round):>26s} "
          f"{100 * (med[(64, 32, 4, 3)] / med[best] - 1):+13.1f}%")
print(f"worst shape: the list's best is {100 * (worst - 1):.1f}% slower than the best contender")
