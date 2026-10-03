# analyze.py
# Item 6: how often does each autotuner setting pick each candidate, what does a pick cost against
# the carefully timed best (ground_truth.csv), and are the autotuner's own timings biased by the
# position of a candidate in the list (the first ones are timed while the GPU clock is still
# ramping up in a fresh process)?
#   python analyze.py > summary.txt
import collections
import csv
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
truth = collections.defaultdict(list)
for r in csv.DictReader(open(HERE / "results" / "ground_truth.csv")):
    cfg = f"({r['q_tile']}, {r['k_tile']}, {r['num_warps']}, {r['num_stages']})"
    truth[(int(r["N"]), r["causal"] == "True", cfg)].append(float(r["ms"]))
truth = {k: statistics.median(v) for k, v in truth.items()}

picks = []
for name in ("picks.jsonl", "picks_warm.jsonl", "picks_long_cold.jsonl"):
    p = HERE / "results" / name
    if p.exists():
        picks += [json.loads(line) for line in p.read_text().splitlines()]

print("== choices per setting, and the cost of each choice against the carefully timed best")
groups = collections.defaultdict(list)
for d in picks:
    groups[(d["setting"], d["N"], d["causal"])].append(d)
for (setting, N, causal), ds in groups.items():
    best = min(v for (n, c, _), v in truth.items() if n == N and c == causal)
    counts = collections.Counter(d["chosen"] for d in ds)
    costs = [100 * (truth[(N, causal, d["chosen"])] / best - 1) for d in ds]
    print(f"{setting:22s} N={N:5d} causal={causal!s:5s} n={len(ds):2d}  "
          + ", ".join(f"{c} x{n} ({100 * (truth[(N, causal, c)] / best - 1):+.1f}%)" for c, n in counts.most_common())
          + f"   mean cost {statistics.mean(costs):+.1f}%, worst {max(costs):+.1f}%")

print("\n== the autotuner's own timing of each candidate, against the careful timing (median over processes)")
for (setting, N, causal), ds in groups.items():
    timed = [d for d in ds if d["bench_time_s"] == d["bench_time_s"]]  # skip runs that read the cache
    if not timed:
        continue
    order = list(timed[0]["timings_ms"])
    parts = []
    for pos, cfg in enumerate(order):
        ratio = statistics.median(d["timings_ms"][cfg] for d in timed) / truth[(N, causal, cfg)]
        parts.append(f"#{pos + 1} {cfg} {100 * (ratio - 1):+.1f}%")
    print(f"{setting:22s} N={N:5d} causal={causal!s:5s}  " + "; ".join(parts))
