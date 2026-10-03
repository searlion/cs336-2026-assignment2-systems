# analyze.py
# Items 3 and 10 (and the article's "what each round was worth" table): the unchanged harness,
# 10 interleaved runs per version instead of 3. Prints, per version and shape, the median and
# the range over runs; the ratio to SDPA measured in the same process (paired per run); the
# ratio between consecutive versions paired by run index; and every autotuner choice.
# Times are recovered from the TFLOP/s columns, which have more significant digits than ms.
#   python analyze.py runs > summary.txt
import collections
import pathlib
import re
import statistics
import sys

VERSIONS = ["v0_baseline", "v1_tiles", "v2_autotune", "v3_causal_skip", "v4_exp2", "v5_final"]
ROW = re.compile(r"^\s*(\d+)\s+(True|False)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+%$")
KEY = re.compile(r"with key as \((\d+), \d+, \d+, (True|False),")
BEST = re.compile(r"best config selected: Q_TILE_SIZE: (\d+), K_TILE_SIZE: (\d+), num_warps: (\d+), "
                  r"num_ctas: \d+, num_stages: (\d+)")

runs = pathlib.Path(sys.argv[1])
data = collections.defaultdict(dict)    # (version, N, causal) -> {run: (ours, sdpa)}
choices = collections.defaultdict(collections.Counter)  # (version, N, causal) -> Counter(config)
for path in sorted(runs.glob("*_run*.txt")):
    version, r = path.stem.rsplit("_run", 1)
    key = None
    for line in path.read_text().splitlines():
        if m := ROW.match(line):
            # The harness prints ms with 3 decimals (a 2% step at N = 512) but TFLOP/s with one
            # (about 0.2%), so recover the time from the TFLOP/s columns.
            n, c = int(m[1]), m[2] == "True"
            flops = 4 * 32 * n * n * 64 / (2 if c else 1)
            ours, sdpa = flops / float(m[5]) * 1e-9, flops / float(m[6]) * 1e-9
            data[(version, n, c)][int(r)] = (ours, sdpa)
        elif m := KEY.search(line):
            key = (int(m[1]), m[2] == "True")
        elif (m := BEST.search(line)) and key:
            choices[(version, *key)][tuple(int(x) for x in m.groups())] += 1
shapes = sorted({(n, c) for _, n, c in data}, key=lambda s: (s[1], s[0]))
n_runs = {v: len({r for (vv, _, _), d in data.items() if vv == v for r in d}) for v in VERSIONS}
print("runs per version:", n_runs)


def fmt_range(xs, f="{:.4f}"):
    return f"{f.format(statistics.median(xs))} [{f.format(min(xs))}-{f.format(max(xs))}]"


print("\n== ms: median [min-max] over runs, and spread (max/min - 1)")
for n, c in shapes:
    print(f"N={n:5d} causal={c!s:5s}")
    for v in VERSIONS:
        xs = [o for o, _ in data[(v, n, c)].values()]
        if xs:
            print(f"   {v:15s} {fmt_range(xs)}  spread {100 * (max(xs) / min(xs) - 1):4.1f}%")
    sd = [s for v in VERSIONS for s in (x[1] for x in data[(v, n, c)].values())]
    print(f"   {'sdpa (all runs)':15s} {fmt_range(sd)}  spread {100 * (max(sd) / min(sd) - 1):4.1f}%  (n={len(sd)})")

print("\n== % of SDPA, paired within each run (100 * sdpa_ms / ours_ms): median [min-max], runs faster than SDPA")
for n, c in shapes:
    parts = []
    for v in ("v1_tiles", "v2_autotune", "v3_causal_skip", "v4_exp2", "v5_final"):
        xs = [100 * s / o for o, s in data[(v, n, c)].values()]
        faster = sum(x > 100 for x in xs)
        parts.append(f"{v[:2]} {fmt_range(xs, '{:.0f}')} ({faster}/{len(xs)})")
    print(f"N={n:5d} causal={c!s:5s}  " + "  ".join(parts))

print("\n== consecutive versions, paired by run index: median [min-max] of new/old - 1 (negative = faster)")
pairs = [("v0_baseline", "v1_tiles"), ("v1_tiles", "v2_autotune"), ("v1_tiles", "v3_causal_skip"),
         ("v3_causal_skip", "v4_exp2"), ("v4_exp2", "v5_final")]
for n, c in shapes:
    parts = []
    for a, b in pairs:
        common = sorted(set(data[(a, n, c)]) & set(data[(b, n, c)]))
        xs = [100 * (data[(b, n, c)][r][0] / data[(a, n, c)][r][0] - 1) for r in common]
        faster = sum(x < 0 for x in xs)
        parts.append(f"{a[:2]}->{b[:2]} {fmt_range(xs, '{:+.1f}')}% ({faster}/{len(xs)} faster)")
    print(f"N={n:5d} causal={c!s:5s}  " + "  ".join(parts))

print("\n== autotuner choices (Q_TILE, K_TILE, num_warps, num_stages): count over runs")
for v in ("v2_autotune", "v5_final"):
    for n, c in shapes:
        ch = choices[(v, n, c)]
        print(f"{v:12s} N={n:5d} causal={c!s:5s} " + ", ".join(f"{k}: {cnt}" for k, cnt in ch.most_common()))
