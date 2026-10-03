# summarize_final.py
# Median of the three harness runs per version and shape (results/final/<version>_run<k>.txt).
import collections
import glob
import pathlib
import statistics

VERSIONS = ["v0_baseline", "v1_tiles", "v2_autotune", "v3_causal_skip", "v4_exp2", "v5_final"]
data = collections.defaultdict(list)
for path in sorted(glob.glob("results/final/*_run*.txt")):
    version = pathlib.Path(path).stem.rsplit("_run", 1)[0]
    for line in pathlib.Path(path).read_text().splitlines()[1:]:
        n, causal, ours, sdpa, tf_ours, tf_sdpa, pct = line.split()
        data[(version, int(n), causal == "True")].append(
            (float(ours), float(sdpa), float(tf_ours), float(tf_sdpa), float(pct.rstrip("%"))))

shapes = sorted({(n, c) for _, n, c in data}, key=lambda s: (s[1], s[0]))
print("median of 3 runs: ours ms / TFLOP/s / % of SDPA (SDPA ms)")
for n, c in shapes:
    print(f"N={n:5d} causal={c!s:5}  " + "  ".join(
        f"{v[:2]}: {statistics.median(x[0] for x in data[(v, n, c)]):.3f} ms {statistics.median(x[2] for x in data[(v, n, c)]):5.1f} "
        f"{statistics.median(x[4] for x in data[(v, n, c)]):4.0f}%" for v in VERSIONS)
          + f"  sdpa {statistics.median(x[1] for v in VERSIONS for x in data[(v, n, c)]):.3f} ms "
            f"{statistics.median(x[3] for v in VERSIONS for x in data[(v, n, c)]):.1f}")
print("spread of ours ms across the 3 runs (max/min - 1):")
for v in VERSIONS:
    spreads = [max(x[0] for x in data[(v, n, c)]) / min(x[0] for x in data[(v, n, c)]) - 1 for n, c in shapes]
    print(f"  {v}: up to {100 * max(spreads):.1f}%")
