# summarize_final.py
# Medians of the harness runs in results/final/ (<version>_run<k>.txt), per version and shape,
# plus SDPA (the median over every harness run) and NVIDIA's example. Writes a table to stdout
# and results/final/summary.csv for the article's tables and chart.
import collections
import csv
import glob
import pathlib
import re
import statistics

VERSIONS = ["v0_naive", "v1_swizzle", "v2_ldmatrix", "v3_p_regs", "v4_cpasync", "v5_tuned", "triton_v5"]
ROW = re.compile(r"^\s*(\d+)\s+(True|False)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)%\s*$")
NV = re.compile(r"^\s*(\d+)\s+(True|False)\s+64x64\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")

data = collections.defaultdict(list)       # (version, N, causal) -> [(ms, tflops)]
sdpa = collections.defaultdict(list)       # (N, causal) -> [(ms, tflops)]
runs = collections.Counter()
for path in sorted(glob.glob("results/final/*_run*.txt")):
    version = pathlib.Path(path).stem.rsplit("_run", 1)[0]
    runs[version] += 1
    for line in pathlib.Path(path).read_text().splitlines():
        if version == "nvidia_example":
            m = NV.match(line)
            if m:
                n, c, ms, _, tf, _ = m.groups()
                data[(version, int(n), c == "True")].append((float(ms), float(tf)))
            continue
        m = ROW.match(line)
        if m:
            n, c, ours, ms_sdpa, tf, tf_sdpa, _ = m.groups()
            data[(version, int(n), c == "True")].append((float(ours), float(tf)))
            sdpa[(int(n), c == "True")].append((float(ms_sdpa), float(tf_sdpa)))

shapes = sorted({(n, c) for _, n, c in data}, key=lambda s: (s[1], s[0]))
names = [v for v in VERSIONS + ["nvidia_example"] if runs[v]]
med = lambda xs, i: statistics.median(x[i] for x in xs)  # noqa: E731
with open("results/final/summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["version", "runs", "N", "causal", "ms", "tflops", "pct_of_sdpa"])
    for n, c in shapes:
        s_ms = med(sdpa[(n, c)], 0)
        for v in names:
            xs = data[(v, n, c)]
            w.writerow([v, len(xs), n, int(c), f"{med(xs, 0):.4f}", f"{med(xs, 1):.1f}", f"{100 * s_ms / med(xs, 0):.0f}"])
        w.writerow(["sdpa", len(sdpa[(n, c)]), n, int(c), f"{s_ms:.4f}", f"{med(sdpa[(n, c)], 1):.1f}", "100"])

print("median over runs: ms / TFLOP/s   (runs per version: " + ", ".join(f"{v} {runs[v]}" for v in names) + ")")
print(f"{'N':>5} {'causal':>6} " + " ".join(f"{v[:12]:>18}" for v in names) + f" {'sdpa':>18}")
for n, c in shapes:
    print(f"{n:>5} {str(c):>6} " + " ".join(
        f"{med(data[(v, n, c)], 0):8.3f} {med(data[(v, n, c)], 1):5.1f}    " for v in names)
        + f"{med(sdpa[(n, c)], 0):8.3f} {med(sdpa[(n, c)], 1):5.1f}")
print("spread of ms across runs (max/min - 1), worst shape:")
for v in names:
    print(f"  {v}: {100 * max(max(x[0] for x in data[(v, n, c)]) / min(x[0] for x in data[(v, n, c)]) - 1 for n, c in shapes):.1f}%")
