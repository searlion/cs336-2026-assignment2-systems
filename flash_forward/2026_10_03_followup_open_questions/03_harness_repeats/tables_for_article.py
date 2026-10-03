# tables_for_article.py
# The article's harness tables, recomputed from the 10-run batch: medians of 10 interleaved runs
# (time recovered from the TFLOP/s columns), TFLOP/s from the median time, and % of SDPA as the
# median of the per-run paired ratios. Prints Markdown ready to paste.
#   python tables_for_article.py runs > tables_for_article.md
import collections
import pathlib
import re
import statistics
import sys

ROW = re.compile(r"^\s*(\d+)\s+(True|False)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+%$")
V = ["v0_baseline", "v1_tiles", "v2_autotune", "v3_causal_skip", "v4_exp2", "v5_final"]
runs = pathlib.Path(sys.argv[1])
ours = collections.defaultdict(dict)
sdpa = collections.defaultdict(dict)
for p in runs.glob("*_run*.txt"):
    v, r = p.stem.rsplit("_run", 1)
    for line in p.read_text().splitlines():
        if m := ROW.match(line):
            n, c = int(m[1]), m[2] == "True"
            fl = 4 * 32 * n * n * 64 / (2 if c else 1)
            ours[(v, n, c)][int(r)] = fl / float(m[5]) * 1e-9
            sdpa[(v, n, c)][int(r)] = fl / float(m[6]) * 1e-9


def ms(v, n, c):
    return statistics.median(ours[(v, n, c)].values())


def tf(v, n, c):
    return 4 * 32 * n * n * 64 / (2 if c else 1) / ms(v, n, c) * 1e-9


def pct(v, n, c):
    return statistics.median(100 * sdpa[(v, n, c)][r] / ours[(v, n, c)][r] for r in ours[(v, n, c)])


def sdpa_ms(n, c):
    return statistics.median(x for v in V for x in sdpa[(v, n, c)].values())


def sdpa_tf(n, c):
    return 4 * 32 * n * n * 64 / (2 if c else 1) / sdpa_ms(n, c) * 1e-9


SHAPES = [(n, False) for n in (512, 1024, 2048, 4096)] + [(n, True) for n in (512, 1024, 2048, 4096)]
yes = lambda c: "yes" if c else "no"  # noqa: E731

print("### Round 3 table (v1 against v2)\n")
print("| N | causal | v1, fixed (ms) | v2, autotuned (ms) | v2 TFLOP/s | v2 % of SDPA |\n|---|---|---|---|---|---|")
for n, c in SHAPES:
    print(f"| {n} | {yes(c)} | {ms('v1_tiles', n, c):.3f} | {ms('v2_autotune', n, c):.3f} | {tf('v2_autotune', n, c):.1f} | {pct('v2_autotune', n, c):.0f}% |")

print("\n### Results table (every version)\n")
print("| N | causal | v0 baseline | v1 tiles | v2 autotune | v3 causal skip | v4 exp2 | v5 final | SDPA |\n|---|---|---|---|---|---|---|---|---|")
for n, c in SHAPES:
    cells = []
    for v in V:
        cell = f"{ms(v, n, c):.3f}"
        if n == 4096:
            cell += f" ({tf(v, n, c):.1f}, {pct(v, n, c):.0f}%)"
        cells.append(cell)
    s = f"{sdpa_ms(n, c):.3f}" + (f" ({sdpa_tf(n, c):.1f})" if n == 4096 else "")
    print(f"| {n} | {yes(c)} | " + " | ".join(cells) + f" | {s} |")

print("\n### What each round was worth, N = 4096 (ratio of 10-run medians; runs in which the newer version was faster)\n")
pairs = [("v0_baseline", "v1_tiles"), ("v1_tiles", "v2_autotune"), ("v1_tiles", "v3_causal_skip"),
         ("v3_causal_skip", "v4_exp2"), ("v4_exp2", "v5_final")]
for a, b in pairs:
    parts = []
    for c in (False, True):
        ratio = ms(a, 4096, c) / ms(b, 4096, c)
        faster = sum(ours[(b, 4096, c)][r] < ours[(a, 4096, c)][r] for r in ours[(b, 4096, c)])
        txt = f"{ratio:.2f}x" if ratio > 1.5 else f"{100 * (ratio - 1):+.1f}%"
        parts.append(f"{'causal' if c else 'non-causal'} {txt} ({faster}/10 runs faster)")
    print(f"- {a} -> {b}: " + ", ".join(parts))

print("\n### Final kernel against the baseline (Round 5 table)\n")
print("| N | causal | baseline (ms) | final (ms) | final TFLOP/s | SDPA TFLOP/s | final % of SDPA |\n|---|---|---|---|---|---|---|")
for n, c in SHAPES:
    print(f"| {n} | {yes(c)} | {ms('v0_baseline', n, c):.3f} | {ms('v5_final', n, c):.3f} | {tf('v5_final', n, c):.1f} | "
          f"{sdpa_tf(n, c):.1f} | {pct('v5_final', n, c):.0f}% |")
