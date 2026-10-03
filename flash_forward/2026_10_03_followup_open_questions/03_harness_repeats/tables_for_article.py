# tables_for_article.py
# The article's harness tables, recomputed from the 10-run batch: medians of 10 interleaved runs,
# TFLOP/s from the median time, and % of SDPA as the median of the per-run paired ratios. Prints
# Markdown ready to paste.
# The harness prints each time twice, rounded: in ms to 3 decimals and as TFLOP/s to 1 decimal.
# Each time is taken from whichever of the two is the more precise for that number (TFLOP/s for
# the short kernels, ms for the long ones).
# A version's own table (Round 1's v1, Round 5's final kernel) shows SDPA as measured in that
# version's runs, so that its "% of SDPA" agrees with its two TFLOP/s columns; the tables that
# compare all versions (the baseline table, Results) pool SDPA over all sixty runs.
#   python tables_for_article.py runs > tables_for_article.md
import collections
import pathlib
import re
import statistics
import sys

ROW = re.compile(r"^\s*(\d+)\s+(True|False)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+%$")
V = ["v0_baseline", "v1_tiles", "v2_autotune", "v3_causal_skip", "v4_exp2", "v5_final"]
runs = pathlib.Path(sys.argv[1])


def best_time(flop, ms_text, tf_text):
    """The time in ms, from the printed ms (3 decimals) or TFLOP/s (1 decimal), whichever is finer."""
    ms_col, tf_col = float(ms_text), float(tf_text)
    return ms_col if 0.0005 / ms_col < 0.05 / tf_col else flop / tf_col * 1e-9


ours = collections.defaultdict(dict)
sdpa = collections.defaultdict(dict)
for p in runs.glob("*_run*.txt"):
    v, r = p.stem.rsplit("_run", 1)
    for line in p.read_text().splitlines():
        if m := ROW.match(line):
            n, c = int(m[1]), m[2] == "True"
            fl = 4 * 32 * n * n * 64 / (2 if c else 1)
            ours[(v, n, c)][int(r)] = best_time(fl, m[3], m[5])
            sdpa[(v, n, c)][int(r)] = best_time(fl, m[4], m[6])


def ms(v, n, c):
    return statistics.median(ours[(v, n, c)].values())


def shown(t, flop):
    """The time as the tables show it: when 3 decimals of ms are finer than 1 decimal of TFLOP/s
    (the long kernels), round it first, so that the TFLOP/s next to it is what a reader gets by
    dividing."""
    return round(t, 3) if 0.0005 / t < 0.05 / (flop / t * 1e-9) else t


def tf(v, n, c):
    flop = 4 * 32 * n * n * 64 / (2 if c else 1)
    return flop / shown(ms(v, n, c), flop) * 1e-9


def pct(v, n, c):
    return statistics.median(100 * sdpa[(v, n, c)][r] / ours[(v, n, c)][r] for r in ours[(v, n, c)])


def sdpa_ms(n, c, versions=V):
    return statistics.median(x for v in versions for x in sdpa[(v, n, c)].values())


def sdpa_tf(n, c, versions=V):
    flop = 4 * 32 * n * n * 64 / (2 if c else 1)
    return flop / shown(sdpa_ms(n, c, versions), flop) * 1e-9


SHAPES = [(n, False) for n in (512, 1024, 2048, 4096)] + [(n, True) for n in (512, 1024, 2048, 4096)]
yes = lambda c: "yes" if c else "no"  # noqa: E731

print("### The baseline, re-measured (v0 and SDPA)\n")
print("| N | causal | today (ms) | today's SDPA (ms) |\n|---|---|---|---|")
for n, c in [(512, False), (1024, False), (2048, False), (4096, False), (4096, True)]:
    print(f"| {n} | {yes(c)} | {ms('v0_baseline', n, c):.3f} | {sdpa_ms(n, c):.3f} |")

print("\n### Round 1 table (v1 against the baseline)\n")
print("| N | causal | baseline (ms) | v1 (ms) | v1 TFLOP/s | SDPA TFLOP/s | v1 % of SDPA |\n|---|---|---|---|---|---|---|")
for n, c in SHAPES:
    print(f"| {n} | {yes(c)} | {ms('v0_baseline', n, c):.3f} | {ms('v1_tiles', n, c):.3f} | {tf('v1_tiles', n, c):.1f} | "
          f"{sdpa_tf(n, c, ['v1_tiles']):.1f} | {pct('v1_tiles', n, c):.0f}% |")
print()

print("### Round 3 table (v1 against v2)\n")
print("| N | causal | v1, fixed (ms) | v2, autotuned (ms) | v2 TFLOP/s | v2 % of SDPA |\n|---|---|---|---|---|---|")
for n, c in SHAPES:
    print(f"| {n} | {yes(c)} | {ms('v1_tiles', n, c):.3f} | {ms('v2_autotune', n, c):.3f} | {tf('v2_autotune', n, c):.1f} | {pct('v2_autotune', n, c):.0f}% |")

print("\n### Round 4 tables (tile-skipping: v1 against v3; exp2: v3 against v4)\n")
print("| N | causal | v1 (ms) | v3 (ms) | v3 TFLOP/s | v3 % of SDPA |\n|---|---|---|---|---|---|")
for n, c in [(512, True), (1024, True), (2048, True), (4096, True), (4096, False)]:
    print(f"| {n} | {yes(c)} | {ms('v1_tiles', n, c):.3f} | {ms('v3_causal_skip', n, c):.3f} | "
          f"{tf('v3_causal_skip', n, c):.1f} | {pct('v3_causal_skip', n, c):.0f}% |")
print("\n| N | causal | v3 (ms) | v4 (ms) | v4 TFLOP/s | v4 % of SDPA |\n|---|---|---|---|---|---|")
for n, c in [(1024, False), (4096, False), (1024, True), (4096, True)]:
    print(f"| {n} | {yes(c)} | {ms('v3_causal_skip', n, c):.3f} | {ms('v4_exp2', n, c):.3f} | "
          f"{tf('v4_exp2', n, c):.1f} | {pct('v4_exp2', n, c):.0f}% |")

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
          f"{sdpa_tf(n, c, ['v5_final']):.1f} | {pct('v5_final', n, c):.0f}% |")
