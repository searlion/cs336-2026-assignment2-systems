# plot_results.py
# TFLOP/s of every version at N = 4096 (median over the harness runs), against SDPA. Each time is
# taken from whichever of the harness's two printed columns is finer (ms to 3 decimals, or TFLOP/s
# to 1 decimal), the median is taken over times, and TFLOP/s is computed from that median, as in
# the article's tables (../2026_10_03_followup_open_questions/03_harness_repeats/tables_for_article.py).
#   python plot_results.py images/results_N4096.png [runs_dir]
# runs_dir defaults to results/final (the 3-run batch); the article uses the 10-run batch in
# ../2026_10_03_followup_open_questions/03_harness_repeats/runs.
import collections
import glob
import pathlib
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"non-causal": "#2a78d6", "causal": "#eb6834"}
plt.rcParams.update({"font.family": "Noto Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

VERSIONS = [("v0_baseline", "v0\nbaseline"), ("v1_tiles", "v1\ntiles"), ("v2_autotune", "v2\nautotune"),
            ("v3_causal_skip", "v3\ncausal skip"), ("v4_exp2", "v4\nexp2"), ("v5_final", "v5\nre-tuned")]
RUNS = sys.argv[2] if len(sys.argv) > 2 else "results/final"
ROW = re.compile(r"^\s*(\d+)\s+(True|False)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+%$")


def best_time(flop, ms_text, tf_text):
    """The time in ms, from the printed ms (3 decimals) or TFLOP/s (1 decimal), whichever is finer."""
    ms_col, tf_col = float(ms_text), float(tf_text)
    return ms_col if 0.0005 / ms_col < 0.05 / tf_col else flop / tf_col * 1e-9


ms = collections.defaultdict(list)
sdpa_ms = collections.defaultdict(list)
runs = set()
for path in glob.glob(f"{RUNS}/*_run*.txt"):
    version, run = pathlib.Path(path).stem.rsplit("_run", 1)
    runs.add(run)
    for line in pathlib.Path(path).read_text().splitlines():  # skips the autotuner's printouts
        m = ROW.match(line)
        if m and m[1] == "4096":
            key = "causal" if m[2] == "True" else "non-causal"
            flop = 4 * 32 * 4096 * 4096 * 64 / (2 if key == "causal" else 1)
            ms[(version, key)].append(best_time(flop, m[3], m[5]))
            sdpa_ms[key].append(best_time(flop, m[4], m[6]))


def tflops(times, key):
    return 4 * 32 * 4096 * 4096 * 64 / (2 if key == "causal" else 1) / statistics.median(times) * 1e-9


tf = {k: tflops(v, k[1]) for k, v in ms.items()}
sdpa = {k: tflops(v, k) for k, v in sdpa_ms.items()}
N_RUNS = {3: "three", 10: "ten"}.get(len(runs), str(len(runs)))

fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=200)
width = 0.36
for i, (v, _) in enumerate(VERSIONS):
    for j, key in enumerate(("non-causal", "causal")):
        y = tf[(v, key)]
        x = i + (j - 0.5) * (width + 0.02)
        ax.bar(x, y, width=width, color=SERIES[key], label=key if i == 0 else None, zorder=2)
        ax.text(x, y + 0.8, f"{y:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
for key, color in SERIES.items():
    ref = sdpa[key]
    ax.axhline(ref, color=color, lw=1, zorder=1)
    ax.text(len(VERSIONS) - 0.45, ref, f" SDPA {key}\n {ref:.1f}", va="center", fontsize=9, color=INK2)
ax.set_xticks(range(len(VERSIONS)), [label for _, label in VERSIONS])
ax.set_xlim(-0.6, len(VERSIONS) + 0.45)
ax.set_ylim(0, 80)
ax.set_ylabel("TFLOP/s (harness accounting)")
ax.grid(axis="y", color=GRID, lw=1, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(length=0)
leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
for t in leg.get_texts():
    t.set_color(INK2)
fig.suptitle("Every round, through the unchanged harness", x=0.02, ha="left", fontsize=13, fontweight="bold")
ax.set_title(f"N = 4096, B = 4, H = 8, D = 64, bf16. Median of {N_RUNS} runs. Lines: SDPA (FlashAttention-2) on the same inputs.",
             loc="left", fontsize=9.5, color=INK2, pad=10)
fig.tight_layout()
fig.savefig(sys.argv[1])
