# plot_results.py
# TFLOP/s of every version at N = 4096 (median of the three harness runs), against SDPA.
#   python plot_results.py images/results_N4096.png
import collections
import glob
import pathlib
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
tf = collections.defaultdict(list)
sdpa = collections.defaultdict(list)
for path in glob.glob("results/final/*_run*.txt"):
    version = pathlib.Path(path).stem.rsplit("_run", 1)[0]
    for line in pathlib.Path(path).read_text().splitlines()[1:]:
        n, causal, _, _, tf_ours, tf_sdpa, _ = line.split()
        if n == "4096":
            key = "causal" if causal == "True" else "non-causal"
            tf[(version, key)].append(float(tf_ours))
            sdpa[key].append(float(tf_sdpa))

fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=200)
width = 0.36
for i, (v, _) in enumerate(VERSIONS):
    for j, key in enumerate(("non-causal", "causal")):
        y = statistics.median(tf[(v, key)])
        x = i + (j - 0.5) * (width + 0.02)
        ax.bar(x, y, width=width, color=SERIES[key], label=key if i == 0 else None, zorder=2)
        ax.text(x, y + 0.8, f"{y:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
for key, color in SERIES.items():
    ref = statistics.median(sdpa[key])
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
ax.set_title("N = 4096, B = 4, H = 8, D = 64, bf16. Median of three runs. Lines: SDPA (FlashAttention-2) on the same inputs.",
             loc="left", fontsize=9.5, color=INK2, pad=10)
fig.tight_layout()
fig.savefig(sys.argv[1])
