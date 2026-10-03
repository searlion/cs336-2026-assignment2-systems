# plot_compile_cost.py
#   python plot_compile_cost.py results/compile_cost.json images/autotune_cost.png
import json
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.family": "Noto Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

data = json.load(open(sys.argv[1]))
runs = [r for r in data["first_call"] if r["cache_results"] == 0]
ks = sorted({r["configs"] for r in runs})


def first(cache):
    return [next(r for r in runs if r["configs"] == k and r["cache"] == cache)["calls"][0]["first_call_s"] for k in ks]


def new_n():
    # A new sequence length re-runs the benchmarks but reuses the compiled kernels: the median of
    # the three new lengths (N = 2048, 1024 and 512), as in the article's table.
    return [statistics.median(c["first_call_s"] for c in next(r for r in runs if r["configs"] == k and r["cache"] == "cold")["calls"][1:4])
            for k in ks]


series = [("first call, empty Triton cache", first("cold")),
          ("first call, kernels already on disk", first("warm")),
          ("each new sequence length", new_n())]
fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=200)
for color, (name, ys) in zip(SERIES, series):
    ax.plot(ks, ys, color=color, lw=2, solid_capstyle="round", label=name)
    ax.scatter(ks, ys, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"{ys[-1]:.1f} s", (ks[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                va="center", fontsize=9.5, color=INK2)
ax.set_xticks(ks, [str(k) for k in ks])
ax.set_xlim(0, ks[-1] + 2.2)
ax.set_ylim(0, None)
ax.set_xlabel("configurations in @triton.autotune")
ax.set_ylabel("seconds before the call returns")
ax.grid(axis="y", color=GRID, lw=1)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(length=0)
leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
for t in leg.get_texts():
    t.set_color(INK2)
fig.suptitle("What autotuning costs on the first call", x=0.02, ha="left", fontsize=13, fontweight="bold")
ax.set_title("bf16, N = 4096, B·H = 32, D = 64. Each config costs ~0.2 s to compile and ~0.11 s to benchmark.",
             loc="left", fontsize=9.5, color=INK2, pad=10)
fig.tight_layout()
fig.savefig(sys.argv[2])
