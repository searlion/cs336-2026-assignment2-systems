# plot_cold_capture.py
# Item 13: the GPU clock next to the kernels in two short Nsight Systems captures of the Round 0
# driver's work (the baseline kernel and SDPA, alternating, N = 4096), taken with
# --gpu-metrics-devices=0 by cold_capture.py: one straight after 3 s idle, one after 1 s of
# warm-up. Writes the figure the article uses in Round 0.
#   python plot_cold_capture.py reports/cold_capture_warmup0.sqlite reports/cold_capture_warmup1.sqlite \
#       ../../2026_10_03_followup/images/nsys_clock_cold_warm.png
import sqlite3
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OURS, SDPA = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "Noto Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def load(path):
    con = sqlite3.connect(path)
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    cid = [m for m, n in con.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS") if "GPC Clock" in n][0]
    # The clock is stored in Hz in a signed 32-bit field: add 2^32 to negative values.
    clock = sorted((t, v + 2 ** 32 if v < 0 else v) for t, v in con.execute(
        f"SELECT timestamp, value FROM GPU_METRICS WHERE metricId={cid}"))
    kernels = [(s, e, "pytorch_flash" in names[d]) for s, e, d in con.execute(
        "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    t0, t1 = kernels[0][0], kernels[-1][1]
    clock = [((t - t0) / 1e6, v / 1e9) for t, v in clock if t0 - 1e6 <= t <= t1 + 1e6]
    kernels = [((s - t0) / 1e6, (e - t0) / 1e6, is_sdpa) for s, e, is_sdpa in kernels]
    return clock, kernels


panels = [(sys.argv[1], "Straight after 3 s idle (no warm-up)"), (sys.argv[2], "After 1 s of warm-up, as in the profiling driver")]
fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6), dpi=200, sharex=True)
for ax, (path, title) in zip(axes, panels):
    clock, kernels = load(path)
    for s, e, is_sdpa in kernels:
        ax.axvspan(s, e, ymin=0, ymax=1, color=SDPA if is_sdpa else OURS, alpha=0.16 if is_sdpa else 0.07, lw=0, zorder=0)
        if is_sdpa:
            ax.text((s + e) / 2, 2.775, f"{e - s:.2f}", ha="center", va="top", fontsize=9, color=INK2)
    ax.plot([t for t, _ in clock], [v for _, v in clock], color=INK, lw=1.2, zorder=2)
    ax.set_ylim(2.40, 2.80)
    ax.set_yticks([2.4, 2.5, 2.6, 2.7, 2.8])
    ax.set_ylabel("GPC clock (GHz)")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=6)
    ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
axes[-1].set_xlabel("Time from the first kernel in the capture (ms)")
handles = [plt.Rectangle((0, 0), 1, 1, color=OURS, alpha=0.2), plt.Rectangle((0, 0), 1, 1, color=SDPA, alpha=0.3),
           plt.Line2D([0], [0], color=INK, lw=1.2)]
leg = axes[0].legend(handles, ["baseline kernel (v0)", "SDPA, labelled with its time in ms", "GPC clock"],
                     frameon=False, fontsize=9, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.02))
for t in leg.get_texts():
    t.set_color(INK2)
fig.suptitle("The GPU clock under two short captures", x=0.02, ha="left", fontsize=13, fontweight="bold")
fig.text(0.02, 0.928, "Nsight Systems GPU metrics (--gpu-metrics-devices=0), sampled at 20 kHz. "
         "The baseline kernel and SDPA alternate, N = 4096, non-causal.", fontsize=9.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.935))
fig.savefig(sys.argv[3])
