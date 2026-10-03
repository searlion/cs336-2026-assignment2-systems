# plot_sweep.py
# Heatmaps from a sweep CSV written by sweep.py.
#
#   python plot_sweep.py results/sweep_v1.csv images/ v1
import collections
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CSV, OUTDIR, TAG = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]

# Sequential blue ramp (light -> dark) and chart chrome.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)
SURFACE, INK, INK2, MUTED, NODATA = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#f0efec"
plt.rcParams.update({
    "font.family": "Noto Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

TILES = [16, 32, 64, 128, 256]
rows = list(csv.DictReader(open(CSV)))
for r in rows:
    for k in ("N", "q_tile", "k_tile", "num_warps", "num_stages"):
        r[k] = int(r[k])
    r["causal"] = r["causal"] == "True"
    r["tflops"] = float(r["tflops"]) if r["status"] == "ok" else None


def ink_on(value, vmin, vmax):
    """White text on the dark half of the ramp, ink on the light half."""
    return "white" if (value - vmin) / (vmax - vmin) > 0.55 else INK


def best_per_tile(N, causal):
    best, why = {}, collections.defaultdict(collections.Counter)
    for r in rows:
        if r["N"] != N or r["causal"] != causal:
            continue
        key = (r["q_tile"], r["k_tile"])
        why[key][r["status"]] += 1
        if r["tflops"] is not None and (key not in best or r["tflops"] > best[key]["tflops"]):
            best[key] = r
    return best, why


def tile_heatmap(ax, N, causal, vmin, vmax, labels=("best", "baseline")):
    best, why = best_per_tile(N, causal)
    winner = max(best.values(), key=lambda r: r["tflops"])
    for i, bq in enumerate(TILES):
        for j, bk in enumerate(TILES):
            r = best.get((bq, bk))
            if r is None:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=NODATA, lw=0))
                reason = "no config\nfits smem" if why[(bq, bk)]["out_of_smem"] else "no config\nruns"
                ax.text(j, i, reason, ha="center", va="center", fontsize=8, color=MUTED)
                continue
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, lw=0,
                                       color=CMAP(max(0.0, (r["tflops"] - vmin) / (vmax - vmin)))))
            show = (("best" in labels and r is winner) or ("baseline" in labels and (bq, bk) == (16, 16))
                    or ("all" in labels))
            if show:
                txt = f"{r['tflops']:.1f}"
                if "all" not in labels and "value_only" not in labels:
                    w = "warp" if r["num_warps"] == 1 else "warps"
                    txt += f"\n{r['num_warps']} {w}, {r['num_stages']} st."
                ax.text(j, i, txt, ha="center", va="center", fontsize=10 if "all" not in labels else 9,
                        color=ink_on(r["tflops"], vmin, vmax),
                        fontweight="bold" if r is winner else "normal")
    # A surface-coloured grid separates the cells (the 2px "surface gap").
    for k in range(len(TILES) + 1):
        ax.axhline(k - 0.5, color=SURFACE, lw=2)
        ax.axvline(k - 0.5, color=SURFACE, lw=2)
    ax.set_xlim(-0.5, len(TILES) - 0.5)
    ax.set_ylim(len(TILES) - 0.5, -0.5)
    ax.set_xticks(range(len(TILES)), TILES)
    ax.set_yticks(range(len(TILES)), TILES)
    ax.tick_params(length=0)
    return winner


def colorbar(fig, ax, vmin, vmax, label="TFLOP/s"):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED)
    cb.set_label(label, color=INK2)


def fig_tiles(N, causal, path, title, subtitle, vmin=20):
    best, _ = best_per_tile(N, causal)
    vmax = max(r["tflops"] for r in best.values())
    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=200)
    winner = tile_heatmap(ax, N, causal, vmin, vmax)
    ax.set_xlabel("K_TILE_SIZE (keys per inner-loop step)")
    ax.set_ylabel("Q_TILE_SIZE (query rows per program)")
    colorbar(fig, ax, vmin, vmax)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK2, pad=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return winner


def fig_warps_stages(N, causal, bq, bk, path, title, subtitle, vmax=None):
    cells = {(r["num_warps"], r["num_stages"]): r for r in rows
             if r["N"] == N and r["causal"] == causal and r["q_tile"] == bq and r["k_tile"] == bk}
    warps, stages = [1, 2, 4, 8, 16], [1, 2, 3, 4, 5]
    ok = [r["tflops"] for r in cells.values() if r["tflops"] is not None]
    vmax = vmax or max(ok)
    top = max((r for r in cells.values() if r["tflops"] is not None), key=lambda r: r["tflops"])
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=200)
    for i, nw in enumerate(warps):
        for j, ns in enumerate(stages):
            r = cells.get((nw, ns))
            if r is None or r["tflops"] is None:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=NODATA, lw=0))
                label = {"out_of_smem": "smem", "skipped_regs": "regs", "wrong": "wrong"}.get(
                    r["status"] if r else "", "n/a")
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color=MUTED)
                continue
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, lw=0,
                                       color=CMAP(min(r["tflops"], vmax) / vmax)))
            ax.text(j, i, f"{r['tflops']:.1f}", ha="center", va="center", fontsize=9,
                    color=ink_on(r["tflops"], 0, vmax), fontweight="bold" if r is top else "normal")
    for k in range(6):
        ax.axhline(k - 0.5, color=SURFACE, lw=2)
        ax.axvline(k - 0.5, color=SURFACE, lw=2)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)
    ax.set_xticks(range(5), stages)
    ax.set_yticks(range(5), warps)
    ax.tick_params(length=0)
    ax.set_xlabel("num_stages")
    ax.set_ylabel("num_warps")
    colorbar(fig, ax, 0, vmax)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK2, pad=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_small_multiples(causal, path, title, subtitle):
    vmax = max(r["tflops"] for r in rows if r["causal"] == causal and r["tflops"] is not None)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), dpi=200)
    for ax, N in zip(axes, (512, 1024, 2048, 4096)):
        w = tile_heatmap(ax, N, causal, 0, vmax, labels=("best", "value_only"))
        ax.set_title(f"N = {N}: best {w['q_tile']}×{w['k_tile']}, {w['num_warps']} warps, "
                     f"{w['num_stages']} stages", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("K_TILE_SIZE")
    axes[0].set_ylabel("Q_TILE_SIZE")
    colorbar(fig, axes[-1], 0, vmax)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.89, subtitle, fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-5


def fig_warps_rule(N, causal, path, title, subtitle):
    """Best TFLOP/s (over K_TILE and num_stages) against num_warps, one line per Q_TILE."""
    warps = [1, 2, 4, 8, 16]
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
    for color, bq in zip(SERIES, TILES):
        ys = []
        for nw in warps:
            vals = [r["tflops"] for r in rows if r["N"] == N and r["causal"] == causal
                    and r["q_tile"] == bq and r["num_warps"] == nw and r["tflops"] is not None]
            ys.append(max(vals) if vals else None)
        xs = [i for i, y in enumerate(ys) if y is not None]
        ys_ok = [y for y in ys if y is not None]
        ax.plot(xs, ys_ok, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round",
                label=f"Q_TILE_SIZE = {bq}")
        k = max(range(len(ys_ok)), key=lambda i: ys_ok[i])
        ax.scatter([xs[k]], [ys_ok[k]], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_xticks(range(len(warps)), warps)
    ax.set_xlim(-0.2, len(warps) - 0.8)
    ax.set_ylim(0, None)
    ax.set_xlabel("num_warps")
    ax.set_ylabel("best TFLOP/s")
    ax.grid(axis="y", color="#e1e0d9", lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(length=0)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK2, pad=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    fig_warps_rule(4096, False, f"{OUTDIR}/warps_rule_{TAG}_N4096.png",
                   "Never more warps than 16-row slices of the Q tile",
                   "Best TFLOP/s over K_TILE_SIZE and num_stages; the dot marks each Q_TILE_SIZE's peak. "
                   "N = 4096, non-causal.")
    for causal, label in ((False, "non-causal"), (True, "causal")):
        w = fig_tiles(4096, causal, f"{OUTDIR}/heatmap_{TAG}_tiles_N4096_{label}.png",
                      f"Best TFLOP/s per tile shape, N = 4096, {label}",
                      "Each cell: best of the num_warps × num_stages grid. B=4, H=8, D=64, bf16, do_bench median.")
        print(label, "winner", w["q_tile"], w["k_tile"], w["num_warps"], w["num_stages"], w["tflops"])
        fig_warps_stages(4096, causal, w["q_tile"], w["k_tile"],
                         f"{OUTDIR}/heatmap_{TAG}_warps_stages_N4096_{label}.png",
                         f"{w['q_tile']}×{w['k_tile']} tiles: num_warps × num_stages",
                         f"TFLOP/s at N = 4096, {label}. Grey cells did not run (reason inside).")
        fig_small_multiples(causal, f"{OUTDIR}/heatmap_{TAG}_tiles_by_N_{label}.png",
                            f"Best tile shape at each sequence length ({label})",
                            "Best TFLOP/s per tile shape at each sequence length; one colour scale across panels.")
