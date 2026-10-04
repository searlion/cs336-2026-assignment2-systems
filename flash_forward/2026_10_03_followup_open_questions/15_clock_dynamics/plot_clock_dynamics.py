# plot_clock_dynamics.py
# Why nsys_clock_cold_warm.png looks the way it does, in three panels drawn from the Nsight
# Systems GPC-clock samples (10 kHz) of gap_capture.py and wake_capture.py:
#   A  from idle, the clock holds at 2.52 GHz until the GPU has been busy for a while, then goes
#      as high as the limits allow: 2.73 GHz for a low-power load, about 2.5 GHz under the 220 W
#      limit for the cold/warm capture's work (v0 and SDPA, N = 4096)
#   B  idle time does not count: woken and left idle for 1 s, the GPU still holds 2.52 GHz until
#      work keeps it busy; busy 10% of the time, it never leaves 2.52 GHz
#   C  during an idle gap the clock climbs to about 2.7 GHz; when the work resumes, the 220 W
#      limit pulls it back down within tens of ms (the "warm" capture's start)
#   python plot_clock_dynamics.py clock_dynamics.png
import bisect
import sqlite3
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OURS, SDPA, LIGHT, HEAVY = "#2a78d6", "#eb6834", "#1f9e89", "#0b0b0b"
plt.rcParams.update({"font.family": "Noto Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.unicode_minus": False})


class Capture:
    def __init__(self, path):
        con = sqlite3.connect(path)
        names = dict(con.execute("SELECT id, value FROM StringIds"))
        cid = [m for m, n in con.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS") if "GPC Clock" in n][0]
        # The clock is stored in Hz in a signed 32-bit field: add 2^32 to negative values.
        self.clock = sorted((t, (v + 2 ** 32 if v < 0 else v) / 1e9) for t, v in con.execute(
            f"SELECT timestamp, value FROM GPU_METRICS WHERE metricId={cid}"))
        self.ct = [t for t, _ in self.clock]
        self.kernels = sorted((s, e, "pytorch_flash" in names[d]) for s, e, d in con.execute(
            "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"))
        self.ranges = {txt: (s, e) for s, e, txt in con.execute(
            "SELECT start, end, text FROM NVTX_EVENTS WHERE text IS NOT NULL AND end IS NOT NULL")}

    def clock_between(self, a, b, t0, scale):
        seg = self.clock[bisect.bisect_left(self.ct, a):bisect.bisect_right(self.ct, b)]
        return [(t - t0) / scale for t, _ in seg], [v for _, v in seg]

    def kernels_in(self, a, b):
        return [k for k in self.kernels if a <= k[0] <= b]


def style(ax, title, xlabel):
    ax.set_ylim(2.40, 2.80)
    ax.set_yticks([2.4, 2.5, 2.6, 2.7, 2.8])
    ax.set_ylabel("GPC clock (GHz)")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=6)
    ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)


gap = Capture("reports/gap_capture.sqlite")
wake = Capture("reports/wake_capture.sqlite")
fig, axes = plt.subplots(3, 1, figsize=(9.2, 10.2), dpi=200)

# A: cold starts, heavy and light
ax = axes[0]
for rng, color, label, lx, ly in (("cold", HEAVY, "v0 + SDPA: held near 2.5 GHz by the 220 W power limit", 0.62, 2.425),
                                  ("spin", LIGHT, "low-power load (58 W): 2.73 GHz", 0.62, 2.755)):
    s, e = gap.ranges[rng]
    k0 = gap.kernels_in(s, e)[0][0]
    x, y = gap.clock_between(k0 - 20e6, e, k0, 1e9)
    ax.plot(x, y, color=color, lw=1.1, zorder=3)
    ax.text(lx, ly, label, fontsize=9, color=INK2, va="center")
ax.text(0.1, 2.62, "both start at\n2.52 GHz, the\ncard's rated\nboost clock", ha="center", va="center", fontsize=8.5, color=INK2)
ax.set_xlim(-0.02, 1.5)
style(ax, "A. From idle: 2.52 GHz until the GPU has been busy for about 0.13 to 0.3 s, then as high as the limits allow",
      "Time from the first kernel (s)")

# B: idle time does not count
ax = axes[1]
for rng, color, label, lx, ly in (("wake, idle 1.0 s, spin", LIGHT, "woken, idle for 1 s, then the low-power load: boosts 0.22 s after the work starts", 0.02, 2.765),
                                  ("5 ms in 50", HEAVY, "5 ms of the low-power load every 50 ms (busy 10% of the time): never boosts", 0.02, 2.455)):
    s, e = wake.ranges[rng]
    k0 = wake.kernels_in(s, e)[0][0]
    x, y = wake.clock_between(k0 - 20e6, e + 50e6, k0, 1e9)
    ax.plot(x, y, color=color, lw=1.1, zorder=3)
    ax.text(lx, ly, label, fontsize=9, color=INK2, va="center")
s, e = wake.ranges["wake, idle 1.0 s, spin"]
ks = wake.kernels_in(s, e)
first_spin = next(a for a, b, _ in ks if b - a > 1e6)
ax.axvspan(0, (first_spin - ks[0][0]) / 1e9, color=MUTED, alpha=0.10, lw=0, zorder=0)
ax.text((first_spin - ks[0][0]) / 2e9, 2.62, "teal line: GPU awake but idle", ha="center", va="center", fontsize=8.5, color=INK2)
ax.set_xlim(-0.02, 2.3)
style(ax, "B. Idle time does not count: the clock rises only once work keeps the GPU busy",
      "Time from the first kernel (s)")

# C: an idle gap, then the work again
ax = axes[2]
s, e = gap.ranges["after 100 ms gap"]
ks = gap.kernels_in(s, e)
prev_end = max(b for a, b, _ in gap.kernels if b <= s)
t0 = ks[0][0]
for a, b, is_sdpa in gap.kernels_in(prev_end - 60e6, t0 + 260e6) + [k for k in gap.kernels if prev_end - 60e6 <= k[0] < s]:
    ax.axvspan((a - t0) / 1e6, (b - t0) / 1e6, color=SDPA if is_sdpa else OURS, alpha=0.16 if is_sdpa else 0.07, lw=0, zorder=0)
    if is_sdpa and (a - t0) / 1e6 > -60:
        ax.text(((a + b) / 2 - t0) / 1e6, 2.775, f"{(b - a) / 1e6:.2f}", ha="center", va="top", fontsize=7.5, color=INK2, rotation=90)
x, y = gap.clock_between(prev_end - 60e6, t0 + 260e6, t0, 1e6)
ax.plot(x, y, color=INK, lw=1.1, zorder=3)
ax.text(-50, 2.47, "idle gap (100 ms):\nthe clock climbs", ha="center", va="center", fontsize=9, color=INK2)
ax.text(165, 2.47, "power limit reached: back to about 2.5 GHz", ha="center", va="center", fontsize=9, color=INK2)
ax.set_xlim(-160, 260)
style(ax, "C. After an idle gap the work starts fast, until the power limit pulls the clock back (SDPA times in ms)",
      "Time from the first kernel after the gap (ms)")
handles = [plt.Rectangle((0, 0), 1, 1, color=OURS, alpha=0.2), plt.Rectangle((0, 0), 1, 1, color=SDPA, alpha=0.3)]
leg = ax.legend(handles, ["baseline kernel (v0)", "SDPA"], frameon=False, fontsize=9, loc="upper left",
                bbox_to_anchor=(0.0, 0.93))
ax.text(258, 2.415, "3.28 and 3.43: occasional slow SDPA calls, at an unchanged clock", ha="right", va="bottom",
        fontsize=8, color=MUTED)
for t in leg.get_texts():
    t.set_color(INK2)

fig.suptitle("What the clock does around idle time", x=0.02, y=0.995, ha="left", fontsize=13, fontweight="bold")
fig.text(0.02, 0.966, "RTX 4070 SUPER, 220 W limit. Nsight Systems GPU metrics, GPC clock sampled at 10 kHz.",
         fontsize=9.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.958), h_pad=2.2)
fig.savefig(sys.argv[1])
