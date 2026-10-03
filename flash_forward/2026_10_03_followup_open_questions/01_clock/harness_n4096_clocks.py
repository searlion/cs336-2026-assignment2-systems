# harness_n4096_clocks.py
# Item 1: the clock during the N = 4096 kernels of one harness run (under nsys GPU metrics), and
# the share of the bf16 -> fp32 peak each reached at its own clock (the harness's TFLOP/s for that
# kernel, from harness_v5_under_nsys.txt).
#
# Only the launches the harness actually timed are used. In the harness's order the N = 4096
# launches come as: ours non-causal (the autotuner timing every candidate, then do_bench timing
# the chosen one), SDPA non-causal, ours causal (same), SDPA causal. The chosen configuration is
# identified by its registers and shared memory (from the CUPTI kernel table), and only the final
# unbroken run of launches with that configuration, after the autotuner has finished, is kept.
#   python harness_n4096_clocks.py reports/harness_v5.sqlite harness_v5_under_nsys.txt
import bisect
import re
import sqlite3
import statistics
import sys

con = sqlite3.connect(sys.argv[1])
names = dict(con.execute("SELECT id, value FROM StringIds"))
cid = [m for m, n in con.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS") if "GPC Clock" in n][0]
# The clock is stored in Hz in a signed 32-bit field: add 2^32 to negative values.
S = sorted((t, v + 2 ** 32 if v < 0 else v) for t, v in con.execute(f"SELECT timestamp, value FROM GPU_METRICS WHERE metricId={cid}"))
ts = [s[0] for s in S]
harness = {}
for line in open(sys.argv[2]):
    m = re.match(r"^\s*4096\s+(True|False)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)", line)
    if m:
        harness[m[1] == "True"] = (float(m[2]), float(m[3]))

rows = con.execute("""SELECT start, end, demangledName, gridX, gridY, gridZ, registersPerThread,
                             staticSharedMemory + dynamicSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL
                      ORDER BY start""").fetchall()
ours = [r for r in rows if (r[3], r[4], r[5]) == (64, 32, 1) and "pytorch_flash" not in names[r[2]]]
sdpa = [r for r in rows if (r[3], r[4], r[5]) == (32, 4, 8) and "pytorch_flash" in names[r[2]]]
# Phases, by order (durations overlap, so they cannot be used to split): ours non-causal, then the
# first block of SDPA launches (non-causal), then ours causal, then the second SDPA block (causal).
first_ours_causal = min(r[0] for r in ours if r[0] > sdpa[0][0])
sdpa_phases = {False: [r for r in sdpa if r[0] < first_ours_causal], True: [r for r in sdpa if r[0] > first_ours_causal]}
phases = {False: [r for r in ours if r[0] < sdpa[0][0]],
          True: [r for r in ours if first_ours_causal <= r[0] < sdpa_phases[True][0][0]]}


def timed_tail(launches):
    """The final unbroken run of launches with the last launch's configuration."""
    cfg = launches[-1][6:8]
    i = len(launches)
    while i > 0 and launches[i - 1][6:8] == cfg:
        i -= 1
    return launches[i:], cfg


def clock(r):
    lo, hi = bisect.bisect_left(ts, r[0]), bisect.bisect_right(ts, r[1])
    return statistics.mean(S[i][1] for i in range(lo, hi)) / 1e6 if hi > lo else None


print(f"{'kernel, N = 4096':26s} {'config (regs, smem B)':>22s} {'launches':>8s} {'clock MHz':>9s} {'ms':>7s} "
      f"{'harness TF/s':>12s} {'peak at clock':>13s} {'% of it':>7s}")
for causal in (False, True):
    for name, launches in (("final (v5)", phases[causal]), ("SDPA", sdpa_phases[causal])):
        if name.startswith("final"):
            launches, cfg = timed_tail(launches)
            cfg_txt = f"{cfg[0]} regs, {cfg[1]}"
        else:
            cfg_txt = "-"
        clks = [c for c in map(clock, launches) if c]
        clk = statistics.median(clks)
        tf = harness[causal][0 if name.startswith("final") else 1]
        peak = 56 * 512 * clk * 1e6 / 1e12
        print(f"{name + (' causal' if causal else ' non-causal'):26s} {cfg_txt:>22s} {len(launches):8d} {clk:9.0f} "
              f"{statistics.median((r[1] - r[0]) / 1e6 for r in launches):7.3f} {tf:12.1f} {peak:13.1f} {100 * tf / peak:6.1f}%")
