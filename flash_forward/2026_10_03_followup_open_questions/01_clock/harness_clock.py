# harness_clock.py
# Item 1: the SM clock while the unchanged harness was running. Reads the nvidia-smi log and the
# start/end timeline written by 03_harness_repeats/run_harness_repeats.sh and reports, per
# version, the SM clock and power over the samples taken while the GPU was busy (power above a
# threshold, so that Python start-up and the gaps between do_bench calls are left out).
#   python harness_clock.py ../03_harness_repeats/runs
import collections
import datetime
import pathlib
import statistics
import sys

runs = pathlib.Path(sys.argv[1])
BUSY_W = 120.0
samples = []
for line in (runs / "nvidia_smi.csv").read_text().splitlines()[1:]:
    ts, clk, mem, pw, temp, reasons = (x.strip() for x in line.split(","))
    samples.append((datetime.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f"), float(clk.split()[0]),
                    float(pw.split()[0]), int(temp), reasons))
windows = collections.defaultdict(list)
start = {}
for line in (runs / "timeline.log").read_text().splitlines():
    d, t, what, version, r = line.split()
    ts = datetime.datetime.strptime(f"{d} {t}", "%Y/%m/%d %H:%M:%S.%f")
    if what == "start":
        start[(version, r)] = ts
    else:
        windows[version].append((start[(version, r)], ts))

print(f"samples with power > {BUSY_W:.0f} W inside each version's harness runs")
print(f"{'version':16s} {'samples':>7s} {'SM MHz median':>13s} {'p10-p90':>11s} {'power W median':>14s} {'temp C max':>10s}  reasons")
allbusy = []
for version, ws in windows.items():
    s = [x for x in samples if any(a <= x[0] <= b for a, b in ws) and x[2] > BUSY_W]
    allbusy += s
    clk = sorted(x[1] for x in s)
    pw = [x[2] for x in s]
    q = lambda p: clk[int(p * (len(clk) - 1))]  # noqa: E731
    print(f"{version:16s} {len(s):7d} {statistics.median(clk):13.0f} {q(.1):5.0f}-{q(.9):<5.0f} "
          f"{statistics.median(pw):14.0f} {max(x[3] for x in s):10d}  {sorted(collections.Counter(x[4] for x in s).items())}")
clk = sorted(x[1] for x in allbusy)
print(f"{'all versions':16s} {len(allbusy):7d} {statistics.median(clk):13.0f} "
      f"{clk[len(clk) // 10]:5.0f}-{clk[9 * len(clk) // 10]:<5.0f}")
print(f"peak bf16->fp32 at the median clock: 56 x 512 x {statistics.median(clk):.0f} MHz = "
      f"{56 * 512 * statistics.median(clk) * 1e6 / 1e12:.1f} TFLOP/s")
