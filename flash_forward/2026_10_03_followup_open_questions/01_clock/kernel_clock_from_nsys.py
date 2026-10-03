# kernel_clock_from_nsys.py
# Item 1 (and 13): the GPU clock during each kernel, from an Nsight Systems capture taken with
# --gpu-metrics-devices=0. GPU metrics sampling records "GPC Clock Frequency" (the SM clock) every
# 1/frequency seconds; this averages the samples that fall inside each kernel and groups kernels
# by name and grid (which identifies the shape in the harness).
#   python kernel_clock_from_nsys.py capture.sqlite
import collections
import sqlite3
import statistics
import sys

con = sqlite3.connect(sys.argv[1])
names = dict(con.execute("SELECT id, value FROM StringIds"))
metric = {mid: name for mid, name in con.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS")}
clock_ids = [mid for mid, n in metric.items() if "GPC Clock" in n]
if not clock_ids:
    sys.exit(f"no GPC clock metric; metrics are: {sorted(set(metric.values()))}")
# The clock is stored in Hz in a signed 32-bit field, so clocks above 2.147 GHz come back negative:
# add 2^32 to recover them.
samples = sorted((t, v + 2 ** 32 if v < 0 else v) for t, v in con.execute(
    f"SELECT timestamp, value FROM GPU_METRICS WHERE metricId IN ({','.join(map(str, clock_ids))})"))
ts = [s[0] for s in samples]
print(f"{len(samples)} clock samples, {metric[clock_ids[0]]!r}")

import bisect  # noqa: E402

groups = collections.defaultdict(list)
kernels = con.execute("""SELECT start, end, demangledName, gridX, gridY, gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start""").fetchall()
for start, end, dem, gx, gy, gz in kernels:
    lo, hi = bisect.bisect_left(ts, start), bisect.bisect_right(ts, end)
    vals = [samples[i][1] for i in range(lo, hi)]
    if not vals:
        continue
    name = names.get(dem, "?")
    short = ("SDPA " if "pytorch_flash" in name else "") + name.split("(")[0].split("<")[0].split("::")[-1][:30]
    groups[(short, (gx, gy, gz))].append((statistics.mean(vals), (end - start) / 1e6, len(vals)))

print(f"{'kernel':34s} {'grid':>16s} {'launches':>8s} {'clock MHz (median of per-launch means)':>40s} {'p10-p90':>11s} {'ms median':>9s}")
for (short, grid), xs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    clk = sorted(x[0] / 1e6 for x in xs)
    ms = statistics.median(x[1] for x in xs)
    print(f"{short:34s} {str(grid):>16s} {len(xs):8d} {statistics.median(clk):40.0f} "
          f"{clk[len(clk) // 10]:5.0f}-{clk[9 * len(clk) // 10]:<5.0f} {ms:9.3f}")
