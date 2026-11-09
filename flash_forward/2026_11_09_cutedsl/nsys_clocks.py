# nsys_clocks.py
# Cross-check of the NVML clocks: Nsight Systems' GPU metrics sample the GPU clock in hardware
# (many times per kernel). For every kernel launch in results/nsys/clocks_final.nsys-rep
# (compare_driver.py: CuTe v5, Triton v5, SDPA at N = 4096, 10 launches per mask), take the
# median GPC clock over the samples inside it, then summarise per kernel and mask.
#
#   python nsys_clocks.py results/nsys/clocks_final
import collections
import sqlite3
import statistics
import subprocess
import sys

NSYS = "/opt/nvidia/nsight-systems/2026.1.3/bin/nsys"
base = sys.argv[1]
subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", base + ".sqlite",
                base + ".nsys-rep"], check=True, capture_output=True)
db = sqlite3.connect(base + ".sqlite")
metrics = dict(db.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS").fetchall())
clock_id = next(i for i, n in metrics.items() if "GPC Clock Frequency" in n)
# The clock is stored in Hz in a signed 32-bit field, so 2.7 GHz wraps negative: undo that.
samples = [(t, (v % 2**32) / 1e6) for t, v in db.execute(
    "SELECT timestamp, value FROM GPU_METRICS WHERE metricId = ? ORDER BY timestamp", (clock_id,)).fetchall()]
kernels = db.execute("""SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
                        JOIN StringIds s ON s.id = k.shortName ORDER BY k.start""").fetchall()


# compare_driver.py launches, in order: CuTe v5 (10 non-causal, then 10 causal), Triton v5
# (the same), SDPA (the same). Triton's and SDPA's short names are both "flash_fwd_kernel",
# so the launch order tells them apart.
seen = collections.Counter()
per = collections.defaultdict(list)
for start, end, name in kernels:
    if "flash" not in name.lower() and "cutlass" not in name.lower():
        continue
    group = "cute_v5" if "cutlass" in name else "flash_fwd_kernel"
    i = seen[group]
    seen[group] += 1
    k = "cute_v5" if group == "cute_v5" else ("triton_v5" if i < 20 else "sdpa")
    causal = (i % 20) >= 10
    inside = [v for t, v in samples if start <= t <= end]
    if inside:
        per[(k, causal)].append((statistics.median(inside), len(inside), (end - start) / 1e6))
print(f"metric: {metrics[clock_id]}")
print(f"{'kernel':10s} {'causal':6s} {'launches':>8s} {'samples/launch':>14s} {'median MHz':>10s} {'min MHz':>8s} {'max MHz':>8s} {'median ms':>9s}")
for (k, c), xs in sorted(per.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    clocks = [x[0] for x in xs]
    print(f"{k:10s} {c!s:6s} {len(xs):8d} {statistics.median(x[1] for x in xs):14.0f} {statistics.median(clocks):10.0f} "
          f"{min(clocks):8.0f} {max(clocks):8.0f} {statistics.median(x[2] for x in xs):9.3f}")
