# summarize_power.py
# Power, temperature, clocks and clock-event reasons during the harness's do_bench calls
# (results/final/power/<version>_run<k>.json from bench_with_power.py), per version and shape.
# Each do_bench call is shorter than NVML's ~0.5 s refresh, so a window holds only 0-1 fresh
# readings; fair_fight.py is the sustained measurement. This table checks the whole set of runs
# for thermal slowdowns and for SM-clock differences between versions.
import collections
import glob
import json
import pathlib
import statistics

rows = collections.defaultdict(list)    # (who, N, causal) -> windows
starts = collections.defaultdict(list)  # who -> temperature at the start of each run
for path in sorted(glob.glob("results/final/power/*_run*.json")):
    version = pathlib.Path(path).stem.rsplit("_run", 1)[0]
    d = json.load(open(path))
    starts[version].append(d["shapes"][0]["ours"]["temp_C_start"])
    for s in d["shapes"]:
        rows[(version, s["N"], s["causal"])].append(s["ours"])
        if version != "nvidia_example":
            rows[("sdpa", s["N"], s["causal"])].append(s["sdpa"])

thermal = [k for k, ws in rows.items() for w in ws if any(r in w["reasons"] for r in ("sw_thermal", "hw_thermal", "hw_slowdown", "hw_power_brake"))]
print("windows with a thermal / hardware slowdown reason:", len(thermal), thermal[:5])
print("temperature at the start of each run (C), median [min-max]:")
for v, t in sorted(starts.items()):
    print(f"  {v:15s} {statistics.median(t):.0f} [{min(t)}-{max(t)}] over {len(t)} runs")
print()
print(f"{'who':15s} {'N':>5s} {'causal':6s} {'power W':>8s} {'max W':>6s} {'max C':>6s} {'SM MHz':>7s} {'min SM':>7s}  power-cap share")
for (who, N, causal), ws in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
    ws = [w for w in ws if w.get("samples")]
    cap = statistics.mean(w["reasons"].get("sw_power_cap", 0) for w in ws)
    print(f"{who:15s} {N:5d} {causal!s:6s} {statistics.median(w['power_W_median'] for w in ws):8.1f} "
          f"{max(w['power_W_max'] for w in ws):6.1f} {max(w['temp_C_max'] for w in ws):6d} "
          f"{statistics.median(w['sm_MHz_median'] for w in ws):7.0f} {min(w['sm_MHz_min'] for w in ws):7d}  {cap:.2f}")
