# analyze_clock_dynamics.py
# What NVML saw in clock_dynamics.py. NVML (and so nvidia-smi) refreshes the clock and power it
# reports only about every 0.5 s on this driver, however often it is asked, so this prints, for
# each trial, every change in the reported values from 0.3 s before the first launch to 3 s after
# the last call: SM clock, instantaneous board power, NVML's 1-second-average power, temperature,
# P-state and the active clock-event reasons (0x1 idle, 0x4 SW power cap). Then the SDPA and v0
# call times (CUDA events, no profiler attached), including the occasional slow SDPA calls.
#   python analyze_clock_dynamics.py > clock_dynamics.txt
import json
import pathlib
import statistics

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
fast = np.load(HERE / "clock_dynamics_samples.npz")["fast"]
meta = json.load(open(HERE / "clock_dynamics_trials.json"))
t = fast[:, 0]
changes = np.flatnonzero(np.any(np.diff(fast[:, 1:4], axis=0) != 0, axis=1)) + 1
gaps = np.diff(t[changes])
print(f"{len(t)} NVML samples, one every {1e3 * np.median(np.diff(t)):.2f} ms (median); the reported clock or power "
      f"changed {len(changes)} times, at a median interval of {np.median(gaps[gaps > 0.05]):.2f} s\n")
for tr in meta["trials"]:
    t0, t1 = tr["t_start"], tr["t_end"]
    print(f"== trial {tr['trial']}, {tr['workload']}: load from 0 to {t1 - t0:.2f} s")
    prev = None
    for r in fast[(t >= t0 - 0.3) & (t <= t1 + 3.0)]:
        key = (r[1], r[2], r[3], r[5], int(r[6]))
        if key != prev:
            print(f"  t = {r[0] - t0:6.3f} s  SM {r[1]:4.0f} MHz  power {r[2]:5.1f} W  1-s average {r[3]:5.1f} W  "
                  f"{r[4]:2.0f} C  P{int(r[5])}  reasons {int(r[6]):#05x}")
            prev = key
    print()
for label in ("sdpa", "v0", "spin"):
    d = [c[2] for tr in meta["trials"] for c in tr["calls"] if c[1] == label]
    med = statistics.median(d)
    slow = [x for x in d if x > med + 0.4]
    print(f"{label:5s} {len(d):6d} calls, median {med:.3f} ms; more than 0.4 ms slower than the median: {len(slow)} "
          f"({100 * len(slow) / len(d):.1f}%)" + (f"; over 3 ms: {sum(x > 3 for x in d)}, slowest {max(d):.2f} ms" if label == "sdpa" else ""))
