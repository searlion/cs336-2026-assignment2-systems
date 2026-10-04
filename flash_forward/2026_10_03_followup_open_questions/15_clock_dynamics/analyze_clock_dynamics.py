# analyze_clock_dynamics.py
# Phases of the clock in each trial of clock_dynamics.py, timed from the first launch (t = 0):
#   wake    the first sample above 1 GHz, and the clock and P-state there
#   hold    how long the clock stays at its first busy value before rising
#   boost   the highest clock in the 0.5 s after it rises
#   cap     the first sample with the "SW power cap" reason, the 1-s average power there, and the
#           lowest clock in the following 0.5 s
#   settle  median clock, instantaneous power and temperature over the last 2 s of the load
#   idle    how long after the last call the GPU is back in P8
# and, from the violation counters, the share of each phase's time that the driver attributes to
# each limiter (low utilisation, power). Also the median SDPA time in each phase (v0+sdpa).
#   python analyze_clock_dynamics.py > clock_dynamics.txt
import json
import pathlib
import statistics

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
S = np.load(HERE / "clock_dynamics_samples.npz")
fast, viol = S["fast"], S["viol"]
t, clk, p_inst, p_avg, temp, pstate, reasons = fast.T
reasons = reasons.astype(np.int64)
meta = json.load(open(HERE / "clock_dynamics_trials.json"))
dt = np.diff(t)
print(f"NVML samples: {len(t)}, median interval {1e3 * np.median(dt):.2f} ms, 99th percentile {1e3 * np.percentile(dt, 99):.2f} ms")
chg = np.flatnonzero(np.diff(clk) != 0)
print(f"clock value changes: {len(chg)}; power (instant) value changes per second while busy: ", end="")
busy = clk > 1000
pc = np.flatnonzero((np.diff(p_inst) != 0) & busy[1:])
print(f"{len(pc) / (busy.sum() * np.median(dt)):.0f}")
print()


def share(policy_col, a, b):
    """Fraction of [a, b] (monotonic s) that the violation counter in column policy_col grew by."""
    if len(viol) == 0:
        return float("nan")
    tv = (viol[:, 0] + viol[:, 1]) / 2
    sel = (tv >= a) & (tv <= b)
    if sel.sum() < 2:
        return float("nan")
    v = viol[sel]
    return (v[-1, policy_col] - v[0, policy_col]) * 1e-9 / (tv[sel][-1] - tv[sel][0])


rows = []
for tr in meta["trials"]:
    t0, t1 = tr["t_start"], tr["t_end"]
    sel = (t >= t0 - 0.2) & (t <= t1 + meta["POST"])
    tt, cc, pi, pa, te, ps, rs = t[sel] - t0, clk[sel], p_inst[sel], p_avg[sel], temp[sel], pstate[sel], reasons[sel]
    i_wake = np.argmax((tt >= 0) & (cc > 1000))
    first = cc[i_wake]
    i_rise = i_wake + np.argmax(cc[i_wake:] != first)
    rise_win = (tt >= tt[i_rise]) & (tt <= tt[i_rise] + 0.5)
    boost = cc[rise_win].max()
    capped = (rs & 0x4).astype(bool) & (tt >= 0) & (tt <= t1 - t0)
    i_cap = np.argmax(capped) if capped.any() else None
    end = t1 - t0
    last2 = (tt >= end - 2) & (tt <= end)
    after = (tt > end) & (ps == 8)
    r = {
        "trial": tr["trial"], "workload": tr["workload"],
        "wake_s": tt[i_wake], "wake_MHz": first, "wake_P": int(ps[i_wake]),
        "hold_s": tt[i_rise] - tt[i_wake], "boost_MHz": boost,
        "cap_s": tt[i_cap] if i_cap is not None else float("nan"),
        "cap_avgW": pa[i_cap] if i_cap is not None else float("nan"),
        "drop_MHz": cc[(tt >= tt[i_cap]) & (tt <= tt[i_cap] + 0.5)].min() if i_cap is not None else float("nan"),
        "settle_MHz": np.median(cc[last2]), "settle_W": np.median(pi[last2]), "settle_C": np.median(te[last2]),
        "hold_W": np.median(pi[(tt >= tt[i_wake] + 0.05) & (tt < tt[i_rise])]) if i_rise > i_wake else float("nan"),
        "boost_W": np.median(pi[rise_win]),
        "idle_s": (tt[after][0] - end) if after.any() else float("nan"),
        "capped_share": capped[last2].mean(),
    }
    a_hold, b_hold = t0 + r["wake_s"], t0 + r["wake_s"] + r["hold_s"]
    r["lowutil_hold"] = share(3, a_hold, b_hold)
    r["power_hold"] = share(2, a_hold, b_hold)
    r["lowutil_last2"] = share(3, t1 - 2, t1)
    r["power_last2"] = share(2, t1 - 2, t1)
    sd = [(c[0] - t0, c[2]) for c in tr["calls"] if c[1] == "sdpa"]
    if sd:
        r["sdpa_hold_ms"] = statistics.median([ms for s, ms in sd if s < r["wake_s"] + r["hold_s"]] or [float("nan")])
        r["sdpa_boost_ms"] = statistics.median([ms for s, ms in sd if r["wake_s"] + r["hold_s"] <= s < r["cap_s"]] or [float("nan")])
        r["sdpa_last2_ms"] = statistics.median([ms for s, ms in sd if s >= end - 2])
    rows.append(r)

hdr = (f"{'workload':13s} {'#':>1s} {'wake':>6s} {'MHz':>5s} {'P':>2s} {'hold s':>6s} {'hold W':>6s} {'boost':>5s} {'boost W':>7s} "
       f"{'cap at':>6s} {'avg W':>5s} {'drop to':>7s} {'settle':>6s} {'W':>4s} {'C':>3s} {'capped':>6s} {'P8 after':>8s}")
print("times in s from the first launch; clocks in MHz; W = instantaneous board power (median), avg W = NVML's 1-s average")
print(hdr)
for r in rows:
    print(f"{r['workload']:13s} {r['trial']:1d} {r['wake_s']:6.3f} {r['wake_MHz']:5.0f} {r['wake_P']:2d} {r['hold_s']:6.3f} {r['hold_W']:6.0f} "
          f"{r['boost_MHz']:5.0f} {r['boost_W']:7.0f} {r['cap_s']:6.3f} {r['cap_avgW']:5.0f} {r['drop_MHz']:7.0f} "
          f"{r['settle_MHz']:6.0f} {r['settle_W']:4.0f} {r['settle_C']:3.0f} {100 * r['capped_share']:5.0f}% {r['idle_s']:8.3f}")
print("\nshare of time each limiter held the clock down (violation counters; not read in the '(nv)' trials)")
print(f"{'workload':13s} {'#':>1s} {'low util. (hold)':>16s} {'power (hold)':>12s} {'low util. (last 2 s)':>20s} {'power (last 2 s)':>16s}")
for r in rows:
    if r["workload"].endswith("(nv)"):
        continue
    print(f"{r['workload']:13s} {r['trial']:1d} {100 * r['lowutil_hold']:15.0f}% {100 * r['power_hold']:11.0f}% "
          f"{100 * r['lowutil_last2']:19.0f}% {100 * r['power_last2']:15.0f}%")
print("\nSDPA time per call (ms, median) in each phase")
for r in rows:
    if "sdpa_hold_ms" in r:
        print(f"{r['workload']:13s} {r['trial']:1d} hold {r['sdpa_hold_ms']:.3f}  boosted {r['sdpa_boost_ms']:.3f}  last 2 s {r['sdpa_last2_ms']:.3f}")
