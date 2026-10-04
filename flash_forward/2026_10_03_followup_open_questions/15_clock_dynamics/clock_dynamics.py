# clock_dynamics.py
# Why, after an idle period, does the RTX 4070 SUPER run at 2,520 MHz for up to a second, then
# jump to about 2.74 GHz, then drop again (nsys_clock_cold_warm.png and clock_ramp.txt)?
# A second process polls NVML every millisecond while this one runs a workload straight from
# idle: the SM clock, the instantaneous and the 1-second-average board power, the temperature,
# the P-state and the active clock-event reasons, plus (every few ms) the driver's cumulative
# "violation" counters, which say how long each limiter (power, low utilisation, ...) has held
# the clock down. Each trial waits until the GPU has sat idle in P8 for IDLE seconds, runs the
# workload for LOAD seconds, then records POST seconds of idle. Workloads, in rotation:
#   v0+sdpa       the work in the cold/warm capture: v0 and SDPA alternating, N = 4096, non-causal
#   v0+sdpa (nv)  the same, without reading the violation counters (a check that reading them
#                 does not change the clock's behaviour)
#   spin          torch.cuda._sleep, one thread spinning on one SM: the GPU is busy but draws little power
# Reading the violation counters wakes an idle GPU from P8 (210 MHz) to P5 (about 1.45 GHz), so
# they are read only while a load runs; the other queries leave it in P8.
#   flock /tmp/claude-gpu.lock python clock_dynamics.py   (writes clock_dynamics_{samples.npz,trials.json})
import json
import multiprocessing as mp
import pathlib
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from nvml_ctypes import Device  # noqa: E402

TRIALS, IDLE, LOAD, POST = 3, 5.0, 6.0, 3.0
VIOLATION_POLICIES = (0, 4, 10, 11)  # power, low utilization, total app clocks, total base clocks


def poll(path, stop, read_violations):
    d = Device()
    fast, viol = [], []

    def violations():
        while not stop.is_set():
            if not read_violations.is_set():
                time.sleep(0.001)
                continue
            t0 = time.monotonic()
            vals = [d.violation_ns(p) for p in VIOLATION_POLICIES]
            viol.append((t0, time.monotonic(), *vals))
            time.sleep(0.002)

    th = threading.Thread(target=violations, daemon=True)
    th.start()
    nxt = time.monotonic()
    while not stop.is_set():
        t = time.monotonic()
        p_inst, p_avg = d.power()
        fast.append((t, d.sm_clock(), p_inst, p_avg, d.temperature(), d.pstate(), d.reasons()))
        nxt += 0.001
        dt = nxt - time.monotonic()
        if dt > 0:
            time.sleep(dt)
        else:
            nxt = time.monotonic()
    th.join()
    np.savez_compressed(path, fast=np.array(fast, dtype=float), viol=np.array(viol, dtype=float))


def wait_idle(d, seconds):
    """Return once the GPU has been in P8 at idle clocks for `seconds` without a break."""
    since, t0 = None, time.monotonic()
    while True:
        now = time.monotonic()
        if now - t0 > 120:
            raise RuntimeError(f"GPU not idle after 120 s: P{d.pstate()} at {d.sm_clock()} MHz")
        if d.pstate() == 8 and d.sm_clock() <= 300:
            since = since or now
            if now - since >= seconds:
                return
        else:
            since = None
        time.sleep(0.01)


def main():
    import torch
    import torch.nn.functional as F
    from common import load_version

    FA = load_version("v0_baseline").FlashAttentionTriton
    B, H, N, D = 4, 8, 4096, 64
    q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    q4, k4, v4 = (t.view(B, H, N, D) for t in (q, k, v))
    workloads = {
        "v0+sdpa": [("v0", lambda: FA.apply(q, k, v, False)),
                    ("sdpa", lambda: F.scaled_dot_product_attention(q4, k4, v4))],
        "spin": [("spin", lambda: torch.cuda._sleep(2_000_000))],
    }
    workloads["v0+sdpa (nv)"] = workloads["v0+sdpa"]
    for calls in workloads.values():  # compile and initialise everything before the first trial
        for _, fn in calls:
            fn()
    torch.cuda.synchronize()

    stop, read_violations = mp.Event(), mp.Event()
    poller = mp.get_context("spawn").Process(target=poll, args=(str(HERE / "clock_dynamics_samples.npz"), stop,
                                                                read_violations))
    poller.start()
    d = Device()
    trials = []
    for trial in range(TRIALS):
        for name, calls in workloads.items():
            wait_idle(d, IDLE)
            print(f"trial {trial} {name}: idle, starting", flush=True)
            log = []
            if not name.endswith("(nv)"):
                read_violations.set()
            t_start = time.monotonic()
            while time.monotonic() - t_start < LOAD:
                for label, fn in calls:
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    fn()
                    e.record()
                    e.synchronize()
                    log.append((time.monotonic(), label, s.elapsed_time(e)))
            t_end = time.monotonic()
            read_violations.clear()
            trials.append({"trial": trial, "workload": name, "t_start": t_start, "t_end": t_end, "calls": log})
            print(f"trial {trial} {name}: {len(log)} calls", flush=True)
            time.sleep(POST)
    stop.set()
    poller.join()
    json.dump({"IDLE": IDLE, "LOAD": LOAD, "POST": POST, "trials": trials}, open(HERE / "clock_dynamics_trials.json", "w"))


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
