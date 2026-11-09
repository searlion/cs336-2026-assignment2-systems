# power_sampler.py
# A background thread that reads power, temperature, SM and memory clocks and the active
# clock-event ("throttle") reasons from NVML every 10 ms, and summarises any time window.
# On this GeForce card the driver refreshes all of these readings only every ~500 ms (checked
# with this sampler and with nvidia-smi -lms 50), so a window shorter than that holds at most one
# fresh reading. nvmlDeviceGetPowerUsage is also a lagging average; the instantaneous field
# (NVML_FI_DEV_POWER_INSTANT) is recorded next to it and is the one summarised as power.
import statistics
import threading
import time

import pynvml as nv

REASONS = {
    nv.nvmlClocksEventReasonGpuIdle: "gpu_idle",
    nv.nvmlClocksEventReasonApplicationsClocksSetting: "app_clocks",
    nv.nvmlClocksEventReasonSwPowerCap: "sw_power_cap",
    nv.nvmlClocksEventReasonHwSlowdown: "hw_slowdown",
    nv.nvmlClocksEventReasonSyncBoost: "sync_boost",
    nv.nvmlClocksEventReasonSwThermalSlowdown: "sw_thermal",
    nv.nvmlClocksEventReasonHwThermalSlowdown: "hw_thermal",
    nv.nvmlClocksEventReasonHwPowerBrakeSlowdown: "hw_power_brake",
    nv.nvmlClocksEventReasonDisplayClockSetting: "display_clocks",
}


class Sampler:
    def __init__(self, period_s=0.010, device=0):
        nv.nvmlInit()
        self.h = nv.nvmlDeviceGetHandleByIndex(device)
        self.period = period_s
        self.samples = []          # (t, instant_W, temp_C, sm_MHz, mem_MHz, reasons_bitmask, average_W)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def info(self):
        h = self.h
        return {"gpu": nv.nvmlDeviceGetName(h),
                "power_limit_W": nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000,
                "max_sm_clock_MHz": nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM),
                "max_mem_clock_MHz": nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_MEM),
                "sample_period_ms": self.period * 1e3}

    def _run(self):
        h = self.h
        while not self._stop.is_set():
            fv = nv.nvmlDeviceGetFieldValues(h, [nv.NVML_FI_DEV_POWER_INSTANT])
            self.samples.append((time.perf_counter(), fv[0].value.uiVal / 1000,
                                 nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                                 nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                                 nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                                 nv.nvmlDeviceGetCurrentClocksEventReasons(h),
                                 nv.nvmlDeviceGetPowerUsage(h) / 1000))
            time.sleep(self.period)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return {"samples": 0}
        col = lambda i: [x[i] for x in s]  # noqa: E731
        counts = {}
        for x in s:
            for bit, name in REASONS.items():
                if x[5] & bit:
                    counts[name] = counts.get(name, 0) + 1
        # distinct readings: the driver refreshes every ~0.5 s, so count changes of the tuple
        fresh = 1 + sum(1 for a, b in zip(s, s[1:]) if a[1:6] != b[1:6])
        return {"samples": len(s), "fresh_readings": fresh, "seconds": round(t1 - t0, 3),
                "power_W_median": round(statistics.median(col(1)), 1), "power_W_max": round(max(col(1)), 1),
                "power_avg_reading_W_median": round(statistics.median(col(6)), 1),
                "temp_C_start": s[0][2], "temp_C_end": s[-1][2], "temp_C_max": max(col(2)),
                "sm_MHz_median": statistics.median(col(3)), "sm_MHz_min": min(col(3)),
                "mem_MHz_median": statistics.median(col(4)), "mem_MHz_min": min(col(4)),
                # fraction of samples in which each clock-event reason was active
                "reasons": {k: round(v / len(s), 3) for k, v in sorted(counts.items())}}


def record_do_bench(windows):
    """Wrap triton.testing.do_bench so every top-level call appends its (start, end) time to
    `windows`. Calls made inside it (Triton's autotuner times its candidates with do_bench
    during the harness's first call) are part of the outer window and are not recorded."""
    import triton.testing
    orig = triton.testing.do_bench
    depth = [0]

    def timed(*a, **k):
        depth[0] += 1
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            depth[0] -= 1
            if depth[0] == 0:
                windows.append((t0, time.perf_counter()))
    triton.testing.do_bench = timed
