# wake_capture.py
# Does the clock leave 2.52 GHz a fixed time after the GPU wakes, or after it has been busy for a
# while? One Nsight Systems capture (GPC clock at 10 kHz) of several cold starts, each from idle
# (P8 for 5 s), each in its own NVTX range:
#   spin             torch.cuda._sleep back to back (busy, little power), three times
#   wake, idle 0.5 s one tiny kernel (wakes the GPU), 0.5 s with nothing to do, then the spin
#   wake, idle 1.0 s the same with 1 s
#   5 ms in 50       5 ms of spin every 50 ms (the GPU busy 10% of the time), for 1.5 s
#   nsys profile --trace=cuda,nvtx --gpu-metrics-devices=0 --gpu-metrics-frequency=10000 \
#       --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite \
#       -o reports/wake_capture -f true python wake_capture.py
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clock_dynamics import wait_idle  # noqa: E402
from nvml_ctypes import Device  # noqa: E402

CYCLES_PER_MS = 2_600_000 // 1000  # torch.cuda._sleep counts clock cycles


def spin(seconds, on_ms=None, period_ms=None):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if on_ms is None:
            torch.cuda._sleep(2 * 1000 * CYCLES_PER_MS)
            torch.cuda.synchronize()
        else:
            t1 = time.monotonic()
            torch.cuda._sleep(on_ms * 1000 * CYCLES_PER_MS)
            torch.cuda.synchronize()
            time.sleep(max(0.0, period_ms / 1000 - (time.monotonic() - t1)))


def wake():
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()


SCENARIOS = [
    ("spin #1", lambda: spin(1.0)),
    ("spin #2", lambda: spin(1.0)),
    ("spin #3", lambda: spin(1.0)),
    ("wake, idle 0.5 s, spin", lambda: (wake(), time.sleep(0.5), spin(1.0))),
    ("wake, idle 1.0 s, spin", lambda: (wake(), time.sleep(1.0), spin(1.0))),
    ("5 ms in 50", lambda: spin(1.5, on_ms=5, period_ms=50)),
]
spin(0.01)
dev = Device()
wait_idle(dev, 5.0)
torch.cuda.cudart().cudaProfilerStart()
for i, (name, fn) in enumerate(SCENARIOS):
    if i:
        wait_idle(dev, 5.0)
    torch.cuda.nvtx.range_push(name)
    fn()
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
