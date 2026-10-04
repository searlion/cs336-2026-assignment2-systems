# profiler_start_pause.py
# How long does cudaProfilerStart() block the program under nsys --capture-range=cudaProfilerApi
# with GPU metrics on? During that time the GPU has no work, so its clock can climb (see panel C
# of clock_dynamics.png). Same command line as the cold/warm capture (cold_capture.py). Appends
# one line per run to reports/profiler_start_pause.txt (before cudaProfilerStop, because nsys
# ends the program there by default).
#   nsys profile --trace=cuda,nvtx --gpu-metrics-devices=0 --capture-range=cudaProfilerApi \
#       -o reports/profiler_start_pause -f true python profiler_start_pause.py
import time

import torch

torch.cuda._sleep(1000)
torch.cuda.synchronize()
t0 = time.perf_counter()
torch.cuda.cudart().cudaProfilerStart()
t1 = time.perf_counter()
with open("reports/profiler_start_pause.txt", "a") as f:
    print(f"cudaProfilerStart blocked for {1e3 * (t1 - t0):.0f} ms", file=f)
torch.cuda._sleep(1000)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
