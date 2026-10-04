# sustained_by_precision.py
# What does the RTX 4070 SUPER reach in each precision when it runs flat out, and at what clock?
# Each workload (an 8192^3 matmul) runs back to back for about 4 s while nvidia-smi samples the
# SM clock and the power every 100 ms. For the last 3 s of each run this prints the median time
# per call, TFLOP/s, the median clock, the peak at that clock (56 SMs x operations per clock per
# SM x clock) and the share of it reached. The per-clock rates are the dense GeForce Ada ones
# (Ada whitepaper; Nsight Compute's peak table agrees except for fp8, see the article):
#   fp16/bf16 -> fp32: 512, fp16 -> fp16: 1024, tf32: 256, int8: 2048, fp8 -> fp32: 1024,
#   fp8 -> fp16: 2048, fp32 on the CUDA cores (FFMA): 256.
# Workloads: the Triton matmul of matmul_variants.py with its fastest config per type
# (best_configs.json), and PyTorch's own (cuBLAS / cuBLASLt) kernels where one exists.
#   flock /tmp/claude-gpu.lock python sustained_by_precision.py > sustained_by_precision.txt
import datetime
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from matmul_variants import VARIANTS, launch, make_inputs  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
M = N = K = 8192
SMS = 56
best = json.load(open(HERE / "best_configs.json"))
RATE = {"bf16_in_fp32_acc": 512, "fp16_in_fp32_acc": 512, "fp16_in_fp16_acc": 1024, "tf32_in_fp32_acc": 256,
        "int8_in_int32_acc": 2048, "fp8e4m3_in_fp32_acc": 1024}

workloads = []  # (label, per-clock rate, function)
for variant, (dt, _, out_dt, _, _) in VARIANTS.items():
    a, b = make_inputs(dt)
    c = torch.empty(M, N, device="cuda", dtype=out_dt)
    cfg = tuple(best[variant]["config"])
    workloads.append((f"Triton {variant}", RATE[variant], (lambda a=a, b=b, c=c, v=variant, cfg=cfg: launch(a, b, c, v, cfg))))

a16, b16 = make_inputs(torch.bfloat16)
workloads.append(("cuBLAS bf16 -> fp32 (torch.matmul)", 512, lambda: a16 @ b16))
h16, g16 = make_inputs(torch.float16)
workloads.append(("cuBLAS fp16 -> fp32 (torch.matmul)", 512, lambda: h16 @ g16))
f32a, f32b = make_inputs(torch.float32)


def tf32_mm():
    torch.backends.cuda.matmul.allow_tf32 = True
    return f32a @ f32b


def fp32_mm():
    torch.backends.cuda.matmul.allow_tf32 = False
    return f32a @ f32b


workloads.append(("cuBLAS tf32 (torch.matmul, allow_tf32)", 256, tf32_mm))
workloads.append(("cuBLAS fp32 on CUDA cores (SGEMM)", 256, fp32_mm))
i8a, i8b = make_inputs(torch.int8)
i8b_cm = i8b.t().contiguous().t()
try:
    torch._int_mm(i8a, i8b_cm)
    workloads.append(("cuBLASLt int8 -> int32 (torch._int_mm)", 2048, lambda: torch._int_mm(i8a, i8b_cm)))
except Exception as e:  # noqa: BLE001
    print(f"torch._int_mm unavailable: {type(e).__name__}: {e}"[:200])
f8a = make_inputs(torch.float8_e4m3fn)[0]
f8b = make_inputs(torch.float8_e4m3fn)[1].t().contiguous().t()
one = torch.ones((), device="cuda")
for fast, rate in ((False, 1024), (True, 2048)):
    try:
        torch._scaled_mm(f8a, f8b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16, use_fast_accum=fast)
        workloads.append((f"cuBLASLt fp8 (torch._scaled_mm, use_fast_accum={fast})", rate,
                          lambda fast=fast: torch._scaled_mm(f8a, f8b, scale_a=one, scale_b=one,
                                                             out_dtype=torch.bfloat16, use_fast_accum=fast)))
    except Exception as e:  # noqa: BLE001
        print(f"torch._scaled_mm(use_fast_accum={fast}) unavailable: {type(e).__name__}: {e}"[:200])
for _, _, fn in workloads:
    fn()
torch.cuda.synchronize()

log = tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False)
smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,clocks.sm,power.draw.instant,temperature.gpu",
                        "--format=csv,noheader,nounits", "-lms", "100"], stdout=log, stderr=subprocess.DEVNULL)
t0 = time.time()
while time.time() - t0 < 2.0:  # warm the GPU up first
    a16 @ b16
    torch.cuda.synchronize()

windows = []
for label, rate, fn in workloads:
    events = []
    start = time.time()
    while time.time() - start < 4.0:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        events.append((time.time(), s, e))
        if len(events) % 8 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    end = time.time()
    ms = statistics.median(s.elapsed_time(e) for t, s, e in events if t > start + 1.0)
    windows.append((label, rate, ms, start + 1.0, end))
    time.sleep(1.0)  # a short gap between workloads
smi.terminate()
smi.wait()

samples = []
for line in open(log.name):
    parts = [p.strip() for p in line.split(",")]
    try:
        ts = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        samples.append((ts, float(parts[1]), float(parts[2]), float(parts[3])))
    except (ValueError, IndexError):
        continue

print(f"{'workload':52s} {'ms':>7s} {'TFLOP/s':>8s} {'clock MHz':>9s} {'W':>5s} {'peak at clock':>13s} {'% of it':>7s}")
for label, rate, ms, a_t, b_t in windows:
    win = [s for s in samples if a_t <= s[0] <= b_t]
    clk = statistics.median(s[1] for s in win)
    watts = statistics.median(s[2] for s in win)
    tf = 2 * M * N * K / ms * 1e-9
    peak = SMS * rate * clk * 1e6 / 1e12
    print(f"{label:52s} {ms:7.3f} {tf:8.1f} {clk:9.0f} {watts:5.0f} {peak:13.1f} {100 * tf / peak:6.1f}%")
print(f"\n{len(samples)} nvidia-smi samples; int8 rows are TOPS, not TFLOP/s; peak = {SMS} SMs x ops/clk/SM x clock")
