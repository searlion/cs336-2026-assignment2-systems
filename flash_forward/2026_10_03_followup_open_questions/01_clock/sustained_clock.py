# sustained_clock.py
# Item 1: what SM clock does the card hold under each workload, and what fraction of the
# tensor-core peak *at that clock* does each reach? Every workload runs back to back for 12 s
# while nvidia-smi samples the SM clock and power every 100 ms; the last 10 s of each window
# give the median clock and power, and CUDA events give the achieved TFLOP/s over the same time.
#
#   python sustained_clock.py results/sustained.json
import datetime
import json
import math
import pathlib
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import launch, load_version  # noqa: E402

B, H, N, D = 4, 8, 4096, 64
SMS, BF16_FP32_OPS_PER_CLK_PER_SM = 56, 512


def attn_flops(causal):
    return 4 * B * H * N * N * D / (2 if causal else 1)


q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
q4, k4, v4 = (t.view(B, H, N, D) for t in (q, k, v))
a, b = (torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16) for _ in range(2))
v4mod, v5mod = load_version("v4_exp2"), load_version("v5_final")

WORKLOADS = [  # (name, fn, flops per call)
    ("cublas bf16 8192^3", lambda: a @ b, 2 * 8192 ** 3),
    ("v5 (autotuned) non-causal", lambda: v5mod.FlashAttentionTriton.apply(q, k, v, False), attn_flops(False)),
    ("v4 64x128x4x2 non-causal", lambda: launch(v4mod, q, k, v, False, (64, 128, 4, 2)), attn_flops(False)),
    ("v4 64x32x4x3 non-causal", lambda: launch(v4mod, q, k, v, False, (64, 32, 4, 3)), attn_flops(False)),
    ("sdpa non-causal", lambda: F.scaled_dot_product_attention(q4, k4, v4), attn_flops(False)),
    ("v5 (autotuned) causal", lambda: v5mod.FlashAttentionTriton.apply(q, k, v, True), attn_flops(True)),
    ("sdpa causal", lambda: F.scaled_dot_product_attention(q4, k4, v4, is_causal=True), attn_flops(True)),
]
for _, fn, _ in WORKLOADS:  # compile and autotune outside the windows
    fn()
torch.cuda.synchronize()
chosen = {causal: str(v5mod.flash_fwd_kernel.cache) for causal in (False, True)}

smi = subprocess.Popen(
    ["nvidia-smi", "--query-gpu=timestamp,clocks.sm,power.draw.instant,temperature.gpu,clocks_event_reasons.active",
     "--format=csv,noheader,nounits", "-lms", "100"], stdout=subprocess.PIPE, text=True)
time.sleep(1.0)
windows = []
for name, fn, flops in WORKLOADS:
    t_start = datetime.datetime.now()
    t0 = time.perf_counter()
    n_calls, gpu_ms, measuring = 0, 0.0, False
    while time.perf_counter() - t0 < 12.0:
        if not measuring and time.perf_counter() - t0 > 2.0:
            measuring, t_meas = True, datetime.datetime.now()
            start_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        for _ in range(10):
            fn()
            n_calls += measuring
        torch.cuda.synchronize()
    end_ev = torch.cuda.Event(enable_timing=True)
    end_ev.record()
    torch.cuda.synchronize()
    gpu_ms = start_ev.elapsed_time(end_ev)
    windows.append(dict(name=name, meas_start=t_meas, end=datetime.datetime.now(),
                        tflops=flops * n_calls / gpu_ms * 1e-9, ms_per_call=gpu_ms / n_calls))
    time.sleep(3.0)  # a short gap, so windows do not run into each other
smi.terminate()
samples = []
for line in smi.stdout.read().splitlines():
    ts, clk, pw, temp, reasons = (x.strip() for x in line.split(","))
    samples.append((datetime.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f"), float(clk), float(pw), int(temp), reasons))

out = []
for w in windows:
    s = sorted(x for x in samples if w["meas_start"] <= x[0] <= w["end"])
    clks = sorted(x[1] for x in s)
    pws = sorted(x[2] for x in s)
    med_clk, med_pw = clks[len(clks) // 2], pws[len(pws) // 2]
    peak = SMS * BF16_FP32_OPS_PER_CLK_PER_SM * med_clk * 1e6 / 1e12
    rec = dict(name=w["name"], n_samples=len(s), sm_clock_mhz_median=med_clk, sm_clock_mhz_min=clks[0],
               sm_clock_mhz_max=clks[-1], power_w_median=med_pw, temp_c_end=s[-1][3],
               clock_event_reasons=sorted({x[4] for x in s}), ms_per_call=round(w["ms_per_call"], 4),
               tflops=round(w["tflops"], 1), bf16_fp32_peak_at_median_clock=round(peak, 1),
               pct_of_peak_at_median_clock=round(100 * w["tflops"] / peak, 1),
               pct_of_72_2=round(100 * w["tflops"] / 72.2, 1))
    out.append(rec)
    print(json.dumps(rec), flush=True)
json.dump(dict(windows=out, v5_autotune_cache=chosen), open(sys.argv[1], "w"), indent=1, default=str)
