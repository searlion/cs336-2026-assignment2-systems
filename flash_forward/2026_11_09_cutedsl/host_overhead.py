# host_overhead.py
# Where the CPU time of one FlashAttentionCuTe.apply call goes, at N = 512 (where the kernel is
# shortest). Times each host-side step of v5's forward on its own, 2000 times, median in µs,
# and the whole call for v5, the Triton kernel and SDPA.
#
#   python host_overhead.py > results/host_overhead.txt
import math
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import cutlass
import cuda.bindings.driver as cuda

from compare_driver_load import load

sys.path.insert(0, "kernels/v5_tuned")
import flashattention_autograd_function_triton as mod  # noqa: E402
cute_fa = mod.FlashAttentionCuTe
triton_fa = load("../2026_10_03_followup/kernels/v5_final")

B, N, D = 32, 512, 64
q, k, v = (torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
dt = cutlass.BFloat16


def t(fn, n=2000):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
        if len(xs) % 100 == 0:
            torch.cuda.synchronize()        # keep the launch queue from filling up
    torch.cuda.synchronize()
    return statistics.median(xs) * 1e6


O = torch.empty_like(q)
L = torch.empty(B, N, device="cuda", dtype=torch.float32)
cute_fa.apply(q, k, v, False)
fn = next(iter(cute_fa._compiled.values()))
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
args = (mod._ptr(q, dt), mod._ptr(k, dt), mod._ptr(v, dt), mod._ptr(O, dt), mod._ptr(L, cutlass.Float32),
        B, N, N, (1.0 / math.sqrt(D)) * mod.LOG2E, stream)
q4, k4, v4 = (x.view(4, 8, N, D) for x in (q, k, v))
rows = [
    ("CuTe v5: whole FlashAttentionCuTe.apply", lambda: cute_fa.apply(q, k, v, False)),
    ("  torch.empty for O and L", lambda: (torch.empty_like(q), torch.empty(B, N, device="cuda", dtype=torch.float32))),
    ("  5 x make_ptr", lambda: [mod._ptr(x, dt) for x in (q, k, v, O, L)]),
    ("  cuda.CUstream(current stream)", lambda: cuda.CUstream(torch.cuda.current_stream().cuda_stream)),
    ("  compiled kernel call (prebuilt arguments)", lambda: fn(*args)),
    ("Triton v5: whole FlashAttentionTriton.apply", lambda: triton_fa.apply(q, k, v, False)),
    ("SDPA", lambda: F.scaled_dot_product_attention(q4, k4, v4)),
]
print(f"median CPU time per call over 2000 calls, N = {N}, B*H = {B}, D = {D} (µs)")
for name, f in rows:
    print(f"{name:45s} {t(f):7.1f}")
