# bench_nvidia_example.py
# Time NVIDIA's CuTe DSL Ampere FlashAttention-2 example (CUTLASS v4.8.1,
# examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py) with the same
# triton.testing.do_bench median as the harness, on the harness's shapes (B=4, H=8, D=64, bf16),
# next to SDPA. It is only a comparison point; none of its code is used in kernels/.
#
#   [POWER_JSON=out.json] python bench_nvidia_example.py /path/to/flash_attention_v2.py [m_block,n_block ...]
# With POWER_JSON set, NVML is sampled every 10 ms and each do_bench window is summarised.
import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F
import triton
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from power_sampler import Sampler, record_do_bench

windows = []
record_do_bench(windows)
sampler = Sampler().start() if os.environ.get("POWER_JSON") else None
power = {"device": sampler.info() if sampler else None, "shapes": []}

spec = importlib.util.spec_from_file_location("nvidia_fa2", sys.argv[1])
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)
blocks = [tuple(int(x) for x in b.split(",")) for b in sys.argv[2:]] or [(64, 64)]

B, H, D = 4, 8, 64


def wrap(t):  # torch (B, H, N, D) storage viewed as the example's (B, N, H, D), no copy
    t = t.permute(0, 2, 1, 3)
    return (from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=3, stride_order=t.dim_order(), divisibility=8))


stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
print(f"{'N':>6} {'causal':>7} {'block':>8} {'nvidia ms':>10} {'sdpa ms':>9} {'nvidia TF/s':>12} {'sdpa TF/s':>10}")
for N in (512, 1024, 2048, 4096):
    for causal in (False, True):
        q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        o = torch.empty_like(q)
        args = [wrap(x) for x in (q, k, v, o)] + [D ** -0.5, stream]
        flops = 4 * B * H * N * N * D / (2 if causal else 1)
        for bm, bn in blocks:
            fn = cute.compile(fa.FlashAttentionForwardAmpere(D, bm, bn, 128, causal), *args)
            fn(*args)
            torch.cuda.synchronize()
            err = (o.float() - F.scaled_dot_product_attention(q, k, v, is_causal=causal).float()).abs().max().item()
            assert err < 2e-2, err
            windows.clear()
            ms = triton.testing.do_bench(lambda: fn(*args), return_mode="median")
            ms_sdpa = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
                                              return_mode="median")
            if sampler:
                (o0, o1), (s0, s1) = windows
                power["shapes"].append({"N": N, "causal": causal, "block": f"{bm}x{bn}", "ours_ms": ms,
                                        "sdpa_ms": ms_sdpa, "ours": sampler.window(o0, o1),
                                        "sdpa": sampler.window(s0, s1)})
            print(f"{N:>6} {str(causal):>7} {f'{bm}x{bn}':>8} {ms:>10.3f} {ms_sdpa:>9.3f} "
                  f"{flops / ms * 1e-9:>12.1f} {flops / ms_sdpa * 1e-9:>10.1f}", flush=True)

if sampler:
    sampler.stop()
    json.dump(power, open(os.environ["POWER_JSON"], "w"), indent=1)
