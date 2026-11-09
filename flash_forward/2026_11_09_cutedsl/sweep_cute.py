# sweep_cute.py
# CuTe DSL has no autotuner, so this does its job by hand for kernels/v5_tuned: every
# (bM, bN, warps, stages) that fits in 99 KB of shared memory is checked for correctness on a
# ragged shape, then timed with triton.testing.do_bench (median) on the 8 harness shapes.
#
#   python sweep_cute.py kernels/v5_tuned results/sweep_v5.csv
import csv
import itertools
import math
import sys
import time

import torch
import torch.nn.functional as F
import triton
import cutlass.utils

sys.path.insert(0, sys.argv[1])
from flashattention_autograd_function_triton import FlashAttentionCuTe  # noqa: E402

SMEM_LIMIT = cutlass.utils.get_smem_capacity_in_bytes("sm_89")     # 101,376 bytes per block
B, H, D = 4, 8, 64


def smem_bytes(bM, bN, stages, D=64, elem=2):
    # Q, plus stages buffers each of K and V; each buffer starts on a 1 KB boundary.
    pad = lambda n: (n + 1023) // 1024 * 1024  # noqa: E731
    return pad(bM * D * elem) + 2 * pad(stages * bN * D * elem)


def configs():
    for bM, bN, warps, stages in itertools.product((64, 128), (32, 64, 128), (4, 8), (1, 2, 3)):
        if bM % (warps * 16) or smem_bytes(bM, bN, stages) > SMEM_LIMIT:
            continue
        yield bM, bN, warps, stages


def correct(cfg):
    FlashAttentionCuTe.CONFIG = cfg
    torch.manual_seed(0)
    for causal in (False, True):
        q, k, v = (torch.randn(2, 300, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        o = FlashAttentionCuTe.apply(q, k, v, causal)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        if not torch.allclose(o.float(), ref.float(), atol=2e-2, rtol=2e-2):
            return False
    return True


def main():
    out = open(sys.argv[2], "w", newline="")
    w = csv.writer(out)
    w.writerow(["bM", "bN", "warps", "stages", "smem_bytes", "N", "causal", "ms", "tflops"])
    tensors = {N: [torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
               for N in (512, 1024, 2048, 4096)}
    # Warm the GPU so the first configurations are not timed at a lower clock.
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        F.scaled_dot_product_attention(*(t.view(B, H, 4096, D) for t in tensors[4096]))
        torch.cuda.synchronize()
    start = time.time()
    for cfg in configs():
        try:
            ok = correct(cfg)
        except Exception as e:  # noqa: BLE001  (a config the compiler rejects)
            print(cfg, "failed to compile:", str(e).splitlines()[0][:120], flush=True)
            continue
        if not ok:
            print(cfg, "WRONG RESULT, skipped", flush=True)
            continue
        FlashAttentionCuTe.CONFIG = cfg
        row = []
        for N in (512, 1024, 2048, 4096):
            q, k, v = tensors[N]
            for causal in (False, True):
                ms = triton.testing.do_bench(lambda: FlashAttentionCuTe.apply(q, k, v, causal), return_mode="median")
                flops = 4 * B * H * N * N * D / (2 if causal else 1)
                w.writerow([*cfg, smem_bytes(*cfg[:2], cfg[3]), N, int(causal), f"{ms:.4f}", f"{flops / ms * 1e-9:.2f}"])
                row.append(f"{flops / ms * 1e-9:5.1f}")
        out.flush()
        print(cfg, " ".join(row), flush=True)
    print(f"sweep took {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
