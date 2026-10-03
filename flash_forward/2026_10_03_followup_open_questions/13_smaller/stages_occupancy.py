# stages_occupancy.py
# Item 13: why are 4 and 5 stages slower than 3 for v1's 64 x 32 tile? The article's guess is
# fewer programs per SM (4 -> 3 -> 2, because each stage adds 8 KB of shared memory). This
# separates the two effects by launching the 3-stage (and 4-stage) kernel with its dynamic shared
# memory padded to what 4 or 5 stages use: the same machine code, at the lower occupancy.
#
# Triton 3.6 passes the shared-memory size to cuLaunchKernelEx from CompiledKernel.packed_metadata
# and sets the function's max-dynamic-shared-memory attribute from metadata.shared when it loads
# the module, so a copy of the CompiledKernel with both replaced, loaded again, launches with
# more shared memory than the code uses.
#
#   python stages_occupancy.py time results/stages_timing.csv
#   python stages_occupancy.py profile        # one launch of each, for ncu's occupancy sections
import copy
import csv
import math
import pathlib
import random
import sys
import time

import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import jit_fn, load_version  # noqa: E402

B, H, D = 4, 8, 64
# (label, num_stages, padded shared bytes or None)
VARIANTS = [
    ("2 stages", 2, None),
    ("3 stages", 3, None),
    ("3 stages, padded to 4-stage smem", 3, 32768),
    ("3 stages, padded to 5-stage smem", 3, 40960),
    ("4 stages", 4, None),
    ("4 stages, padded to 5-stage smem", 4, 40960),
    ("5 stages", 5, None),
]


def build(mod, q, k, v, o, L, causal):
    fn = jit_fn(mod)
    N = q.shape[1]
    kernels = {}
    for label, ns, pad in VARIANTS:
        c = fn.warmup(q, k, v, o, L, *q.stride(), *k.stride(), *v.stride(), *o.stride(), *L.stride(),
                      N, N, 1.0 / math.sqrt(D), D=D, Q_TILE_SIZE=64, K_TILE_SIZE=32, is_causal=causal,
                      num_warps=4, num_stages=ns, grid=(1,))
        c._init_handles()
        natural = c.metadata.shared
        if pad is not None:
            assert pad >= natural
            c = copy.copy(c)
            c.metadata = c.metadata._replace(shared=pad)
            c.packed_metadata = (c.metadata.num_warps, c.metadata.num_ctas, pad)
            c.module = c.function = c._run = None
            c._init_handles()
        kernels[label] = (c, natural)
    args = (q, k, v, o, L, *q.stride(), *k.stride(), *v.stride(), *o.stride(), *L.stride(),
            N, N, 1.0 / math.sqrt(D), D, 64, 32, causal)
    grid = (triton.cdiv(N, 64), q.shape[0], 1)
    return kernels, (lambda c: c[grid](*args))


def time_all(out_path):
    mod = load_version("v1_tiles")
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "N", "causal", "variant", "natural_smem", "launch_smem", "regs", "ms", "tflops"])
        for N in (4096, 512):
            torch.manual_seed(0)
            q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
            o = torch.empty_like(q)
            L = torch.empty((B * H, N), device="cuda", dtype=torch.float32)
            kernels, run = build(mod, q, k, v, o, L, False)
            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            for label, (c, natural) in kernels.items():
                run(c)
                assert torch.allclose(o, ref, atol=2e-2, rtol=2e-2), label
            labels = list(kernels)
            for r in range(10):
                random.Random(r).shuffle(labels)
                for label in labels:
                    c, natural = kernels[label]
                    ms = triton.testing.do_bench(lambda: run(c), return_mode="median")
                    w.writerow([r, N, False, label, natural, c.metadata.shared, c.n_regs, f"{ms:.4f}",
                                f"{4 * B * H * N * N * D / ms * 1e-9:.2f}"])
                fh.flush()
            print(f"N={N} done", flush=True)


def profile():
    mod = load_version("v1_tiles")
    N = 4096
    q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o = torch.empty_like(q)
    L = torch.empty((B * H, N), device="cuda", dtype=torch.float32)
    kernels, run = build(mod, q, k, v, o, L, False)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        for c, _ in kernels.values():
            run(c)
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for label, (c, _) in kernels.items():
        torch.cuda.nvtx.range_push(label)
        run(c)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    if sys.argv[1] == "time":
        time_all(sys.argv[2])
    else:
        profile()
