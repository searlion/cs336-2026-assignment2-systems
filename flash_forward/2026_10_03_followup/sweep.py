# sweep.py
# Sweep Q_TILE_SIZE x K_TILE_SIZE x num_warps x num_stages for the unchanged kernel,
# timing each configuration exactly the way bench_flashattention.py does:
# FlashAttentionTriton.apply under triton.testing.do_bench, median, same shapes.
#
#   python sweep.py kernels/v1_tiles results/sweep_v1.csv
import csv
import itertools
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn.functional as F
import triton

KERNEL_DIR = sys.argv[1]
OUT = sys.argv[2]
sys.path.insert(0, KERNEL_DIR)
import flashattention_autograd_function_triton as mod  # noqa: E402

TILES = (16, 32, 64, 128, 256)
WARPS = (1, 2, 4, 8, 16)
STAGES = (1, 2, 3, 4, 5)
NS = (512, 1024, 2048, 4096)
B, H, D = 4, 8, 64
CONFIGS = list(itertools.product(TILES, TILES, WARPS, STAGES))
if os.environ.get("SWEEP_SMOKE"):  # a handful of configs, to check the script itself
    CONFIGS, NS = CONFIGS[::61], (512,)


def fp32_regs_per_thread(bq, bk, nw):
    # The S tile and the O accumulator are fp32 and spread over 32 * num_warps threads.
    return (bq * bk + bq * D) / (32 * nw)


def launch(q, k, v, causal, bq, bk, nw, ns, warmup=False):
    """The same launch as FlashAttentionTriton.forward, returning the compiled kernel."""
    o = torch.empty_like(q)
    L = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    N = q.shape[1]
    fn = mod.flash_fwd_kernel.warmup if warmup else mod.flash_fwd_kernel[(triton.cdiv(N, bq), q.shape[0])]
    extra = dict(grid=(1,)) if warmup else {}
    return fn(q, k, v, o, L,
              *q.stride(), *k.stride(), *v.stride(), *o.stride(), *L.stride(),
              N, N, 1.0 / math.sqrt(D),
              D=D, Q_TILE_SIZE=bq, K_TILE_SIZE=bk, is_causal=causal,
              num_warps=nw, num_stages=ns, **extra)


def precompile(job):
    """Runs in a worker process: compile into the on-disk Triton cache, do not launch."""
    (bq, bk, nw, ns), causal = job
    q = torch.empty(1, 4096, D, device="cuda", dtype=torch.bfloat16)
    t0 = time.perf_counter()
    try:
        launch(q, q, q, causal, bq, bk, nw, ns, warmup=True)
        return job, time.perf_counter() - t0, ""
    except Exception as e:  # noqa: BLE001
        return job, time.perf_counter() - t0, type(e).__name__


def main():
    jobs = [(c, causal) for c in CONFIGS for causal in (False, True)
            if fp32_regs_per_thread(c[0], c[1], c[2]) <= 512]
    print(f"{len(CONFIGS)} configs, {len(jobs)} (config, causal) compiles after the register filter")
    t0 = time.perf_counter()
    compile_s = {}
    with ProcessPoolExecutor(max_workers=6, mp_context=torch.multiprocessing.get_context("spawn")) as ex:
        for job, secs, err in ex.map(precompile, jobs):
            compile_s[job] = (secs, err)
    print(f"precompiled in {time.perf_counter() - t0:.0f} s")

    fields = ["N", "causal", "q_tile", "k_tile", "num_warps", "num_stages", "status",
              "ms", "tflops", "regs", "spills", "smem", "compile_s"]
    new_file = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            w.writeheader()
        for causal in (False, True):
            for N in NS:
                torch.manual_seed(0)
                q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
                ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                flops = 4 * B * H * N * N * D / (2 if causal else 1)
                for bq, bk, nw, ns in CONFIGS:
                    row = dict(N=N, causal=causal, q_tile=bq, k_tile=bk, num_warps=nw, num_stages=ns)
                    secs, err = compile_s.get(((bq, bk, nw, ns), causal), (0.0, "skipped_regs"))
                    row["compile_s"] = f"{secs:.2f}"
                    if err:
                        w.writerow(row | dict(status=err))
                        continue
                    try:
                        ck = launch(q, k, v, causal, bq, bk, nw, ns)
                    except triton.runtime.errors.OutOfResources as e:
                        w.writerow(row | dict(status="out_of_smem" if "shared" in str(e) else "out_of_resources"))
                        continue
                    row.update(regs=ck.n_regs, spills=ck.n_spills, smem=ck.metadata.shared)
                    mod.FlashAttentionTriton.Q_TILE_SIZE, mod.FlashAttentionTriton.K_TILE_SIZE = bq, bk
                    mod.FlashAttentionTriton.NUM_WARPS, mod.FlashAttentionTriton.NUM_STAGES = nw, ns
                    o = mod.FlashAttentionTriton.apply(q, k, v, causal)
                    if not torch.allclose(o, ref, atol=2e-2, rtol=2e-2):
                        w.writerow(row | dict(status="wrong"))
                        continue
                    ms = triton.testing.do_bench(
                        lambda: mod.FlashAttentionTriton.apply(q, k, v, causal), return_mode="median")
                    w.writerow(row | dict(status="ok", ms=f"{ms:.4f}", tflops=f"{flops / ms * 1e-9:.2f}"))
                fh.flush()
                print(f"done N={N} causal={causal} at {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
