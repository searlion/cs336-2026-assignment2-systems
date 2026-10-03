# matmul_variants.py
# One Triton matmul with six input/accumulator types, to see how Nsight Compute's tensor-pipe
# utilisation ("SM: Pipe Tensor Cycles Active") relates to the per-type operation counters
# that the tensor-core roofline divides by its peak.
#
#   python matmul_variants.py time      # pick the fastest tile config per variant, report TFLOP/s
#   python matmul_variants.py profile   # one launch of each variant's best config, for ncu
import json
import pathlib
import sys
import time

import torch
import triton
import triton.language as tl

HERE = pathlib.Path(__file__).resolve().parent
M = N = K = 8192


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
                  ACC: tl.constexpr, INPUT_PRECISION: tl.constexpr):
    # Grouped program order, so neighbouring programs share tiles of A in L2.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    group = GROUP_M * num_pid_n
    first_m = (pid // group) * GROUP_M
    size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m
    a_bp = tl.make_block_ptr(a_ptr, (M, K), (stride_am, stride_ak), (pid_m * BM, 0), (BM, BK), (1, 0))
    b_bp = tl.make_block_ptr(b_ptr, (K, N), (stride_bk, stride_bn), (0, pid_n * BN), (BK, BN), (1, 0))
    acc = tl.zeros((BM, BN), dtype=ACC)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(a_bp), tl.load(b_bp), acc, input_precision=INPUT_PRECISION, out_dtype=ACC)
        a_bp = tl.advance(a_bp, (0, BK))
        b_bp = tl.advance(b_bp, (BK, 0))
    c_bp = tl.make_block_ptr(c_ptr, (M, N), (stride_cm, stride_cn), (pid_m * BM, pid_n * BN), (BM, BN), (1, 0))
    tl.store(c_bp, acc.to(c_ptr.dtype.element_ty))


# name: (input dtype, accumulator, output dtype, input_precision, ncu ops counter for this path)
VARIANTS = {
    "bf16_in_fp32_acc": (torch.bfloat16, tl.float32, torch.float32, "tf32", "sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off"),
    "fp16_in_fp32_acc": (torch.float16, tl.float32, torch.float32, "tf32", "sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off"),
    "fp16_in_fp16_acc": (torch.float16, tl.float16, torch.float16, "tf32", "sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off"),
    "tf32_in_fp32_acc": (torch.float32, tl.float32, torch.float32, "tf32", "sm__ops_path_tensor_src_tf32_dst_fp32_sparsity_off"),
    "int8_in_int32_acc": (torch.int8, tl.int32, torch.int32, "tf32", "sm__ops_path_tensor_src_int8_sparsity_off"),
    "fp8e4m3_in_fp32_acc": (torch.float8_e4m3fn, tl.float32, torch.float32, "tf32", "sm__ops_path_tensor_src_fp8_sparsity_off"),
}
# (BM, BN, BK, num_warps, num_stages); BK is in elements, so 8-bit inputs get a longer one.
CONFIGS = [(128, 128, 32, 4, 4), (128, 128, 32, 8, 4), (128, 128, 64, 8, 3), (128, 256, 32, 8, 3),
           (256, 128, 32, 8, 3), (128, 128, 64, 4, 3), (128, 256, 64, 8, 3), (128, 128, 128, 8, 3),
           (64, 128, 32, 4, 4), (128, 64, 64, 4, 4)]


def make_inputs(dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    if dtype == torch.int8:
        a = torch.randint(-4, 4, (M, K), device="cuda", dtype=torch.int8, generator=g)
        b = torch.randint(-4, 4, (K, N), device="cuda", dtype=torch.int8, generator=g)
    else:
        a = (torch.randn(M, K, device="cuda", generator=g) * 0.1).to(dtype)
        b = (torch.randn(K, N, device="cuda", generator=g) * 0.1).to(dtype)
    return a, b


def launch(a, b, c, variant, cfg):
    _, acc, _, prec, _ = VARIANTS[variant]
    bm, bn, bk, nw, ns = cfg
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    matmul_kernel[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(),
                        BM=bm, BN=bn, BK=bk, GROUP_M=8, ACC=acc, INPUT_PRECISION=prec,
                        num_warps=nw, num_stages=ns)


def time_all():
    results = {}
    for variant, (dt, _, out_dt, _, _) in VARIANTS.items():
        a, b = make_inputs(dt)
        c = torch.empty(M, N, device="cuda", dtype=out_dt)
        best = None
        for cfg in CONFIGS:
            try:
                launch(a, b, c, variant, cfg)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001  (out of shared memory, unsupported, ...)
                print(f"{variant} {cfg}: {type(e).__name__}", flush=True)
                continue
            ms = triton.testing.do_bench(lambda: launch(a, b, c, variant, cfg), warmup=100, rep=500,
                                         return_mode="median")
            tf = 2 * M * N * K / ms * 1e-9
            print(f"{variant} {cfg}: {ms:.3f} ms {tf:.1f} TFLOP/s", flush=True)
            if best is None or ms < best[1]:
                best = (cfg, ms, tf)
        # Sanity check of the result against torch (exact for int8, loose for the floats).
        ref = (a.float() @ b.float()) if dt != torch.int8 else (a.int().float() @ b.int().float())
        launch(a, b, c, variant, best[0])
        err = (c.float() - ref).abs().max().item()
        results[variant] = dict(config=best[0], ms=round(best[1], 4), tflops=round(best[2], 1),
                                max_abs_err_vs_fp32_matmul=err, ref_abs_max=ref.abs().max().item())
        print(variant, results[variant], flush=True)
    # cuBLAS for reference: bf16 (fp32 accumulation) and fp16 with reduced-precision reduction allowed.
    a, b = make_inputs(torch.bfloat16)
    ms = triton.testing.do_bench(lambda: a @ b, warmup=100, rep=500, return_mode="median")
    results["cublas_bf16"] = dict(ms=round(ms, 4), tflops=round(2 * M * N * K / ms * 1e-9, 1))
    json.dump(results, open(HERE / "best_configs.json", "w"), indent=1)
    print(json.dumps(results, indent=1))


def profile():
    best = json.load(open(HERE / "best_configs.json"))
    runs = []
    for variant, (dt, _, out_dt, _, _) in VARIANTS.items():
        a, b = make_inputs(dt)
        c = torch.empty(M, N, device="cuda", dtype=out_dt)
        cfg = tuple(best[variant]["config"])
        launch(a, b, c, variant, cfg)
        runs.append((variant, a, b, c, cfg))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:  # clocks up before the capture
        for variant, a, b, c, cfg in runs:
            launch(a, b, c, variant, cfg)
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for variant, a, b, c, cfg in runs:
        torch.cuda.nvtx.range_push(variant)
        launch(a, b, c, variant, cfg)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    {"time": time_all, "profile": profile}[sys.argv[1]]()
