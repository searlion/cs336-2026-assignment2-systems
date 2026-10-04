# fp16_acc_gap.py
# Why is the fp16-accumulate matmul about 1.55x as fast as the fp32-accumulate one, and not 2x,
# when its tensor-core peak per clock is twice as high (1,024 against 512 operations per clock
# per SM)? Four 8192^3 matmuls with fp16 inputs:
#   Triton fp16 -> fp32 acc and Triton fp16 -> fp16 acc: matmul_variants.py, each with its
#     fastest configuration (best_configs.json);
#   cuBLAS fp32 acc: torch.matmul;
#   cuBLAS fp16 acc: cublasGemmEx with CUBLAS_COMPUTE_16F, called through ctypes (PyTorch has no
#     fp16-accumulate matmul).
#   python fp16_acc_gap.py check       # max error of each against an fp32 matmul
#   python fp16_acc_gap.py profile     # one launch of each inside an NVTX range, for ncu
#   python fp16_acc_gap.py sustained   # each back to back for 4 s at the 220 W limit, with the clock
#   python fp16_acc_gap.py table reports/fp16_acc_gap.ncu-rep   # the comparison table from ncu
# The profile was taken with
#   ncu --profile-from-start off --nvtx --section SpeedOfLight --section LaunchStats --section Occupancy
#       --section WarpStateStats --section SchedulerStats --metrics <the tensor-path ops counters and
#       the totals printed by table()> -o reports/fp16_acc_gap python fp16_acc_gap.py profile
# Outputs: fp16_acc_gap_check.txt, fp16_acc_gap_ncu.txt, fp16_acc_gap_sustained.txt.
import csv
import ctypes
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
from matmul_variants import launch, make_inputs  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
M = N = K = 8192
SMS = 56
best = json.load(open(HERE / "best_configs.json"))

# cuBLAS through ctypes: the library PyTorch already loaded.
lib = next((pathlib.Path(torch.__file__).parent.parent / "nvidia").glob("cu*/lib/libcublas.so.*"))
cublas = ctypes.CDLL(str(lib))
cublas.cublasGemmEx.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 5 + [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
CUDA_R_16F, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT = 2, 64, -1
handle = ctypes.c_void_p()
assert cublas.cublasCreate_v2(ctypes.byref(handle)) == 0
assert cublas.cublasSetStream_v2(handle, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)) == 0
ONE, ZERO = ctypes.c_uint16(0x3C00), ctypes.c_uint16(0)  # 1.0 and 0.0 in fp16: alpha and beta are fp16 here


def cublas_fp16_acc(a, b, c):
    # Row-major C = A B is column-major C^T = B^T A^T, so B goes first.
    status = cublas.cublasGemmEx(handle, 0, 0, N, M, K, ctypes.byref(ONE), b.data_ptr(), CUDA_R_16F, N,
                                 a.data_ptr(), CUDA_R_16F, K, ctypes.byref(ZERO), c.data_ptr(), CUDA_R_16F, N,
                                 CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT)
    assert status == 0, status


a, b = make_inputs(torch.float16)
c32 = torch.empty(M, N, device="cuda", dtype=torch.float32)
c16 = torch.empty(M, N, device="cuda", dtype=torch.float16)
cfg32 = tuple(best["fp16_in_fp32_acc"]["config"])
cfg16 = tuple(best["fp16_in_fp16_acc"]["config"])
# (label, operations per clock per SM, function)
WORKLOADS = [
    (f"Triton fp16 -> fp32 acc {cfg32}", 512, lambda: launch(a, b, c32, "fp16_in_fp32_acc", cfg32)),
    (f"Triton fp16 -> fp16 acc {cfg16}", 1024, lambda: launch(a, b, c16, "fp16_in_fp16_acc", cfg16)),
    ("cuBLAS fp32 acc (torch.matmul)", 512, lambda: torch.matmul(a, b)),
    ("cuBLAS fp16 acc (CUBLAS_COMPUTE_16F)", 1024, lambda: cublas_fp16_acc(a, b, c16)),
]


def check():
    ref = a.float() @ b.float()
    for (label, _, fn), out in zip(WORKLOADS, [c32, c16, None, c16]):  # None: torch.matmul returns its result
        result = fn()
        torch.cuda.synchronize()
        out = result if out is None else out
        print(f"{label:44s} max abs error vs fp32 matmul {(out.float() - ref).abs().max().item():.4f}"
              f" (largest |value| {ref.abs().max().item():.2f})")


def profile():
    for _, _, fn in WORKLOADS:
        fn()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:  # clocks up before the capture
        for _, _, fn in WORKLOADS:
            fn()
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for label, _, fn in WORKLOADS:
        torch.cuda.nvtx.range_push(label)
        fn()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def sustained():
    for _, _, fn in WORKLOADS:
        fn()
    torch.cuda.synchronize()
    log = tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False)
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,clocks.sm,power.draw.instant",
                            "--format=csv,noheader,nounits", "-lms", "100"], stdout=log, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 2.0:  # warm the GPU up first
        torch.matmul(a, b)
        torch.cuda.synchronize()
    windows = []
    for label, rate, fn in WORKLOADS:
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
        ms = statistics.median(s.elapsed_time(e) for t, s, e in events if t > start + 1.0)
        windows.append((label, rate, ms, start + 1.0, time.time()))
        time.sleep(1.0)
    smi.terminate()
    smi.wait()
    samples = []
    for line in open(log.name):
        parts = [p.strip() for p in line.split(",")]
        try:
            ts = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            samples.append((ts, float(parts[1]), float(parts[2])))
        except (ValueError, IndexError):
            continue
    print(f"{'workload':44s} {'ms':>7s} {'TFLOP/s':>8s} {'clock MHz':>9s} {'W':>5s} {'peak at clock':>13s} {'% of it':>7s}")
    for label, rate, ms, t_a, t_b in windows:
        win = [s for s in samples if t_a <= s[0] <= t_b]
        clk = statistics.median(s[1] for s in win)
        watts = statistics.median(s[2] for s in win)
        tf = 2 * M * N * K / ms * 1e-9
        peak = SMS * rate * clk * 1e6 / 1e12
        print(f"{label:44s} {ms:7.3f} {tf:8.1f} {clk:9.0f} {watts:5.0f} {peak:13.1f} {100 * tf / peak:6.1f}%")


def table(report):
    ncu = "/opt/nvidia/nsight-compute/2026.2.1/ncu"
    out = subprocess.run([ncu, "--import", report, "--page", "raw", "--csv"], capture_output=True, text=True).stdout
    rows = list(csv.reader(out.splitlines()))
    header, kernels = rows[0], rows[2:]

    def v(r, k):
        return float(r[header.index(k)].replace(",", ""))

    labels = [label for label, _, _ in WORKLOADS]
    print("Nsight Compute, default --clock-control (clock locked at 2.52 GHz unless power pulls it lower)")
    for label, r in zip(labels, kernels):
        print(f"\n{label}: {r[header.index('Kernel Name')]}")
        path = "fp16" if v(r, "sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.sum") > 0 else "fp32"
        ops = f"sm__ops_path_tensor_src_fp16_dst_{path}_sparsity_off.sum"
        ms = v(r, "gpu__time_duration.sum")
        print(f"  {ms:.3f} ms = {2 * M * N * K / ms * 1e-9:.1f} TFLOP/s at {v(r, 'gpc__cycles_elapsed.avg.per_second'):.3f} GHz;"
              f" tensor ops {v(r, ops + '.pct_of_peak_sustained_elapsed'):.1f}% of the {v(r, ops + '.peak_sustained') / SMS:.0f}"
              f" ops/clk/SM peak (fp16 -> {path})")
        print(f"  {v(r, 'launch__registers_per_thread'):.0f} registers/thread, {v(r, 'launch__block_size'):.0f} threads/block,"
              f" {v(r, 'sm__warps_active.avg.per_cycle_active'):.1f} warps active per SM"
              f" ({v(r, 'sm__warps_active.avg.per_cycle_active') / 4:.1f} per scheduler),"
              f" issue slots busy {v(r, 'smsp__issue_active.avg.pct_of_peak_sustained_active'):.1f}%")
        print(f"  instructions {v(r, 'smsp__inst_executed.sum') / 1e6:.0f} M, L2 sectors {v(r, 'lts__t_sectors.sum') / 1e6:.0f} M,"
              f" shared-memory wavefronts {v(r, 'l1tex__data_pipe_lsu_wavefronts_mem_shared.sum') / 1e6:.0f} M,"
              f" DRAM read {v(r, 'dram__bytes_read.sum'):.2f} GB")
        stalls = sorted(((v(r, k), k) for k in header if k.startswith("smsp__average_warps_issue_stalled_")
                         and k.endswith("_per_issue_active.ratio")), reverse=True)[:5]
        print("  cycles stalled per issued instruction: " + ", ".join(
            f"{k.split('stalled_')[1].split('_per_issue')[0]} {x:.2f}" for x, k in stalls))


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table(sys.argv[2])
    else:
        {"check": check, "profile": profile, "sustained": sustained}[sys.argv[1]]()
