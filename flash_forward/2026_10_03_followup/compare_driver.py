# compare_driver.py
# Run several kernel versions (and SDPA) back to back in one process, each inside its own
# NVTX range, so a single nsys timeline or ncu report holds all of them side by side.
#
#   python compare_driver.py --N 4096 --causal 0 kernels/v0_baseline kernels/v1_tiles sdpa
import argparse
import importlib.util
import pathlib
import time

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("versions", nargs="+", help="kernel directories, or 'sdpa'")
parser.add_argument("--N", type=int, default=4096)
parser.add_argument("--causal", type=int, nargs="+", default=[0])
parser.add_argument("--iters", type=int, default=3)
args = parser.parse_args()


def load(kernel_dir):
    path = pathlib.Path(kernel_dir) / "flashattention_autograd_function_triton.py"
    spec = importlib.util.spec_from_file_location(pathlib.Path(kernel_dir).name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlashAttentionTriton


B, H, D = 4, 8, 64
q, k, v = (torch.randn(B * H, args.N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
q4, k4, v4 = (t.view(B, H, args.N, D) for t in (q, k, v))

runs = []
for name in args.versions:
    if name == "sdpa":
        fn = lambda causal: F.scaled_dot_product_attention(q4, k4, v4, is_causal=causal)  # noqa: E731
    else:
        fa = load(name)
        fn = (lambda fa: lambda causal: fa.apply(q, k, v, causal))(fa)
    label = pathlib.Path(name).name
    for causal in args.causal:
        fn(bool(causal))  # compile / autotune outside the measured region
        runs.append((f"{label} causal={bool(causal)}", fn, bool(causal)))
torch.cuda.synchronize()

# Keep the GPU busy for about a second so its clocks have ramped up before the capture.
t0 = time.perf_counter()
while time.perf_counter() - t0 < 1.0:
    for _, fn, causal in runs:
        fn(causal)
    torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStart()
for label, fn, causal in runs:
    for _ in range(args.iters):
        torch.cuda.nvtx.range_push(label)
        fn(causal)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
