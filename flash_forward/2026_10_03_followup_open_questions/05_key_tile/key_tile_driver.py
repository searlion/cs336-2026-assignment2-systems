# key_tile_driver.py
# Items 5 and 8: why do 128-key tiles win after Round 4? The same kernel (v4) and the same
# mask with 32-, 64- and 128-key tiles and 2 or 3 stages, plus an ablation that removes the
# `O_acc = alpha * O_acc` rescale (numerically wrong, timing only), so that the rescale's
# share of the difference between tile shapes can be measured rather than inferred.
#
#   python key_tile_driver.py time results/timing.csv    # 10 interleaved rounds of do_bench
#   python key_tile_driver.py profile                    # one launch per config, for ncu
import csv
import pathlib
import random
import sys
import time

import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import KERNELS, launch, load_version  # noqa: E402

CONFIGS = [(64, 32, 4, 3), (64, 32, 4, 2), (64, 64, 4, 3), (64, 64, 4, 2), (64, 128, 4, 2), (64, 128, 4, 3)]
B, H, N, D = 4, 8, 4096, 64


def no_rescale_variant():
    src = (KERNELS / "v4_exp2" / "flashattention_autograd_function_triton.py").read_text()
    old = "        O_acc = alpha * O_acc\n"
    assert src.count(old) == 1
    d = HERE / "variants" / "v4_no_rescale"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "flashattention_autograd_function_triton.py"
    path.write_text(src.replace(old, "        # (ablation) no O_acc rescale: wrong results, timing only\n"))
    return load_version("v4_no_rescale", path)


def time_all(out_path, rounds=10):
    kernels = {"v4": load_version("v4_exp2"), "v4_no_rescale": no_rescale_variant()}
    q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    jobs = [(name, cfg, causal) for name in kernels for cfg in CONFIGS for causal in (False, True)]
    for name, cfg, causal in jobs:  # compile everything first
        launch(kernels[name], q, k, v, causal, cfg)
    torch.cuda.synchronize()
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "kernel", "q_tile", "k_tile", "num_warps", "num_stages", "causal", "ms", "tflops"])
        for r in range(rounds):
            random.Random(r).shuffle(jobs)
            for name, cfg, causal in jobs:
                ms = triton.testing.do_bench(lambda: launch(kernels[name], q, k, v, causal, cfg),
                                             return_mode="median")
                flops = 4 * B * H * N * N * D / (2 if causal else 1)
                w.writerow([r, name, *cfg, causal, f"{ms:.4f}", f"{flops / ms * 1e-9:.2f}"])
            fh.flush()
            print(f"round {r} done", flush=True)


def profile():
    mod = load_version("v4_exp2")
    q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    runs = [(cfg, causal) for causal in (False, True) for cfg in CONFIGS]
    for cfg, causal in runs:
        launch(mod, q, k, v, causal, cfg)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        for cfg, causal in runs:
            launch(mod, q, k, v, causal, cfg)
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for cfg, causal in runs:
        torch.cuda.nvtx.range_push("x".join(map(str, cfg)) + f" causal={causal}")
        launch(mod, q, k, v, causal, cfg)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    if sys.argv[1] == "time":
        time_all(sys.argv[2])
    else:
        profile()
