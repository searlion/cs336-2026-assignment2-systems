# variants_experiment.py
# One-line variants of the v4 inner loop, timed like the harness and checked for accuracy
# against a float64 reference.
#
#   python variants_experiment.py <scratch dir> results/variants.json
import importlib.util
import json
import math
import pathlib
import sys

import torch
import triton

SCRATCH = pathlib.Path(sys.argv[1])
BASE = pathlib.Path("kernels/v4_exp2/flashattention_autograd_function_triton.py").read_text()

VARIANTS = {
    "v4 (P cast to bf16 after the row sum)": [],
    "no P cast: fp32 P, V upcast, TF32 dot": [(
        "        P_ij = P_ij.to(V_j.dtype)\n        O_acc = alpha * O_acc\n        O_acc = tl.dot(P_ij, V_j, acc=O_acc)\n",
        "        O_acc = alpha * O_acc\n        O_acc = tl.dot(P_ij, V_j.to(tl.float32), acc=O_acc)\n")],
    "P cast before the row sum": [(
        "        l_acc = alpha * l_acc + tl.sum(P_ij, axis=1, keep_dims=True)\n\n        P_ij = P_ij.to(V_j.dtype)\n",
        "        P_ij = P_ij.to(V_j.dtype)\n        l_acc = alpha * l_acc + tl.sum(P_ij.to(tl.float32), axis=1, keep_dims=True)\n\n")],
    "reverse query-tile order": [(
        "    query_tile_index = tl.program_id(0)\n",
        "    # Start the longest causal rows first so the last wave is short.\n"
        "    query_tile_index = tl.num_programs(0) - 1 - tl.program_id(0)\n")],
}


def load(name, src):
    d = SCRATCH / name.replace(" ", "_").replace(":", "").replace(",", "").replace("(", "").replace(")", "")
    d.mkdir(parents=True, exist_ok=True)
    (d / "flashattention_autograd_function_triton.py").write_text(src)
    spec = importlib.util.spec_from_file_location("variant_" + d.name, d / "flashattention_autograd_function_triton.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.FlashAttentionTriton


def reference(q, k, v, causal):
    s = (q.double() @ k.double().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        s = s.masked_fill(~torch.ones(s.shape[-2:], dtype=torch.bool, device=q.device).tril(), float("-inf"))
    return torch.softmax(s, dim=-1) @ v.double()


results = []
B, H, D = 4, 8, 64
for name, edits in VARIANTS.items():
    src = BASE
    for old, new in edits:
        assert src.count(old) == 1, (name, old)
        src = src.replace(old, new)
    fa = load(name, src)
    for N in (1024, 4096):
        for causal in (False, True):
            torch.manual_seed(0)
            q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
            o = fa.apply(q, k, v, causal)
            ref = reference(q[:2], k[:2], v[:2], causal)  # two heads are enough for the error
            err = (o[:2].double() - ref).abs()
            ms = triton.testing.do_bench(lambda: fa.apply(q, k, v, causal), return_mode="median")
            flops = 4 * B * H * N * N * D / (2 if causal else 1)
            rec = dict(variant=name, N=N, causal=causal, ms=round(ms, 4),
                       tflops=round(flops / ms * 1e-9, 1), max_err=float(err.max()), mean_err=float(err.mean()))
            results.append(rec)
            print(rec, flush=True)
json.dump(results, open(sys.argv[2], "w"), indent=1)
