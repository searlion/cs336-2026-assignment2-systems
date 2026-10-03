# pcast_accuracy.py
# Item 9: the P-cast accuracy comparison, on all 32 heads and three seeds instead of 2 heads
# and one seed, with each variant also run with an fp32 output buffer (test-only: the kernel
# casts O to the output's element type on store), to separate the rounding of P from the
# rounding of O. The floor is the error of rounding the exact float64 output to bf16.
#
#   python pcast_accuracy.py results.json
import json
import math
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import KERNELS, launch, load_version  # noqa: E402

BASE = (KERNELS / "v4_exp2" / "flashattention_autograd_function_triton.py").read_text()
VARIANTS = {  # the same three as the article's table, from variants_experiment.py
    "v4: sum fp32 P, cast, multiply": [],
    "cast first, sum the bf16 P": [(
        "        l_acc = alpha * l_acc + tl.sum(P_ij, axis=1, keep_dims=True)\n\n        P_ij = P_ij.to(V_j.dtype)\n",
        "        P_ij = P_ij.to(V_j.dtype)\n        l_acc = alpha * l_acc + tl.sum(P_ij.to(tl.float32), axis=1, keep_dims=True)\n\n")],
    "no cast: fp32 P, V upcast, TF32": [(
        "        P_ij = P_ij.to(V_j.dtype)\n        O_acc = alpha * O_acc\n        O_acc = tl.dot(P_ij, V_j, acc=O_acc)\n",
        "        O_acc = alpha * O_acc\n        O_acc = tl.dot(P_ij, V_j.to(tl.float32), acc=O_acc)\n")],
}
CONFIG = (64, 32, 4, 3)
B, H, D = 4, 8, 64


def load(name, edits):
    src = BASE
    for old, new in edits:
        assert src.count(old) == 1, (name, old)
        src = src.replace(old, new)
    d = HERE / "variants" / name.split(":")[0].replace(" ", "_").replace(",", "")
    d.mkdir(parents=True, exist_ok=True)
    (d / "flashattention_autograd_function_triton.py").write_text(src)
    return load_version(d.name, d / "flashattention_autograd_function_triton.py")


def reference(q, k, v, causal, chunk=4):
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    for i in range(0, q.shape[0], chunk):
        s = (q[i:i + chunk].double() @ k[i:i + chunk].double().transpose(-1, -2)) / math.sqrt(D)
        if causal:
            s.masked_fill_(~torch.ones(s.shape[-2:], dtype=torch.bool, device=q.device).tril(), float("-inf"))
        out[i:i + chunk] = torch.softmax(s, dim=-1) @ v[i:i + chunk].double()
    return out


def stats(err):
    return dict(mean=err.mean().item(), rms=err.pow(2).mean().sqrt().item(), max=err.max().item())


mods = {name: load(name, edits) for name, edits in VARIANTS.items()}
results = []
for N in (1024, 4096):
    for causal in (False, True):
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
            ref = reference(q, k, v, causal)
            floor = stats((ref.to(torch.bfloat16).double() - ref).abs())
            rec = dict(N=N, causal=causal, seed=seed, floor_bf16_rounding_of_exact_output=floor)
            for name, mod in mods.items():
                o_bf16, _ = launch(mod, q, k, v, causal, CONFIG)
                o_fp32, _ = launch(mod, q, k, v, causal, CONFIG,
                                   o=torch.empty(q.shape, device="cuda", dtype=torch.float32))
                rec[name] = dict(bf16_output=stats((o_bf16.double() - ref).abs()),
                                 fp32_output=stats((o_fp32.double() - ref).abs()),
                                 # how often the kernel's bf16 output differs from bf16(exact)
                                 frac_elements_not_equal_to_rounded_exact=(
                                     o_bf16 != ref.to(torch.bfloat16)).float().mean().item())
            results.append(rec)
            print(json.dumps(rec), flush=True)
json.dump(results, open(sys.argv[1], "w"), indent=1)
