# bitwise_check.py
# Is a new kernel version bitwise identical to v5? O and L are compared with torch.equal on the
# 8 harness shapes (B*H = 32, D = 64, N = 512..4096, both masks) and on ragged shapes (the test
# file's, plus a few more), in bf16 and fp32. Where they differ, the max |difference| against v5
# and against an fp32 reference (computed from the same rounded inputs) is printed for both.
#
#   python bitwise_check.py kernels/v6_mbarrier [--config 64,64,4,2,1]
import argparse
import math

import torch

from kload import load_module

p = argparse.ArgumentParser()
p.add_argument("kernel_dir")
p.add_argument("--ref", default="kernels/v5_tuned")
p.add_argument("--config", default=None, help="override the new version's config tuple")
args = p.parse_args()
new, ref = load_module(args.kernel_dir), load_module(args.ref)
if args.config:
    new.FlashAttentionCuTe.CONFIG = tuple(int(x) for x in args.config.split(","))

SHAPES = [(32, N, N, 64) for N in (512, 1024, 2048, 4096)] + [
    (4, 128, 128, 64), (2, 100, 100, 64), (1, 37, 53, 32), (1, 200, 96, 128),
    (3, 300, 300, 64), (2, 1000, 1000, 64), (1, 4097, 4097, 64), (2, 777, 1500, 64), (2, 65, 65, 128)]


def fp32_reference(q, k, v, causal):
    s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        s = s.masked_fill(~torch.ones(s.shape[-2:], dtype=torch.bool, device=q.device).tril(), float("-inf"))
    return torch.softmax(s, -1) @ v.float(), torch.logsumexp(s, -1)


def run(mod, q, k, v, c):
    o = mod.FlashAttentionCuTe.apply(q, k, v, c)
    L = [t for t in o.grad_fn.saved_tensors if t.shape == q.shape[:2]][0] if o.grad_fn else None
    return o, L


all_equal = True
for dtype in (torch.bfloat16, torch.float32):
    for B, Nq, Nk, D in SHAPES:
        for c in (False, True):
            if c and Nq != Nk:
                continue
            if dtype == torch.float32 and B * Nq * Nk > 32 * 1024 * 1024:
                continue                            # fp32 inputs: small shapes only (the tests' range)
            torch.manual_seed(B * 7 + Nq * 3 + Nk + D + c)
            q, k, v = (torch.randn(B, n, D, device="cuda", dtype=dtype, requires_grad=True)
                       for n in (Nq, Nk, Nk))
            o1, L1 = run(new, q, k, v, c)
            o0, L0 = run(ref, q, k, v, c)
            eq = torch.equal(o1, o0) and torch.equal(L1, L0)
            all_equal &= eq
            line = f"{str(dtype)[6:]:8s} B={B:2d} Nq={Nq:4d} Nk={Nk:4d} D={D:3d} causal={c!s:5}: " + \
                   ("bitwise identical" if eq else "DIFFERENT")
            if not eq:
                w = torch.float16 if dtype == torch.float32 else dtype   # what the kernels compute in
                ro, rL = fp32_reference(*(x.detach().to(w) for x in (q, k, v)), c)
                line += (f"  max|O-O_v5|={(o1.float() - o0.float()).abs().max().item():.3g}"
                         f"  max|L-L_v5|={(L1 - L0).abs().max().item():.3g}"
                         f"  vs fp32 ref: new O {(o1.float() - ro).abs().max().item():.3g} / v5 O "
                         f"{(o0.float() - ro).abs().max().item():.3g}, new L {(L1 - rL).abs().max().item():.3g}"
                         f" / v5 L {(L0 - rL).abs().max().item():.3g}")
            print(line, flush=True)
print("ALL BITWISE IDENTICAL" if all_equal else "NOT ALL IDENTICAL")
