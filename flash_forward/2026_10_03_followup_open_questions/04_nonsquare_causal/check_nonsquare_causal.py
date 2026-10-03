# check_nonsquare_causal.py
# Item 4: non-square causal attention and D = 16, which the unchanged test file does not cover.
# Every version is compared with a float64 reference that uses an explicit top-left mask
# (row i sees keys j <= i, PyTorch SDPA's is_causal convention), through the version's own
# FlashAttentionTriton.apply and, for the two-loop kernels, through explicit tile configs too,
# because the loop bounds depend on how Q_TILE_SIZE and K_TILE_SIZE relate.
#
#   python check_nonsquare_causal.py results.csv
import csv
import pathlib
import sys

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import VERSIONS, launch, load_version, reference  # noqa: E402

SHAPES = [  # (batch, n_queries, n_keys, head_dim)
    (1, 200, 96, 64), (1, 96, 200, 64), (1, 37, 53, 32), (1, 53, 37, 32), (2, 100, 100, 16),
    (2, 64, 128, 16), (2, 128, 64, 16), (1, 1, 77, 64), (1, 77, 1, 64), (1, 130, 257, 64),
    (1, 257, 130, 64), (1, 333, 1000, 64), (1, 1000, 333, 64), (1, 200, 96, 128), (1, 96, 200, 128),
    (1, 128, 128, 64),  # square, as a control
]
# Explicit configs for the two-loop kernels: key tile smaller than, equal to and larger than
# the query tile, including all of v5's autotune list.
TWO_LOOP_CONFIGS = [(64, 32, 4, 3), (64, 32, 4, 4), (64, 32, 4, 5), (64, 128, 4, 2), (64, 32, 4, 2),
                    (64, 64, 4, 3), (128, 64, 4, 2), (16, 64, 1, 3), (32, 128, 2, 2), (128, 32, 8, 3)]
ONE_LOOP_CONFIGS = [(16, 16, 4, 3), (64, 32, 4, 3), (64, 16, 4, 5)]
TOL = {torch.bfloat16: dict(atol=2e-2, rtol=2e-2), torch.float32: dict(atol=1e-2, rtol=1e-2)}


def check(o, L, ref_o, ref_L, dtype):
    ok_o = torch.allclose(o.double(), ref_o, **TOL[dtype])
    ok_L = torch.allclose(L.double(), ref_L, **TOL[dtype])
    return ok_o and ok_L, (o.double() - ref_o).abs().max().item(), (L.double() - ref_L).abs().max().item()


def main():
    out = csv.writer(open(sys.argv[1], "w", newline=""))
    out.writerow(["version", "path", "config", "B", "Nq", "Nk", "D", "dtype", "causal", "status", "max_err_O", "max_err_L"])
    fails = []

    # Which convention does SDPA use? Compare it with both references on the non-square shapes.
    print("SDPA is_causal=True against the two conventions (max |error|):")
    for B, Nq, Nk, D in SHAPES:
        if Nq == Nk:
            continue
        torch.manual_seed(0)
        q, k, v = (torch.randn(B, n, D, device="cuda") for n in (Nq, Nk, Nk))
        o = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True)
        tl_o, _ = reference(q, k, v, True, "top-left")
        br_o, _ = reference(q, k, v, True, "bottom-right")
        print(f"  {(B, Nq, Nk, D)}: vs top-left {(o - tl_o).abs().max().item():.1e}, "
              f"vs bottom-right {(o - br_o).nan_to_num(0).abs().max().item():.1e}")

    for name in VERSIONS:
        mod = load_version(name)
        two_loop = hasattr(mod, "_attend_key_tiles")
        configs = TWO_LOOP_CONFIGS if two_loop else ONE_LOOP_CONFIGS
        n_ok = n_all = 0
        for B, Nq, Nk, D in SHAPES:
            for dtype in (torch.bfloat16, torch.float32):
                for causal in (True, False):
                    torch.manual_seed(0)
                    q = torch.randn(B, Nq, D, device="cuda", dtype=dtype, requires_grad=True)
                    k = torch.randn(B, Nk, D, device="cuda", dtype=dtype, requires_grad=True)
                    v = torch.randn(B, Nk, D, device="cuda", dtype=dtype, requires_grad=True)
                    ref_o, ref_L = reference(q.detach(), k.detach(), v.detach(), causal)
                    runs = [("apply", "default")] + [("config", c) for c in configs]
                    for path, cfg in runs:
                        try:
                            if path == "apply":
                                o = mod.FlashAttentionTriton.apply(q, k, v, causal)
                                L = [t for t in o.grad_fn.saved_tensors if t.shape == (B, Nq)][0]
                            else:
                                o, L = launch(mod, q.detach(), k.detach(), v.detach(), causal, cfg)
                            torch.cuda.synchronize()
                        except triton.runtime.errors.OutOfResources:
                            out.writerow([name, path, cfg, B, Nq, Nk, D, str(dtype)[6:], causal, "out_of_resources", "", ""])
                            continue
                        ok, eo, el = check(o.detach(), L, ref_o, ref_L, dtype)
                        n_all += 1
                        n_ok += ok
                        out.writerow([name, path, cfg, B, Nq, Nk, D, str(dtype)[6:], causal,
                                      "pass" if ok else "FAIL", f"{eo:.2e}", f"{el:.2e}"])
                        if not ok:
                            fails.append((name, path, cfg, (B, Nq, Nk, D), str(dtype)[6:], causal, eo, el))
        print(f"{name}: {n_ok}/{n_all} cases pass")
    print(f"{len(fails)} failures")
    for f in fails:
        print("  FAIL", f)


if __name__ == "__main__":
    main()
