# probe.py: does tl.range(..., warp_specialize=True) do anything on sm_89 (Triton 3.6.0)?
# Compiles kernel/ (Triton v5 with only that flag added to the key-tile loop) at 64 x 128, 2 stages,
# and compares the TTGIR and SASS with the unchanged kernel, and checks the result.
import sys, math, re, torch, triton
sys.path.insert(0, "../..")
from kload import load_module
import torch.nn.functional as F
out = []
for name, d in [("v5_final", "../../../2026_10_03_followup/kernels/v5_final"), ("tl.range, no flag", "kernel_range"), ("tl.range, warp_specialize=True", "kernel")]:
    m = load_module(d)
    kern = m.flash_fwd_kernel.fn
    B, N, D = 32, 4096, 64
    q, k, v = (torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o = torch.empty_like(q); L = torch.empty(B, N, device="cuda", dtype=torch.float32)
    c = kern.warmup(q, k, v, o, L, *q.stride(), *k.stride(), *v.stride(), *o.stride(), *L.stride(), N, N,
                    1 / math.sqrt(D), D=D, Q_TILE_SIZE=64, K_TILE_SIZE=128, is_causal=False, num_warps=4,
                    num_stages=2, grid=(N // 64, B))
    c._init_handles()
    ttgir = c.asm["ttgir"]
    out.append(f"{name}: regs={c.n_regs} spills={c.n_spills} shared={c.metadata.shared} "
               f"num_warps={c.metadata.num_warps} "
               f"'warp_specialize' in TTGIR: {'ttg.warp_specialize' in ttgir}; "
               f"loop attrs: {sorted(set(re.findall(r'tt\.[a-z_]*warp[a-z_]*', ttgir)))}; "
               f"cubin sha: {hash(bytes(c.asm['cubin']))}")
    m.flash_fwd_kernel.configs = [triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 128}, num_warps=4, num_stages=2)]
    ok = torch.allclose(m.FlashAttentionTriton.apply(q, k, v, False).float(),
                        F.scaled_dot_product_attention(q, k, v).float(), atol=2e-2, rtol=2e-2)
    out.append(f"  correct vs SDPA: {ok}")
print("\n".join(out))
