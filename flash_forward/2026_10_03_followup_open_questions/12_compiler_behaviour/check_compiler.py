# check_compiler.py
# Item 12: the compiler behaviours the article relies on, as checks that can be re-run after a
# Triton upgrade. Each check compiles one configuration (no launch) and looks for the evidence in
# the TTGIR or PTX. Prints PASS/FAIL with the line that decided it.
#   python check_compiler.py > check_compiler.txt
import math
import pathlib
import re
import sys

import torch
import triton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import jit_fn, load_version  # noqa: E402

N, B = 4096, 32
results = []


def compile_cfg(version, cfg, D=64, dtype=torch.bfloat16, causal=False):
    bq, bk, nw, ns = cfg
    q = torch.empty(B, N, D, device="cuda", dtype=dtype)
    o = torch.empty_like(q)
    L = torch.empty(B, N, device="cuda", dtype=torch.float32)
    c = jit_fn(load_version(version)).warmup(
        q, q, q, o, L, *q.stride(), *q.stride(), *q.stride(), *o.stride(), *L.stride(), N, N, 1 / math.sqrt(D),
        D=D, Q_TILE_SIZE=bq, K_TILE_SIZE=bk, is_causal=causal, num_warps=nw, num_stages=ns, grid=(1,))
    c._init_handles()
    return c


def check(name, ok, evidence):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {evidence}")


def mma_warps(ttgir):
    m = re.search(r"#mma = #ttg\.nvidia_mma<\{[^}]*warpsPerCTA = \[(\d+), (\d+)\]", ttgir)
    return [int(m[1]), int(m[2])] if m else None


print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}\n")

# 1. The warp layouts behind the "never more warps than 16-row slices" rule.
for cfg, want in [((16, 16, 4, 3), [1, 4]), ((64, 32, 8, 3), [8, 1]), ((32, 32, 2, 3), [1, 2]),
                  ((64, 32, 4, 3), [4, 1]), ((128, 128, 4, 2), [4, 1]), ((128, 64, 8, 3), [8, 1])]:
    got = mma_warps(compile_cfg("v1_tiles", cfg).asm["ttgir"])
    check(f"warpsPerCTA for {cfg} is {want}", got == want, f"got {got}")

# 2. Q kept in shared memory and re-read in the loop; 3. num_stages = s gives s - 1 K and V buffers;
# and the shared-memory formula 128 * (Q_TILE + 2 (s - 1) K_TILE) for bf16, D = 64.
for s in (2, 3, 4, 5):
    c = compile_cfg("v1_tiles", (64, 32, 4, s))
    ttgir = c.asm["ttgir"]
    q_alloc = re.search(r"ttg\.local_alloc %\S+ : \(tensor<64x64xbf16[^)]*\)", ttgir)
    kv = re.findall(r"ttg\.local_alloc : \(\) -> !ttg\.memdesc<(\d+)x32x64xbf16", ttgir)
    formula = 128 * (64 + 2 * (s - 1) * 32)
    if s == 3:
        check("Q tile is a ttg.local_alloc (shared memory), read with ttg.local_load in the loop",
              q_alloc is not None and "ttg.local_load" in ttgir, q_alloc[0][:90] if q_alloc else "no Q local_alloc")
    check(f"num_stages={s}: K and V buffers are {s - 1}x32x64", kv == [str(s - 1)] * 2, f"buffers {kv}")
    check(f"num_stages={s}: shared memory {c.metadata.shared} == formula {formula}", c.metadata.shared == formula, "")

# 4. exp lowering: tl.exp -> mul by log2(e) + ex2.approx.f32 (no ftz); tl.math.exp2 -> ex2.approx.ftz.f32.
ptx3 = compile_cfg("v3_causal_skip", (64, 32, 4, 3)).asm["ptx"]
ptx4 = compile_cfg("v4_exp2", (64, 32, 4, 3)).asm["ptx"]
n3, n3ftz = len(re.findall(r"ex2\.approx\.f32", ptx3)), len(re.findall(r"ex2\.approx\.ftz\.f32", ptx3))
n4, n4ftz = len(re.findall(r"ex2\.approx\.f32", ptx4)), len(re.findall(r"ex2\.approx\.ftz\.f32", ptx4))
check("tl.exp (v3) lowers to mul.f32 by 0f3FB8AA3B + ex2.approx.f32 (non-ftz)",
      n3 > 0 and n3ftz == 0 and "0f3FB8AA3B" in ptx3, f"v3 PTX: {n3} ex2.approx.f32, {n3ftz} ex2.approx.ftz.f32")
check("tl.math.exp2 (v4) lowers to ex2.approx.ftz.f32", n4ftz > 0 and n4 == 0,
      f"v4 PTX: {n4} ex2.approx.f32, {n4ftz} ex2.approx.ftz.f32")

# 5. The P layout conversion: free (convert_layout #mma -> dot_op, no shared memory) when warps own
# whole rows; a round trip through shared memory when they do not.
for cfg, want_smem in [((64, 32, 4, 3), False), ((16, 16, 4, 3), True)]:
    ttgir = compile_cfg("v1_tiles", cfg).asm["ttgir"]
    p_alloc = re.search(r"ttg\.local_alloc %P_ij\S* : \(tensor<\d+x\d+xbf16, #mma>\)", ttgir)
    conv = re.search(r"ttg\.convert_layout %P_ij\S* : tensor<\d+x\d+xbf16, #mma> -> tensor<[^>]*dot_op", ttgir)
    ok = (p_alloc is not None) == want_smem and (conv is not None) != want_smem
    check(f"P for {cfg} goes {'through shared memory' if want_smem else 'straight to the dot operand'}", ok,
          (p_alloc or conv)[0][:110] if (p_alloc or conv) else "neither found")

print(f"\n{sum(results)}/{len(results)} checks pass")
