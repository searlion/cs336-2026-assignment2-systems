# sass_counts.py
# Compile one kernel version for the benchmark shape (bf16, D = 64) with its cubin kept, then
# count the SASS instructions the article talks about, for the whole kernel and for each loop
# (a loop is the code between a backward branch and its target).
#
#   python sass_counts.py kernels/v0_naive --causal 0 [--config 64,64,4,1]
import argparse
import os

# --keep-cubin also drops a copy in the working directory; send that one to a scratch folder.
os.environ.setdefault("CUTE_DSL_DUMP_DIR", "results/cubin/_dump")
import collections
import importlib.util
import math
import pathlib
import re
import subprocess

import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

CUDA_BIN = "/usr/local/cuda-13.0/bin"
OPS = ["HMMA", "LDSM", "LDS", "STS", "LDG", "STG", "LDGSTS", "LDGDEPBAR", "DEPBAR", "BAR", "MUFU.EX2",
       "SHFL.BFLY", "FFMA", "FMUL", "FMNMX", "F2FP", "LDL", "STL",
       # added for round 6: mbarrier arrivals (cp.async-triggered and plain) and warp syncs
       "ARRIVES", "SYNCS", "ATOMS", "WARPSYNC"]

parser = argparse.ArgumentParser()
parser.add_argument("kernel_dir")
parser.add_argument("--causal", type=int, default=0)
parser.add_argument("--config", default=None, help="bM,bN,warps[,stages]")
parser.add_argument("--D", type=int, default=64)
args = parser.parse_args()

path = pathlib.Path(args.kernel_dir) / "flashattention_autograd_function_triton.py"
spec = importlib.util.spec_from_file_location(pathlib.Path(args.kernel_dir).name, path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

B, N, D = 32, 4096, args.D
q, k, v = (torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
o = torch.empty_like(q)
L = torch.empty(B, N, device="cuda", dtype=torch.float32)
cfg = (tuple(int(x) for x in args.config.split(",")) if args.config
       else mod.FlashAttentionCuTe.config(N, N, D, bool(args.causal), torch.bfloat16))
dt = cutlass.BFloat16
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
cargs = (mod._ptr(q, dt), mod._ptr(k, dt), mod._ptr(v, dt), mod._ptr(o, dt), mod._ptr(L, cutlass.Float32),
         B, N, N, (1.0 / math.sqrt(D)) * mod.LOG2E, stream)
fn = cute.compile(mod.FlashFwd(dt, D, bool(args.causal), *cfg), *cargs,
                  options="--keep-cubin --generate-line-info")

out = pathlib.Path("results/cubin") / pathlib.Path(args.kernel_dir).name
out.mkdir(parents=True, exist_ok=True)
tag = f"D{D}_{'causal' if args.causal else 'noncausal'}_" + "x".join(map(str, cfg))
cubin = out / f"{tag}.cubin"
cubin.write_bytes(fn.__cubin__)
sass = subprocess.run([f"{CUDA_BIN}/cuobjdump", "-sass", str(cubin)], capture_output=True, text=True).stdout
(out / f"{tag}.sass").write_text(sass)
res = subprocess.run([f"{CUDA_BIN}/cuobjdump", "-res-usage", str(cubin)], capture_output=True, text=True).stdout
(out / f"{tag}.res").write_text(res)

LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.]+)([^;]*);")
# "@!PT" means "never": ptxas pads around cp.async with never-executed LDS; leave those out.
instrs = [(int(m.group(1), 16), m.group(3), m.group(4)) for m in map(LINE.search, sass.splitlines())
          if m and (m.group(2) or "").strip() != "@!PT"]


def count(seq):
    c = collections.Counter()
    for _, op, _ in seq:
        base = op.split(".")[0]
        for name in OPS:
            if name == op or (("." not in name) and name == base) or op.startswith(name + "."):
                c[name] += 1
                break
    c["total"] = len(seq)
    return c


loops = []
for addr, op, rest in instrs:
    if op.startswith("BRA"):
        m = re.search(r"0x([0-9a-f]+)", rest)
        if m and int(m.group(1), 16) < addr:
            loops.append((int(m.group(1), 16), addr))
regs = re.search(r"REG:(\d+)", res)
shared = re.search(r"SHARED:(\d+)", res)
local = re.search(r"LOCAL:(\d+)", res)
print(f"{args.kernel_dir} causal={args.causal} config={cfg}: registers {regs.group(1)}, "
      f"static shared {shared.group(1)} B, local {local.group(1)} B")
rows = [("whole kernel", count(instrs))] + [
    (f"loop {i} ({lo:#x}-{hi:#x})", count([x for x in instrs if lo <= x[0] <= hi])) for i, (lo, hi) in enumerate(loops)]
print(f"{'':24s}" + "".join(f"{n:>10s}" for n in OPS + ["total"]))
for name, c in rows:
    print(f"{name:24s}" + "".join(f"{c[n]:>10d}" for n in OPS + ["total"]))


# Added for round 6: how evenly the tensor-core instructions are spread through each loop. "gap" is
# the number of instructions between consecutive HMMAs (the loop wraps around); a long HMMA-free run
# is a stretch the warp's scheduler must cover with other warps.
def hmma_gaps(seq):
    pos = [i for i, (_, op, _) in enumerate(seq) if op.startswith("HMMA")]
    if len(pos) < 2:
        return None
    gaps = [b - a - 1 for a, b in zip(pos, pos[1:])] + [len(seq) - pos[-1] - 1 + pos[0]]
    return max(gaps), sum(gaps) / len(gaps), sum(1 for g in gaps if g >= 8)


print(f"{'':24s}{'HMMA-free run, longest':>24s}{'mean gap':>10s}{'gaps >= 8':>11s}")
for i, (lo, hi) in enumerate(loops):
    g = hmma_gaps([x for x in instrs if lo <= x[0] <= hi])
    if g:
        print(f"{f'loop {i} ({lo:#x}-{hi:#x})':24s}{g[0]:>24d}{g[1]:>10.1f}{g[2]:>11d}")
