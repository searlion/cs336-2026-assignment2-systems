# inspect_kernel.py
# Compile one configuration of a kernel version and print what the compiler did with it:
# registers, spills, shared memory, the MMA layout, and every layout conversion.
#
#   python inspect_kernel.py kernels/v1_tiles 16,16,4,3
#   python inspect_kernel.py kernels/v1_tiles 128,64,4,3 --causal 1 --sass
import argparse
import collections
import math
import re
import sys

import torch
import triton

parser = argparse.ArgumentParser()
parser.add_argument("kernel_dir")
parser.add_argument("config", help="Q_TILE,K_TILE,num_warps,num_stages")
parser.add_argument("--N", type=int, default=4096)
parser.add_argument("--D", type=int, default=64)
parser.add_argument("--dtype", default="bfloat16")
parser.add_argument("--causal", type=int, default=0)
parser.add_argument("--sass", action="store_true", help="print a histogram of SASS opcodes")
parser.add_argument("--ttgir", action="store_true", help="dump the full TTGIR")
parser.add_argument("--dump-sass", default=None, help="write the full SASS to this file")
args = parser.parse_args()

sys.path.insert(0, args.kernel_dir)
import flashattention_autograd_function_triton as mod  # noqa: E402

bq, bk, nw, ns = (int(x) for x in args.config.split(","))
B, N, D = 32, args.N, args.D
q, k, v = (torch.randn(B, N, D, device="cuda", dtype=getattr(torch, args.dtype)) for _ in range(3))
o = torch.empty_like(q)
L = torch.empty((B, N), device="cuda", dtype=torch.float32)
kern = mod.flash_fwd_kernel
if hasattr(kern, "configs"):  # unwrap @triton.autotune if present
    kern = kern.fn
# warmup() compiles without launching, so this is safe to run next to a benchmark.
compiled = kern.warmup(
    q, k, v, o, L,
    *q.stride(), *k.stride(), *v.stride(), *o.stride(), *L.stride(),
    N, N, 1.0 / math.sqrt(D),
    D=D, Q_TILE_SIZE=bq, K_TILE_SIZE=bk, is_causal=bool(args.causal),
    num_warps=nw, num_stages=ns, grid=(triton.cdiv(N, bq), B),
)
compiled._init_handles()  # loads the module (no launch) so register and spill counts are filled in
print(f"config {args.config}: regs/thread={compiled.n_regs} spills={compiled.n_spills} "
      f"shared={compiled.metadata.shared} B")

ttgir = compiled.asm["ttgir"]
if args.ttgir:
    print(ttgir)
for line in ttgir.splitlines():
    if line.startswith("#") and ("mma" in line or "blocked" in line or "shared" in line):
        print("  layout:", line.strip()[:160])
convs = collections.Counter()
for line in ttgir.splitlines():
    m = re.search(r"ttg\.(convert_layout|local_alloc|local_load|local_store|async_copy_global_to_local)", line)
    if m:
        types = re.findall(r"tensor<([^>]*)>", line)
        convs[(m.group(1), " -> ".join(t.split(",")[-1].strip() for t in types[-2:]))] += 1
for (op, ty), n in sorted(convs.items()):
    print(f"  {n:3d} x {op:32s} {ty}")

if args.sass:
    # Triton keeps the cubin; nvdisasm (CUDA toolkit) turns it into SASS.
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(compiled.asm["cubin"])
        f.flush()
        sass = subprocess.run(["nvdisasm", "-c", "-g", f.name], capture_output=True, text=True).stdout
    if args.dump_sass:
        open(args.dump_sass, "w").write(sass)
    ops = collections.Counter()
    for line in sass.splitlines():
        m = re.match(r"\s*(?:/\*[0-9a-f]+\*/)?\s*(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)\s", line)
        if m:
            ops[m.group(1)] += 1
    print("  static SASS opcode counts:", ", ".join(f"{k}={v}" for k, v in ops.most_common(40)))
