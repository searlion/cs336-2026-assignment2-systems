# smem_fp32_d128.py
# Item 13: the fp32, D = 128 failure needed 107,520 bytes of shared memory against the formula's
# 98,304 (Q: 64x128x4 = 32,768; K and V: 2 stages-in-flight x 2 x 32x128x4 = 65,536). The TTGIR
# shows a buffer for P (64x32x4 = 8,192). Where are the other 1,024?
#
# Triton assigns shared-memory offsets in the AllocateSharedMemory pass, after the TTGIR that
# compiled.asm["ttgir"] holds, so this compiles the configuration in a child process with
# MLIR_ENABLE_DUMP=1 (IR printed after every pass) and reads the `allocation.offset` attributes
# from the IR after that pass, together with any scratch buffers it assigned.
#   python smem_fp32_d128.py > smem_fp32_d128.txt
import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
CHILD = r'''
import math, sys, torch
sys.path.insert(0, "%s")
from common import jit_fn, load_version
q = torch.empty(1, 256, 128, device="cuda", dtype=torch.float32)
L = torch.empty(1, 256, device="cuda", dtype=torch.float32)
c = jit_fn(load_version("v1_tiles")).warmup(q, q, q, q, L, *q.stride(), *q.stride(), *q.stride(), *q.stride(),
    *L.stride(), 256, 256, 1 / math.sqrt(128), D=128, Q_TILE_SIZE=64, K_TILE_SIZE=32, is_causal=False,
    num_warps=4, num_stages=3, grid=(1,))
print("metadata.shared =", c.metadata.shared, file=sys.stderr)
''' % str(HERE.parent)

with tempfile.TemporaryDirectory() as cache:
    env = dict(os.environ, MLIR_ENABLE_DUMP="1", TRITON_CACHE_DIR=cache)
    err = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True).stderr
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports" / "fp32_d128_mlir_dump.txt").write_text(err)
print(re.search(r"metadata\.shared = \d+", err)[0])
dumps = re.split(r"// -----// IR Dump (?:Before|After) ", err)
after = [d for d in dumps if "allocation.offset" in d]
print(f"{len(dumps)} IR dumps; {len(after)} contain allocation.offset; using the first: {after[0].splitlines()[0]}")
ir = after[0]
m = re.search(r"ttg\.shared = (\d+)", ir)
print("module ttg.shared =", m[1] if m else "?")
for line in ir.splitlines():
    if "allocation.offset" in line:
        off = re.search(r"allocation\.offset = (\d+)", line)[1]
        op = re.search(r"(ttg\.\w+|tt\.\w+|ttng\.\w+|arith\.\w+)", line)
        types = re.findall(r"(?:memdesc|tensor)<[^>]*>", line)
        print(f"  offset {int(off):6d}  {op[1] if op else '?':28s} {' -> '.join(types)[:150]}")
