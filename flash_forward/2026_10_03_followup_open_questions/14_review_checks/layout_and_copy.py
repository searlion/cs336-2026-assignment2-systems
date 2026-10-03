# layout_and_copy.py
# Two checks raised in review:
# 1. Which axis does Triton 3.6 put the warps on for the attention kernel's first tl.dot? Compile
#    v1's body for several (dtype, D, tile, warps) combinations and print warpsPerCTA of the #mma
#    layout and the shared memory. Claim to test: the layout follows the shape (query tile against
#    head dimension), not the input dtype.
# 2. Device-to-device copy bandwidth, counting the bytes read plus the bytes written, in GB/s
#    (1e9 bytes) and GiB/s (2^30 bytes): torch's copy_ between two large buffers, timed back to
#    back with CUDA events after a warm-up.
#   flock /tmp/claude-gpu.lock python layout_and_copy.py > layout_and_copy.txt
import math
import pathlib
import re
import statistics
import sys

import torch
import triton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import jit_fn, load_version  # noqa: E402

v1 = load_version("v1_tiles")
print("== warpsPerCTA of the first tl.dot's #mma layout")
cases = [(torch.bfloat16, 64, (64, 32, 4, 3)), (torch.float32, 64, (64, 32, 4, 3)),
         (torch.bfloat16, 128, (64, 32, 4, 3)), (torch.float16, 128, (64, 32, 4, 3)),
         (torch.float32, 128, (64, 32, 4, 2)), (torch.bfloat16, 128, (128, 32, 4, 3)),
         (torch.bfloat16, 32, (32, 32, 2, 3)), (torch.bfloat16, 64, (32, 32, 2, 3)),
         (torch.bfloat16, 16, (16, 16, 4, 3)), (torch.bfloat16, 64, (16, 16, 4, 3)),
         (torch.bfloat16, 64, (64, 32, 8, 3))]
for dtype, D, (bq, bk, nw, ns) in cases:
    N = 256
    q = torch.empty(1, N, D, device="cuda", dtype=dtype)
    o = torch.empty_like(q)
    L = torch.empty(1, N, device="cuda", dtype=torch.float32)
    ck = jit_fn(v1).warmup(q, q, q, o, L, *q.stride(), *q.stride(), *q.stride(), *o.stride(), *L.stride(),
                           N, N, 1 / math.sqrt(D), D=D, Q_TILE_SIZE=bq, K_TILE_SIZE=bk, is_causal=False,
                           num_warps=nw, num_stages=ns, grid=(N // bq, 1))
    mma = re.search(r"#mma\d* = #ttg\.nvidia_mma<\{[^}]*warpsPerCTA = \[(\d+), (\d+)\]", ck.asm["ttgir"])
    axis = "query axis" if int(mma[2]) == 1 else ("key axis" if int(mma[1]) == 1 else "both axes")
    print(f"{str(dtype)[6:]:9s} D={D:3d} tile {bq:3d}x{bk:<3d} warps {nw:2d} stages {ns}: "
          f"warpsPerCTA [{mma[1]}, {mma[2]}] ({axis}), shared {ck.metadata.shared} B, Q_TILE {'>=' if bq >= D else '<'} D")

print("\n== device-to-device copy (bytes read + bytes written)")
for mib in (256, 1024):
    n = mib * 2 ** 20 // 4
    src = torch.randn(n, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(20):
        dst.copy_(src)
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        dst.copy_(src)
        e.record()
        times.append((s, e))
    torch.cuda.synchronize()
    ms = statistics.median(s.elapsed_time(e) for s, e in times)
    moved = 2 * n * 4
    print(f"{mib:5d} MiB buffer: {ms:.3f} ms, {moved / ms / 1e6:.0f} GB/s = {moved / ms * 1e3 / 2 ** 30:.0f} GiB/s")
