# swizzle_partition_repro.py  (nvidia-cutlass-dsl 4.8.1, RTX 4070 SUPER)
#
# A (64 x 64) tile V sits in shared memory, optionally with a Swizzle<3,3,3> layout. sVt is
# its transposed view (a composition), and tsV = tiled_mma.get_slice(t).partition_B(sVt) is one
# thread's share of it as the B operand of P @ V, shape (values, N-blocks, K-blocks). The same
# path NVIDIA's FA2 example uses, before it switches to ldmatrix. Each value is read two ways:
#   slice: tsV[None, None, k][i]   (take the k-th K-block, then index it)
#   full:  tsV[v, n, k]             (one full coordinate)
# and compared with the expected element of V, found through an identity tensor partitioned the
# same way. fp32 storage keeps every element distinct (the layouts are those of the bf16 MMA).
#
#   python repro/swizzle_partition_repro.py   -> results/swizzle_partition_repro.txt
import torch
import cutlass
import cutlass.cute as cute
import cutlass.memory
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack

R = C = 64


def smem_layout(swizzle):
    if not swizzle:
        return cute.make_layout((R, C), stride=(C, 1))
    atom = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1)))
    return cute.tile_to_shape(atom, (R, C), (0, 1))


class Repro:
    def __init__(self, swizzle):
        self.swizzle = swizzle

    @cute.kernel
    def kernel(self, g: cute.Tensor, out: cute.Tensor, mma: cute.TiledMma, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        sV = cutlass.memory.SmemAllocator().allocate(Smem).s.get_tensor(smem_layout(self.swizzle))
        for i in range(tidx, R * C, 128):
            sV[i // C, i % C] = g[i // C, i % C]
        cute.arch.sync_threads()
        sVt = cute.composition(sV, cute.make_layout((C, R), stride=(R, 1)))     # (D, N) view of V
        thr = mma.get_slice(tidx)
        tsV = thr.partition_B(sVt)
        tcV = thr.partition_B(cute.make_identity_tensor((C, R)))
        j = 0
        for k in cutlass.range_constexpr(cute.size(tsV, mode=[2])):
            sl = tsV[None, None, k]                    # the k-th K-block, as a slice
            for n in cutlass.range_constexpr(cute.size(tsV, mode=[1])):
                for v in cutlass.range_constexpr(cute.size(tsV, mode=[0])):
                    c = tcV[v, n, k]
                    out[tidx, j, 0] = sl[v, n]                  # slice, then index
                    out[tidx, j, 1] = tsV[v, n, k]              # full coordinate
                    out[tidx, j, 2] = g[c[1], c[0]]             # expected
                    out[tidx, j, 3] = tsV[v, n, 0]              # the same value in K-block 0
                    j = j + 1

    @cute.jit
    def __call__(self, g: cute.Tensor, out: cute.Tensor):
        @cute.struct
        class Smem:
            s: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, R * C], 1024]
        mma = cute.make_tiled_mma(warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)), (4, 1, 1),
                                  permutation_mnk=(64, 16, 16))
        self.kernel(g, out, mma, Smem).launch(grid=[1, 1, 1], block=[128, 1, 1])


g = torch.arange(R * C, device="cuda", dtype=torch.float32).view(R, C)
for swizzle in (False, True):
    out = torch.zeros(128, 128, 4, device="cuda")   # every warp holds all of B: 128 threads x 128 values
    Repro(swizzle)(from_dlpack(g), from_dlpack(out))
    torch.cuda.synchronize()
    sl, full, want, k0 = out.unbind(-1)
    k = torch.arange(128, device="cuda") // 32       # K-block of each value (4 blocks of 32 values)
    wrong = sl != want
    print(f"swizzle={swizzle!s:5}  slice: {int(wrong.sum()):5d} wrong of {want.numel()}  "
          f"(in K-block 0: {int(wrong[:, k == 0].sum())}; wrong values equal to K-block 0's value: "
          f"{int((wrong & (sl == k0)).sum())})   full coordinate: {int((full != want).sum())} wrong")
