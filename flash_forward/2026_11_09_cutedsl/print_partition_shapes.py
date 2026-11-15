# print_partition_shapes.py
# Prints the layouts that v0's setup code gives thread tidx, as CuTe DSL computes them when it compiles
# the kernel: tSsQ = partition_A(sQ), tSsK = partition_B(sK), acc_S. Same tiled MMA and shared-memory
# layouts as kernels/v0_naive. Run: .venv/bin/python print_partition_shapes.py
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp

D = bM = bN = 64
WARPS = 4
dtype = cutlass.BFloat16


@cute.struct
class Smem:
    sQ: cute.struct.Align[cute.struct.MemRange[dtype, bM * D], 128]
    sK: cute.struct.Align[cute.struct.MemRange[dtype, bN * D], 128]


@cute.kernel
def probe(tiled_mma: cute.TiledMma):
    tidx, _, _ = cute.arch.thread_idx()
    storage = cutlass.memory.SmemAllocator().allocate(Smem)
    sQ = storage.sQ.get_tensor(cute.make_layout((bM, D), stride=(D, 1)))      # row-major, as in v0
    sK = storage.sK.get_tensor(cute.make_layout((bN, D), stride=(D, 1)))
    thr_mma = tiled_mma.get_slice(tidx)
    tSsQ = thr_mma.partition_A(sQ)
    tSsK = thr_mma.partition_B(sK)
    acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), cutlass.Float32)
    # Python print runs once, while CuTe traces the kernel: the layouts are known at compile time.
    print("sQ   ", sQ.layout)
    print("tSsQ ", tSsQ.layout)
    print("tSsK ", tSsK.layout)
    print("acc_S", acc_S.layout)
    print("cute.size(tSsQ, mode=[2]) =", cute.size(tSsQ, mode=[2]))


@cute.jit
def main():
    tiled_mma = cute.make_tiled_mma(
        warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
        (WARPS, 1, 1), permutation_mnk=(WARPS * 16, 16, 16))
    probe(tiled_mma).launch(grid=(1, 1, 1), block=[WARPS * 32, 1, 1])


if __name__ == "__main__":
    cute.compile(main)
