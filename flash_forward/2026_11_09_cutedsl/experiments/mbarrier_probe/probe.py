# probe.py: do cp.async + mbarrier (cp.async.mbarrier.arrive.noinc, mbarrier.test_wait.parity)
# compile and run on sm_89 from CuTe DSL 4.8.1? The DSL's own wrappers refuse (check_arch >= sm_90),
# so this calls the NVVM ops underneath them. Each of 128 threads cp.asyncs 16 bytes of a 2 KB tile
# into shared memory, arrives on an mbarrier when its copies land, waits for phase 0, and then reads
# a *different* thread's 16 bytes: correct only if the mbarrier made every thread's copies visible.
# Repeated over 2 phases into the same buffer to exercise the parity flip.
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cutlass_dsl import dsl_user_op, while_generate
import cutlass.memory
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack


import os
WRAPPERS = os.environ.get("USE_DSL_WRAPPERS") == "1"     # the DSL's own cute.arch functions


def arrive_noinc(ptr):
    if WRAPPERS:
        cute.arch.cp_async_mbarrier_arrive_noinc(ptr)
    else:
        _nvvm.cp_async_mbarrier_arrive(ptr.to_llvm_ptr(), noinc=True)


def test_wait(ptr, phase):
    if WRAPPERS:
        return cute.arch.mbarrier_test_wait(ptr, phase)
    return cutlass.Boolean(_nvvm.mbarrier_wait_parity(ptr.to_llvm_ptr(), cutlass.Int32(phase).ir_value(),
                                                      _nvvm.MBarrierWaitKind.TEST))


@cute.kernel
def k(gA: cute.Tensor, gB: cute.Tensor, Smem: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    st = cutlass.memory.SmemAllocator().allocate(Smem)
    s = st.buf.get_tensor(cute.make_layout((2, 128, 8), stride=(1024, 8, 1)))
    mbar = st.mbar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(mbar, 128)
    cute.arch.sync_threads()
    atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), cutlass.Float16,
                               num_bits_per_copy=128)
    for it in cutlass.range_constexpr(2):
        cute.copy(atom, gA[it, tidx, None], s[it, tidx, None])
        arrive_noinc(mbar)
        done = test_wait(mbar, it % 2)
        while not done:
            done = test_wait(mbar, it % 2)
        other = (tidx * 37 + 11) % 128
        for e in cutlass.range_constexpr(8):
            gB[it, tidx, e] = s[it, other, e]


@cute.jit
def run(gA: cute.Tensor, gB: cute.Tensor, stream: cuda.CUstream):
    @cute.struct
    class Smem:
        mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 1], 8]
        buf: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, 2048], 128]
    k(gA, gB, Smem).launch(grid=[1, 1, 1], block=[128, 1, 1], stream=stream)


a = torch.randn(2, 128, 8, device="cuda", dtype=torch.float16)
b = torch.zeros_like(a)
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
A, B = from_dlpack(a, assumed_align=16), from_dlpack(b, assumed_align=16)
f = cute.compile(run, A, B, stream, options="--keep-ptx --keep-cubin")
f(A, B, stream)
torch.cuda.synchronize()
idx = (torch.arange(128) * 37 + 11) % 128
print("correct:", torch.equal(b, a[:, idx.cuda(), :]))
ptx = f.__ptx__
print("\n".join(l.strip() for l in ptx.splitlines() if "mbarrier" in l or "cp.async" in l or ".target" in l))
open("probe.cubin", "wb").write(f.__cubin__)
