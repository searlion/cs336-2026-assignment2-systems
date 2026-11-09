# arrive_probe.py: which plain mbarrier.arrive forms compile for sm_89 from CuTe DSL 4.8.1?
import sys, torch, cutlass, cutlass.cute as cute, cutlass.memory
from cutlass._mlir.dialects import llvm
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack
MODE = sys.argv[1]


def arrive(ptr):
    if MODE == "dsl":
        cute.arch.mbarrier_arrive(ptr)
    else:
        llvm.inline_asm(None, [ptr.toint().ir_value()],
                        "{\n.reg .b64 state;\nmbarrier.arrive.shared::cta.b64 state, [$0];\n}", "r,~{memory}",
                        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.kernel
def k(gB: cute.Tensor, Smem: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    st = cutlass.memory.SmemAllocator().allocate(Smem)
    mbar = st.mbar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(mbar, 128)
    cute.arch.sync_threads()
    arrive(mbar)
    cute.arch.sync_threads()
    if tidx == 0:
        gB[0] = 1


@cute.jit
def run(gB: cute.Tensor, stream: cuda.CUstream):
    @cute.struct
    class Smem:
        mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 1], 8]
    k(gB, Smem).launch(grid=[1, 1, 1], block=[128, 1, 1], stream=stream)


b = torch.zeros(4, device="cuda", dtype=torch.int32)
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
B = from_dlpack(b)
try:
    f = cute.compile(run, B, stream, options="--keep-ptx")
    f(B, stream); torch.cuda.synchronize()
    print(MODE, "OK:", [l.strip() for l in f.__ptx__.splitlines() if "mbarrier.arrive" in l])
except Exception as e:
    print(MODE, "FAILED:", str(e).splitlines()[0][:200])
