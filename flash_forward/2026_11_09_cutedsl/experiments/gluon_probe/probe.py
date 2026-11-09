# probe.py: which of the round-6 tricks can Triton 3.6's experimental Gluon layer express on sm_89?
#  1. cp.async completion tracked by an mbarrier (cp.async.mbarrier.arrive.noinc + mbarrier wait),
#     a K tile split into 16-wide k-slices read from shared memory, and the bf16 MMA (mma_v2,
#     fp32 accumulate) chained over the slices: S = Q K^T for one 64 x 64 tile.
#  2. warp_specialize: a 1-warp producer partition that issues the cp.async and arrives on the
#     mbarrier, and the 4-warp default partition that waits and computes.
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp, mbarrier, mma_v2

BLK = ttgl.constexpr(ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0]))
BLK1 = ttgl.constexpr(ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[1, 1], order=[1, 0]))
import os
WS_REGS = ttgl.constexpr(int(os.environ.get('WS_REGS', '48')))   # worker_num_regs
MMA = ttgl.constexpr(ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]))
SMEM = ttgl.constexpr(ttgl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0]))


@gluon.jit
def _ptrs(base, layout: ttgl.constexpr):
    r = ttgl.arange(0, 64, layout=ttgl.SliceLayout(1, layout))
    c = ttgl.arange(0, 64, layout=ttgl.SliceLayout(0, layout))
    return base + r[:, None] * 64 + c[None, :]


@gluon.jit
def _qk_sliced(q_s, k_s):
    acc = ttgl.zeros([64, 64], ttgl.float32, MMA)
    for kk in ttgl.static_range(4):                        # four 16-wide k-slices
        a = q_s.slice(16 * kk, 16, dim=1).load(ttgl.DotOperandLayout(0, MMA, 2))
        b = k_s.slice(16 * kk, 16, dim=1).permute([1, 0]).load(ttgl.DotOperandLayout(1, MMA, 2))
        acc = mma_v2(a, b, acc)
    return acc


@gluon.jit
def qk_mbarrier(q_ptr, k_ptr, s_ptr):
    q_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [64, 64], SMEM)
    k_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [64, 64], SMEM)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=128)
    ttgl.thread_barrier()
    cp.async_copy_global_to_shared(q_s, _ptrs(q_ptr, BLK))
    cp.async_copy_global_to_shared(k_s, _ptrs(k_ptr, BLK))
    cp.mbarrier_arrive(bar, increment_count=False)          # cp.async.mbarrier.arrive.noinc
    mbarrier.wait(bar, 0)
    acc = _qk_sliced(q_s, k_s)
    ttgl.store(_ptrs(s_ptr, MMA), acc)


@gluon.jit
def _consumer(q_s, k_s, bar, s_ptr):
    mbarrier.wait(bar, 0)
    acc = _qk_sliced(q_s, k_s)
    ttgl.store(_ptrs(s_ptr, MMA), acc)


@gluon.jit
def _producer(q_s, k_s, bar, q_ptr, k_ptr):
    cp.async_copy_global_to_shared(q_s, _ptrs(q_ptr, BLK1))
    cp.async_copy_global_to_shared(k_s, _ptrs(k_ptr, BLK1))
    cp.mbarrier_arrive(bar, increment_count=False)


@gluon.jit
def qk_warp_specialized(q_ptr, k_ptr, s_ptr):
    q_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [64, 64], SMEM)
    k_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [64, 64], SMEM)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=32)                            # the producer warp's 32 threads
    ttgl.thread_barrier()
    ttgl.warp_specialize([(_consumer, (q_s, k_s, bar, s_ptr)), (_producer, (q_s, k_s, bar, q_ptr, k_ptr))],
                         [1], [WS_REGS])


q, k = (torch.randn(64, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2))
ref = q.float() @ k.float().T
import sys
for kern in [{'mbarrier': qk_mbarrier, 'ws': qk_warp_specialized}[sys.argv[1]]]:
    s = torch.zeros(64, 64, device="cuda", dtype=torch.float32)
    try:
        h = kern[(1,)](q, k, s, num_warps=4, **({"maxnreg": int(os.environ["MAXNREG"])} if "MAXNREG" in os.environ else {}))
        torch.cuda.synchronize()
        ptx = h.asm["ptx"]
        ops = sorted({w for l in ptx.splitlines() for w in l.split()
                      if w.startswith(("mbarrier.", "cp.async", "mma.sync", "setmaxnreg", "bar.sync", "barrier."))})
        print(f"{kern.fn.__name__}: compiled for {h.metadata.target}, num_warps={h.metadata.num_warps}, "
              f"max |S - ref| = {(s - ref).abs().max().item():.3g}")
        print("   PTX ops:", ", ".join(ops))
    except Exception as e:  # noqa: BLE001
        print(f"{kern.fn.__name__}: FAILED: {type(e).__name__}: {str(e).strip().splitlines()[-1][:300]}")
