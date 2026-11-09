# flashattention_autograd_function_triton.py  (CuTe DSL, v6_mbarrier)
#
# v5 with its block-wide synchronisation replaced by mbarriers. v5 waits for its cp.async copies
# with cp.async.wait_group and then a __syncthreads (BAR.SYNC) that every warp must reach: twice
# per key block with one buffer, once with a ring. Here K and V each have their own ring (KS and VS
# buffers, which may differ) and every buffer has two mbarriers:
#   full  - each thread arrives through cp.async.mbarrier.arrive.noinc, which fires when that
#           thread's copies have landed; 128 arrivals complete it, so a warp that waits on it knows
#           the whole tile is in shared memory, without waiting for the other warps.
#   empty - each warp arrives (lane 0, after __syncwarp) when it has finished reading the buffer;
#           the copy that refills it waits for 4 arrivals.
# A warp therefore waits only for the data it needs and for the buffer it overwrites, never for
# the other warps to reach the same line. The maths is v5's, in v5's order, so O and L are
# bitwise the same as v5's at the same bM x bN.
#
# CuTe DSL 4.8.1's own wrappers for cp.async.mbarrier.arrive and mbarrier.test_wait refuse
# sm_89 (they check for sm_90+), although PTX has both since sm_80. The helpers below call the
# NVVM operations underneath them; on sm_89 the wait is a spin on mbarrier.test_wait (an LDS of
# the barrier word in SASS), since mbarrier.try_wait needs sm_90; the plain arrive is inline PTX.
import math
from types import SimpleNamespace

import torch
import cutlass
import cutlass.cute as cute
import cutlass.memory
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_ptr
import cuda.bindings.driver as cuda
from cutlass._mlir.dialects import llvm, nvvm as _nvvm

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
MASK_VALUE = -1.0e6          # same as the Triton kernel: finite, so no -inf - -inf = NaN


def mn_view(t):
    """View an MMA accumulator ((2,2), MMA_M, MMA_N) as (rows, cols) owned by this thread."""
    l = cute.make_layout(t.layout.shape)
    mn = cute.make_layout(((l.shape[0][1], l.shape[1]), (l.shape[0][0], l.shape[2])),
                          stride=((l.stride[0][1], l.stride[1]), (l.stride[0][0], l.stride[2])))
    return cute.make_tensor(t.iterator, cute.composition(t.layout, mn))


def smem_layout(rows, cols):
    """A (rows, cols) row-major tile built from 8-row blocks whose 16-byte chunks are permuted
    by XOR with the row number (Swizzle<3,3,3> for 64-element rows, <2,3,3> for 32)."""
    width = 64 if cols % 64 == 0 else 32
    atom = cute.make_composed_layout(cute.make_swizzle(3 if width == 64 else 2, 3, 3), 0,
                                     cute.make_layout((8, width), stride=(width, 1)))
    return cute.tile_to_shape(atom, (rows, cols), (0, 1))


def smem_layout_staged(rows, cols, stages):
    """smem_layout with a third mode for the buffer index."""
    width = 64 if cols % 64 == 0 else 32
    atom = cute.make_composed_layout(cute.make_swizzle(3 if width == 64 else 2, 3, 3), 0,
                                     cute.make_layout((8, width), stride=(width, 1)))
    return cute.tile_to_shape(atom, (rows, cols, stages), (0, 1, 2))


def as_a_operand(acc):
    """Re-view an accumulator ((2,2), MMA_M, MMA_N) as an A operand ((2,2,2), MMA_M, MMA_N/2).

    One 16x8 accumulator tile gives each thread 2 rows x 2 columns. An A operand of a 16x8x16
    MMA gives each thread the same 2 rows x 2 columns, plus the same pattern 8 columns to the
    right. So two neighbouring accumulator tiles along N are exactly one A tile along K: group
    the N mode in pairs and move the pair into the value mode. No data moves."""
    l = acc.layout
    pairs = cute.logical_divide(l, (None, None, 2))       # N mode -> (2, MMA_N / 2)
    return cute.make_tensor(acc.iterator, cute.make_layout(
        ((pairs.shape[0], pairs.shape[2][0]), pairs.shape[1], pairs.shape[2][1]),
        stride=((pairs.stride[0], pairs.stride[2][0]), pairs.stride[1], pairs.stride[2][1])))


def quad_reduce(v, op):
    """The 4 lanes of a quad hold the same rows: combine their partial results."""
    v = op(v, cute.arch.shuffle_sync_bfly(v, offset=2, mask=-1, mask_and_clamp=31))
    return op(v, cute.arch.shuffle_sync_bfly(v, offset=1, mask=-1, mask_and_clamp=31))


def full_arrive(bar):
    """cp.async.mbarrier.arrive.noinc: arrive on bar when this thread's earlier cp.asyncs land."""
    _nvvm.cp_async_mbarrier_arrive(bar.to_llvm_ptr(), noinc=True)


def full_arrive_inc(bar):
    """cp.async.mbarrier.arrive (counting): pending count +1 now, -1 when the copies land."""
    _nvvm.cp_async_mbarrier_arrive(bar.to_llvm_ptr(), noinc=False)


@cute.jit
def mbar_wait(bar, phase):
    """Spin until the phase with this parity has completed (mbarrier.test_wait.parity)."""
    done = cutlass.Boolean(_nvvm.mbarrier_wait_parity(bar.to_llvm_ptr(), cutlass.Int32(phase).ir_value(),
                                                      _nvvm.MBarrierWaitKind.TEST))
    while not done:
        done = cutlass.Boolean(_nvvm.mbarrier_wait_parity(bar.to_llvm_ptr(), cutlass.Int32(phase).ir_value(),
                                                          _nvvm.MBarrierWaitKind.TEST))


def arrive(bar):
    """mbarrier.arrive (release), as inline PTX: cute.arch.mbarrier_arrive emits a form NVVM rejects
    for sm_89 (results/round6/cute_mbarrier_probe.txt)."""
    llvm.inline_asm(None, [bar.toint().ir_value()],
                    "{\n.reg .b64 state;\nmbarrier.arrive.shared::cta.b64 state, [$0];\n}", "r,~{memory}",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def warp_arrive(bar):
    """One arrival per warp: every lane has finished its reads (syncwarp), then lane 0 arrives."""
    cute.arch.sync_warp()
    if cute.arch.lane_idx() == 0:
        arrive(bar)



class FlashFwd:
    def __init__(self, dtype, D, causal, bM=64, bN=64, warps=4, k_stages=1, v_stages=1):
        self.dtype, self.D, self.causal = dtype, D, causal
        self.bM, self.bN, self.warps = bM, bN, warps
        self.KS, self.VS = k_stages, v_stages
        self.threads = warps * 32

    @cute.jit
    def __call__(self, pQ: cute.Pointer, pK: cute.Pointer, pV: cute.Pointer, pO: cute.Pointer,
                 pL: cute.Pointer, B: cutlass.Int32, Nq: cutlass.Int32, Nk: cutlass.Int32,
                 scale_log2: cutlass.Float32, stream: cuda.CUstream):
        D, bM, bN, dtype = self.D, self.bM, self.bN, self.dtype
        q_stride, k_stride = cute.assume(Nq * D, divby=D), cute.assume(Nk * D, divby=D)
        mQ = cute.make_tensor(pQ, cute.make_layout((B, Nq, D), stride=(q_stride, D, 1)))
        mK = cute.make_tensor(pK, cute.make_layout((B, Nk, D), stride=(k_stride, D, 1)))
        mV = cute.make_tensor(pV, cute.make_layout((B, Nk, D), stride=(k_stride, D, 1)))
        mO = cute.make_tensor(pO, cute.make_layout((B, Nq, D), stride=(q_stride, D, 1)))
        mL = cute.make_tensor(pL, cute.make_layout((B, Nq), stride=(Nq, 1)))

        @cute.struct
        class Smem:
            sQ: cute.struct.Align[cute.struct.MemRange[dtype, bM * D], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[dtype, bN * D * self.KS], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[dtype, bN * D * self.VS], 1024]
            # full K, empty K, full V, empty V: one 8-byte mbarrier per buffer each
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2 * (self.KS + self.VS)], 8]

        vec = 128 // dtype.width
        tpr = D // vec
        g2s = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), dtype,
                                num_bits_per_copy=128),
            cute.make_layout((self.threads // tpr, tpr), stride=(tpr, 1)),
            cute.make_layout((1, vec)))
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
            (self.warps, 1, 1), permutation_mnk=(self.warps * 16, 16, 16))

        grid = (cute.ceil_div(Nq, bM), B, 1)
        self.kernel(mQ, mK, mV, mO, mL, Nq, Nk, scale_log2, g2s, tiled_mma, Smem).launch(
            grid=grid, block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor,
               mL: cute.Tensor, Nq: cutlass.Int32, Nk: cutlass.Int32, scale_log2: cutlass.Float32,
               g2s: cute.TiledCopy, tiled_mma: cute.TiledMma, Smem: cutlass.Constexpr):
        D, bM, bN, dtype = self.D, self.bM, self.bN, self.dtype
        KS, VS = self.KS, self.VS
        tidx, _, _ = cute.arch.thread_idx()
        m_block, b, _ = cute.arch.block_idx()
        q_start = m_block * bM

        gQ = cute.local_tile(mQ[b, None, None], (bM, D), (m_block, 0))
        gK = cute.local_tile(mK[b, None, None], (bN, D), (None, 0))
        gV = cute.local_tile(mV[b, None, None], (bN, D), (None, 0))
        gO = cute.local_tile(mO[b, None, None], (bM, D), (m_block, 0))
        cQ = cute.local_tile(cute.make_identity_tensor((Nq, D)), (bM, D), (m_block, 0))
        cK = cute.local_tile(cute.make_identity_tensor((Nk, D)), (bN, D), (None, 0))

        storage = cutlass.memory.SmemAllocator().allocate(Smem)
        sQ = storage.sQ.get_tensor(smem_layout(bM, D))
        sK = storage.sK.get_tensor(smem_layout_staged(bN, D, KS))
        sV = storage.sV.get_tensor(smem_layout_staged(bN, D, VS))
        sVt = cute.composition(sV, cute.make_layout((D, bN, VS), stride=(bN, 1, bN * D)))
        mbar = storage.mbar.data_ptr()
        bars = SimpleNamespace(full_k=mbar, empty_k=mbar + KS, full_v=mbar + 2 * KS, empty_v=mbar + 2 * KS + VS)

        thr_g2s = g2s.get_slice(tidx)
        tQgQ, tQsQ, tQcQ = thr_g2s.partition_S(gQ), thr_g2s.partition_D(sQ), thr_g2s.partition_S(cQ)
        tKgK, tKsK, tKcK = thr_g2s.partition_S(gK), thr_g2s.partition_D(sK), thr_g2s.partition_S(cK)
        tVgV, tVsV = thr_g2s.partition_S(gV), thr_g2s.partition_D(sV)

        thr_mma = tiled_mma.get_slice(tidx)
        tSsQ, tSsK = thr_mma.partition_A(sQ), thr_mma.partition_B(sK[None, None, 0])
        tOsVt = thr_mma.partition_B(sVt[None, None, 0])
        tSrQ, tSrK = thr_mma.make_fragment_A(tSsQ), thr_mma.make_fragment_B(tSsK)
        tOrVt = thr_mma.make_fragment_B(tOsVt)
        acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), cutlass.Float32)
        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, D)), cutlass.Float32)
        acc_O.fill(0.0)
        tScS = thr_mma.partition_C(cute.make_identity_tensor((bM, bN)))

        def ldsm(transpose):
            return cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=4), dtype)
        s2r_A = cute.make_tiled_copy_A(ldsm(False), tiled_mma)
        s2r_K = cute.make_tiled_copy_B(ldsm(False), tiled_mma)
        s2r_V = cute.make_tiled_copy_B(ldsm(True), tiled_mma)
        thr_A, thr_K, thr_V = s2r_A.get_slice(tidx), s2r_K.get_slice(tidx), s2r_V.get_slice(tidx)
        ops = SimpleNamespace(
            s2r_A=s2r_A, s2r_K=s2r_K, s2r_V=s2r_V,
            sQ=thr_A.partition_S(sQ), rQ=thr_A.retile(tSrQ),
            sK=thr_K.partition_S(sK), rK=thr_K.retile(tSrK),
            sVt=thr_V.partition_S(sVt), rVt=thr_V.retile(tOrVt))

        n_rows = acc_O.shape[0][0] * acc_O.shape[1]
        row_max = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        n_full = Nk // bN
        unmasked_end = n_full
        masked_end = cute.ceil_div(Nk, bN)
        if cutlass.const_expr(self.causal):
            unmasked_end = min(q_start // bN, n_full)
            masked_end = cute.ceil_div(min(q_start + bM, Nk), bN)

        # Barriers: full ones count the 128 threads' copies, empty ones the 4 warps' reads.
        if tidx == 0:
            for s in cutlass.range_constexpr(KS):
                cute.arch.mbarrier_init(bars.full_k + s, self.threads)
                cute.arch.mbarrier_init(bars.empty_k + s, self.warps)
            for s in cutlass.range_constexpr(VS):
                cute.arch.mbarrier_init(bars.full_v + s, self.threads)
                cute.arch.mbarrier_init(bars.empty_v + s, self.warps)
        # Q: cp.async, rows past the end zero-filled (plain stores, made visible by the barrier
        # below); its copies are covered by K_0's full barrier.
        for m in cutlass.range_constexpr(cute.size(tQsQ, mode=[1])):
            if cute.elem_less(tQcQ[0, m, 0][0], Nq):
                cute.copy(g2s, tQgQ[None, m, None], tQsQ[None, m, None])
            else:
                tQsQ[None, m, None].fill(0)
        cute.arch.sync_threads()      # the barriers are initialised; the only BAR.SYNC in the kernel

        kv = SimpleNamespace(g2s=g2s, gK=tKgK, sK=tKsK, gV=tVgV, sV=tVsV, cK=tKcK,
                             n_full=n_full, n_end=masked_end, Nk=Nk)
        # Prologue: fill both rings, K before V for each block. (With one V buffer, attend issues
        # V_j itself, once V_{j-1} is no longer needed.)
        for st in cutlass.range_constexpr(max(KS, VS)):
            if st < masked_end:
                if cutlass.const_expr(st < KS):
                    self.load(kv.gK, kv.sK[None, None, None, st], kv, st, bars.full_k + st)
                if cutlass.const_expr(st < VS and VS > 1):
                    self.load(kv.gV, kv.sV[None, None, None, st], kv, st, bars.full_v + st)

        for j in range(0, unmasked_end):
            self.attend(j, False, kv, bars, tiled_mma, ops, tSrQ, tSrK,
                        tOrVt, acc_S, acc_O, tScS, row_max, row_sum,
                        q_start, Nk, scale_log2)
        for j in range(unmasked_end, masked_end):
            self.attend(j, True, kv, bars, tiled_mma, ops, tSrQ, tSrK,
                        tOrVt, acc_S, acc_O, tScS, row_max, row_sum,
                        q_start, Nk, scale_log2)

        acc_O_mn = mn_view(acc_O)
        tOgO = mn_view(thr_mma.partition_C(gO))
        tOcO = mn_view(thr_mma.partition_C(cute.make_identity_tensor((bM, D))))
        rO = cute.make_rmem_tensor_like(acc_O, dtype)
        rO_mn = mn_view(rO)
        for r in cutlass.range_constexpr(n_rows):
            row_sum[r] = quad_reduce(row_sum[r], lambda x, y: x + y)
            inv = cute.arch.rcp_approx(row_sum[r])
            rO_mn[r, None].store((acc_O_mn[r, None].load() * inv).to(dtype))
            row = q_start + tOcO[r, 0][0]
            if row < Nq:
                cute.autovec_copy(rO_mn[r, None], tOgO[r, None])
                if tidx % 4 == 0:
                    mL[b, row] = (
                        row_max[r] + cute.math.log2(row_sum[r], fastmath=True)) * LN2

    @cute.jit
    def load(self, gX, sX, kv, j, full_bar):
        """Start the cp.async copies of key block j of K or V into one buffer and arrive on that
        buffer's full barrier when they land. A ragged last block zero-fills rows past the end with
        plain stores, so there the thread also arrives normally (release) after them."""
        if j < kv.n_full:
            cute.copy(kv.g2s, gX[None, None, None, j], sX)
            full_arrive(full_bar)
        else:
            for n in cutlass.range_constexpr(cute.size(sX, mode=[1])):
                if cute.elem_less(kv.cK[0, n, 0, j][0], kv.Nk):
                    cute.copy(kv.g2s, gX[None, n, None, j], sX[None, n, None])
                else:
                    sX[None, n, None].fill(0)
            full_arrive_inc(full_bar)
            arrive(full_bar)

    @cute.jit
    def attend(self, j, masked: cutlass.Constexpr, kv, bars, tiled_mma,
               ops, tSrQ, tSrK, tOrVt, acc_S, acc_O, tScS,
               row_max, row_sum, q_start, Nk, scale_log2):
        """Fold key block j into the running state."""
        bN, KS, VS = self.bN, self.KS, self.VS
        ks, vs = j % KS, j % VS

        mbar_wait(bars.full_k + ks, (j // KS) % 2)                 # K_j (and Q) have landed
        if cutlass.const_expr(VS == 1):
            # One V buffer: V_j goes in once every warp has finished P V_{j-1}.
            if j > 0:
                mbar_wait(bars.empty_v, (j - 1) % 2)
            self.load(kv.gV, kv.sV[None, None, None, 0], kv, j, bars.full_v)
        if cutlass.const_expr(KS > 1):
            # Refill the K buffer of block j - 1 with block j - 1 + KS: every warp finished
            # reading it one key block ago, so this wait rarely waits.
            nb = j - 1 + KS
            if j > 0 and nb < kv.n_end:
                mbar_wait(bars.empty_k + (j - 1) % KS, ((j - 1) // KS) % 2)
                self.load(kv.gK, kv.sK[None, None, None, (j - 1) % KS], kv, nb, bars.full_k + (j - 1) % KS)

        # S = Q K^T, operands read from shared memory with ldmatrix.
        acc_S.fill(0.0)
        for k in cutlass.range_constexpr(cute.size(ops.sQ, mode=[2])):
            cute.copy(ops.s2r_A, ops.sQ[None, None, k], ops.rQ[None, None, k])
            cute.copy(ops.s2r_K, ops.sK[None, None, k, ks], ops.rK[None, None, k])
            cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)
        warp_arrive(bars.empty_k + ks)                             # this warp is done with K_j

        if cutlass.const_expr(KS == 1):
            # One K buffer: K_{j+1} goes in once every warp has finished Q K_j^T.
            if j + 1 < kv.n_end:
                mbar_wait(bars.empty_k, j % 2)
                self.load(kv.gK, kv.sK[None, None, None, 0], kv, j + 1, bars.full_k)
        if cutlass.const_expr(VS > 1):
            nb = j - 1 + VS
            if j > 0 and nb < kv.n_end:
                mbar_wait(bars.empty_v + (j - 1) % VS, ((j - 1) // VS) % 2)
                self.load(kv.gV, kv.sV[None, None, None, (j - 1) % VS], kv, nb, bars.full_v + (j - 1) % VS)

        if cutlass.const_expr(masked):
            for i in cutlass.range_constexpr(cute.size(acc_S)):
                col = j * bN + tScS[i][1]
                drop = col >= Nk
                if cutlass.const_expr(self.causal):
                    drop = drop | (col > q_start + tScS[i][0])
                if drop:
                    acc_S[i] = MASK_VALUE

        acc_S_mn, acc_O_mn = mn_view(acc_S), mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_max)):
            s = acc_S_mn[r, None].load()
            m_new = quad_reduce(s.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0), cute.arch.fmax)
            m_new = cute.arch.fmax(row_max[r], m_new * scale_log2)
            p = cute.math.exp2(s * scale_log2 - m_new, fastmath=True)
            alpha = cute.math.exp2(row_max[r] - m_new, fastmath=True)
            row_sum[r] = row_sum[r] * alpha + p.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * alpha)
            row_max[r] = m_new
            acc_S_mn[r, None].store(p)

        rP = cute.make_rmem_tensor_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = as_a_operand(rP)
        mbar_wait(bars.full_v + vs, (j // VS) % 2)                 # V_j has landed
        for k in cutlass.range_constexpr(cute.size(tOrP, mode=[2])):
            cute.copy(ops.s2r_V, ops.sVt[None, None, k, vs], ops.rVt[None, None, k])
            cute.gemm(tiled_mma, acc_O, tOrP[None, None, k], tOrVt[None, None, k], acc_O)
        warp_arrive(bars.empty_v + vs)                             # this warp is done with V_j


# ----------------------------------------------------------------------------------------------
# Host side: compile once per (D, dtype, causal, config), keep the executor, pass raw pointers.
_TORCH_TO_CUTE = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


def _ptr(t, dtype):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


class FlashAttentionCuTe(torch.autograd.Function):
    CONFIG = None        # (bM, bN, warps, k_stages, v_stages) override, used by the profiling scripts
    _compiled = {}

    @staticmethod
    def config(N_q, N_k, D, is_causal, dtype):
        # (bM, bN, warps, K buffers, V buffers): v5's table with each ring depth S as (S, S).
        # Other depths at N = 4096 (results/round6/ab_v6_configs.txt) were no closer to v5.
        if N_k <= 512:
            return (64, 32, 4, 3, 3)
        return (64, 64, 4, 2, 2) if is_causal else (64, 64, 4, 1, 1)

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.ndim == 3, "expects (batch, seq, head_dim); flatten (B, H, N, D) to (B*H, N, D)"
        B, N_q, D = Q.shape
        N_k = K.shape[1]
        assert D in (32, 64, 128)
        # The tensor cores take bf16 or fp16. fp32 inputs are rounded to fp16 (the same 10-bit
        # mantissa as the TF32 that Triton's fp32 tl.dot uses), and O is cast back to fp32.
        work = Q.dtype if Q.dtype in _TORCH_TO_CUTE else torch.float16
        q, k, v = (x.detach().to(work).contiguous() for x in (Q, K, V))
        O = torch.empty_like(q)
        L = torch.empty((B, N_q), device=Q.device, dtype=torch.float32)

        cfg = FlashAttentionCuTe.CONFIG or FlashAttentionCuTe.config(N_q, N_k, D, is_causal, work)
        dt = _TORCH_TO_CUTE[work]
        key = (D, work, bool(is_causal), cfg)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = (_ptr(q, dt), _ptr(k, dt), _ptr(v, dt), _ptr(O, dt), _ptr(L, cutlass.Float32),
                B, N_q, N_k, (1.0 / math.sqrt(D)) * LOG2E, stream)
        fn = FlashAttentionCuTe._compiled.get(key)
        if fn is None:
            fn = cute.compile(FlashFwd(dt, D, bool(is_causal), *cfg), *args)
            FlashAttentionCuTe._compiled[key] = fn
        fn(*args)

        if O.dtype != Q.dtype:
            O = O.to(Q.dtype)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("forward only")


FlashAttentionTriton = FlashAttentionCuTe     # the name the harness and the tests import
