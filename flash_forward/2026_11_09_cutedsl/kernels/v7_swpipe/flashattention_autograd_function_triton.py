# flashattention_autograd_function_triton.py  (CuTe DSL, v7_swpipe)
#
# v5 with the key-block loop software-pipelined inside each warp. In v5 a warp computes
# S_j = Q K_j^T, then the softmax of S_j, then P_j V_j: the softmax has no tensor-core work next to
# it, so the warp's tensor cores idle through it unless other warps fill the gap. Here iteration j
# computes S_{j+1} = Q K_{j+1}^T in 16-wide k-slices, and after each slice one piece of the softmax
# of S_j (row maxima; exp and sum of one row; rescale of O and the bf16 P), so independent
# tensor-core and softmax work sit side by side in one basic block. Then P_j V_j. S is carried
# across iterations in registers (two S tiles live at once). K_{j+1} must be in shared memory a
# whole iteration earlier, so K and V each get a 2-deep ring, with one barrier per key block; the
# last key block is peeled (it has no next S). The maths is v5's, in v5's order, so O and L are
# bitwise the same as v5's at the same bM x bN.
import math
from types import SimpleNamespace

import torch
import cutlass
import cutlass.cute as cute
import cutlass.memory
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_ptr
import cuda.bindings.driver as cuda

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


class FlashFwd:
    def __init__(self, dtype, D, causal, bM=64, bN=64, warps=4):
        self.dtype, self.D, self.causal = dtype, D, causal
        self.bM, self.bN, self.warps = bM, bN, warps
        self.stages = 2                                  # K and V rings
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
            sK: cute.struct.Align[cute.struct.MemRange[dtype, bN * D * self.stages], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[dtype, bN * D * self.stages], 1024]

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
        D, bM, bN, dtype, S = self.D, self.bM, self.bN, self.dtype, self.stages
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
        sK = storage.sK.get_tensor(smem_layout_staged(bN, D, S))
        sV = storage.sV.get_tensor(smem_layout_staged(bN, D, S))
        sVt = cute.composition(sV, cute.make_layout((D, bN, S), stride=(bN, 1, bN * D)))

        thr_g2s = g2s.get_slice(tidx)
        tQgQ, tQsQ, tQcQ = thr_g2s.partition_S(gQ), thr_g2s.partition_D(sQ), thr_g2s.partition_S(cQ)
        tKgK, tKsK, tKcK = thr_g2s.partition_S(gK), thr_g2s.partition_D(sK), thr_g2s.partition_S(cK)
        tVgV, tVsV = thr_g2s.partition_S(gV), thr_g2s.partition_D(sV)

        thr_mma = tiled_mma.get_slice(tidx)
        tSsQ, tSsK = thr_mma.partition_A(sQ), thr_mma.partition_B(sK[None, None, 0])
        tOsVt = thr_mma.partition_B(sVt[None, None, 0])
        tSrQ, tSrK = thr_mma.make_fragment_A(tSsQ), thr_mma.make_fragment_B(tSsK)
        tOrVt = thr_mma.make_fragment_B(tOsVt)
        acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), cutlass.Float32)       # S_j
        acc_N = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), cutlass.Float32)       # S_{j+1}
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

        # Prologue. Group 1: Q and K_0. Group 2: K_1 and V_0.
        for m in cutlass.range_constexpr(cute.size(tQsQ, mode=[1])):
            if cute.elem_less(tQcQ[0, m, 0][0], Nq):
                cute.copy(g2s, tQgQ[None, m, None], tQsQ[None, m, None])
            else:
                tQsQ[None, m, None].fill(0)
        kv = SimpleNamespace(g2s=g2s, gK=tKgK, sK=tKsK, gV=tVgV, sV=tVsV, cK=tKcK,
                             n_full=n_full, n_end=masked_end, Nk=Nk)
        self.start_load(kv.gK, kv.sK[None, None, None, 0], kv, 0)
        cute.arch.cp_async_commit_group()
        if 1 < masked_end:
            self.start_load(kv.gK, kv.sK[None, None, None, 1], kv, 1)
        self.start_load(kv.gV, kv.sV[None, None, None, 0], kv, 0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)              # Q and K_0 have landed ...
        cute.arch.sync_threads()                      # ... for every thread
        acc_S.fill(0.0)
        for k in cutlass.range_constexpr(cute.size(ops.sQ, mode=[2])):
            self.qk_slice(k, 0, tiled_mma, ops, tSrQ, tSrK, acc_S)

        # Every key block but the last has a next S to compute; the last one is peeled.
        st = SimpleNamespace(acc_S=acc_S, acc_N=acc_N, acc_O=acc_O, row_max=row_max, row_sum=row_sum)
        last = masked_end - 1
        for j in range(0, min(unmasked_end, last)):
            self.attend(j, False, True, kv, st, tiled_mma, ops, tSrQ, tSrK, tOrVt, tScS, q_start, Nk, scale_log2)
        for j in range(unmasked_end, last):
            self.attend(j, True, True, kv, st, tiled_mma, ops, tSrQ, tSrK, tOrVt, tScS, q_start, Nk, scale_log2)
        if last >= unmasked_end:
            self.attend(last, True, False, kv, st, tiled_mma, ops, tSrQ, tSrK, tOrVt, tScS, q_start, Nk, scale_log2)
        else:
            self.attend(last, False, False, kv, st, tiled_mma, ops, tSrQ, tSrK, tOrVt, tScS, q_start, Nk, scale_log2)

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
    def start_load(self, gX, sX, kv, j):
        """Start the cp.async copies of key block j of K or V into its shared buffer. Only a
        ragged last block checks rows; rows past the end are zero-filled."""
        if j < kv.n_full:
            cute.copy(kv.g2s, gX[None, None, None, j], sX)
        else:
            for n in cutlass.range_constexpr(cute.size(sX, mode=[1])):
                if cute.elem_less(kv.cK[0, n, 0, j][0], kv.Nk):
                    cute.copy(kv.g2s, gX[None, n, None, j], sX[None, n, None])
                else:
                    sX[None, n, None].fill(0)

    def qk_slice(self, k, slot, tiled_mma, ops, tSrQ, tSrK, acc):
        """One 16-wide k-slice of S = Q K^T: ldmatrix Q and K, then the MMAs."""
        cute.copy(ops.s2r_A, ops.sQ[None, None, k], ops.rQ[None, None, k])
        cute.copy(ops.s2r_K, ops.sK[None, None, k, slot], ops.rK[None, None, k])
        cute.gemm(tiled_mma, acc, tSrQ[None, None, k], tSrK[None, None, k], acc)

    @cute.jit
    def attend(self, j, masked: cutlass.Constexpr, has_next: cutlass.Constexpr, kv, st, tiled_mma,
               ops, tSrQ, tSrK, tOrVt, tScS, q_start, Nk, scale_log2):
        """Fold key block j (S_j already in st.acc_S) into the running state, and compute S_{j+1}."""
        bN, S = self.bN, self.stages
        cute.arch.cp_async_wait_group(0)              # K_{j+1} and V_j have landed ...
        cute.arch.sync_threads()                      # ... and every warp is done with K_j, V_{j-1}
        if j + 2 < kv.n_end:
            self.start_load(kv.gK, kv.sK[None, None, None, j % S], kv, j + 2)
        if j + 1 < kv.n_end:
            self.start_load(kv.gV, kv.sV[None, None, None, (j + 1) % S], kv, j + 1)
        cute.arch.cp_async_commit_group()

        acc_S, acc_N, acc_O, row_max, row_sum = st.acc_S, st.acc_N, st.acc_O, st.row_max, st.row_sum
        if cutlass.const_expr(masked):
            for i in cutlass.range_constexpr(cute.size(acc_S)):
                col = j * bN + tScS[i][1]
                drop = col >= Nk
                if cutlass.const_expr(self.causal):
                    drop = drop | (col > q_start + tScS[i][0])
                if drop:
                    acc_S[i] = MASK_VALUE

        # The softmax of S_j in pieces (v5's arithmetic, split by row), to sit between the k-slices
        # of S_{j+1}: row maxima and rescale factors; exp2 and sum of each row; rescale O, P to bf16.
        acc_S_mn, acc_O_mn = mn_view(acc_S), mn_view(acc_O)
        n_rows = cute.size(row_max)
        m_new = [None] * n_rows
        alpha = [None] * n_rows
        rP = cute.make_rmem_tensor_like(acc_S, self.dtype)

        def piece_max():
            for r in cutlass.range_constexpr(n_rows):
                s = acc_S_mn[r, None].load()
                m = quad_reduce(s.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0), cute.arch.fmax)
                m_new[r] = cute.arch.fmax(row_max[r], m * scale_log2)
                alpha[r] = cute.math.exp2(row_max[r] - m_new[r], fastmath=True)

        def piece_row(r):
            def f():
                p = cute.math.exp2(acc_S_mn[r, None].load() * scale_log2 - m_new[r], fastmath=True)
                row_sum[r] = row_sum[r] * alpha[r] + p.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
                acc_S_mn[r, None].store(p)
            return f

        def piece_rescale():
            for r in cutlass.range_constexpr(n_rows):
                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * alpha[r])
                row_max[r] = m_new[r]
            rP.store(acc_S.load().to(self.dtype))

        pieces = [piece_max] + [piece_row(r) for r in range(n_rows)] + [piece_rescale]
        n_k = cute.size(ops.sQ, mode=[2])
        if cutlass.const_expr(has_next):
            acc_N.fill(0.0)
            nxt = (j + 1) % S
            done = 0
            for k in cutlass.range_constexpr(n_k):
                self.qk_slice(k, nxt, tiled_mma, ops, tSrQ, tSrK, acc_N)
                upto = (k + 1) * len(pieces) // n_k          # pieces due after this slice
                for i in cutlass.range_constexpr(done, upto):
                    pieces[i]()
                done = upto
        else:
            for f in pieces:
                f()

        # O += P_j V_j
        tOrP = as_a_operand(rP)
        cur = j % S
        for k in cutlass.range_constexpr(cute.size(tOrP, mode=[2])):
            cute.copy(ops.s2r_V, ops.sVt[None, None, k, cur], ops.rVt[None, None, k])
            cute.gemm(tiled_mma, acc_O, tOrP[None, None, k], tOrVt[None, None, k], acc_O)
        if cutlass.const_expr(has_next):
            acc_S.store(acc_N.load())                 # S_{j+1} becomes the next iteration's S_j


# ----------------------------------------------------------------------------------------------
# Host side: compile once per (D, dtype, causal, config), keep the executor, pass raw pointers.
_TORCH_TO_CUTE = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


def _ptr(t, dtype):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


class FlashAttentionCuTe(torch.autograd.Function):
    CONFIG = None        # (bM, bN, warps) override, used by the profiling scripts
    _compiled = {}

    @staticmethod
    def config(N_q, N_k, D, is_causal, dtype):
        # (bM, bN, warps): v5's tile sizes (so the results match v5's bit for bit); K and V always
        # have 2-deep rings here.
        if N_k <= 512:
            return (64, 32, 4)
        return (64, 64, 4)

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
