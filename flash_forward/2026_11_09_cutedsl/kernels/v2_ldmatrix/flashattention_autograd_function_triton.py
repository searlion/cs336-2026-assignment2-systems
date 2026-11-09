# flashattention_autograd_function_triton.py  (CuTe DSL, v2_ldmatrix)
#
# v1 with one change: the MMA operands (Q, K, V^T and P) are read from shared memory with
# ldmatrix, which loads four 8x8 tiles per instruction for the whole warp, instead of with
# plain 32-bit and 16-bit loads. Still, as in v1:
#   - global -> shared copies go through registers (LDG then STS) and block the thread,
#   - P is written to shared memory and read back as the A operand of P @ V,
#   - one buffer each for K and V, with barriers around every load.
# The file keeps the Triton file's name so the unchanged harness and tests can import it.
import math
from types import SimpleNamespace

import torch
import cutlass
import cutlass.cute as cute
import cutlass.memory
from cutlass.cute.nvgpu import warp
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


def quad_reduce(v, op):
    """The 4 lanes of a quad hold the same rows: combine their partial results."""
    v = op(v, cute.arch.shuffle_sync_bfly(v, offset=2, mask=-1, mask_and_clamp=31))
    return op(v, cute.arch.shuffle_sync_bfly(v, offset=1, mask=-1, mask_and_clamp=31))


class FlashFwd:
    def __init__(self, dtype, D, causal, bM=64, bN=64, warps=4):
        self.dtype, self.D, self.causal = dtype, D, causal
        self.bM, self.bN, self.warps = bM, bN, warps
        self.threads = warps * 32

    @cute.jit
    def __call__(self, pQ: cute.Pointer, pK: cute.Pointer, pV: cute.Pointer, pO: cute.Pointer,
                 pL: cute.Pointer, B: cutlass.Int32, Nq: cutlass.Int32, Nk: cutlass.Int32,
                 scale_log2: cutlass.Float32, stream: cuda.CUstream):
        D, bM, bN, dtype = self.D, self.bM, self.bN, self.dtype
        # (batch, rows, D) row-major tensors; N and B are runtime values, D is fixed.
        # cute.assume tells the compiler each batch starts 16-byte aligned (D is a multiple of 8).
        q_stride, k_stride = cute.assume(Nq * D, divby=D), cute.assume(Nk * D, divby=D)
        mQ = cute.make_tensor(pQ, cute.make_layout((B, Nq, D), stride=(q_stride, D, 1)))
        mK = cute.make_tensor(pK, cute.make_layout((B, Nk, D), stride=(k_stride, D, 1)))
        mV = cute.make_tensor(pV, cute.make_layout((B, Nk, D), stride=(k_stride, D, 1)))
        mO = cute.make_tensor(pO, cute.make_layout((B, Nq, D), stride=(q_stride, D, 1)))
        mL = cute.make_tensor(pL, cute.make_layout((B, Nq), stride=(Nq, 1)))

        @cute.struct
        class Smem:
            sQ: cute.struct.Align[cute.struct.MemRange[dtype, bM * D], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[dtype, bN * D], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[dtype, bN * D], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[dtype, bM * bN], 1024]

        # Global -> shared: each thread moves 8 elements (16 bytes) per step, through registers.
        vec = 128 // dtype.width
        tpr = D // vec                                  # threads per row of a tile
        g2s = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128),
            cute.make_layout((self.threads // tpr, tpr), stride=(tpr, 1)),
            cute.make_layout((1, vec)))
        # Tensor-core MMA: 16x8x16 bf16 atoms, fp32 accumulate, warps stacked along the rows.
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
        tidx, _, _ = cute.arch.thread_idx()
        m_block, b, _ = cute.arch.block_idx()
        q_start = m_block * bM

        # This block's tiles of global memory, and matching tiles of coordinates.
        gQ = cute.local_tile(mQ[b, None, None], (bM, D), (m_block, 0))          # (bM, D)
        gK = cute.local_tile(mK[b, None, None], (bN, D), (None, 0))             # (bN, D, n_blocks)
        gV = cute.local_tile(mV[b, None, None], (bN, D), (None, 0))
        gO = cute.local_tile(mO[b, None, None], (bM, D), (m_block, 0))
        cQ = cute.local_tile(cute.make_identity_tensor((Nq, D)), (bM, D), (m_block, 0))
        cK = cute.local_tile(cute.make_identity_tensor((Nk, D)), (bN, D), (None, 0))

        # Shared memory: swizzled tiles (the only change from v0).
        storage = cutlass.memory.SmemAllocator().allocate(Smem)
        sQ = storage.sQ.get_tensor(smem_layout(bM, D))
        sK = storage.sK.get_tensor(smem_layout(bN, D))
        sV = storage.sV.get_tensor(smem_layout(bN, D))
        sP = storage.sP.get_tensor(smem_layout(bM, bN))
        # V viewed as (D, bN), the B operand of P @ V. No data moves.
        sVt = cute.composition(sV, cute.make_layout((D, bN), stride=(bN, 1)))

        thr_g2s = g2s.get_slice(tidx)
        tQgQ, tQsQ, tQcQ = thr_g2s.partition_S(gQ), thr_g2s.partition_D(sQ), thr_g2s.partition_S(cQ)
        tKgK, tKsK, tKcK = thr_g2s.partition_S(gK), thr_g2s.partition_D(sK), thr_g2s.partition_S(cK)
        tVgV, tVsV = thr_g2s.partition_S(gV), thr_g2s.partition_D(sV)

        thr_mma = tiled_mma.get_slice(tidx)
        tSsQ, tSsK = thr_mma.partition_A(sQ), thr_mma.partition_B(sK)
        tOsP, tOsVt = thr_mma.partition_A(sP), thr_mma.partition_B(sVt)
        tSrQ, tSrK = thr_mma.make_fragment_A(tSsQ), thr_mma.make_fragment_B(tSsK)
        tOrP, tOrVt = thr_mma.make_fragment_A(tOsP), thr_mma.make_fragment_B(tOsVt)
        tPsP = thr_mma.partition_C(sP)                                          # where this thread writes P
        acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), cutlass.Float32)
        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, D)), cutlass.Float32)
        acc_O.fill(0.0)
        tScS = thr_mma.partition_C(cute.make_identity_tensor((bM, bN)))         # (row, col) of each acc_S value

        # ldmatrix copies, laid out to match the MMA fragments (V^T uses the transposing form).
        def ldsm(transpose):
            return cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=4), dtype)
        s2r_A = cute.make_tiled_copy_A(ldsm(False), tiled_mma)
        s2r_K = cute.make_tiled_copy_B(ldsm(False), tiled_mma)
        s2r_V = cute.make_tiled_copy_B(ldsm(True), tiled_mma)
        thr_A, thr_K, thr_V = s2r_A.get_slice(tidx), s2r_K.get_slice(tidx), s2r_V.get_slice(tidx)
        ops = SimpleNamespace(
            s2r_A=s2r_A, s2r_K=s2r_K, s2r_V=s2r_V,
            sQ=thr_A.partition_S(sQ), rQ=thr_A.retile(tSrQ),        # (copy view of smem, of registers)
            sK=thr_K.partition_S(sK), rK=thr_K.retile(tSrK),
            sP=thr_A.partition_S(sP), rP=thr_A.retile(tOrP),
            sVt=thr_V.partition_S(sVt), rVt=thr_V.retile(tOrVt))

        n_rows = acc_O.shape[0][0] * acc_O.shape[1]                             # rows this thread owns
        row_max = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        # Q: once, with rows past the end filled with zeros.
        for m in cutlass.range_constexpr(cute.size(tQsQ, mode=[1])):
            if cute.elem_less(tQcQ[0, m, 0][0], Nq):
                cute.copy(g2s, tQgQ[None, m, None], tQsQ[None, m, None])
            else:
                tQsQ[None, m, None].fill(0)

        # Which key blocks need the mask (same split as the Triton kernel).
        n_full = Nk // bN
        unmasked_end = n_full
        masked_end = cute.ceil_div(Nk, bN)
        if cutlass.const_expr(self.causal):
            unmasked_end = min(q_start // bN, n_full)
            masked_end = cute.ceil_div(min(q_start + bM, Nk), bN)

        for j in range(0, unmasked_end):
            self.attend(j, False, g2s, tKgK, tKsK, tKcK, tVgV, tVsV, tiled_mma, ops, tSrQ, tSrK,
                        tOrP, tOrVt, tPsP, acc_S, acc_O, tScS, row_max, row_sum,
                        q_start, Nk, scale_log2)
        for j in range(unmasked_end, masked_end):
            self.attend(j, True, g2s, tKgK, tKsK, tKcK, tVgV, tVsV, tiled_mma, ops, tSrQ, tSrK,
                        tOrP, tOrVt, tPsP, acc_S, acc_O, tScS, row_max, row_sum,
                        q_start, Nk, scale_log2)

        # Epilogue: O = acc_O / l, L = (m + log2 l) * ln 2, rows past Nq are not written.
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
    def attend(self, j, masked: cutlass.Constexpr, g2s, tKgK, tKsK, tKcK, tVgV, tVsV, tiled_mma,
               ops, tSrQ, tSrK, tOrP, tOrVt, tPsP, acc_S, acc_O, tScS,
               row_max, row_sum, q_start, Nk, scale_log2):
        """Fold key block j into the running state."""
        bN = self.bN
        cute.arch.sync_threads()                      # everyone is done with the previous K, V
        for n in cutlass.range_constexpr(cute.size(tKsK, mode=[1])):
            if cutlass.const_expr(masked):
                if cute.elem_less(tKcK[0, n, 0, j][0], Nk):
                    cute.copy(g2s, tKgK[None, n, None, j], tKsK[None, n, None])
                    cute.copy(g2s, tVgV[None, n, None, j], tVsV[None, n, None])
                else:
                    tKsK[None, n, None].fill(0)
                    tVsV[None, n, None].fill(0)
            else:
                cute.copy(g2s, tKgK[None, n, None, j], tKsK[None, n, None])
                cute.copy(g2s, tVgV[None, n, None, j], tVsV[None, n, None])
        cute.arch.sync_threads()                      # K, V are in shared memory

        # S = Q K^T, operands read from shared memory with ldmatrix.
        acc_S.fill(0.0)
        for k in cutlass.range_constexpr(cute.size(ops.sQ, mode=[2])):
            cute.copy(ops.s2r_A, ops.sQ[None, None, k], ops.rQ[None, None, k])
            cute.copy(ops.s2r_K, ops.sK[None, None, k], ops.rK[None, None, k])
            cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)

        if cutlass.const_expr(masked):
            for i in cutlass.range_constexpr(cute.size(acc_S)):
                col = j * bN + tScS[i][1]
                drop = col >= Nk
                if cutlass.const_expr(self.causal):
                    drop = drop | (col > q_start + tScS[i][0])
                if drop:
                    acc_S[i] = MASK_VALUE

        # Online softmax, one row at a time; m is in log2 units of the scaled scores.
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

        # P goes to shared memory as bf16, then comes back as the A operand of P @ V.
        rP = cute.make_rmem_tensor_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        cute.autovec_copy(rP, tPsP)
        cute.arch.sync_warp()                         # each warp reads back only its own rows
        for k in cutlass.range_constexpr(cute.size(ops.sP, mode=[2])):
            cute.copy(ops.s2r_A, ops.sP[None, None, k], ops.rP[None, None, k])
            cute.copy(ops.s2r_V, ops.sVt[None, None, k], ops.rVt[None, None, k])
            cute.gemm(tiled_mma, acc_O, tOrP[None, None, k], tOrVt[None, None, k], acc_O)
        cute.arch.sync_warp()                         # done reading P before the next block writes it


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
            bM, bN, warps = cfg
            fn = cute.compile(FlashFwd(dt, D, bool(is_causal), bM, bN, warps), *args)
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
