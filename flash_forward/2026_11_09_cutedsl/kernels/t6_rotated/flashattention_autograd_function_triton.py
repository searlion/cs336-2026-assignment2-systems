# flashattention_autograd_function_triton.py  (Triton, t6_rotated)
#
# The Triton control for CuTe v7_swpipe: Triton v5 with its key-block loop rotated, so that
# iteration b computes S_{b+1} = Q K_{b+1}^T before the softmax of S_b and carries it to the next
# iteration (the last block is peeled). Everything else, including the autotuning list, is v5's.
# If this gives Triton the gain v7 gives CuTe, that gain is not CuTe-only.
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_pv(O_acc, l_acc, m_acc, S_ij, V_j, q_pos, k_start, N_KEYS, qk_scale,
                K_TILE_SIZE: tl.constexpr, MASKED: tl.constexpr, is_causal: tl.constexpr):
    # Triton v5's loop body after Q K^T, unchanged: mask, online softmax, O += P V.
    if MASKED:
        k_pos = (k_start + tl.arange(0, K_TILE_SIZE))[None, :]
        keep = k_pos < N_KEYS
        if is_causal:
            keep = keep & (k_pos <= q_pos)
        S_ij = tl.where(keep, S_ij, -1e6)
    m_new = tl.maximum(m_acc, tl.max(S_ij, axis=1, keep_dims=True) * qk_scale)
    P_ij = tl.math.exp2(S_ij * qk_scale - m_new)
    alpha = tl.math.exp2(m_acc - m_new)
    l_acc = alpha * l_acc + tl.sum(P_ij, axis=1, keep_dims=True)
    P_ij = P_ij.to(V_j.dtype)
    O_acc = alpha * O_acc
    O_acc = tl.dot(P_ij, V_j, acc=O_acc)
    return O_acc, l_acc, m_new


@triton.jit
def _attend_rotated(
    O_acc, l_acc, m_acc, S_ij, Q_i, q_pos,
    K_block_ptr, V_block_ptr,
    first, last, N_KEYS, qk_scale,
    K_TILE_SIZE: tl.constexpr,
    MASKED: tl.constexpr,
    is_causal: tl.constexpr,
):
    # The Triton control for CuTe v7 (loop rotation): key blocks [first, last), each of which has a
    # next block. S_ij holds Q K_b^T on entry to iteration b; the iteration computes Q K_{b+1}^T
    # *before* the softmax of S_ij, so the two are independent and the compiler may overlap them,
    # and carries it to the next iteration. K_{b+1} may be the ragged last block, so it is loaded
    # with a bounds check.
    K_block_ptr = K_block_ptr.advance(((first + 1) * K_TILE_SIZE, 0))
    V_block_ptr = V_block_ptr.advance((first * K_TILE_SIZE, 0))
    for b in range(first, last):
        K_n = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        if MASKED:
            V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        else:
            V_j = tl.load(V_block_ptr)
        S_next = tl.dot(Q_i, tl.trans(K_n))
        O_acc, l_acc, m_acc = _softmax_pv(O_acc, l_acc, m_acc, S_ij, V_j, q_pos, b * K_TILE_SIZE, N_KEYS,
                                          qk_scale, K_TILE_SIZE=K_TILE_SIZE, MASKED=MASKED, is_causal=is_causal)
        S_ij = S_next
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    return O_acc, l_acc, m_acc, S_ij


# From the second sweep (after tile-skipping and exp2), chosen so that every benchmark shape
# is within 1% of its best configuration in that sweep: 64x32 tiles with 3, 4 or 5 stages
# for short sequences, 64x128 with 2 stages for long non-causal ones, and 64x32 with 2
# stages, the only one that fits in 99 KB of shared memory for fp32 inputs at D = 128.
AUTOTUNE_CONFIGS = [
    triton.Config({"Q_TILE_SIZE": bq, "K_TILE_SIZE": bk}, num_warps=nw, num_stages=ns)
    for bq, bk, nw, ns in [(64, 32, 4, 3), (64, 32, 4, 4), (64, 32, 4, 5), (64, 128, 4, 2), (64, 32, 4, 2)]
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N_QUERIES", "N_KEYS", "D", "is_causal"])
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
        offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D), strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,), strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,), order=(0,),
    )

    O_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_acc = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
    m_acc = tl.full((Q_TILE_SIZE, 1), value=float("-inf"), dtype=tl.float32)

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    q_start = query_tile_index * Q_TILE_SIZE
    q_pos = (q_start + tl.arange(0, Q_TILE_SIZE))[:, None]

    # Key tiles that end at or before full_stop lie entirely inside the tensor.
    full_stop = (N_KEYS // K_TILE_SIZE) * K_TILE_SIZE
    if is_causal:
        # Every key below the first query of this tile is visible to every row,
        # so those tiles need no mask. The tiles that straddle the diagonal get
        # the mask, and nothing beyond the last query of the tile is visited.
        unmasked_stop = tl.minimum((q_start // K_TILE_SIZE) * K_TILE_SIZE, full_stop)
        masked_stop = tl.minimum(q_start + Q_TILE_SIZE, N_KEYS)
    else:
        # Only a ragged last key tile needs the mask.
        unmasked_stop = full_stop
        masked_stop = N_KEYS

    # Fold log2(e) into the softmax scale once, so the inner loop can use exp2.
    qk_scale = scale * 1.4426950408889634

    # Key blocks in block units: [0, n_unmasked) need no mask, [n_unmasked, n_total) do.
    n_unmasked = unmasked_stop // K_TILE_SIZE
    n_total = tl.cdiv(masked_stop, K_TILE_SIZE)
    S_ij = tl.dot(Q_i, tl.trans(tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")))
    O_acc, l_acc, m_acc, S_ij = _attend_rotated(
        O_acc, l_acc, m_acc, S_ij, Q_i, q_pos, K_block_ptr, V_block_ptr,
        0, tl.minimum(n_unmasked, n_total - 1), N_KEYS, qk_scale,
        K_TILE_SIZE=K_TILE_SIZE, MASKED=False, is_causal=is_causal,
    )
    O_acc, l_acc, m_acc, S_ij = _attend_rotated(
        O_acc, l_acc, m_acc, S_ij, Q_i, q_pos, K_block_ptr, V_block_ptr,
        n_unmasked, n_total - 1, N_KEYS, qk_scale,
        K_TILE_SIZE=K_TILE_SIZE, MASKED=True, is_causal=is_causal,
    )
    # The last key block has no next block (peeled).
    b_last = n_total - 1
    V_last = tl.load(V_block_ptr.advance((b_last * K_TILE_SIZE, 0)), boundary_check=(0, 1), padding_option="zero")
    if b_last >= n_unmasked:
        O_acc, l_acc, m_acc = _softmax_pv(O_acc, l_acc, m_acc, S_ij, V_last, q_pos, b_last * K_TILE_SIZE, N_KEYS,
                                          qk_scale, K_TILE_SIZE=K_TILE_SIZE, MASKED=True, is_causal=is_causal)
    else:
        O_acc, l_acc, m_acc = _softmax_pv(O_acc, l_acc, m_acc, S_ij, V_last, q_pos, b_last * K_TILE_SIZE, N_KEYS,
                                          qk_scale, K_TILE_SIZE=K_TILE_SIZE, MASKED=False, is_causal=is_causal)

    O_i = (O_acc / l_acc).to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))

    # m and log2(l) are in log2 units; the backward pass expects natural-log L.
    L_i = tl.reshape((m_acc + tl.math.log2(l_acc)) * 0.6931471805599453, (Q_TILE_SIZE,))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.ndim == 3, "expects (batch, seq, head_dim); flatten (B, H, N, D) to (B*H, N, D)"
        assert Q.stride(-1) == 1 and K.stride(-1) == 1 and V.stride(-1) == 1
        B, N_q, D = Q.shape
        N_k = K.shape[1]
        assert D in (16, 32, 64, 128), "block_shape dims must be powers of two"

        O = torch.empty_like(Q)
        L = torch.empty((B, N_q), device=Q.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(N_q, meta["Q_TILE_SIZE"]), B)  # noqa: E731

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            1.0 / math.sqrt(D),
            D=D,
            is_causal=is_causal,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("tiled backward is left for future articles")
