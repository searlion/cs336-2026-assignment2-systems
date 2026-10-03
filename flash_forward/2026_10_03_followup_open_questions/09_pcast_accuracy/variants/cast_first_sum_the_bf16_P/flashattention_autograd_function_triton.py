# flashattention_autograd_function_triton.py
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attend_key_tiles(
    O_acc, l_acc, m_acc, Q_i, q_pos,
    K_block_ptr, V_block_ptr,
    start, stop, N_KEYS, qk_scale,
    K_TILE_SIZE: tl.constexpr,
    MASKED: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Fold the key tiles [start, stop) into the running state. MASKED is a
    # compile-time flag: the unmasked copy of this loop has no compare, no
    # select and no bounds checks on its loads.
    K_block_ptr = K_block_ptr.advance((start, 0))
    V_block_ptr = V_block_ptr.advance((start, 0))
    for k_start in range(start, stop, K_TILE_SIZE):
        if MASKED:
            K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        else:
            K_j = tl.load(K_block_ptr)
            V_j = tl.load(V_block_ptr)

        S_ij = tl.dot(Q_i, tl.trans(K_j))                   # raw scores, not yet scaled

        if MASKED:
            k_pos = (k_start + tl.arange(0, K_TILE_SIZE))[None, :]
            keep = k_pos < N_KEYS
            if is_causal:
                keep = keep & (k_pos <= q_pos)
            S_ij = tl.where(keep, S_ij, -1e6)

        # m is kept in log2 units of the scaled scores. qk_scale > 0, so the
        # row max of the raw scores times qk_scale is the max of the scaled ones,
        # and the scale is applied to Q_TILE values here instead of Q_TILE * K_TILE.
        m_new = tl.maximum(m_acc, tl.max(S_ij, axis=1, keep_dims=True) * qk_scale)
        # One FFMA and one ex2 per score: exp(scale*s - m) == 2^(s*qk_scale - m').
        P_ij = tl.math.exp2(S_ij * qk_scale - m_new)
        alpha = tl.math.exp2(m_acc - m_new)
        P_ij = P_ij.to(V_j.dtype)
        l_acc = alpha * l_acc + tl.sum(P_ij.to(tl.float32), axis=1, keep_dims=True)

        O_acc = alpha * O_acc
        O_acc = tl.dot(P_ij, V_j, acc=O_acc)
        m_acc = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    return O_acc, l_acc, m_acc


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

    O_acc, l_acc, m_acc = _attend_key_tiles(
        O_acc, l_acc, m_acc, Q_i, q_pos, K_block_ptr, V_block_ptr,
        0, unmasked_stop, N_KEYS, qk_scale,
        K_TILE_SIZE=K_TILE_SIZE, MASKED=False, is_causal=is_causal,
    )
    O_acc, l_acc, m_acc = _attend_key_tiles(
        O_acc, l_acc, m_acc, Q_i, q_pos, K_block_ptr, V_block_ptr,
        unmasked_stop, masked_stop, N_KEYS, qk_scale,
        K_TILE_SIZE=K_TILE_SIZE, MASKED=True, is_causal=is_causal,
    )

    O_i = (O_acc / l_acc).to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))

    # m and log2(l) are in log2 units; the backward pass expects natural-log L.
    L_i = tl.reshape((m_acc + tl.math.log2(l_acc)) * 0.6931471805599453, (Q_TILE_SIZE,))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    Q_TILE_SIZE = 64
    K_TILE_SIZE = 32
    NUM_WARPS = 4
    NUM_STAGES = 3

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.ndim == 3, "expects (batch, seq, head_dim); flatten (B, H, N, D) to (B*H, N, D)"
        assert Q.stride(-1) == 1 and K.stride(-1) == 1 and V.stride(-1) == 1
        B, N_q, D = Q.shape
        N_k = K.shape[1]
        assert D in (16, 32, 64, 128), "block_shape dims must be powers of two"

        O = torch.empty_like(Q)
        L = torch.empty((B, N_q), device=Q.device, dtype=torch.float32)
        grid = (triton.cdiv(N_q, FlashAttentionTriton.Q_TILE_SIZE), B)
        # The tuned config assumes bf16 rows of D = 64 (128 bytes). fp32 at D = 128 has
        # 512-byte rows: with 3 stages Triton asks for 107,520 bytes of shared memory,
        # more than the 101,376 an Ada block may have, so drop to 2 stages there.
        num_stages = FlashAttentionTriton.NUM_STAGES if D * Q.element_size() <= 256 else 2

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
            Q_TILE_SIZE=FlashAttentionTriton.Q_TILE_SIZE,
            K_TILE_SIZE=FlashAttentionTriton.K_TILE_SIZE,
            is_causal=is_causal,
            num_warps=FlashAttentionTriton.NUM_WARPS,
            num_stages=num_stages,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("tiled backward is left for future articles")
