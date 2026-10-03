# flashattention_autograd_function_triton.py
import math
import torch
import triton
import triton.language as tl


# The per-shape winners of the sweep (Q_TILE, K_TILE, num_warps, num_stages). They differ
# only in the pipeline depth and, for long causal runs, the key tile.
AUTOTUNE_CONFIGS = [
    triton.Config({"Q_TILE_SIZE": bq, "K_TILE_SIZE": bk}, num_warps=nw, num_stages=ns)
    for bq, bk, nw, ns in [(64, 32, 4, 3), (64, 32, 4, 4), (64, 32, 4, 2), (64, 16, 4, 5)]
]


# One tuning run per distinct (N_QUERIES, N_KEYS, D, is_causal) and input dtype; Triton adds
# the dtypes of the tensor arguments to the key on its own.
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

    # Block pointers: a (rows, D) tile into each tensor for this batch element.
    # Q and O tiles start at this program's query tile; K and V start at row 0
    # and are advanced inside the loop.
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

    # Running state, kept in fp32 regardless of input dtype.
    O_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_acc = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
    m_acc = tl.full((Q_TILE_SIZE, 1), value=float("-inf"), dtype=tl.float32)

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    q_pos = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE))[:, None]

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_pos = (j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE))[None, :]

        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale          # (Q_TILE, K_TILE), fp32

        # FIX 2: zero-padded keys past N_KEYS score 0, not -inf. Mask them.
        keep = k_pos < N_KEYS
        if is_causal:
            keep = keep & (k_pos <= q_pos)
        S_ij = tl.where(keep, S_ij, -1e6)

        m_new = tl.maximum(m_acc, tl.max(S_ij, axis=1, keep_dims=True))
        P_ij = tl.exp(S_ij - m_new)
        alpha = tl.exp(m_acc - m_new)
        l_acc = alpha * l_acc + tl.sum(P_ij, axis=1, keep_dims=True)

        # FIX 1: the cast must be assigned. tl.dot needs both operands in the
        # same dtype; the fp32 accumulator is passed separately via acc=.
        P_ij = P_ij.to(V_j.dtype)
        O_acc = alpha * O_acc
        O_acc = tl.dot(P_ij, V_j, acc=O_acc)
        m_acc = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = (O_acc / l_acc).to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))

    L_i = tl.reshape(m_acc + tl.log(l_acc), (Q_TILE_SIZE,))
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
        # The grid depends on the Q tile the autotuner picks, so it is a function of the config.
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
