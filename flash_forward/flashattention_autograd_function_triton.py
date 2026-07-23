import triton
import triton.language as tl
import torch
import einops
import math

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
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
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    # Each program only takes a slice of Q_block, with query_tile_index * Q_TILE_SIZE offset
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    # Each program processes the entire K_block, hence offset begins at (0,0)
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )
    # Use K_TILE_SIZE as the inner j loop always load the K tile and V tile together for the same key block
    # Each program processes the entire V_block, hence offset begins at (0,0)
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    # Initialise the on-chip accumulators
    # Accumulate in fp32 for numerical stability
    O_acc = tl.zeros(
        shape=(Q_TILE_SIZE, D), 
        dtype=tl.float32
    )
    l_acc = tl.zeros(
        shape=(Q_TILE_SIZE,1), 
        dtype=tl.float32
    )
    m_acc = tl.full(
        shape=(Q_TILE_SIZE,1), 
        value=-float('inf'), 
        dtype=tl.float32
    )
    Q_i = tl.load(
        pointer=Q_block_ptr, 
        boundary_check=(0,1),
        padding_option="zero"
    )

    # For causal mask
    row_query_positions = (tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE)[:, None]

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # For causal mask
        column_query_positions = (tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE)[None, :]
        # tl.device_print("rows:", row_query_positions)
        # tl.device_print("columns:", column_query_positions)
        keep_mask = column_query_positions <= row_query_positions
        # tl.device_print("keep mask:", keep_mask)

        K_j = tl.load(
            pointer=K_block_ptr,
            boundary_check=(0,1),
            padding_option="zero"
        )
        V_j = tl.load(
            pointer=V_block_ptr,
            boundary_check=(0,1),
            padding_option="zero"
        )
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            S_ij = tl.where(keep_mask, S_ij, -1e6)
        # S_ij is shape [BLOCK_M, BLOCK_N]
        # m_ip is shape [BLOCK_M]
        # Reduce along the last dimension (axis=1) and keep dims for broadcasting
        s_max = tl.max(S_ij, axis=1, keep_dims=True)  # shape [BLOCK_M, 1]
        # Expand/broadcast m_ip to match shape [BLOCK_M, 1] or let Triton handle it
        m_ij = tl.maximum(m_acc, s_max)  # shape [BLOCK_M, 1]
        P_tilde_ij = tl.exp(S_ij - m_ij)
        mip_mik_diff = tl.exp(m_acc - m_ij)
        l_ij = mip_mik_diff * l_acc + tl.sum(P_tilde_ij, axis=1, keep_dims=True)
        # Cast P_tilde_ij to bf16 AFTER softmax, i.e. the line just before this comment
        P_tilde_ij.to(V_j.type.element_ty)
        O_ij = mip_mik_diff * O_acc + tl.dot(P_tilde_ij, V_j)
        l_acc = l_ij
        m_acc = m_ij
        O_acc = O_ij
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))
    O_i = O_acc / l_acc
    O_i.to(O_block_ptr.type.element_ty)
    tl.store(
        pointer=O_block_ptr, 
        value=O_i,
        boundary_check=(0,1) 
    )
    L_i = m_acc + tl.log(l_acc)
    L_i = tl.reshape(L_i, (Q_TILE_SIZE,))
    L_i.to(L_block_ptr.type.element_ty)
    tl.store(
        pointer=L_block_ptr,
        value=L_i,
        boundary_check=(0,)
    )


class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        D, output_dims = Q.shape[-1], Q.shape[:-1]
        batch, N_queries = output_dims[0], output_dims[1]
        N_keys = K.shape[1]
        Q_TILE_SIZE=16
        K_TILE_SIZE=16
        O = torch.empty_like(Q)
        L = torch.empty(output_dims, device=Q.device, dtype=torch.float32)
        ctx.is_causal = is_causal
        flash_fwd_kernel[(triton.cdiv(N_queries, Q_TILE_SIZE), batch)](
            Q_ptr=Q, K_ptr=K,V_ptr=V,
            O_ptr=O, L_ptr=L,
            stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
            stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
            stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
            stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2),
            stride_lb=L.stride(0), stride_lq=L.stride(1),
            N_QUERIES=N_queries, N_KEYS=N_keys, scale=1/math.sqrt(D),
            D=D, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=ctx.is_causal
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward():
        raise NotImplementedError
    