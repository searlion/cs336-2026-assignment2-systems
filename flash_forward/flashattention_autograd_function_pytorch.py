import torch
import einops
import math

class FlashAttentionPytorch(torch.autograd.Function):

    def __init__(self):
        self.tile_size = 16

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        tile_size = 16
        # Split Q, K, V into T_q, T_k and T_v tiles
        Q_tq = einops.rearrange(Q, "... (Tq Bq) d -> ... Tq Bq d", Bq=tile_size)
        K_tk = einops.rearrange(K, "... (Tk Bk) d -> ... Tk Bk d", Bk=tile_size)
        V_tk = einops.rearrange(V, "... (Tk Bv) d -> ... Tk Bv d", Bv=tile_size)
        Output = torch.empty(Q.shape)
        LogSumExp = torch.empty(Q.shape[:-1])
        for i in range(0, Q_tq.shape[-3]):
            Q_i = Q_tq[..., i, :, :]
            bq_dim = Q_i.shape[-2]
            O_i0 = torch.zeros((bq_dim, Q_tq.shape[-1]))
            O_ip = O_i0
            l_i0 = torch.zeros((bq_dim, 1))
            l_ip = l_i0
            m_i0 = torch.full((bq_dim, 1), -torch.inf)
            m_ip = m_i0
            for j in range(0, K_tk.shape[-3]):
                K_j = K_tk[..., j, :, :]
                V_j = V_tk[..., j, :, :]
                S_ij = einops.einsum(Q_i, K_j, "... Bq d, ... Bk d -> ... Bq Bk")/math.sqrt(Q_i.shape[-1])
                m_ij = torch.maximum(m_ip, torch.amax(S_ij, dim=-1, keepdim=True))
                P_tilde_ij = torch.exp(S_ij - m_ij)
                mip_mij_diff = torch.exp(m_ip - m_ij)
                l_ij = mip_mij_diff * l_ip + torch.sum(P_tilde_ij, dim=-1, keepdim=True)
                O_ij = mip_mij_diff * O_ip + einops.einsum(P_tilde_ij, V_j, "... Bq Bk, ... Bk d -> ... Bq d")
                l_ip = l_ij
                m_ip = m_ij
                O_ip = O_ij
            O_i = O_ij / l_ij
            L_i = m_ij + torch.log(l_ij)
            Output[..., (i*tile_size):(i*tile_size+tile_size), :] = O_i
            LogSumExp[..., (i*tile_size):(i*tile_size+tile_size)] = einops.rearrange(L_i, "... Bq 1 -> ... Bq")            
        # N = einops.einsum(Q, K, "... Nq d, ... Nk d -> ... Nq Nk")
        # d = K.shape[-1]
        # S = N / math.sqrt(d)
        # L = torch.logsumexp(S, dim=-1)
        # ctx.save_for_backward(L)
        # P = torch.softmax(S, dim=-1)
        # O = einops.einsum(P, V, "... Nq Nk, ... Nk d -> ... Nq d")
        ctx.save_for_backward(LogSumExp)
        return Output

    @staticmethod
    def backward():
        raise NotImplementedError