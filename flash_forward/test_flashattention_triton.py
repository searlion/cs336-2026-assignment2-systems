from flashattention_autograd_function_triton import FlashAttentionTriton
import torch

batch = 1
N_queries = 16
N_keys = 16
D = 16
q = torch.rand((1, N_queries, D), device="cuda")
k = torch.rand((1, N_keys, D), device="cuda")
v = torch.rand((1, N_keys, D), device="cuda")
o = FlashAttentionTriton.apply(q, k, v, True)
# self_attention.forward(Q=q, K=k, V=v, is_causal=True)