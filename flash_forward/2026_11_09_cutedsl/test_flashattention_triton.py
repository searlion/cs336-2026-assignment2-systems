# test_flashattention_triton.py
import math
import pytest
import torch
import torch.nn.functional as F
from flashattention_autograd_function_triton import FlashAttentionTriton


def reference(q, k, v, is_causal):
    o = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if is_causal:
        tril = torch.ones(s.shape[-2:], dtype=torch.bool, device=q.device).tril()
        s = s.masked_fill(~tril, float("-inf"))
    return o, torch.logsumexp(s, dim=-1)


# (batch, n_queries, n_keys, head_dim). Deliberately includes shapes that
# are not multiples of the 16-row tile and non-square attention.
SHAPES = [
    (4, 128, 128, 64),     # the assignment's shape
    (2, 1024, 1024, 64),
    (2, 100, 100, 64),     # ragged last tile on both axes
    (1, 37, 53, 32),       # ragged and non-square
    (1, 200, 96, 128),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("B,Nq,Nk,D", SHAPES)
def test_forward_matches_sdpa(B, Nq, Nk, D, is_causal, dtype):
    if is_causal and Nq != Nk:
        pytest.skip("keep causal tests square so the mask convention is unambiguous")
    torch.manual_seed(0)
    q = torch.randn(B, Nq, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, Nk, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B, Nk, D, device="cuda", dtype=dtype, requires_grad=True)

    o = FlashAttentionTriton.apply(q, k, v, is_causal)
    # L is not returned; pull it out of the saved tensors, as the assignment test does.
    L = [t for t in o.grad_fn.saved_tensors if t.shape == (B, Nq)][0]

    o_ref, L_ref = reference(q, k, v, is_causal)
    # fp32 tl.dot uses TF32 on tensor cores by default; bf16 rounds P before P@V.
    tol = dict(atol=2e-2, rtol=2e-2) if dtype == torch.bfloat16 else dict(atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(o, o_ref, **tol)
    torch.testing.assert_close(L, L_ref, **tol)
