# bench_flashattention.py
import torch
import torch.nn.functional as F
import triton
from flashattention_autograd_function_triton import FlashAttentionTriton


def attn_flops(batch, n, d, is_causal):
    # QK^T and PV: two matmuls of n*n*d multiply-adds each, 2 flops per MAC.
    f = 4 * batch * n * n * d
    return f / 2 if is_causal else f


def bench(B, H, N, D, is_causal, dtype=torch.bfloat16):
    q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=dtype) for _ in range(3))
    ms_ours = triton.testing.do_bench(
        lambda: FlashAttentionTriton.apply(q, k, v, is_causal), return_mode="median")

    q4, k4, v4 = (t.view(B, H, N, D) for t in (q, k, v))
    ms_sdpa = triton.testing.do_bench(
        lambda: F.scaled_dot_product_attention(q4, k4, v4, is_causal=is_causal),
        return_mode="median")

    flops = attn_flops(B * H, N, D, is_causal)
    tflops = lambda ms: flops / ms * 1e-9
    return ms_ours, ms_sdpa, tflops(ms_ours), tflops(ms_sdpa)


if __name__ == "__main__":
    print(f"{'N':>6} {'causal':>7} {'ours ms':>9} {'sdpa ms':>9} {'ours TF/s':>10} {'sdpa TF/s':>10} {'% sdpa':>7}")
    for N in (512, 1024, 2048, 4096):
        for causal in (False, True):
            a, b, c, d = bench(B=4, H=8, N=N, D=64, is_causal=causal)
            print(f"{N:>6} {str(causal):>7} {a:>9.3f} {b:>9.3f} {c:>10.1f} {d:>10.1f} {100 * b / a:>6.0f}%")
