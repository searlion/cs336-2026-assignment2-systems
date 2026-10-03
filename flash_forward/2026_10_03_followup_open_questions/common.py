# common.py
# Shared helpers for the open-question experiments: where the follow-up's code lives, how to
# load one kernel version under a unique module name, and how to launch its kernel with an
# explicit (Q_TILE, K_TILE, num_warps, num_stages), bypassing class attributes and autotuning.
import importlib.util
import math
import pathlib

import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
FU = HERE.parent / "2026_10_03_followup"
KERNELS = FU / "kernels"
VERSIONS = ["v0_baseline", "v1_tiles", "v2_autotune", "v3_causal_skip", "v4_exp2", "v5_final"]


def load_version(name, path=None):
    """Import kernels/<name>/flashattention_autograd_function_triton.py (or `path`) as its own module."""
    path = pathlib.Path(path) if path else KERNELS / name / "flashattention_autograd_function_triton.py"
    spec = importlib.util.spec_from_file_location(f"fa_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jit_fn(module):
    """The @triton.jit function, unwrapped from @triton.autotune if the version autotunes."""
    kern = module.flash_fwd_kernel
    return kern.fn if hasattr(kern, "configs") else kern


def launch(module, q, k, v, causal, config, o=None, L=None):
    """The same launch as FlashAttentionTriton.forward, with an explicit config.

    `o` may be passed in a different dtype from q (the kernel casts to O's element type on
    store), which the P-cast experiment uses to write O in fp32."""
    bq, bk, nw, ns = config
    B, Nq, D = q.shape
    Nk = k.shape[1]
    o = torch.empty_like(q) if o is None else o
    L = torch.empty((B, Nq), device=q.device, dtype=torch.float32) if L is None else L
    jit_fn(module)[(triton.cdiv(Nq, bq), B)](
        q, k, v, o, L,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(), *L.stride(),
        Nq, Nk, 1.0 / math.sqrt(D),
        D=D, Q_TILE_SIZE=bq, K_TILE_SIZE=bk, is_causal=causal,
        num_warps=nw, num_stages=ns,
    )
    return o, L


def reference(q, k, v, causal, align="top-left"):
    """float64 attention and logsumexp with an explicit mask. Top-left: row i sees keys j <= i
    (PyTorch SDPA's is_causal). Bottom-right: row i sees keys j <= i + Nk - Nq (FlashAttention>=2.1)."""
    s = (q.double() @ k.double().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        nq, nk = s.shape[-2:]
        diag = 0 if align == "top-left" else nk - nq
        keep = torch.ones(nq, nk, dtype=torch.bool, device=q.device).tril(diagonal=diag)
        s = s.masked_fill(~keep, float("-inf"))
    return torch.softmax(s, dim=-1) @ v.double(), torch.logsumexp(s, dim=-1)
