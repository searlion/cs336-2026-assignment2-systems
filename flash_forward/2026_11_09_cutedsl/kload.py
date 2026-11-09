# kload.py: load a kernel directory's flashattention_autograd_function_triton.py as its own module
# (several versions side by side in one process), and optionally pin a Triton kernel's config.
import importlib.util
import pathlib

import triton


def load_module(kernel_dir):
    p = pathlib.Path(kernel_dir)
    spec = importlib.util.spec_from_file_location(f"k_{p.name}_{abs(hash(str(p.resolve())))}",
                                                  p / "flashattention_autograd_function_triton.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load(kernel_dir, triton_config=None):
    """Return the autograd Function. triton_config = (bq, bk, warps, stages) pins an autotuned Triton
    kernel to that one configuration (an autotuner with one config uses it without timing)."""
    m = load_module(kernel_dir)
    if triton_config is not None and hasattr(m, "flash_fwd_kernel"):
        bq, bk, nw, ns = triton_config
        m.flash_fwd_kernel.configs = [triton.Config({"Q_TILE_SIZE": bq, "K_TILE_SIZE": bk},
                                                    num_warps=nw, num_stages=ns)]
    return m.FlashAttentionTriton
