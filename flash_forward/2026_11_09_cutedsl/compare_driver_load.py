# compare_driver_load.py: compare_driver.py's load(), importable (compare_driver.py runs on import).
import importlib.util
import pathlib


def load(kernel_dir):
    path = pathlib.Path(kernel_dir) / "flashattention_autograd_function_triton.py"
    spec = importlib.util.spec_from_file_location(pathlib.Path(kernel_dir).name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlashAttentionTriton
