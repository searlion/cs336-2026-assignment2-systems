from tests.test_fsdp import _test_fsdp_correctness
import torch.distributed as dist
import torch.multiprocessing as mp

if __name__ == "__main__":
    world_size = 3
    mp.spawn(
        _test_fsdp_correctness,
        args=(world_size, None),
        nprocs=world_size,
        join=True,
    )