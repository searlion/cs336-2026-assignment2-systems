import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

os.environ["OMP_NUM_THREADS"] = "1"
warmup_iter = 5
num_of_iters = 100

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # os.environ["NCCL_DEBUG"] = "WARN"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
def distributed_demo(rank, world_size):
    setup(rank, world_size)
    # print(torch.get_num_threads())
    data = torch.rand((1000000000,), device="cpu", dtype=torch.float32)
    print(f"rank {rank} data (before all-reduce): {data}")
    i = 0
    j = 0
    while i < warmup_iter:
        dist.all_reduce(data, async_op=False)
        i += 1
    dist.barrier()
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    while j < num_of_iters:
        dist.all_reduce(data, async_op=False)
        j += 1
    # dist.barrier()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time/num_of_iters} seconds")
    print(f"rank {rank} data (after all-reduce): {data}")
    dist.destroy_process_group()
if __name__ == "__main__":
    world_size = 6
    # Spawn nprocs processes that run fn with the provided args
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)