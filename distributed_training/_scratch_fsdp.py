from tests.test_fsdp import _test_fsdp_correctness

test = _test_fsdp_correctness(rank=0, world_size=1, compute_dtype=None)

if __name__ == "__main__":
    print("hello")