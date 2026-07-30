import torch
import torch.distributed as dist

class Naive_DDP(torch.nn.Module):

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        assert dist.is_initialized()
        # print("BEFORE:", dist.get_rank(), module.fc1.weight[0, :3])
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor=tensor, src=0, async_op=False)
        # print("AFTER:", dist.get_rank(), module.fc1.weight[0, :3])

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        print("BEFORE:", dist.get_rank(), self.module.fc1.weight.grad[0, :3])
        for tensor in self.module.parameters():
            if tensor.grad is not None:
                dist.all_reduce(tensor=tensor.grad.mul_(1.0/dist.get_world_size()), op=dist.ReduceOp.SUM, async_op=False)
