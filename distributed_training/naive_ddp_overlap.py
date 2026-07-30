import torch
import torch.distributed as dist

class Naive_DDP_overlap(torch.nn.Module):

    def backward_pass(self, tensor: torch.Tensor) -> None:
        tensor.grad.mul_(1.0/dist.get_world_size())
        work = dist.all_reduce(tensor=tensor.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.registration.append(work)

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        assert dist.is_initialized()
        # print("BEFORE:", dist.get_rank(), module.fc1.weight[0, :3])
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor=tensor, src=0, async_op=False)
        # print("AFTER:", dist.get_rank(), module.fc1.weight[0, :3])
        self.registration = []
        for tensor in self.module.parameters():
            if tensor.requires_grad and tensor.grad_fn is None:
                tensor.register_post_accumulate_grad_hook(self.backward_pass)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        print("BEFORE:", dist.get_rank(), self.module.fc1.weight.grad[0, :3])
        while self.registration:
            work = self.registration.pop()
            work.wait()
