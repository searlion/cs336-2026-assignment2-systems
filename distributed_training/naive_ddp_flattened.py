import torch
import torch.distributed as dist

class Naive_DDP_flatten(torch.nn.Module):

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        assert dist.is_initialized()
        # print("BEFORE:", dist.get_rank(), module.fc1.weight[0, :3])
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor=tensor, src=0, async_op=False)
        # print("AFTER:", dist.get_rank(), module.fc1.weight[0, :3])
        sum_of_parameters_requiring_grad = sum(tensor.numel() for tensor in self.module.parameters() if tensor.requires_grad)
        dtype = next(self.module.parameters()).dtype
        device = next(self.module.parameters()).device
        # Must be zeroed as autograd accumulation is +=
        self.flat_buffer = torch.zeros(sum_of_parameters_requiring_grad, dtype=dtype, device=device)

    def forward(self, *args, **kwargs):
        current_offset = 0
        for tensor in self.module.parameters():
            if tensor.requires_grad:
                number_of_parameters = tensor.numel()
                grad_slice = self.flat_buffer[current_offset : current_offset + number_of_parameters]
                tensor.grad = grad_slice.view(tensor.shape)
                current_offset += number_of_parameters
        self.flat_buffer.zero_()
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # print("BEFORE FLATTENED:", dist.get_rank(), self.module.fc1.weight.grad[0, :3])
        print("BEFORE POINTER:", self.module.fc1.weight.grad.data_ptr())
        dist.all_reduce(tensor=self.flat_buffer.mul_(1.0/dist.get_world_size()), op=dist.ReduceOp.SUM, async_op=False)
        print("AFTER POINTER:", self.module.fc1.weight.grad.data_ptr())
        offset = 0
        base = self.flat_buffer.data_ptr()
        esize = self.flat_buffer.element_size()
        for name, p in self.module.named_parameters():
            if p.requires_grad:
                assert p.grad.data_ptr() == base + offset * esize, f"Grad for {name} is not at expected offset in flat buffer"
                offset += p.numel()
        assert offset == self.flat_buffer.numel()
        # assert self.module.fc1.weight.grad.data_ptr() == self.flat_buffer.data_ptr()
        # assert self.module.fc2.fc.weight.grad.data_ptr() == self.flat_buffer.data_ptr() + 400
        # for tensor in self.module.parameters():
        #     if tensor.grad is not None:
        #         dist.all_reduce(tensor=tensor.grad.mul_(1.0/dist.get_world_size()), op=dist.ReduceOp.SUM, async_op=False)
