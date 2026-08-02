import torch
from torch import nn
import torch.distributed as dist
from cs336_basics import model
import math
from dataclasses import dataclass
from torch import Size
import os
from typing import Any
from functools import partial
from torch.autograd.grad_mode import _unsafe_preserve_version_counter as _preserve_vc

@dataclass
class Record:
    original_shape: Size
    unpadded_numel: int
    shard_numel: int
    clamp_jagged_tensor_size: int
    owner_attribute_name: str
    owner_module: model.Embedding | model.Linear
    gathered_full_tensor: torch.Tensor | None = None
    work_handle: dist.Work | None = None
    param_view: torch.Tensor | None = None
    param_name: str = ""

def _free_storage(t: torch.Tensor) -> None:
    s = t.untyped_storage()
    if s.size() > 0:
        with _preserve_vc(t):
            s.resize_(0)

def _alloc_storage(t: torch.Tensor) -> None:
    s = t.untyped_storage()
    needed = t.numel() * t.element_size()
    if s.size() != needed:
        with _preserve_vc(t):
            s.resize_(needed)

class FSDP(nn.Module):


    def all_gather_full_tensor(self, mods: nn.Module, args: Any = None) -> None:
            metadata_record = self.metadata_record[mods]
            numel = metadata_record.shard_numel*self.world_size
            shard = mods.weight_shard
            # Master weights (the shard) stay in fp32; the gathered copy that forward/backward
            # actually compute with lives in compute_dtype when mixed precision is requested.
            gather_dtype = self.compute_dtype if self.compute_dtype is not None else shard.dtype
            if metadata_record.gathered_full_tensor is None:
                metadata_record.gathered_full_tensor = torch.empty(numel, dtype=gather_dtype, device=shard.device)
            # We store the full_weight_instead of full_weight unpadded so that when we apply resize_(0), it works with the original reference and not the view()
            full_weight = metadata_record.gathered_full_tensor
            _alloc_storage(full_weight)
            send = shard.detach().to(gather_dtype)
            with _preserve_vc(full_weight):
                dist.all_gather_into_tensor(output_tensor=full_weight, input_tensor=send, async_op=False)
            if metadata_record.param_view is None:
                full_weight_unpadded = full_weight[:metadata_record.unpadded_numel].view(metadata_record.original_shape)
                full_weight_unpadded.requires_grad_(True)
                full_weight_unpadded.register_post_accumulate_grad_hook(
                    partial(self.reduce_scatter_grad, metadata_record=metadata_record)
                    )
                metadata_record.param_view = full_weight_unpadded
            setattr(mods, metadata_record.owner_attribute_name, metadata_record.param_view)


    def free_full_tensor(self, mods: nn.Module, args: Any = None, output: Any = None) -> None:
        metadata_record = self.metadata_record[mods]
        _free_storage(metadata_record.gathered_full_tensor)
        if metadata_record.owner_attribute_name in mods.__dict__:
            delattr(mods, metadata_record.owner_attribute_name)

    def reduce_scatter_grad(self, tensor: torch.Tensor, metadata_record : Record) -> None:
        ## 1. Take the full gradient off the tensor it was handed.
        grad = tensor.grad
        tensor.grad = None
        if grad is None:
             return
        grad = grad * (1.0 / self.world_size)
        # 2. Flatten and pad it to shard_numel * world_size, detached.
        flat = torch.zeros(metadata_record.shard_numel * self.world_size, dtype=grad.dtype, device=grad.device)
        flat[: metadata_record.unpadded_numel] = grad.detach().flatten()
        del grad
        # 3. Allocate a shard_numel output.
        out = torch.empty(metadata_record.shard_numel, dtype=flat.dtype, device=flat.device)
        # 4. reduce_scatter_tensor(output, input) -- with averaging, not plain sum.
        dist.reduce_scatter_tensor(output=out, input=flat, op=dist.ReduceOp.SUM)
        del flat
        # 5. Cast to fp32 if needed and assign to weight_shard.grad.
        shard = metadata_record.owner_module.weight_shard
        out = out.to(shard.dtype)
        shard.grad = out if shard.grad is None else shard.grad.add_(out)
        _free_storage(metadata_record.gathered_full_tensor)
        # Drop the stale `weight` attribute the backward pre-hook re-installed; it now
        # points at storage of size 0.
        owner = metadata_record.owner_module
        if metadata_record.owner_attribute_name in owner.__dict__:
            delattr(owner, metadata_record.owner_attribute_name)

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self._debug = os.environ.get("debug", False)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.module = module
        self.compute_dtype = compute_dtype
        list_of_modules = list(module.named_modules())
        self.metadata_record = {}
        before_delete = sum(p.numel() for p in self.module.parameters())
        for name, mods in list_of_modules:
            if isinstance(mods, model.Embedding) or isinstance(mods, model.Linear):
                shard_size = math.ceil(mods.weight.numel() / self.world_size)
                # For handling jagged sharded tensors at the tail-end, clamp to 0 minimum
                clamp_jagged_tensor_size = max(min(shard_size, mods.weight.numel() - self.rank*shard_size), 0)
                dtype = mods.weight.dtype
                device = mods.weight.device
                flat_buffer = torch.zeros(shard_size, dtype=dtype, device=device)
                flat_buffer[0 : clamp_jagged_tensor_size] = torch.flatten(mods.weight)[self.rank*shard_size : (self.rank+1)*shard_size]
                self.metadata_record[mods] = Record(
                    original_shape=mods.weight.shape,
                    unpadded_numel=mods.weight.numel(),
                    shard_numel=shard_size,
                    clamp_jagged_tensor_size=clamp_jagged_tensor_size,
                    owner_attribute_name="weight",
                    owner_module=mods,
                    param_name=f"{name}.weight",
                )
                mods.register_parameter("weight_shard", nn.Parameter(flat_buffer))
                if self._debug:
                    with torch.no_grad():
                        ## Validate sharding was implemented correctly
                        full_weight = torch.empty(shard_size*self.world_size, dtype=dtype, device=device)
                        # full_weight[shard_size*self.rank : shard_size*self.rank + clamp_jagged_tensor_size] = mods.weight_shard[0 : clamp_jagged_tensor_size]
                        dist.all_gather_into_tensor(output_tensor=full_weight, input_tensor=mods.weight_shard, async_op=False)
                        assert torch.allclose(full_weight[:self.metadata_record[mods].unpadded_numel].view(self.metadata_record[mods].original_shape), mods.weight)
                delattr(mods, "weight")

                ## Attach pre and post forward hooks
                mods.register_forward_pre_hook(self.all_gather_full_tensor)
                mods.register_forward_hook(self.free_full_tensor)
                mods.register_full_backward_pre_hook(self.all_gather_full_tensor)

        # Parameters FSDP sharded; everything else is replicated and needs DDP-style
        # gradient averaging in finish_gradient_synchronization().
        self._sharded_param_ids = {id(rec.owner_module.weight_shard) for rec in self.metadata_record.values()}

        after_delete = sum(p.numel() for p in self.module.parameters())
        if self.rank == 0 and self._debug:
            print(f"memory savings: {before_delete} vs {after_delete}")
        
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        out = {}
        for rec in self.metadata_record.values():
            shard = rec.owner_module.weight_shard
            full = torch.empty(
                rec.shard_numel * self.world_size, dtype=shard.dtype, device=shard.device
            )
            dist.all_gather_into_tensor(full, shard.detach(), async_op=False)
            out[rec.param_name] = full[: rec.unpadded_numel].view(rec.original_shape).clone()

        for name, p in self.module.named_parameters():
            if name.endswith("weight_shard"):
                continue
            out[name] = p.detach().clone()
        return out

    def finish_gradient_synchronization(self):
        """Average the gradients of the *replicated* (non-sharded) parameters.

        Sharded parameters already had their gradients averaged by the reduce-scatter
        in `reduce_scatter_grad`. Replicated parameters (RMSNorm weights, biases, ...)
        are identical on every rank, but each rank only saw its own slice of the batch,
        so their gradients are per-rank partial results and must be all-reduced with a
        mean, exactly like DDP does.
        """
        handles = []
        for p in self.module.parameters():
            if id(p) in self._sharded_param_ids or p.grad is None:
                continue
            p.grad.div_(self.world_size)
            handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True))
        for handle in handles:
            handle.wait()
