import torch
from torch import nn
import torch.distributed as dist
from cs336_basics import model
import math
from dataclasses import dataclass
from torch import Size

@dataclass
class Record:
    original_shape: Size
    unpadded_numel: int
    shard_numel: int
    owner_attribute_name: str
    owner_module: model.Embedding | model.Linear
    gathered_full_tensor: torch.Tensor | None = None
    work_handle: dist.Work | None = None

class FSDP(nn.Module):

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.module = module
        self.compute_dtype = compute_dtype
        list_of_modules = list(module.named_modules())
        self.metadata_record = {}
        for _, mods in list_of_modules:
            if isinstance(mods, model.Embedding) or isinstance(mods, model.Linear):
                shard_size = math.ceil(mods.weight.numel() / self.world_size)
                dtype = mods.weight.dtype
                device = mods.weight.device
                flat_buffer = torch.zeros(shard_size, dtype=dtype, device=device)
                flat_buffer[0 : min(shard_size, mods.weight.numel() - self.rank*shard_size)] = torch.flatten(mods.weight)[self.rank*shard_size : (self.rank+1)*shard_size]
                self.metadata_record[mods] = Record(
                    original_shape=mods.weight.shape,
                    unpadded_numel=mods.weight.numel(),
                    shard_numel=shard_size,
                    owner_attribute_name="weight",
                    owner_module=mods
                )
                mods.register_parameter("weight_shard", nn.Parameter(flat_buffer))
                delattr(mods, "weight")

    def forward(self, *inputs, **kwargs):
        pass

    def finish_gradient_synchronization(self):
        pass