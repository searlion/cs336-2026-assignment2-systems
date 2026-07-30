from torch.optim import Optimizer
from typing import Any, Type
import torch.distributed as dist
import copy
import torch

class OptimizerStateSharding(Optimizer):

    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.settings = kwargs
        self.wrapped_optimizers = None
        self.ownership = {}
        super().__init__(params, kwargs)


    def step(self, closure=None, **kwargs):
        # If self.wrapped_optimizer is None, this rank owns no parameters, so there is nothing for this rank to compute.
        # Since no optimizer state exists, no update to apply. Skipping is right.
        if self.wrapped_optimizers is not None:
            self.wrapped_optimizers.step(closure, **kwargs)
        # The following broadcast is done after backward pass, and updating of parameters using local optimizers has been completed.
        # Hence, no_grad suppresses the recording of the broadcast operation in the graph. There's no need for autograd to track the write because
        # the write isnt part of any computation you will ever differentiate.
        with torch.no_grad():
            for param_group in self.param_groups:
                for param in param_group["params"]:
                    dist.broadcast(tensor=param,src=self.ownership[id(param)], async_op=False)

    def add_param_group(self, param_group: dict[str, Any]):
        local_params = []
        for idx, value in enumerate(param_group["params"]):
            owner = idx % self.world_size
            if id(value) not in self.ownership:
                self.ownership[id(value)] = owner
                if owner == self.rank:
                    local_params.append(value)

        super().add_param_group(param_group)
        wrapped_param_groups = copy.copy(param_group)
        wrapped_param_groups["params"] = local_params
        if local_params:
            if self.wrapped_optimizers is None:
                self.wrapped_optimizers = self.optimizer_cls(
                    local_params,
                    **self.settings
                )
            else:
                self.wrapped_optimizers.add_param_group(wrapped_param_groups)
