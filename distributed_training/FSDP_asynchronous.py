"""FSDP with overlapped communication.

Same sharding scheme and the same memory model as `FSDP.py` (flat padded shards,
gathered weight freed after forward and re-gathered before backward), but every
collective is issued with `async_op=True` and waited on as late as possible, so
communication overlaps with compute:

  1. All-gathers are *prefetched `prefetch_depth` modules ahead* (default 2, i.e.
     both k+1 and k+2 are in flight while module k computes; k-1 and k-2 during
     backward). This needs the execution order of the sharded modules, which is
     recorded on the first forward pass and reused from then on. Depth D keeps up
     to D+1 gathered weights resident at once, so it trades memory for latency
     hiding: depth 1 only covers one layer of compute, while a deeper queue keeps
     the wire busy when a single layer is too fast to cover a collective.
  2. Reduce-scatters are issued from the post-accumulate-grad hook but not waited
     on there. The handles are drained in `finish_gradient_synchronization()`, so
     a layer's gradient communication overlaps with the rest of the backward pass.
  3. The replicated-parameter all-reduces are issued together and waited on as a
     batch (as in `FSDP.py`).

The correctness constraint throughout is that *every rank issues collectives on
the process group in the same order*. That holds here because all ranks run the
same module execution order and the same autograd graph.
"""

import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch import Size, nn
from torch.autograd.grad_mode import _unsafe_preserve_version_counter as _preserve_vc

from cs336_basics import model


@dataclass
class Record:
    original_shape: Size
    unpadded_numel: int
    shard_numel: int
    clamp_jagged_tensor_size: int
    owner_attribute_name: str
    owner_module: model.Embedding | model.Linear
    gathered_full_tensor: torch.Tensor | None = None
    param_view: torch.Tensor | None = None
    param_name: str = ""
    # --- in-flight all-gather ---
    gather_handle: dist.Work | None = None
    # The input buffer of an async collective must stay alive until it completes.
    gather_send: torch.Tensor | None = None
    # Version counter of the output buffer *before* the collective was issued.
    gather_version: int = 0
    # --- in-flight reduce-scatter ---
    work_handle: dist.Work | None = None
    reduce_in: torch.Tensor | None = None
    reduce_out: torch.Tensor | None = None


def _set_version(t: torch.Tensor, version: int) -> None:
    """Force `t`'s autograd version counter back to `version`.

    `_preserve_vc` cannot be used across an async collective. A collective issued
    with `async_op=True` bumps the output tensor's version counter when it
    *completes* on a background thread, which is somewhere in the middle of
    unrelated autograd work -- not inside any context manager we control. The
    version therefore has to be captured before the collective is issued and
    restored after it has been waited on, spanning the whole in-flight window.
    """
    torch._C._autograd._unsafe_set_version_counter((t,), (version,))


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
    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
        prefetch_depth: int = 2,
    ):
        super().__init__()
        self._debug = os.environ.get("debug", False)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.module = module
        self.compute_dtype = compute_dtype
        # How many modules ahead to start all-gathers. 1 = only the next module,
        # 2 = the next two (N+1 and N+2), 0 disables prefetching entirely.
        self.prefetch_depth = prefetch_depth
        self.metadata_record: dict[nn.Module, Record] = {}

        # Execution order of the sharded modules, learned on the first forward and
        # used to decide what to prefetch. `named_modules()` order is *definition*
        # order, which is not guaranteed to match execution order.
        self._fwd_order: list[nn.Module] = []
        self._order_index: dict[nn.Module, int] = {}
        self._order_ready = False
        # Reduce-scatters issued this backward pass, drained in finish_gradient_synchronization.
        self._pending_reduce: list[Record] = []

        before_delete = sum(p.numel() for p in self.module.parameters())
        for name, mods in module.named_modules():
            if not isinstance(mods, (model.Embedding, model.Linear)):
                continue
            shard_size = math.ceil(mods.weight.numel() / self.world_size)
            clamp_jagged_tensor_size = max(min(shard_size, mods.weight.numel() - self.rank * shard_size), 0)
            flat_buffer = torch.zeros(shard_size, dtype=mods.weight.dtype, device=mods.weight.device)
            flat_buffer[0:clamp_jagged_tensor_size] = torch.flatten(mods.weight)[
                self.rank * shard_size : (self.rank + 1) * shard_size
            ]
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
            delattr(mods, "weight")

            mods.register_forward_pre_hook(self.unshard_forward)
            mods.register_forward_hook(self.free_full_tensor)
            mods.register_full_backward_pre_hook(self.unshard_backward)

        self._sharded_param_ids = {id(rec.owner_module.weight_shard) for rec in self.metadata_record.values()}

        after_delete = sum(p.numel() for p in self.module.parameters())
        if self.rank == 0 and self._debug:
            print(f"memory savings: {before_delete} vs {after_delete}")

    # ------------------------------------------------------------------ gather

    def _issue_gather(self, mods: nn.Module) -> None:
        """Start (but do not wait on) the all-gather for `mods`."""
        rec = self.metadata_record[mods]
        if rec.gather_handle is not None:
            return  # already in flight
        shard = mods.weight_shard
        gather_dtype = self.compute_dtype if self.compute_dtype is not None else shard.dtype
        if rec.gathered_full_tensor is None:
            rec.gathered_full_tensor = torch.empty(
                rec.shard_numel * self.world_size, dtype=gather_dtype, device=shard.device
            )
        full_weight = rec.gathered_full_tensor
        # Snapshot the version before anything touches the buffer; restored in
        # _wait_gather once the collective has landed. See _set_version.
        rec.gather_version = full_weight._version
        _alloc_storage(full_weight)
        # Held on the Record so the buffer outlives the call; a local would be
        # freed as soon as this function returns, while the collective still reads it.
        rec.gather_send = shard.detach().to(gather_dtype)
        rec.gather_handle = dist.all_gather_into_tensor(
            output_tensor=full_weight, input_tensor=rec.gather_send, async_op=True
        )

    def _wait_gather(self, mods: nn.Module) -> None:
        """Block until `mods`'s weight is gathered, then install it on the module."""
        rec = self.metadata_record[mods]
        if rec.gather_handle is None:
            self._issue_gather(mods)
        full_weight = rec.gathered_full_tensor
        rec.gather_handle.wait()
        # The async write is now definitely done, so this is the first safe moment
        # to hide it from autograd.
        _set_version(full_weight, rec.gather_version)
        rec.gather_handle = None
        rec.gather_send = None

        if rec.param_view is None:
            full_weight_unpadded = full_weight[: rec.unpadded_numel].view(rec.original_shape)
            full_weight_unpadded.requires_grad_(True)
            full_weight_unpadded.register_post_accumulate_grad_hook(
                partial(self.reduce_scatter_grad, metadata_record=rec)
            )
            rec.param_view = full_weight_unpadded
        setattr(mods, rec.owner_attribute_name, rec.param_view)

    def _prefetch(self, mods: nn.Module, reverse: bool) -> None:
        """Issue the all-gathers for the next `prefetch_depth` modules.

        With the default depth of 2, running module k leaves both k+1 and k+2 in
        flight (k-1 and k-2 in backward), so a collective that outlasts a single
        layer's compute is still covered.
        """
        if not self._order_ready:
            return
        i = self._order_index.get(mods)
        if i is None:
            return
        step = -1 if reverse else 1
        for d in range(1, self.prefetch_depth + 1):
            j = i + step * d
            if 0 <= j < len(self._fwd_order):
                self._issue_gather(self._fwd_order[j])

    def unshard_forward(self, mods: nn.Module, args: Any = None) -> None:
        if not self._order_ready and mods not in self._order_index:
            self._order_index[mods] = len(self._fwd_order)
            self._fwd_order.append(mods)
        # Wait for *this* module first, then start the next one, so the next
        # collective is in flight while this module computes.
        self._wait_gather(mods)
        self._prefetch(mods, reverse=False)

    def unshard_backward(self, mods: nn.Module, grad_output: Any = None) -> None:
        self._wait_gather(mods)
        self._prefetch(mods, reverse=True)

    def free_full_tensor(self, mods: nn.Module, args: Any = None, output: Any = None) -> None:
        rec = self.metadata_record[mods]
        _free_storage(rec.gathered_full_tensor)
        if rec.owner_attribute_name in mods.__dict__:
            delattr(mods, rec.owner_attribute_name)

    def _drain_gathers(self) -> None:
        """Retire any all-gather that was prefetched but never consumed.

        Without this, a prefetched-but-unused buffer would still be marked
        in-flight on the next iteration and would hand back weights from *before*
        the last optimizer step.
        """
        for rec in self.metadata_record.values():
            if rec.gather_handle is None:
                continue
            rec.gather_handle.wait()
            _set_version(rec.gathered_full_tensor, rec.gather_version)
            rec.gather_handle = None
            rec.gather_send = None
            _free_storage(rec.gathered_full_tensor)
            if rec.owner_attribute_name in rec.owner_module.__dict__:
                delattr(rec.owner_module, rec.owner_attribute_name)

    # ------------------------------------------------------- reduce / all-reduce

    def reduce_scatter_grad(self, tensor: torch.Tensor, metadata_record: Record) -> None:
        """Issue the gradient reduce-scatter; the wait happens later."""
        grad = tensor.grad
        tensor.grad = None
        if grad is None:
            return
        rec = metadata_record
        grad = grad * (1.0 / self.world_size)
        flat = torch.zeros(rec.shard_numel * self.world_size, dtype=grad.dtype, device=grad.device)
        flat[: rec.unpadded_numel] = grad.detach().flatten()
        del grad
        out = torch.empty(rec.shard_numel, dtype=flat.dtype, device=flat.device)
        # Both buffers stay referenced until the wait in finish_gradient_synchronization.
        rec.reduce_in, rec.reduce_out = flat, out
        rec.work_handle = dist.reduce_scatter_tensor(
            output=out, input=flat, op=dist.ReduceOp.SUM, async_op=True
        )
        self._pending_reduce.append(rec)

        # Safe to free immediately: the collective reads `flat`, which is a copy.
        _free_storage(rec.gathered_full_tensor)
        if rec.owner_attribute_name in rec.owner_module.__dict__:
            delattr(rec.owner_module, rec.owner_attribute_name)

    def finish_gradient_synchronization(self):
        """Drain every in-flight collective and publish the gradients."""
        for rec in self._pending_reduce:
            rec.work_handle.wait()
            shard = rec.owner_module.weight_shard
            out = rec.reduce_out.to(shard.dtype)
            shard.grad = out if shard.grad is None else shard.grad.add_(out)
            rec.work_handle = rec.reduce_in = rec.reduce_out = None
        self._pending_reduce.clear()

        handles = []
        for p in self.module.parameters():
            if id(p) in self._sharded_param_ids or p.grad is None:
                continue
            p.grad.div_(self.world_size)
            handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True))
        for handle in handles:
            handle.wait()

    # ----------------------------------------------------------------- module

    def forward(self, *inputs, **kwargs):
        self._drain_gathers()
        # Prime the queue: start the first `prefetch_depth` gathers before any
        # compute happens, so layer 0 is not waiting on a cold collective.
        for mods in self._fwd_order[: self.prefetch_depth]:
            self._issue_gather(mods)

        out = self.module(*inputs, **kwargs)

        if not self._order_ready:
            self._order_ready = True
        elif torch.is_grad_enabled():
            # Backward starts at the last layer and walks forwards, so prime the
            # tail of the queue now; those gathers overlap with the loss computation.
            for mods in self._fwd_order[len(self._fwd_order) - self.prefetch_depth :]:
                self._issue_gather(mods)
        return out

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        out = {}
        for rec in self.metadata_record.values():
            shard = rec.owner_module.weight_shard
            full = torch.empty(rec.shard_numel * self.world_size, dtype=shard.dtype, device=shard.device)
            dist.all_gather_into_tensor(full, shard.detach(), async_op=False)
            out[rec.param_name] = full[: rec.unpadded_numel].view(rec.original_shape).clone()

        for name, p in self.module.named_parameters():
            if name.endswith("weight_shard"):
                continue
            out[name] = p.detach().clone()
        return out
