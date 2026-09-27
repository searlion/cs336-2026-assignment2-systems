"""FSDP built on DTensor.

Same contract as `FSDP.py`, but the sharding, the all-gather, the padding for
uneven shards, and the gradient reduce-scatter are all delegated to DTensor.

The whole scheme rests on one line in `unshard()`:

    full = shard.redistribute(placements=[Replicate()]).to_local(grad_placements=[Partial()])

  * `redistribute(Shard(0) -> Replicate())` is an all-gather in the forward pass.
  * `to_local(grad_placements=[Partial()])` declares that the gradient arriving
    at this point is a *partial* result (each rank holds its own contribution,
    which must be summed across ranks). Autograd therefore runs the redistribute
    backwards as `Partial -> Shard(0)`, which is exactly a reduce-scatter.

So the reduce-scatter is never written by hand: it is the derivative of the
all-gather. That removes the post-accumulate-grad hook, the flatten/pad
arithmetic, the storage resizing, and the manual `dist.*` calls from `FSDP.py`.

Communication overlap: `redistribute(..., async_op=True)` returns immediately with
an `AsyncCollectiveTensor`, which blocks only when its data is first read. That
gives overlap in both directions from a single flag, because
`Redistribute.backward` reuses `ctx.async_op`:
  * forward  -- all-gathers for the next `prefetch_depth` modules are issued while
    the current module computes (see `_issue_unshard` / `_prefetch`);
  * backward -- the reduce-scatter is issued inside autograd and is not waited on
    until the gradient is read in `finish_gradient_synchronization()`.
"""

import os
import weakref
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor, init_device_mesh

from cs336_basics import model


def _storage_ptr(t: torch.Tensor) -> int | None:
    """Address of the buffer `t` aliases, or None if it has no real storage.

    Added for FULL_SHARD. Two wrinkles this absorbs:
      * `t` may be an `AsyncCollectiveTensor`, a wrapper subclass with no storage of
        its own -- its `.elem` holds the real all-gather output buffer, which is
        allocated at issue time (`wait_tensor` only synchronizes, never reallocates),
        so the address is already final. We only *read* through the wrapper;
        unwrapping with `.wait()` would drop the outer autograd node.
      * Some saved tensors (meta/fake tensors, subclasses) refuse `data_ptr()`
        outright. Those are never gathered weights, so None is the right answer.
    """
    inner = getattr(t, "elem", t)
    try:
        return inner.untyped_storage().data_ptr()
    except RuntimeError:
        return None


class FSDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
        prefetch_depth: int = 2,
        reshard_after_forward: bool | None = None,
    ):
        super().__init__()
        self._debug = os.environ.get("debug", False)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.module = module
        self.compute_dtype = compute_dtype
        # --- added for FULL_SHARD via saved_tensors_hooks ---
        # False (default) == SHARD_GRAD_OP: the gathered weight is retained by the
        # autograd graph from forward until backward.
        # True == FULL_SHARD: the graph saves the *shard* instead, and the full
        # weight is rematerialized during backward. See `_pack` / `_unpack`.
        if reshard_after_forward is None:
            reshard_after_forward = os.environ.get("FSDP_RESHARD", "0") == "1"
        self.reshard_after_forward = reshard_after_forward
        # storage address of a live gathered weight -> the module that owns it.
        # Registered in `unshard`, removed in `reshard`, so an entry only exists
        # while that module's forward is running and its buffer is provably alive.
        # Without that windowing a freed buffer's address could be reused by a
        # later activation and misidentified as a weight.
        self._weight_storage: dict[int, nn.Module] = {}
        self._weight_ptr: dict[nn.Module, int] = {}
        # module -> weakref to the rematerialized full weight (backward only).
        # Weak so the buffer dies as soon as the backward node that unpacked it
        # releases its view, which is what actually bounds peak memory.
        self._remat: dict[nn.Module, Any] = {}
        self._remat_count = 0  # instrumentation: rematerializing all-gathers issued
        # --- added for communication/compute overlap ---
        # How many modules ahead to issue all-gathers. 2 == N+1 and N+2 in flight
        # while module N computes. 0 disables prefetch (fully serialized).
        self.prefetch_depth = prefetch_depth
        # Full weights already gathered (or in flight), keyed by module.
        self._unsharded: dict[nn.Module, torch.Tensor] = {}
        # Execution order of the sharded modules, learned on the first forward.
        # `named_modules()` gives definition order, which need not match.
        self._fwd_order: list[nn.Module] = []
        self._order_index: dict[nn.Module, int] = {}
        self._order_ready = False

        # A 1-D mesh over all ranks. DTensor issues its collectives on this mesh's
        # process group, so `Shard(0)` means "split dim 0 across all ranks".
        device_type = next(module.parameters()).device.type
        self.mesh = init_device_mesh(device_type, (self.world_size,))

        before_shard = sum(p.numel() for p in self.module.parameters())
        # module -> the name the *full* parameter had, e.g. "linear1.weight"
        self.sharded_modules: dict[nn.Module, str] = {}

        for name, mods in module.named_modules():
            if not isinstance(mods, (model.Embedding, model.Linear)):
                continue
            # distribute_tensor splits dim 0 across the mesh and handles a ragged
            # tail on its own -- no ceil-divide, no padding buffer, no clamping.
            shard = nn.Parameter(distribute_tensor(mods.weight.detach(), self.mesh, [Shard(0)]))
            delattr(mods, "weight")
            mods.register_parameter("weight_shard", shard)
            self.sharded_modules[mods] = f"{name}.weight"

            mods.register_forward_pre_hook(self.unshard)
            mods.register_forward_hook(self.reshard)
            # NOTE: no backward pre-hook. The gathered weight is kept alive by the
            # autograd graph from forward until backward, so there is nothing to
            # re-gather. See the note on `reshard` for the memory consequence.

        self._sharded_param_ids = {id(mods.weight_shard) for mods in self.sharded_modules}

        after_shard = sum(p.numel() for p in self.module.parameters())
        if self.rank == 0 and self._debug:
            print(f"memory savings: {before_shard} vs {after_shard}")

    def _issue_unshard(self, mods: nn.Module) -> None:
        """Start the all-gather for `mods` without blocking on it.

        Added for overlap: `async_op=True` makes `redistribute` return an
        `AsyncCollectiveTensor` immediately. Nothing blocks until the tensor is
        actually read -- which happens inside the module's own forward, by which
        point the collective has had a whole layer (or two) of compute to finish in.
        """
        if mods in self._unsharded:
            return  # already gathered or in flight
        shard = mods.weight_shard
        if self.compute_dtype is not None:
            # Cast the *shard* before gathering, so the collective moves
            # compute_dtype bytes. Master weights stay fp32; this cast's backward
            # puts the reduce-scattered gradient back into fp32.
            shard = shard.to(self.compute_dtype)
        full = shard.redistribute(placements=[Replicate()], async_op=True)
        self._unsharded[mods] = full.to_local(grad_placements=[Partial()])

    def _prefetch(self, mods: nn.Module) -> None:
        """Added for overlap: issue all-gathers `prefetch_depth` modules ahead."""
        if not self._order_ready:
            return  # execution order not known until the first forward finishes
        i = self._order_index.get(mods)
        if i is None:
            return
        for d in range(1, self.prefetch_depth + 1):
            j = i + d
            if j < len(self._fwd_order):
                self._issue_unshard(self._fwd_order[j])

    def unshard(self, mods: nn.Module, args: Any = None) -> None:
        """Forward pre-hook: materialize the full weight from its shard."""
        # Added for overlap: record execution order on the first pass so that
        # later passes know what "the next two modules" means.
        if not self._order_ready and mods not in self._order_index:
            self._order_index[mods] = len(self._fwd_order)
            self._fwd_order.append(mods)

        self._issue_unshard(mods)  # no-op when a prefetch already started it
        local = self._unsharded.pop(mods)
        if self.reshard_after_forward:
            # Added for FULL_SHARD: mark this buffer so `_pack` recognizes any tensor
            # (or view -- einsum saves a permuted view, not the tensor itself) that
            # aliases it as "this module's gathered weight".
            # `local` may still be an AsyncCollectiveTensor, which has no storage of
            # its own. Read the inner buffer's address instead: it is allocated at
            # issue time and `wait_tensor` only synchronizes, never reallocates, so
            # the address is already final. Unwrapping via `.wait()` here would drop
            # the outer autograd node, so we only *read* through the wrapper.
            ptr = _storage_ptr(local)
            self._weight_storage[ptr] = mods
            self._weight_ptr[mods] = ptr
        setattr(mods, "weight", local)
        self._prefetch(mods)  # added for overlap: keep the queue full

    def reshard(self, mods: nn.Module, args: Any = None, output: Any = None) -> None:
        """Forward post-hook: drop the module's reference to the full weight.

        This does NOT free the memory: the autograd graph still holds the gathered
        tensor so it can compute grad_input in backward. This implementation is
        therefore equivalent to PyTorch FSDP's `reshard_after_forward=False`
        (a.k.a. `ShardingStrategy.SHARD_GRAD_OP`) -- parameters and optimizer state
        are sharded, but the gathered copies live from forward until backward.
        `FSDP.py` trades complexity for the stronger guarantee by freeing the
        storage here and re-gathering in a backward pre-hook.

        With `reshard_after_forward=True` the graph never saved the gathered tensor
        in the first place (see `_pack`), so dropping the reference here *does* free
        it, and this becomes true FULL_SHARD.
        """
        if self.reshard_after_forward:
            # Added for FULL_SHARD: close the identification window. Past this point
            # the buffer may be freed and its address recycled.
            ptr = self._weight_ptr.pop(mods, None)
            if ptr is not None:
                self._weight_storage.pop(ptr, None)
        if "weight" in mods.__dict__:
            delattr(mods, "weight")

    # ------------------------------------------------------------------
    # Added for FULL_SHARD: pack the saved weight down to its shard, and
    # rematerialize it during backward.
    #
    # `redistribute` stays inside the autograd graph and under grad mode, so the
    # reduce-scatter is still derived automatically -- unlike a custom
    # autograd.Function, which disables grad mode in its forward and forces you to
    # hand-write the Partial() -> Shard(0) gradient yourself.
    # ------------------------------------------------------------------

    def _pack(self, t: torch.Tensor):
        """Called for every tensor autograd is about to save.

        Anything aliasing a live gathered weight is replaced by (module, geometry).
        Everything else -- activations, RMSNorm weights -- passes through untouched.
        """
        mods = self._weight_storage.get(_storage_ptr(t))
        if mods is None:
            return t
        # Record the *view* geometry, not just the module: einsum saves a permuted
        # view of the weight, and unpack must hand back the identical layout.
        return (mods, t.shape, t.stride(), t.storage_offset())

    def _unpack(self, x):
        if not isinstance(x, tuple):
            return x
        mods, shape, stride, offset = x
        ref = self._remat.get(mods)
        base = ref() if ref is not None else None
        if base is None:
            base = self._regather(mods)
            self._remat[mods] = weakref.ref(base)
        return torch.as_strided(base, shape, stride, offset)

    @torch.no_grad()
    def _regather(self, mods: nn.Module) -> torch.Tensor:
        """Re-run the forward all-gather to recover the full weight.

        Values-only: no optimizer step happens between forward and backward, so this
        reproduces the forward weight exactly. No graph is needed because the edges
        from this weight back to the shard were already recorded in forward.
        """
        self._remat_count += 1
        shard = mods.weight_shard.detach()
        if self.compute_dtype is not None:
            shard = shard.to(self.compute_dtype)
        return shard.redistribute(placements=[Replicate()]).to_local()

    def _drain_unsharded(self) -> None:
        """Added for overlap: retire prefetches that were never consumed.

        In the normal flow every prefetched module runs later in the same forward,
        so this is empty. It matters if a forward pass raised part-way through:
        the stale entries would otherwise be handed out after an optimizer step
        and silently supply pre-update weights.
        """
        for tensor in self._unsharded.values():
            wait = getattr(tensor, "wait", None)  # AsyncCollectiveTensor only
            if wait is not None:
                wait()
        self._unsharded.clear()

    def forward(self, *inputs, **kwargs):
        self._drain_unsharded()
        # Added for overlap: prime the queue so module 0 does not wait on a cold
        # collective. Empty on the very first pass, when the order is still unknown.
        for mods in self._fwd_order[: self.prefetch_depth]:
            self._issue_unshard(mods)

        if self.reshard_after_forward:
            # Added for FULL_SHARD: the hooks are scoped to the forward region, but
            # the `_unpack` closure they record runs later, during backward.
            with saved_tensors_hooks(self._pack, self._unpack):
                out = self.module(*inputs, **kwargs)
        else:
            out = self.module(*inputs, **kwargs)
        self._order_ready = True
        return out

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        out = {}
        for mods, param_name in self.sharded_modules.items():
            # full_tensor() = all-gather back to a plain, replicated torch.Tensor.
            out[param_name] = mods.weight_shard.detach().full_tensor()
        for name, p in self.module.named_parameters():
            if name.endswith("weight_shard"):
                continue
            out[name] = p.detach().clone()
        return out

    def finish_gradient_synchronization(self):
        """Turn the summed gradients into means, and all-reduce replicated ones.

        Sharded parameters were already reduce-scattered by autograd, but with
        `Partial()` == sum, so they only need a local division here -- no
        communication. Replicated parameters (RMSNorm weights) are untouched by
        DTensor and still need the DDP-style all-reduce.

        With async_op=True the `div_` below is also the *wait* point for the
        backward reduce-scatter: reading an `AsyncCollectiveTensor` blocks until
        its collective lands, so the gradient comms of the early layers overlap
        with the rest of the backward pass.
        """
        for mods in self.sharded_modules:
            grad = mods.weight_shard.grad
            if grad is not None:
                grad.div_(self.world_size)

        handles = []
        for p in self.module.parameters():
            if id(p) in self._sharded_param_ids or p.grad is None:
                continue
            p.grad.div_(self.world_size)
            handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True))
        for handle in handles:
            handle.wait()
