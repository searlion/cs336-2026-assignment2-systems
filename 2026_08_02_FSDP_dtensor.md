# FSDP with DTensor — `distributed_training/FSDP_dtensor.py`

A rewrite of `distributed_training/FSDP.py` where the sharding, the all-gather, the padding for
uneven shards, and the gradient reduce-scatter are all delegated to DTensor. Same public contract
(`forward`, `gather_full_params`, `finish_gradient_synchronization`).

A note on "simpler", since the raw line counts do not show it: 160 code lines vs 152 for
`FSDP.py`. The DTensor file is not shorter *overall*, because roughly 40 lines of prefetch
scaffolding were added to it (execution-order recording, `_issue_unshard`, `_prefetch`,
`_drain_unsharded`) that the synchronous manual version does not have — scaffolding that is
orthogonal to DTensor and appears in the [async manual version](2026_08_02_FSDP_asynchronous.md)
too. What actually collapses is the **FSDP core**: the sharding math, the collectives, the
gradient path, and the memory management go from ~70 lines of fiddly index arithmetic and private
autograd APIs to about 9 lines, with no `Record` dataclass and no storage helpers. That is where
the simplicity is, and it is the part most likely to harbour bugs.

## How to run it

```bash
CUDA_VISIBLE_DEVICES="" FSDP_IMPL=dtensor uv run pytest tests/test_fsdp.py
```

`tests/adapters.py::get_fsdp` now dispatches on `FSDP_IMPL` (`manual` is the default, so the plain
`uv run pytest tests/test_fsdp.py` still exercises `FSDP.py`).

`CUDA_VISIBLE_DEVICES=""` is **required on this machine** — see [the backend
limitation](#the-one-real-limitation-gloo--cuda) below. Result:

```
tests/test_fsdp.py::test_fsdp_correctness[fp32]    PASSED
tests/test_fsdp.py::test_fsdp_correctness[fp16]    PASSED
tests/test_fsdp.py::test_fsdp_gradient_sync[fp32]  PASSED
tests/test_fsdp.py::test_fsdp_gradient_sync[fp16]  PASSED
```

Also verified at `world_size=5` (jagged shards: 128 rows / 5 ranks → 26,26,26,26,24) and
`world_size=3`, both dtypes.

For the `FULL_SHARD` variant (frees the gathered weight after forward — see
[peak memory](#the-real-trade-off-peak-memory)), add `FSDP_RESHARD=1`:

```bash
CUDA_VISIBLE_DEVICES="" FSDP_IMPL=dtensor FSDP_RESHARD=1 uv run pytest tests/test_fsdp.py
```

Also 4 passed, stable over 5 repeats. The default is unchanged.

---

## The one idea the whole file rests on

```python
full = shard.redistribute(placements=[Replicate()]).to_local(grad_placements=[Partial()])
```

* `redistribute(Shard(0) → Replicate())` is **an all-gather in the forward pass**.
* `to_local(grad_placements=[Partial()])` declares that the gradient arriving at this point is a
  *partial* result — each rank holds its own contribution, and they must be summed. Autograd
  therefore runs the redistribute backwards as `Partial → Shard(0)`, which **is** a reduce-scatter.

So the reduce-scatter is never written. It is the derivative of the all-gather, and PyTorch derives
it. I verified the semantics standalone before building on it — two ranks, an intentionally uneven
5×3 tensor, and a different loss scale per rank so the gradients would differ if they weren't
summed:

```
[0] local shard shape torch.Size([3, 3]), global torch.Size([5, 3])   # uneven split, handled
[1] local shard shape torch.Size([2, 3]), global torch.Size([5, 3])
[0] gathered full == original: True                                   # forward all-gather
[0] grad placements (Shard(dim=0),) local [3.0, 3.0, ... ]            # rank0 gave 1, rank1 gave 2
[1] grad placements (Shard(dim=0),) local [3.0, 3.0, ... ]            # -> 3.0 == summed, sharded
```

`grad_placements` is the load-bearing argument. Without it the gradient of a `Replicate` tensor is
treated as already-replicated, and `Replicate → Shard(0)` is just local chunking with **no
communication** — you would silently drop every other rank's contribution.

---

## What disappears, line by line

### 1. Sharding: 6 lines → 1

`FSDP.py` shards the *flattened* weight into equal `ceil(numel / world_size)` chunks and pads the
ragged tail by hand:

```python
shard_size = math.ceil(mods.weight.numel() / self.world_size)
clamp_jagged_tensor_size = max(min(shard_size, mods.weight.numel() - self.rank * shard_size), 0)
flat_buffer = torch.zeros(shard_size, dtype=dtype, device=device)
flat_buffer[0:clamp_jagged_tensor_size] = torch.flatten(mods.weight)[
    self.rank * shard_size : (self.rank + 1) * shard_size
]
mods.register_parameter("weight_shard", nn.Parameter(flat_buffer))
```

DTensor version:

```python
shard = nn.Parameter(distribute_tensor(mods.weight.detach(), self.mesh, [Shard(0)]))
```

`distribute_tensor` splits **dim 0** (rows) rather than the flat buffer, and handles a ragged tail
internally. The entire `clamp_jagged_tensor_size` concept — the thing that needs a `world_size=5`
test to exercise — is gone. Note the shards are now *unequal in size* across ranks (26 vs 24 rows)
where the manual version made them equal-and-padded; DTensor pads internally only for the duration
of a collective.

### 2. The gradient path: 26 lines → 0

`FSDP.py` needs a post-accumulate-grad hook registered on a cached view, which then flattens, pads,
allocates, reduce-scatters, and casts:

```python
full_weight_unpadded.register_post_accumulate_grad_hook(
    partial(self.reduce_scatter_grad, metadata_record=metadata_record))
...
def reduce_scatter_grad(self, tensor, metadata_record):
    grad = tensor.grad
    tensor.grad = None
    ...
    flat = torch.zeros(rec.shard_numel * self.world_size, ...)
    flat[: rec.unpadded_numel] = grad.detach().flatten()
    out = torch.empty(rec.shard_numel, ...)
    dist.reduce_scatter_tensor(output=out, input=flat, op=dist.ReduceOp.SUM)
    shard.grad = out.to(shard.dtype) if shard.grad is None else ...
```

The DTensor version has **no equivalent code at all**. `weight_shard.grad` shows up as a
`Shard(0)` DTensor because autograd put it there. Everything the manual hook does by hand — the
flatten, the pad, the collective, the dtype cast back to fp32 — is a step in the derivative chain
of `shard.to(compute_dtype).redistribute(...).to_local(...)`.

### 3. Storage manipulation: gone

`FSDP.py` carries `_free_storage` / `_alloc_storage`, which call `storage.resize_(0)` and back
under `_unsafe_preserve_version_counter` to hide the mutation from autograd. That machinery — the
part of the manual implementation most likely to produce a baffling
"variable needed for gradient computation has been modified by an inplace operation" — does not
exist here. Nor does the backward pre-hook that re-gathers the freed weight.

### 4. `gather_full_params`

```python
# FSDP.py
full = torch.empty(rec.shard_numel * self.world_size, dtype=shard.dtype, device=shard.device)
dist.all_gather_into_tensor(full, shard.detach(), async_op=False)
out[rec.param_name] = full[: rec.unpadded_numel].view(rec.original_shape).clone()

# FSDP_dtensor.py
out[param_name] = mods.weight_shard.detach().full_tensor()
```

---

## What does *not* disappear

**Replicated parameters still need a manual all-reduce.** DTensor only manages what you handed it.
`norm1.weight` / `norm2.weight` are ordinary `nn.Parameter`s, so the DDP-style averaging from
[the bug-1 fix](2026_08_02_FSDP_BUGS.md) is carried over unchanged. This is worth internalizing:
"I used DTensor" is not the same as "gradients are synchronized." Whether they *could* be DTensors
is a real question with a non-obvious answer — see [the next section](#could-replicated-parameters-be-dtensors-too).

**The mean.** `Partial()` defaults to *sum*, so the reduce-scatter produces a sum and
`finish_gradient_synchronization` divides by `world_size` — a local op, no communication. (You
could instead use `Partial(reduce_op="avg")`, but that maps to `ReduceOp.AVG`, which is NCCL-only.)

**Mixed precision** is still explicit, and still the same idea: cast the *shard* before gathering
so the wire carries `compute_dtype` bytes, and let the cast's backward return the gradient to fp32.

```python
if self.compute_dtype is not None:
    shard = shard.to(self.compute_dtype)
```

Torch 2.11's `redistribute` also accepts `forward_dtype=` / `backward_dtype=` kwargs that do this
natively; I kept the explicit `.to()` because it makes the master-weight/compute-weight split
visible and works on older versions.

---

## Could replicated parameters be DTensors too?

Yes — but every route costs more than the four lines of manual all-reduce it removes. The reason is
worth recording, because it explains where the DTensor/plain-tensor boundary in this file actually
sits and why it sits there.

All three experiments below run on 2 CPU ranks computing `y = sum(x * w)` with a replicated `w`,
where rank 0 sees rows of `1.0` and rank 1 sees rows of `10.0`. The correct global gradient is
`2*1 + 2*10 = 22.0`.

### A. You cannot swap the parameter type in isolation

Make `w` a `Replicate()` DTensor, leave the activation a plain local tensor, and DTensor refuses:

```
[0] A REJECTED: aten.mul.Tensor: got mixed torch.Tensor and DTensor,
    need to convert all torch.Tensor to DTensor before calling distributed operators!
```

DTensor is all-or-nothing at each op. Promoting one parameter forces every activation that reaches
it to be promoted too.

That rejection is also the deeper reason a manual all-reduce is needed at all. **DTensor tracks
placements, not semantics.** It has no way to know your batch is sharded, because you never told
it — the batch is a plain tensor. Had it silently accepted the plain activation as replicated, it
would have concluded `grad_w` was replicated, skipped the reduction, and handed back unaveraged
gradients with no error. The hard failure is DTensor declining to guess, which is the right call.

### B. Go all-in, and the reduction becomes a placement conversion

Promote the activation to `Shard(0)` and DTensor tracks the whole chain:

```
[0] B  out (Shard(dim=0),)   loss (Partial(sum),)
[0] B -> grad (Partial(sum),)  local=[2.0,  2.0,  2.0]   full=[22.0, 22.0, 22.0]
[1] B -> grad (Partial(sum),)  local=[20.0, 20.0, 20.0]  full=[22.0, 22.0, 22.0]
```

The gradient of a replicated parameter under a sharded batch is `Partial(sum)` — derived from the
contraction over the sharded dimension, the same machinery that derives the reduce-scatter for the
sharded weights. `Partial(sum) -> Replicate()` *is* the all-reduce.

Note what DTensor does **not** do: it leaves `.grad` in `Partial` and never auto-redistributes it to
match the parameter's placement. That is deliberate. Staying lazy keeps gradient accumulation free
(`Partial + Partial` is a local add) and collapses N accumulation steps into one all-reduce at the
point you finally read it.

### C. The middle path — activations stay plain

The `to_local(grad_placements=...)` trick that carries the sharded weights works on a replicated one
too, and it does *not* require promoting the activations:

```python
w = nn.Parameter(distribute_tensor(norm.weight, mesh, [Replicate()]))
w_local = w.to_local(grad_placements=[Partial()])   # a plain Tensor, usable with plain activations
```
```
[0] C w_local type=Tensor
[0] C -> grad (Partial(sum),)  local=[2.0,  2.0,  2.0]
[1] C -> grad (Partial(sum),)  local=[20.0, 20.0, 20.0]
```

`finish_gradient_synchronization`'s replicated branch would then collapse to:

```python
for p in replicated_params:
    p.grad = p.grad.redistribute(placements=[Replicate()]).div_(self.world_size)
```

### Why none of these is adopted

- **B** requires wrapping the input batch, giving every op in `cs336_basics.model` a sharding rule
  (RoPE buffers, masks, `arange`, and custom indexing are where this bites), and handling a
  `Partial(sum)` loss. It also directly contradicts the design decision at `FSDP_dtensor.py:114`,
  where `.to_local(grad_placements=[Partial()])` drops back to plain tensors *precisely so the model
  body runs unmodified*. Going all-in undoes the thing that made this file short.
- **C** trades one `dist.all_reduce` for one `redistribute` — same single collective — while adding
  `__init__` sharding and forward hooks on every RMSNorm. Net more code.
- Either way the `div_` survives: `Partial(sum) -> Replicate` lowers to `ReduceOp.SUM`, and
  `Partial(reduce_op="avg")` lowers to `ReduceOp.AVG`, which gloo does not support.

The honest framing is that this file draws a deliberate boundary: **DTensor owns the parameter
dimension, plain tensors own the activation dimension.** Replicated parameters never cross that
boundary — they are never redistributed — so DTensor has no reason to be involved and the DDP
all-reduce stays hand-written. Erasing that last manual collective means moving the boundary to
include the data, which is the 2D-parallel design (an FSDP mesh dim crossed with a TP mesh dim).
That is a different program, not a tidier version of this one.

---

## The real trade-off: peak memory

This is the one place where the DTensor version is genuinely *weaker*, and it is worth being
precise about rather than glossing over.

`FSDP.py` frees the gathered weight in a forward post-hook and re-gathers it in a backward
pre-hook, so at most one full weight is resident at a time. By default the DTensor version does not:
the gathered tensor is held by the autograd graph from forward until backward, so `reshard()` only
drops the module's reference — it does not free memory.

`reshard_after_forward=True` (also settable with `FSDP_RESHARD=1`) recovers `FULL_SHARD`, using
`saved_tensors_hooks`. Both modes are in the file; the default is unchanged.

| | `FSDP.py` | `FSDP_dtensor.py` default | `FSDP_dtensor.py` `reshard_after_forward=True` |
|---|---|---|---|
| Params sharded | yes | yes | yes |
| Optimizer state sharded | yes | yes | yes |
| Gathered weight freed after forward | **yes** | no | **yes** |
| Peak transient weight memory | one layer | all layers | one layer |
| All-gathers per step | 2 per layer | 1 per layer | 2 per layer |
| Reduce-scatter written by hand | yes | **no** | **no** |
| Equivalent PyTorch FSDP mode | `FULL_SHARD` | `SHARD_GRAD_OP` | `FULL_SHARD` |

The default is not a defect so much as a different, legitimate point on the curve — PyTorch ships
exactly this trade as a first-class strategy. You pay memory (all gathered weights resident) and
save communication (half the all-gathers).

### Not a DTensor limitation — a `redistribute()` limitation

Worth separating the two. `redistribute()` allocates its own output and exposes no `out=`:

```
DTensor.redistribute(self, device_mesh=None, placements=None, *, async_op=False)
```

So the buffer autograd ends up holding is one you never get a handle on, and there is nothing to
free. But DTensor itself is not the obstacle — torch's own DTensor-based FSDP2
(`torch.distributed.fsdp.fully_shard`) implements `reshard_after_forward=True` by *not* using
`redistribute` for the forward gather:

| step | FSDP2 (torch 2.6) | this repo's `FSDP.py` |
|---|---|---|
| gather | `dist.all_gather_into_tensor` into an owned buffer — `_fsdp_collectives.py:165` | `FSDP.py:56` |
| param given to the module | `torch.as_strided(...)`, a view into that buffer — `_fsdp_param.py:505` | `full_weight[:n].view(shape)`, `FSDP.py:58` |
| free after forward | `free_storage(t)` → `storage.resize_(0)` — `_fsdp_param.py:674, 858` | `_free_storage`, `FSDP.py:26` |
| version counter | `torch.autograd._unsafe_preserve_version_counter` — `_fsdp_param.py:515` | `_preserve_vc`, `FSDP.py:11` |

That is the manual version's trick, in torch, on DTensor params. (`_unsafe_preserve_version_counter`
exists in torch *because* FSDP2 needed it.) DTensor answers **what the parameter is** — spec,
placement, ragged-tail padding, gradient placement. It deliberately does not answer **when the
memory exists**; FSDP2 answers that outside DTensor, by owning the buffer.

### How the `saved_tensors_hooks` route works

Rather than fight for the buffer, intercept what the graph *saves*. `redistribute` stays in the
graph and under grad mode, so the reduce-scatter is still derived automatically.

```python
def _pack(self, t):
    mods = self._weight_storage.get(_storage_ptr(t))
    if mods is None:
        return t                                        # activations pass through
    return (mods, t.shape, t.stride(), t.storage_offset())

def _unpack(self, x):
    if not isinstance(x, tuple):
        return x
    mods, shape, stride, offset = x
    base = ...                                          # re-gather, cached weakly
    return torch.as_strided(base, shape, stride, offset)
```

Four things this has to get right, each of which cost a debugging round:

1. **Match on storage, not identity.** `einsum` saves a *permuted view* of the weight, not the
   tensor you installed — the same `AsStridedBackward0` that produced the version-counter error in
   [the async writeup](2026_08_02_FSDP_asynchronous.md). An `id()` map misses it entirely.
2. **Record the view geometry.** `_unpack` must hand back the identical shape/stride/offset, which
   is why it ends in `as_strided` — the same call FSDP2 makes at `_fsdp_param.py:505`.
3. **Window the identification.** The `data_ptr -> module` entry is added in `unshard` and removed
   in `reshard`, so it exists only while that module's forward runs and its buffer is provably
   alive. Leave it registered and a freed buffer's address can be recycled by a later activation
   and misidentified as a weight.
4. **`AsyncCollectiveTensor` has no storage of its own.** Both `unshard` and `_pack` see the
   wrapper, and `.untyped_storage()` on it raises `Attempted to access the data pointer on an
   invalid python storage`. `_storage_ptr` reads through `.elem` — the real buffer, allocated at
   issue time — rather than calling `.wait()`, which would drop the outer autograd node.

The cache is a `weakref`, so the rematerialized weight dies as soon as the backward node that
unpacked it releases its view. That is what actually bounds peak memory; a strong cache would leave
every weight resident by the end of backward and give back the whole win.

**Measured.** Toy model, 2 CPU ranks, counting gathered buffers still alive between forward and
backward:

```
reshard_after_forward=False          reshard_after_forward=True
  gathered in forward       : 4        gathered in forward       : 4
  ALIVE between fwd and bwd : 3        ALIVE between fwd and bwd : 0
  re-gathers during bwd     : 0        re-gathers during bwd     : 3
```

Both correct: `FSDP_IMPL=dtensor FSDP_RESHARD=1 uv run pytest tests/test_fsdp.py` → 4 passed,
stable over 5 repeats. Two details in those numbers are worth reading:

- **4 gathered but only 3 retained**, even in the default mode. `Embedding.forward` is an index op
  whose backward saves the *indices*, not the weight — so the embedding never had a full copy to
  free. The "all layers resident" cost is really "all layers whose backward needs the weight."
- **3 re-gathers for 3 packed weights** — exactly one per weight, no duplicates, confirming the
  weakref cache coalesces the multiple saved views of a single weight.

### Why not a custom `torch.autograd.Function`

A tempting alternative: `redistribute` in forward, `save_for_backward(shard)` only, re-gather in
backward. It works — I ran it and the gradients match — but it gives up the thing that made the
DTensor version worth writing, because **grad mode is off inside `Function.forward`**:

```
[inside Function.forward] torch.is_grad_enabled() = False
```

So `redistribute` builds no graph node and its derivative is *not* generated. You write the
reduce-scatter yourself, and you must know the placement algebra to write it correctly. Same script,
one placement label changed on the gradient returned from `backward`:

```
from_local(g, mesh, [Partial()]).redistribute([Shard(0)])     -> [22.0 ...]  correct
from_local(g, mesh, [Replicate()]).redistribute([Shard(0)])   -> rank0 [2.0 ...], rank1 [20.0 ...]  WRONG
```

`Replicate() -> Shard(0)` is a **slice**; only `Partial() -> Shard(0)` is a **reduce-scatter**. Get
it wrong and each rank keeps its own unsummed contribution — a silent wrong answer, no crash.

Two further costs. The Function must also own the matmul (if it merely returned the gathered weight,
the downstream matmul's backward would save it and nothing would be freed), so you hand-write the
module's backward per layer type — easy for `Linear`, a scatter-add for `Embedding`. And the
re-gather fires *inside* the backward node, i.e. exactly when needed, so there is no overlap;
`FSDP.py:141` at least gathers a module ahead via `register_full_backward_pre_hook`.

| route | frees weight | reduce-scatter auto-derived | hand-written per-module backward | overlap |
|---|---|---|---|---|
| `FSDP_dtensor.py` default | no | yes | no | yes (prefetch) |
| `FSDP_dtensor.py` `reshard_after_forward=True` | yes | yes | no | forward only |
| custom `autograd.Function` | yes | **no** | **yes** | no |
| `torch.utils.checkpoint` | yes | yes | no | no |

The honest caveat on the implemented route: `_unpack` fires exactly when the backward node needs the
weight, so the re-gather is synchronous and unoverlapped. Recovering backward overlap means driving
a prefetch from a backward pre-hook that pre-populates the cache `_unpack` reads from — the same
machinery `FSDP.py` already has, which is the point at which the two implementations converge.

---

## Communication overlap (added)

Originally this file was fully synchronous. It now prefetches, driven by `prefetch_depth`
(default 2 — see [the async writeup](2026_08_02_FSDP_asynchronous.md) for the equivalent machinery
in the manual implementation). Every added block is marked `added for overlap` in the source.

```python
full = shard.redistribute(placements=[Replicate()], async_op=True)
self._unsharded[mods] = full.to_local(grad_placements=[Partial()])
```

`async_op=True` returns an `AsyncCollectiveTensor` immediately; it blocks only when its data is
first read, which happens inside the owning module's `forward`. `_prefetch` then issues the
gathers for the next `prefetch_depth` modules after the current one has been handed over, so those
collectives fly while the current layer computes.

One nice property falls out of reading `Redistribute.backward` in the PyTorch source: it forwards
`ctx.async_op` to its own `redistribute_local_tensor` call. So the single `async_op=True` also
makes the **backward reduce-scatter** asynchronous, with the wait deferred to the first read of the
gradient — which is the `grad.div_(self.world_size)` in `finish_gradient_synchronization`. One flag
buys overlap in both directions.

Prefetching needs to know execution order, and `named_modules()` gives *definition* order, which is
not guaranteed to match. So the order is recorded during the first forward pass and used from the
second onwards (`_fwd_order` / `_order_index` / `_order_ready`).

`_drain_unsharded()` at the top of `forward` retires prefetches that were never consumed. In the
normal flow it is a no-op; it matters if a forward raised part-way through, since a stale entry
would otherwise be handed out after an optimizer step and silently supply pre-update weights.

Note there is no *backward* prefetch here, because there is nothing to prefetch: the weights are
still resident from forward. That is the flip side of the memory trade-off above.

---

## The one real limitation: gloo + CUDA

`FSDP_IMPL=dtensor` **segfaults** on this machine unless you set `CUDA_VISIBLE_DEVICES=""`. This is
not a bug in the implementation. DTensor issues its collectives through *functional collectives*
(`torch.distributed._functional_collectives`), and that path crashes with the gloo backend on CUDA
tensors. Isolated to a 3-line repro, independent of any FSDP code:

```
plain c10d all_gather (CUDA, gloo)  -> OK      [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
funcol  all_gather    (CUDA, gloo)  -> SIGSEGV
funcol  all_gather    (CPU,  gloo)  -> OK      [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
```

It also segfaults at `world_size=1`, so it is not about two ranks sharing one GPU — it is
gloo + CUDA + functional collectives, full stop. `tests/common.py::_setup_process_group` hardcodes
`backend="gloo"` and then picks `cuda:0` whenever `torch.cuda.is_available()`, which lands exactly
on the broken combination. `CUDA_VISIBLE_DEVICES=""` makes it return `"cpu"` instead.

The manual and async implementations are unaffected because they call `dist.*` directly rather than
through funcol. On a real multi-GPU node with NCCL — DTensor's intended configuration — this does
not arise.

---

## Summary

| | `FSDP.py` (manual) | `FSDP_dtensor.py` |
|---|---|---|
| Uneven shards | manual ceil-divide + pad + clamp | `distribute_tensor` handles it |
| Shard layout | flattened, equal + padded | dim-0 rows, unequal |
| All-gather | `dist.all_gather_into_tensor` | `redistribute(→ Replicate())` |
| Reduce-scatter | hand-written in a post-accumulate-grad hook | derivative of the all-gather |
| Memory freeing | `storage.resize_(0)` + version-counter suppression | none by default; `saved_tensors_hooks` under `reshard_after_forward=True` |
| Backward pre-hook | required (re-gather) | not needed |
| Replicated-param sync | manual all-reduce | manual all-reduce (unchanged) |
| Peak memory | `FULL_SHARD` | `SHARD_GRAD_OP`, or `FULL_SHARD` with `reshard_after_forward=True` |
| Works with gloo+CUDA | yes | **no** (funcol segfault) |

The DTensor version is the one to reach for when you want the shortest correct implementation.
`reshard_after_forward=True` closes the memory gap without giving up the derived reduce-scatter, at
the cost of a second all-gather per layer and no backward overlap. The manual version is the one to
read when you want to know what `redistribute` is actually doing.
