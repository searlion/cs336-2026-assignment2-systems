# Asynchronous FSDP — `distributed_training/FSDP_asynchronous.py`

Same sharding scheme and the same memory model as `distributed_training/FSDP.py` — flat padded
shards, gathered weight freed after forward and re-gathered before backward (`FULL_SHARD`) — but
every collective is issued with `async_op=True` and waited on as late as possible, so communication
overlaps with compute.

## How to run it

```bash
FSDP_IMPL=async uv run pytest tests/test_fsdp.py
```

```
tests/test_fsdp.py::test_fsdp_correctness[fp32]    PASSED
tests/test_fsdp.py::test_fsdp_correctness[fp16]    PASSED
tests/test_fsdp.py::test_fsdp_gradient_sync[fp32]  PASSED
tests/test_fsdp.py::test_fsdp_gradient_sync[fp16]  PASSED
```

Passes on both CUDA+gloo and CPU, and at `world_size=5` (jagged shards) and `world_size=3`, both
dtypes. Unlike the [DTensor version](2026_08_02_FSDP_dtensor.md), this one calls `dist.*` directly,
so it is unaffected by the funcol/gloo/CUDA segfault.

---

## What "asynchronous" buys, and where

The manual version issues every collective with `async_op=False`, so the pattern per layer is
`gather → wait → compute → free`. The wire is idle while a layer computes, and the GPU is idle
while a collective runs. Three changes fix that:

| | `FSDP.py` | `FSDP_asynchronous.py` |
|---|---|---|
| All-gather | `async_op=False`, waited immediately | `async_op=True`, prefetched `prefetch_depth` modules ahead |
| Reduce-scatter | `async_op=False` inside the grad hook | `async_op=True`, drained in `finish_gradient_synchronization` |
| Replicated all-reduce | already async, batched | unchanged |
| `Record.work_handle` | declared, never used | actually used |
| Concurrent in-flight collectives | 1 | `prefetch_depth` |

### 1. All-gather prefetch, `prefetch_depth` modules ahead

`prefetch_depth` defaults to 2, so while module k computes, the gathers for **k+1 and k+2** are both
in flight (k−1 and k−2 during backward). Depth 1 only covers a single layer of compute, which is not
enough when one layer is faster than one collective; depth 0 disables prefetch and reproduces the
synchronous behaviour.

```python
def _prefetch(self, mods, reverse):
    if not self._order_ready:
        return
    i = self._order_index.get(mods)
    step = -1 if reverse else 1
    for d in range(1, self.prefetch_depth + 1):
        j = i + step * d
        if 0 <= j < len(self._fwd_order):
            self._issue_gather(self._fwd_order[j])
```

Order matters inside the hook: **wait for this module first, then issue the next ones**, so the
newly issued collectives fly during this module's compute:

```python
def unshard_forward(self, mods, args=None):
    ...
    self._wait_gather(mods)          # this layer's weight is ready
    self._prefetch(mods, reverse=False)  # k+1, k+2 now in flight during this layer's compute
```

Measured, 2 ranks, 3 steps, 4 sharded modules:

```
depth=0: max concurrent in-flight all-gathers = 1, total gathers = 24
depth=1: max concurrent in-flight all-gathers = 1, total gathers = 24
depth=2: max concurrent in-flight all-gathers = 2, total gathers = 24
depth=3: max concurrent in-flight all-gathers = 3, total gathers = 24
```

Depth D gives D concurrent collectives, and the total gather count is unchanged (4 modules × 2
passes × 3 steps = 24) — prefetching adds no redundant communication, it just moves it earlier.
Depth 1 shows a max of 1 because the module's own gather is waited on before the next is issued;
that one collective still overlaps a full layer of compute.

**Prefetching needs execution order**, and `named_modules()` gives *definition* order, which is not
guaranteed to match. The order is therefore recorded during the first forward pass
(`_fwd_order` / `_order_index`) and used from the second onwards. `forward()` also primes the queue
at both ends:

```python
for mods in self._fwd_order[: self.prefetch_depth]:   # layer 0 doesn't wait on a cold collective
    self._issue_gather(mods)
out = self.module(*inputs, **kwargs)
...
elif torch.is_grad_enabled():
    # backward starts at the last layer, so start its gather now -- it overlaps
    # with the loss computation
    for mods in self._fwd_order[len(self._fwd_order) - self.prefetch_depth :]:
        self._issue_gather(mods)
```

### 2. Asynchronous reduce-scatter

The post-accumulate-grad hook now issues the collective and returns; the handle and **both**
buffers are parked on the `Record`:

```python
rec.reduce_in, rec.reduce_out = flat, out
rec.work_handle = dist.reduce_scatter_tensor(
    output=out, input=flat, op=dist.ReduceOp.SUM, async_op=True)
self._pending_reduce.append(rec)
_free_storage(rec.gathered_full_tensor)   # safe: the collective reads `flat`, a copy
```

and `finish_gradient_synchronization` drains them:

```python
for rec in self._pending_reduce:
    rec.work_handle.wait()
    shard = rec.owner_module.weight_shard
    out = rec.reduce_out.to(shard.dtype)
    shard.grad = out if shard.grad is None else shard.grad.add_(out)
```

So a layer's gradient communication overlaps with the rest of the backward pass — the last layer's
reduce-scatter is in flight while the first layer's gradients are still being computed. This is
what `Record.work_handle` was always there for; the synchronous version declared it and never used
it.

Both buffers must be held on the `Record`, not left as locals. An async collective reads its input
buffer *after* the issuing function returns, so a local `flat` could be freed out from under an
in-flight reduce-scatter. Same reason `gather_send` exists on the `Record`.

---

## The bug this exposed, and why it is the interesting part of the file

Adding prefetch immediately broke the tests:

```
RuntimeError: one of the variables needed for gradient computation has been modified by an
inplace operation: [torch.cuda.FloatTensor [1, 128, 64]], which is output 0 of
AsStridedBackward0, is at version 2; expected version 0 instead.
```

`FSDP.py` already writes into a buffer that autograd has saved — that is the whole point of the
free-and-re-gather trick — and it hides those writes with
`_unsafe_preserve_version_counter`, which snapshots `tensor._version` on entry and restores it on
exit. I wrapped the async issue and the async wait in that context manager the same way. It did not
work, and the reason is specific to async collectives:

```
before issue            0
right after issue       0
after sleep (no wait!)  1   <- the version was bumped with wait() never called
after wait              1
```

**An async collective bumps the output tensor's version counter when it *completes* on a background
thread — not when it is issued, and not when you call `wait()`.** With a prefetch, that completion
lands in the middle of unrelated autograd work, outside any context manager. Two separate
`preserve_vc` windows (one around the issue, one around the wait) leave a gap, and the bump lands
in the gap and survives.

Instrumenting the real run shows it precisely — the version jumps 0 → 2 between the two calls, with
none of my code executing in between:

```
_issue_gather   (0, 0) -> (0, 0)     # issued during lm_head's backward pre-hook
_wait_gather    (2, 2) -> (2, 2)     # linear2's own pre-hook; already at version 2 on entry
```

The fix is to make the suppression window span the **entire in-flight interval**: capture the
version before the collective is issued, restore it after the wait completes.

```python
def _set_version(t: torch.Tensor, version: int) -> None:
    torch._C._autograd._unsafe_set_version_counter((t,), (version,))

# _issue_gather
rec.gather_version = full_weight._version           # snapshot BEFORE issuing
_alloc_storage(full_weight)
rec.gather_handle = dist.all_gather_into_tensor(..., async_op=True)

# _wait_gather
rec.gather_handle.wait()
_set_version(full_weight, rec.gather_version)       # first safe moment to hide the write
```

### Why this is a fix and not a mitigation

`wait()` is a hard synchronization point: when it returns, the collective has completed, so any
version bump it caused has *already* landed. Restoring the version after `wait()` therefore erases
every bump from that collective no matter when the background thread got to it. There is no
remaining timing dependence — which is exactly what the old pattern lacked, since its restore
happened at the close of a window the bump could fall outside of.

Verified adversarially. Injecting a 50 ms sleep between issue and wait forces the completion to
land in the gap — the precise condition that broke the old code — and then running 3 full
train steps at each prefetch depth:

```
old suppression, depth=0: FAIL: one of the variables needed for gradient computation ...
old suppression, depth=1: FAIL
old suppression, depth=2: FAIL
old suppression, depth=3: FAIL
new suppression, depth=0: PASS
new suppression, depth=1: PASS
new suppression, depth=2: PASS
new suppression, depth=3: PASS
```

Two things to read off this. First, the fix holds at every depth under a deliberately widened
window. Second — and this is the point worth keeping — **the old pattern fails at `depth=0` too**
once the window is widened. `depth=0` passing before the fix was timing, not correctness: with no
prefetch the issue and the wait are adjacent, so the completion usually landed inside the `wait()`'s
`preserve_vc` window. If you take one thing from this file, take that:
**`_unsafe_preserve_version_counter` is a synchronous tool, and pairing it with an asynchronous
collective is a latent race even when the tests are green.**

### The invariant the fix still relies on

During the in-flight window the version counter *is* transiently wrong — the fix erases the bump
afterwards, it does not prevent it. That is safe only because of an ordering invariant: the buffer
for module j is written by a collective that is always awaited in module j's own hook, and that
hook runs before module j's backward node — the only node that unpacks j's saved weight. So no
version check ever observes the transient state. A graph where a module's saved tensors could be
unpacked while its gather is still in flight (double-backward, or a hand-built graph that reorders
those nodes) would need the wait moved earlier, not just the restore.

---

## Correctness constraints that async introduces

**1. Every rank must issue collectives on the process group in the same order.** Gloo and NCCL match
collectives positionally, not by tag. If rank 0 issues `[gather(k−1), reduce_scatter(k)]` while
rank 1 issues them in the other order, they pair up wrongly — corrupt data or a hang, not an error
message. This holds here because all ranks execute the same module order and the same autograd
graph, so both the prefetch sequence and the backward node order are identical. It is worth being
aware that a data-dependent model (early exit, MoE routing, dynamic control flow) would break this
assumption, and the standard remedy is separate process groups for gathers and reduce-scatters.

**2. Async buffers must outlive the call that issued them** — hence `gather_send`, `reduce_in`,
`reduce_out` on the `Record`.

**3. A prefetched-but-unconsumed gather must be retired.** `_drain_gathers()` at the top of
`forward` waits on and frees anything left in flight. Without it, a stale handle would still be
marked in-flight on the next iteration and would hand back weights from *before* the last optimizer
step. In the normal flow it is a no-op; it matters when a forward pass raises part-way through, and
it is the reason the backward-prefetch at the end of `forward` is guarded by
`torch.is_grad_enabled()`.

---

## The memory cost of prefetching

Prefetch depth D keeps up to D+1 gathered weights resident instead of 1. That is a real regression
against `FSDP.py`'s peak memory, and it is the dial you trade:

| depth | in-flight collectives | resident gathered weights | comms hidden |
|---|---|---|---|
| 0 | 1 (blocking) | 1 | none |
| 1 | 1 | 2 | one layer of compute |
| 2 (default) | 2 | 3 | two layers of compute |
| 3 | 3 | 4 | three layers |

Still far below the [DTensor version](2026_08_02_FSDP_dtensor.md), which retains *all* gathered
weights from forward to backward. Depth 2 is a reasonable default: enough to cover a collective
that outlasts a single layer, cheap enough that the resident set stays O(1) in model depth.

---

## Caveat on measured speedup

This file is correct and the overlap is real in the sense that the collectives are genuinely in
flight during compute — but do not expect a wall-clock win *on this setup*. With gloo, "async"
collectives run on a background thread that stages CUDA tensors through host memory, so the overlap
is limited. Real overlap wants NCCL, which runs the collective on its own CUDA stream; there you
would additionally need stream management (`torch.cuda.Stream` plus `record_stream` on the comm
buffers) so the caching allocator does not recycle a buffer that a collective is still reading.
That is the next thing to add if you take this to a multi-GPU box.

---

## File map

| Concern | Where |
|---|---|
| Prefetch depth | `FSDP.__init__(prefetch_depth=2)` |
| Issue a gather without waiting | `_issue_gather` |
| Wait + install weight + restore version | `_wait_gather` |
| Prefetch next D modules | `_prefetch` |
| Forward / backward entry points | `unshard_forward`, `unshard_backward` |
| Retire unconsumed prefetches | `_drain_gathers` |
| Issue async reduce-scatter | `reduce_scatter_grad` |
| Drain all collectives | `finish_gradient_synchronization` |
| The version-counter subtlety | `_set_version` docstring |
