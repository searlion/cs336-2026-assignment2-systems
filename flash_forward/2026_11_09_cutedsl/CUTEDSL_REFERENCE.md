# CuTe DSL reference for an FA2 forward on RTX 4070 SUPER (sm_89), nvidia-cutlass-dsl 4.8.1

Everything here was checked against the installed 4.8.1 on this GPU (2026-11-08) unless a line says otherwise.
Source keys used below:
- **[ex-fa2]** `examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py` (CUTLASS tag v4.8.1, identical on main)
- **[ex-gemm]** `examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py` (v4.8.1)
- **[ex-add]** `examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_add.py` (v4.8.1)
- **[docs]** https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ (`media/docs/pythonDSL/*.rst`; main == v4.8.1 for these files)
- **[verified]** a smoke test run here with `.venv/bin/python` (code in §11)

Upstream: https://github.com/NVIDIA/cutlass. Use the `v4.8.1` tag. The old flat path `examples/python/CuTeDSL/ampere/` no longer exists; the examples now live under `cute/ampere/kernel/...`.

**Source decision:** use the official docs and examples. The mcpmarket "cute-gpu-kernel-architect" skill is `ncky/autocute`, 0 stars, written April 2026, and pins no version. It is a workflow index that links to the official docs, with no Ampere MMA, ldmatrix or swizzle content. It also uses `make_fragment_like` throughout; that still works, but `make_rmem_tensor` is the current name.

## 1. Install / run
- Venv: `/home/lewis/github/cs336/cs336-2026-assignment2-systems/flash_forward/2026_11_09_cutedsl/.venv` (torch 2.11+cu130, cutlass-dsl 4.8.1, cuda-bindings 13.5). Run with `.venv/bin/python`. Imports:
  `import cutlass, cutlass.cute as cute; from cutlass.cute.nvgpu import cpasync, warp; from cutlass.cute.runtime import from_dlpack; import cuda.bindings.driver as cuda`
- The DSL targets the current GPU automatically: the PTX `.target` is `sm_89` [verified]. Override with `--gpu-arch` in the options [docs: dsl_jit_compilation_options].
- `apache-tvm-ffi` is **not** installed, so `--enable-tvm-ffi` is unavailable. Pass `from_dlpack` tensors instead (§4).

## 2. Programming model
- `@cute.jit` is host-side code that gets traced and JIT-compiled; launch kernels from it. `@cute.kernel` is device code, launched with `kernel(args...).launch(grid=[x,y,z], block=[x,y,z], stream=stream, smem=bytes_optional)`. Allowed calls: Python → jit; jit → jit, kernel, or plain Python; kernel → jit or plain Python (inlined). Calling a kernel from Python or from another kernel raises an error [docs: dsl_introduction].
- Both decorators work on methods (`@cute.jit def __call__(self, ...)` plus `@cute.kernel def kernel(self, ...)`). That is the class pattern [ex-fa2] uses; Python attributes on `self` are compile-time constants.
- Helpers called inside a kernel can be plain Python functions or `@cute.jit`. Plain functions get inlined; [ex-fa2] does this for `_make_acc_tensor_mn_view` and `_threadquad_reduce`.
- Indices: `tidx, _, _ = cute.arch.thread_idx()`, `cute.arch.block_idx()`, `cute.arch.block_dim()`, `cute.arch.lane_idx()`, `cute.arch.warp_idx()`.
- **Compile-time vs runtime** [docs: dsl_control_flow, limitations]:
  - `if cutlass.const_expr(cond):` and `for i in cutlass.range_constexpr(n):` are evaluated and unrolled in Python, so no IR branch or loop is generated. Use them for tile loops over fragment modes.
  - A plain `for i in range(dyn_n):` or `if dyn_cond:` becomes runtime `scf.for`/`scf.if`. `cutlass.range(n, unroll=..., unroll_full=True)` is a runtime loop with an unroll hint.
  - Variables assigned inside runtime control flow must already exist before it and keep the same type. There is no early `return` or `break` from runtime ifs. You can't read `_`.
  - `a if cond else b` needs `a` and `b` of the same type.
  - A runtime-valued result can't be returned from a `@cute.jit` function called from Python.
  - Function parameters annotated `cutlass.Constexpr` are compile-time. Use them for flags like `is_causal` and for the `SharedStorage` class.
- Numeric types: `cutlass.Float32`, `BFloat16`, `Float16`, `Int32`, `Boolean`. Constants: `cutlass.Float32.inf`, `cutlass.Float32.zero`. Convert a TensorSSA with `.to(cutlass.BFloat16)`.
- Debugging: `cute.printf("x=%f\n", v)` prints at runtime on the device. Python `print(t.type)` or `print(layout)` runs at trace time and shows static layouts [ex-add].

## 3. Layouts, tensors, partitioning (the CuTe algebra)
- `cute.make_layout(shape, stride=...)`; `cute.make_ordered_layout(shape, order=(1,0))` (row-major); `cute.size(x, mode=[i])`; `cute.cosize(layout)` (number of smem elements needed); `cute.ceil_div`.
- Tensor indexing: `mQ[b, None, h, None]` slices; `None` keeps the mode. Shapes and strides are available as `t.shape`, `t.layout.shape`, `t.layout.stride`, `t.element_type`, `t.iterator`.
- `cute.make_tensor(iterator, layout)` re-views the same storage. [ex-fa2] uses it to reuse sQ storage for sO: `cute.make_tensor(sQ.iterator, sO_layout)`.
- `cute.composition(t, layout)` re-indexes a view. The transposed-V trick: `sVt = cute.composition(sV, cute.make_layout((D, bN), stride=(bN, 1)))` [ex-fa2].
- `cute.local_tile(t, tiler, coord)` picks the CTA tile. Put `None` in `coord` to keep the tile-index mode, for example `gK = local_tile(mK[b,None,h,None], (bN, D), (None, 0))` has shape `(bN, D, n_blocks)` [ex-fa2].
- `cute.local_partition(t, thr_layout, tidx)` splits a tile across threads [verified §11 t4].
- `cute.zipped_divide(t, tiler)` gives `((TileM,TileN),(RestM,RestN))` [ex-add].
- **Tiled copies.** Build one with `cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)`. In 4.8.1, `cute.make_tiled_copy(atom, layout_tv, tiler_mn)` takes a TV layout and tiler, not thr/val layouts as in C++ [verified signature].
  - Use: `thr = tiled.get_slice(tidx)`, then `thr.partition_S(src)` and `thr.partition_D(dst)`. Shapes come out as `(CPY_ATOM, CPY_M, CPY_K[, rest])`.
  - `cute.copy(tiled_or_atom, src, dst, pred=bool_tensor)` does the copy; `pred` is optional.
- MMA-matched copies: `cute.make_tiled_copy_A/B/C(copy_atom, tiled_mma)`. Use `thr.retile(frag)` to view an MMA fragment in the copy's layout.
- Register tensors:
  - `cute.make_rmem_tensor(shape_or_layout, dtype)` replaces the removed `make_fragment` (4.3 changelog).
  - `cute.make_rmem_tensor_like(t)`; `cute.make_fragment_like(t, dtype)` keeps t's layout.
  - `.fill(x)`; `.load()` returns a TensorSSA (vector value); `.store(ssa)`; `t[i] = v`.
- TensorSSA supports `+ - * /` with scalars, `.reduce(cute.ReductionOp.MAX|ADD, init, 0)` and `.to(dtype)`. Use `cute.math.exp2(ssa, fastmath=True)`.
- Copy helpers: `cute.autovec_copy(src, dst)` (auto-vectorized, works for rmem→gmem and gmem→rmem) and `cute.basic_copy(src, dst)` (element-wise).
- Predication and bounds: `idT = cute.make_identity_tensor(mQ.layout.shape)` gives a tensor of coordinates. Tile and partition it exactly like the data, then compare with `cute.elem_less(coord, shape)` [ex-fa2, ex-add].

## 4. Torch interop (from_dlpack)
- `from_dlpack(t, assumed_align=16)` gives a cute Tensor with a **static** layout, so the kernel is compiled for that exact shape.
- For one compile that serves many shapes:
  ```python
  def to_cute(t, lead):  # lead = index of the stride-1 dim
      div = 128 // (t.element_size() * 8)          # 8 for bf16 -> dynamic dims stay 16B-divisible
      return (from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=lead)
              .mark_compact_shape_dynamic(mode=lead, stride_order=t.dim_order(), divisibility=div))
  ```
  Bare `.mark_layout_dynamic()` is not enough for cp.async. It loses the 16-byte guarantee on row strides, and compilation fails with `src ptr alignment (16 bits) does not meet requirement (128 bits) of atom simt_async_copy` [verified; pattern from ex-fa2 `create_tensor`].
- Non-contiguous views work. A torch `(B,H,N,D)` tensor passed as `t.permute(0,2,1,3)` (a `(B,N,H,D)` view) through `to_cute(view, 3)` runs [ex-fa2] correctly with no copy [verified].
- Streams: `stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)`. Pass it as a `cuda.CUstream` argument and forward it to `.launch(stream=stream)`.
- Raw `torch.Tensor` arguments are only accepted if the jit parameter is **not** annotated `cute.Tensor` (otherwise `ARG_ANNOTATION_MISMATCH`). Even then they are slow; see §5.

## 5. Compilation and caching [docs: dsl_jit_caching; verified timings]
- `compiled = cute.compile(fn_or_obj, *example_args, options="...")` returns an executor. Call it as `compiled(*args)` with the same argument kinds. `cute.compile` always compiles; you own the cache, for example a dict keyed on `(D, causal, dtype)`.
- Measured host cost per call (vector add):

  | How it's called | Per call |
  |---|---|
  | compiled executor, prebuilt cute tensors | **6 µs** |
  | compiled executor, `from_dlpack` on every call | 18 µs |
  | calling the `@cute.jit` function directly | **~4 ms** (re-traces Python every time even on a cache hit) |
  | `@cute.jit` with raw torch tensors | ~6 ms |

  So: compile once, keep the executor, and wrap tensors once if you can.
- The disk cache lives at `/tmp/$USER/cutlass_python_cache` (`CUTE_DSL_CACHE_DIR`; disable with `CUTE_DSL_DISABLE_FILE_CACHING=True`). The cache key includes env vars, so toggling `CUTE_DSL_LINEINFO` or `CUTE_DSL_DEBUG` recompiles.
- Typed options also work: `from cutlass.cute import KeepPTX, KeepCUBIN, GenerateLineInfo, OptLevel, PtxasOptions`, then `cute.compile[KeepPTX, GenerateLineInfo](fn, ...)`.

## 6. Ampere tensor-core MMA (bf16, m16n8k16, fp32 acc) [ex-fa2, verified]
```python
tiled_mma = cute.make_tiled_mma(
    warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
    (num_warps, 1, 1),                                 # warps tiled along M
    permutation_mnk=(num_warps * 16, 16, 16))          # N extent 16, pairs with ldmatrix.x4 on B (as in ex-fa2)
thr_mma = tiled_mma.get_slice(tidx)
tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))      # (MMA, MMA_M, MMA_K) regs
tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))      # B is (N,K) "K-major": S = Q K^T
acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), cutlass.Float32); acc_S.fill(0.0)
cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)   # D = A*B + C
```
- In PTX this becomes `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`; in SASS, `HMMA.16816.F32.BF16` [verified].
- The accumulator shape per thread is `((2,2), MMA_M, MMA_N)`. Each thread holds 2 rows × 2 adjacent columns per 16×8 atom. The 4 lanes of a quad (`lane//4` = row group) share rows, which is why row reductions need a quad shuffle (§9).
- For P@V, use V as the B operand with shape `(D, bN)`: take the `sVt` composition view and build `thr_mma.make_fragment_B(thr_mma.partition_B(sVt))`.
- Reusing `acc_S` as the A operand of P@V needs a re-view from `(4, MMA_M, MMA_N)` to `((4,2), MMA_M, MMA_N/2)`, after converting to bf16 [ex-fa2 L875-896]:
  ```python
  rP = cute.make_fragment_like(acc_S, dtype); rP.store(acc_S.load().to(dtype))
  d = cute.logical_divide(rP.layout, (None, None, 2))
  tOrS = cute.make_tensor(rP.iterator, cute.make_layout(
      ((d.shape[0], d.shape[2][0]), d.shape[1], d.shape[2][1]),
      stride=((d.stride[0], d.stride[2][0]), d.stride[1], d.stride[2][1])))
  ```
- Other ops in `cutlass.cute.nvgpu.warp`: `MmaTF32Op`, `MmaFP8Op` (sm_89 has fp8 mma), plus the ldmatrix and stmatrix ops.

## 7. smem → registers: ldmatrix [ex-fa2, verified]
```python
s2r_Q = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), dtype), tiled_mma)
s2r_V = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True,  num_matrices=4), dtype), tiled_mma)
thr = s2r_Q.get_slice(tidx); tSsQ = thr.partition_S(sQ); tSrQ_v = thr.retile(tSrQ)
cute.copy(s2r_Q, tSsQ[None, None, k], tSrQ_v[None, None, k])  # per k-block (16 wide)
```
- V is stored `(bN, D)` row-major in smem, but P@V needs it as B=(D, bN), so use `transpose=True` (`ldmatrix.trans`).
- Register double-buffering: load k-block `k+1` before issuing the gemm on `k`, inside `cutlass.range_constexpr` [ex-fa2 L830-849, ex-gemm L702-721].
- SASS shows `LDSM.16.M88.4` [verified].

## 8. gmem → smem: cp.async and pipelining [ex-fa2, ex-gemm, verified]
```python
atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), dtype, num_bits_per_copy=128)
tpr = bK // (128 // dtype.width)                        # threads per smem row (8 bf16 per thread)
g2s = cute.make_tiled_copy_tv(atom, cute.make_layout((nthr // tpr, tpr), stride=(tpr, 1)), cute.make_layout((1, 8)))
cute.copy(g2s, tKgK[None, None, None, n_block], tKsK, pred=tKVpKV)   # issue
cute.arch.cp_async_commit_group()                                     # close a group
cute.arch.cp_async_wait_group(N)                                      # wait until <= N groups pending
cute.arch.sync_threads()                                              # make smem visible CTA-wide
```
- `LoadCacheMode.GLOBAL` emits `cp.async.cg` (L2 only, 16 B); in SASS that is `LDGSTS.E.BYPASS.LTC128B.128` [verified].
- **FA2 schedule** [ex-fa2 `compute_one_n_block`]:
  1. `wait(0)` and sync, so K_j is in smem.
  2. Issue V_j and commit.
  3. Compute S = Q K_j^T via ldmatrix + MMA.
  4. `wait(0)` and sync, so V_j is in smem.
  5. If more blocks remain, issue K_{j-1} and commit.
  6. Online softmax.
  7. O += P V_j.

  K/V each have a single buffer, and loads overlap compute. [ex-fa2] iterates n_blocks in reverse, so the masked (last) block comes first.
- **Multi-stage ring** [ex-gemm, ≥3 stages]: give smem a stage mode `(bM, bK, STAGES)` and index it with a runtime stage `t[None, None, None, stage]`. Prefetch `STAGES-1` tiles, then in the loop call `wait_group(STAGES-2)`, sync, compute, and issue the next tile. §11's GEMM uses 2 stages with `wait_group(1)` and commits even when there was nothing to load, so group counting stays uniform.
- Out-of-bounds handling: use `pred=` on the copy. For rows beyond the end, [ex-fa2] does `tKsK[None, n, None].fill(0)` instead, so no garbage or NaN reaches the MMA.

## 9. smem allocation and swizzle [ex-fa2, verified]
```python
atom = cute.make_composed_layout(cute.make_swizzle(B, 3, 3), 0, cute.make_layout((8, kblk), stride=(kblk, 1)))
#   kblk=64 (128B rows) -> Swizzle<3,3,3>;  kblk=32 (64B rows) -> Swizzle<2,3,3>   [ex-fa2 L226-232]
sQ_layout = cute.tile_to_shape(atom, (bM, D), (0, 1))            # add a 3rd mode for stages: (bM,bK,S),(0,1,2)
@cute.struct
class SharedStorage:
    sQ: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sQ_layout)], 1024]
    sK: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sK_layout)], 1024]
# in kernel (SharedStorage passed as a cutlass.Constexpr arg):
storage = cutlass.memory.SmemAllocator().allocate(SharedStorage)
sQ = storage.sQ.get_tensor(sQ_layout)
```
- `cutlass.utils.SmemAllocator` still works but emits a DeprecationWarning; use `cutlass.memory.SmemAllocator` (needs `import cutlass.memory`) [verified]. [ex-fa2] still uses the old name.
- Dynamic smem is sized from the struct automatically. Nothing needed to be passed to `.launch(smem=...)` even past 48 KB: [ex-fa2] ran with D=128, 128×128 tiles, 96 KB [verified].
- The smem limit on sm_89 is **99 KB/CTA**: `cutlass.utils.get_smem_capacity_in_bytes("sm_89") == 101376`. [ex-fa2] `can_implement` checks against `"sm_80"` (163 KB), so on this GPU it would accept configs that cannot launch. Check against `"sm_89"`.
- Barriers: `cute.arch.sync_threads()` for the whole CTA, or `cutlass.pipeline.NamedBarrier(barrier_id=1, num_threads=n).arrive_and_wait()` [ex-fa2]; `cute.arch.sync_warp()`.

## 10. Online softmax pieces (row view, quad shuffles, exp2) [ex-fa2 L922-1142, verified]
```python
def mn_view(t):   # ((2,2),MMA_M,MMA_N) accumulator -> (rows, cols) view for this thread
    l = cute.make_layout(t.layout.shape)
    mn = cute.make_layout(((l.shape[0][1], l.shape[1]), (l.shape[0][0], l.shape[2])),
                          stride=((l.stride[0][1], l.stride[1]), (l.stride[0][0], l.stride[2])))
    return cute.make_tensor(t.iterator, cute.composition(t.layout, mn))
def quad_reduce(v, op):  # lanes 4r..4r+3 own the same rows
    v = op(v, cute.arch.shuffle_sync_bfly(v, offset=2, mask=-1, mask_and_clamp=31))
    return op(v, cute.arch.shuffle_sync_bfly(v, offset=1, mask=-1, mask_and_clamp=31))
for r in cutlass.range_constexpr(cute.size(row_max)):       # row_max: rmem (acc.shape[0][0]*acc.shape[1],)
    row = acc_S_mn[r, None].load()
    m = quad_reduce(row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0), cute.arch.fmax)
    m = cute.arch.fmax(row_max[r], m)                       # skip on first block
    p = cute.math.exp2(row * scale_log2 - m * scale_log2, fastmath=True)   # scale_log2 = scale*log2(e)
    corr = cute.math.exp2((row_max[r] - m) * scale_log2, fastmath=True)
    row_sum[r] = row_sum[r] * corr + p.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
    acc_O_mn[r, None] = acc_O_mn[r, None].load() * corr; row_max[r] = m; acc_S_mn[r, None] = p
# epilogue: row_sum[r] = quad_reduce(row_sum[r], add); acc_O_mn[r,None] *= cute.arch.rcp_approx(row_sum[r])
```
- `exp2(fastmath=True)` lowers to `ex2.approx`, and the shuffles to `shfl.sync.bfly` [verified PTX].
- [ex-fa2] defers the row_sum quad reduction to the epilogue; each thread keeps its partial sum.
- Causal masking with a row max of `-inf` produces NaN. [ex-fa2] sets the max to 0.0 when it is `-inf` (L1013-1016) and guards `row_sum == 0 or nan` in normalize (L1062).
- LSE for a backward pass is not computed by [ex-fa2]. It equals `m*scale + log(row_sum)` per row, written by lanes with `tidx % 4 == 0`.
- For whole-warp reductions: `cute.arch.warp_redux_sync(v, "fmax"|"add"|...)` (int `redux.sync` is sm_80+; float kinds may need newer arches, so not tested here), or a 5-step `shuffle_sync_bfly` with offsets 16..1. `cute.arch.shuffle_sync_down` and `shuffle_sync_up` are also available.

## 11. Working smoke tests (all pass on sm_89 with 4.8.1)
**t1, vector add** (dynamic layout, compile once, artifacts kept):
```python
@cute.kernel
def vadd_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx(); bidx, _, _ = cute.arch.block_idx(); bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    if i < n:
        gC[i] = gA[i] + gB[i]
@cute.jit
def vadd(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream: cuda.CUstream):
    n = cute.size(mA, mode=[0])
    vadd_kernel(mA, mB, mC, n).launch(grid=[(n + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream)
A, B, C = [from_dlpack(x, assumed_align=16).mark_layout_dynamic() for x in (a, b, c)]
f = cute.compile(vadd, A, B, C, stream, options="--keep-ptx --keep-cubin --generate-line-info")
f(A, B, C, stream)   # also reused unchanged for a different N
```
**t2, bf16 GEMM tile + fused row softmax.** Exercises cp.async (2-stage), Swizzle<2,3,3> smem, ldmatrix.x4, m16n8k16 MMA, quad shuffles and exp2. It gives the reference result for `C = A@B^T` (M,N,K = 256,192,512), and correct P and rowsum for N=64. It is the skeleton of the FA2 inner loop:
```python
bM, bN, bK, STAGES, THREADS = 64, 64, 32, 2, 128
class TileGemm:
    def __init__(self, do_softmax): self.do_softmax = do_softmax
    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mL: cute.Tensor,
                 scale_log2: cutlass.Float32, stream: cuda.CUstream):
        dtype = mA.element_type
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, bK), stride=(bK, 1)))
        sA_layout = cute.tile_to_shape(atom, (bM, bK, STAGES), (0, 1, 2))
        sB_layout = cute.tile_to_shape(atom, (bN, bK, STAGES), (0, 1, 2))
        @cute.struct
        class Smem:
            sA: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sA_layout)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sB_layout)], 1024]
        vec = 128 // dtype.width; tpr = bK // vec
        g2s = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), dtype, num_bits_per_copy=128),
            cute.make_layout((THREADS // tpr, tpr), stride=(tpr, 1)), cute.make_layout((1, vec)))
        tiled_mma = cute.make_tiled_mma(warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
                                        (THREADS // 32, 1, 1), permutation_mnk=(THREADS // 32 * 16, 16, 16))
        grid = (cute.ceil_div(mA.shape[0], bM), cute.ceil_div(mB.shape[0], bN), 1)
        self.kernel(mA, mB, mC, mL, scale_log2, sA_layout, sB_layout, g2s, tiled_mma, Smem).launch(
            grid=grid, block=[THREADS, 1, 1], stream=stream)
    @cute.kernel
    def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mL: cute.Tensor, scale_log2: cutlass.Float32,
               sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout, g2s: cute.TiledCopy,
               tiled_mma: cute.TiledMma, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx(); bm, bn, _ = cute.arch.block_idx()
        gA = cute.local_tile(mA, (bM, bK), (bm, None)); gB = cute.local_tile(mB, (bN, bK), (bn, None))
        gC = cute.local_tile(mC, (bM, bN), (bm, bn))
        st = cutlass.memory.SmemAllocator().allocate(Smem)
        sA = st.sA.get_tensor(sA_layout); sB = st.sB.get_tensor(sB_layout)
        thr_g2s = g2s.get_slice(tidx)
        tAgA, tAsA = thr_g2s.partition_S(gA), thr_g2s.partition_D(sA)   # (CPY,CPY_M,CPY_K,k|stage)
        tBgB, tBsB = thr_g2s.partition_S(gB), thr_g2s.partition_D(sB)
        thr_mma = tiled_mma.get_slice(tidx)
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA[None, None, 0]))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB[None, None, 0]))
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), cutlass.Float32); acc.fill(0.0)
        ldsm = warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
        s2r_A = cute.make_tiled_copy_A(cute.make_copy_atom(ldsm, mA.element_type), tiled_mma)
        s2r_B = cute.make_tiled_copy_B(cute.make_copy_atom(ldsm, mB.element_type), tiled_mma)
        tA, tB = s2r_A.get_slice(tidx), s2r_B.get_slice(tidx)
        tCsA, tCrA_v = tA.partition_S(sA), tA.retile(tCrA)
        tCsB, tCrB_v = tB.partition_S(sB), tB.retile(tCrB)
        k_tiles = cute.size(gA, mode=[2])
        cute.copy(g2s, tAgA[None, None, None, 0], tAsA[None, None, None, 0])
        cute.copy(g2s, tBgB[None, None, None, 0], tBsB[None, None, None, 0])
        cute.arch.cp_async_commit_group()
        for kt in range(k_tiles):
            if kt + 1 < k_tiles:
                nxt = (kt + 1) % STAGES
                cute.copy(g2s, tAgA[None, None, None, kt + 1], tAsA[None, None, None, nxt])
                cute.copy(g2s, tBgB[None, None, None, kt + 1], tBsB[None, None, None, nxt])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
            cute.arch.sync_threads()
            cur = kt % STAGES
            for k in cutlass.range_constexpr(cute.size(tCsA, mode=[2])):
                cute.copy(s2r_A, tCsA[None, None, k, cur], tCrA_v[None, None, k])
                cute.copy(s2r_B, tCsB[None, None, k, cur], tCrB_v[None, None, k])
                cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            cute.arch.sync_threads()
        if cutlass.const_expr(self.do_softmax):
            acc_mn = mn_view(acc)
            tCcC_mn = mn_view(thr_mma.partition_C(cute.make_identity_tensor((bM, bN))))
            for r in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
                row = acc_mn[r, None].load()
                mx = quad_reduce(row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0), cute.arch.fmax)
                p = cute.math.exp2(row * scale_log2 - mx * scale_log2, fastmath=True)
                s = quad_reduce(p.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0), lambda x, y: x + y)
                acc_mn[r, None] = p
                if tidx % 4 == 0:
                    mL[bm * bM + tCcC_mn[r, 0][0]] = s
        cute.autovec_copy(acc, thr_mma.partition_C(gC))   # rmem fp32 -> gmem, no bounds check
# host: args = (to_cute(A,1), to_cute(B,1), to_cute(C,1), to_cute(L,0), scale*log2(e), stream)
#       g = cute.compile(TileGemm(False), *args); g(*args)
```
Resulting kernel: 56 registers, 16 HMMA, 8 LDGSTS, 10 LDSM, no local memory (`cuobjdump -res-usage`).

## 12. NVIDIA's FA2 example on this GPU [verified]
Run `.venv/bin/python flash_attention_v2.py --dtype BFloat16 --batch_size 4 --seqlen_q 4096 --seqlen_k 4096 --num_head 8 --head_dim 64 [--is_causal]`. Its correctness check against torch SDPA passes. The script doesn't print time; `run()` returns µs (`cutlass.cute.testing.benchmark`). Results for bf16, B=4, H=8, N=4096, D=64, 128 threads, 100 iterations:

| config (m×n block) | non-causal | causal |
|---|---|---|
| 128×64 (CLI default) | 2.17 ms | 1.12 ms |
| 128×128 | 2.05 ms | 1.12 ms |
| 64×64 | **1.99 ms** | **1.07 ms** |
| torch SDPA flash backend | 2.10 ms | 1.15 ms |

- Layout is BSHD `(B, N, H, D)`, with grid `(ceil(N/bM), B, H)`. Torch BHSD tensors work via `t.permute(0,2,1,3)` (no copy): causal 1.15 ms, error ≤ 7.8e-3.
- Constraints: fp16/bf16 only, D % 8 == 0, `(2*bM) % threads == 0`. No LSE output.
- The 4.8.1 run passes `softmax_scale` as a Python float; it becomes a `Float32` runtime arg.

## 13. Dumping PTX / cubin / SASS, ncu source view [docs: debugging, dsl_jit_compilation_options; verified]
- Per compile: `cute.compile(fn, *args, options="--keep-ptx --keep-cubin --generate-line-info")` writes `cutlass_<fn>_<argsig>.sm_89.{ptx,cubin}` into the CWD (or `CUTE_DSL_DUMP_DIR`). In memory: `f.__ptx__`, `f.__cubin__` (bytes), `f.__mlir__`. `__ptx__` is **None unless `--keep-ptx`** was given.
- Environment variables: `CUTE_DSL_KEEP=ir,ptx,cubin`, `CUTE_DSL_DUMP_DIR=...`, `CUTE_DSL_LINEINFO=1`, `CUTE_DSL_PRINT_IR=1`, `CUTE_DSL_DEBUG=1`.
- `CUTE_DSL_KEEP=sass` and `--keep-sass` **fail here**: they need nvdisasm ≥ 13.4, and the local toolkit is 13.0. Disassemble the kept cubin yourself instead: `/usr/local/cuda-13.0/bin/cuobjdump -sass x.cubin`, `nvdisasm -g -c x.cubin` (with line info, shows `//## File ".../your.py", line N`), `cuobjdump -res-usage x.cubin`.
- Register and spill report at compile time: `options="remarks{ptx}"`. To cap registers: `--ptxas-options '--maxrregcount=128'`.
- ncu works without admin here (`/opt/nvidia/nsight-compute/2026.2.1/ncu`):
  `ncu -k regex:<ClassName or fn name> -c 1 --set full --import-source yes -o rep python script.py`
  Kernel symbols look like `kernel_cutlass_kernel___main__TileGemm_object_at__tensorptrbf16...`; [ex-fa2]'s contains `FlashAttentionForwardAmpere`. With `--generate-line-info`, `ncu -i rep.ncu-rep --page source --print-source cuda` maps to the Python file [verified]. Note that `-k regex:kernel` also matches torch kernels.
- Ready-made benchmark helper: `cutlass.cute.testing.benchmark(compiled, workspace_generator=..., stream=..., warmup_iterations=, iterations=)` returns µs [ex-fa2].

## 14. Gotchas hit or confirmed with 4.8.1
1. Example paths moved to `examples/python/CuTeDSL/cute/ampere/kernel/{attention,dense_gemm,elementwise}/`.
2. `mark_layout_dynamic()` without `mark_compact_shape_dynamic(..., divisibility=8)` breaks 128-bit cp.async (alignment error at compile time).
3. A `cute.Tensor` annotation rejects raw `torch.Tensor`. Calling `@cute.jit` functions directly re-traces every call (~4 ms); always `cute.compile` once.
4. `cutlass.utils.SmemAllocator` is deprecated in favour of `cutlass.memory.SmemAllocator`. `make_fragment` is gone; use `make_rmem_tensor`. `cutlass.range_dynamic` and `dynamic_expr` are deprecated; use `range` / `cutlass.range`. `make_tiled_copy` takes `(atom, layout_tv, tiler)`; prefer `make_tiled_copy_tv` [changelog].
5. `__ptx__` is None without `--keep-ptx`. SASS dumping needs nvdisasm ≥ 13.4, so use cuobjdump on the kept cubin.
6. sm_89 smem is 99 KB per CTA; don't trust `get_smem_capacity_in_bytes("sm_80")` from the example.
7. Runtime `if`/`for` need pre-declared, type-stable variables. Use `cutlass.const_expr` and `range_constexpr` for anything over fragment modes (they must be static to index register tensors without local-memory spills).
8. The online-softmax `-inf` max in causal rows gives NaN: guard it as in §10.
9. Keep `permutation_mnk=(W*16,16,16)` as in [ex-fa2]/§11. It makes the tiled-MMA N extent 16, which pairs with `LdMatrix8x8x16bOp(num_matrices=4)` on B (verified working; other permutations untested).
10. mbarriers on sm_89 (round 6, `results/round6/cute_mbarrier_probe.txt`). `cute.arch.cp_async_mbarrier_arrive_noinc`, `mbarrier_wait`, `mbarrier_try_wait` and `mbarrier_test_wait` raise `CONFIG_UNSUPPORTED_ARCH` (they check for sm_90+), and `cute.arch.mbarrier_arrive` fails in NVVM ("unsupported operation"), although PTX has all of these, except `try_wait`, since sm_80. What works: `cute.arch.mbarrier_init`; `_nvvm.cp_async_mbarrier_arrive(ptr.to_llvm_ptr(), noinc=True)`; a spin on `_nvvm.mbarrier_wait_parity(ptr.to_llvm_ptr(), phase, _nvvm.MBarrierWaitKind.TEST)`; and inline PTX `mbarrier.arrive.shared::cta.b64` via `llvm.inline_asm` (`from cutlass._mlir.dialects import llvm, nvvm as _nvvm`). In SASS the wait is a polling `LDS` of the barrier word plus a branch, and the cp.async arrive is `ARRIVES.LDGSTSBAR.64`. See `kernels/v6_mbarrier`.
