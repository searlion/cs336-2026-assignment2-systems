# Open questions for 2026_10_03_FOLLOWUP.md

Findings in the follow-up article that are worth checking before publishing. Each item says where the claim sits in the article, what was actually measured, what is my inference on top of the measurement, and how to check it. Items are ordered by how much they could change what the article says.

Status labels:

- **Wrong in the article**: a check after writing showed the article needs a correction.
- **Measured, explanation inferred**: the numbers are measured, the reason given for them is my interpretation.
- **Within noise**: the measured difference is about the size of run-to-run variation.
- **Untested**: a claim or recommendation that no experiment covered.
- **Scope**: true for this setup; may not carry over.

Commands assume you are in `flash_forward/2026_10_03_followup/`, with the repo's `.venv` Python.

| # | Finding | Status | Article sections affected |
|---|---|---|---|
| 1 | The 72 TFLOP/s ceiling uses Nsight Compute's 2.52 GHz, not the clock the harness runs at | Wrong in the article | The setup; checklist item 2 |
| 2 | Nsight Compute's tensor-pipe row reads exactly half of the roofline's "Peak %" | Measured, explanation inferred | Round 2 step 2; checklist item 7 |
| 3 | "Faster than SDPA" at N = 4096 | Within noise | Headline; Round 5; Results |
| 4 | Non-square causal attention in the new two-loop kernel | Untested | Round 4 (causal tile-skipping) |
| 5 | Why 128-key tiles win after the inner-loop changes | Measured, explanation inferred | Round 4 (end of `exp2`); Round 5 |
| 6 | The suggested fixes for autotuner noise | Untested | Round 3 |
| 7 | How the baseline duplicates work (the 1.5×) | Measured, explanation inferred | Round 0 step 6 |
| 8 | Where the barrier stalls come from | Measured, explanation inferred | Round 2 step 4; Round 5 |
| 9 | P-cast accuracy comparison | Measured on a small sample | Round 4 (the P cast) |
| 10 | Speed-ups at N = 512 | Within noise | Rounds 1, 3, 5; Results |
| 11 | "Within 1% of the best on every shape" for the final autotune list | Within noise | Round 5 |
| 12 | Compiler behaviour the article relies on | Scope | Rounds 0, 1, 4 |
| 13 | Smaller interpretations | Inferred | Various |

---

## 1. The tensor-core ceiling is computed at the wrong clock

**Status: wrong in the article.**

**The article says** (The setup → The GPU): the bf16-with-fp32-accumulation peak is 512 FLOP per clock per SM, "so 72.2 TFLOP/s at the 2.52 GHz Nsight Compute ran at", then treats 72 TFLOP/s as the ceiling for the harness: "cuBLAS reaching 69.4 TFLOP/s on a large matmul says that ceiling is real. In the previous article SDPA reached 68.4 TFLOP/s at N = 4096, which is about 95% of it." Checklist item 2 repeats the 72 TFLOP/s.

**What I found after writing.** With `nvidia-smi dmon -s pc` sampling during a sustained 8192³ bf16 cuBLAS matmul loop, the card held **2,670 to 2,685 MHz at 219 W**. Nsight Compute 2026.2.1 locks the clocks during profiling (`--clock-control`, default `boost`), and that lock is what reported 2.52 GHz. The harness does not run under that lock.

At 2.67 GHz the peak is 56 SMs × 512 × 2.67 GHz ≈ **76.6 TFLOP/s**, so:

| | TFLOP/s | % of 72.2 (as in the article) | % of 76.6 (at 2.67 GHz) |
|---|---|---|---|
| cuBLAS bf16 matmul | 69.4 | 96% | 91% |
| SDPA, N = 4096 non-causal (median) | 68.4 | 95% | 89% |
| Final kernel, N = 4096 non-causal (median) | 70.3 | 97% | 92% |

**Not affected:** the "% of peak" numbers read from Nsight Compute (87.5%, 89.2%, 90.2% and so on). Nsight Compute computes its peak at its own locked clock, so those percentages are consistent. The headline "within 10% of the tensor cores' peak" stays true on either basis.

**Still open:** I measured the clock under cuBLAS, not under the attention kernels. The attention kernels may draw different power and hold a different clock, which also changes as the card warms up during a run.

**How to check**

```bash
# terminal 1: power and SM clock, once per second
nvidia-smi dmon -s pc -d 1
# terminal 2
PYTHONPATH=kernels/v5_final python bench_flashattention.py
```

Or profile at the card's own clocks and read "SM Frequency" in the Speed of Light section:

```bash
ncu --clock-control none --section SpeedOfLight --nvtx --nvtx-include "ours N=4096 causal=False/" \
    --launch-count 1 python profile_driver.py kernels/v5_final --N 4096 --iters 1
```

Earlier in the session, `nvidia-smi --query-gpu=clocks.sm` returned idle-looking values (285 MHz) while the GPU drew 218 W. A later query under load agreed with `dmon` (2,670 MHz). I do not know why the first reading was wrong; `dmon` was consistent.

---

## 2. Nsight Compute's tensor-pipe row reads half the roofline's "Peak %"

**Status: measured, explanation inferred.**

**The article says** (Round 2, step 2; checklist item 7): for bf16 inputs with fp32 accumulation on GeForce Ada, "about 50% on that row means the tensor cores are saturated", the "Latency Issue" rule is a false alarm, and compute-boundness should be read from the roofline instead. It calls this "the most important thing I learnt from Nsight Compute on this card".

**Measured.** For every kernel profiled, the pipe counter is exactly half the roofline's percentage:

| Kernel | `sm__pipe_tensor_cycles_active` (% of peak, elapsed) | Roofline Peak % (`sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off`) |
|---|---|---|
| v0 baseline | 22.8 | 45.5 |
| v1 tuned | 43.8 | 87.5 |
| SDPA | 44.6 | 89.2 |

A third counter, `sm__inst_executed_pipe_tensor_op_hmma_v2`, reads 11.4% for the baseline, a quarter of the roofline figure. Different tensor counters use different scales.

**Inferred.** My explanation is that the pipe counter's 100% corresponds to the fp16-in/fp16-accumulate rate (1,024 operations per clock per SM, per the report's peak table), twice the bf16-in/fp32-accumulate peak (512). I did not find NVIDIA documentation that says so.

**Why it matters.** If the explanation is wrong, the article's advice about which number to trust needs rewording. The factor of two itself is measured and safe to report.

**How to check**

1. Read the metric definitions: `ncu --query-metrics --chip ad104 | grep -i pipe_tensor`, and the Nsight Compute Profiling Guide's metrics section.
2. Run a direct experiment: write a minimal Triton matmul in two variants, bf16 inputs with the default fp32 accumulator, and fp16 inputs with `tl.dot(a, b, out_dtype=tl.float16)`. Profile each with:

   ```bash
   ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
   sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed,\
   sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.sum.pct_of_peak_sustained_elapsed \
       python your_matmul.py
   ```

   If the explanation holds, the fp16-accumulate kernel's pipe percentage and ops percentage will roughly agree. The bf16 kernel's pipe percentage will stay at half its ops percentage.

---

## 3. "Faster than SDPA" at N = 4096 is inside the noise

**Status: within noise.**

**The article says** (headline table, Round 5, Results): the final kernel is at 103% of SDPA non-causal (1.954 ms against 2.008 ms) and 104% causal (1.077 ms against 1.121 ms) at N = 4096.

**Measured.** These are medians of three interleaved runs. The margins are 2.7% and 3.9%. The article itself says it treats differences under about 3% as noise, and SDPA's single runs at N = 4096 ranged from 2.0 to 2.2 ms during the session.

**What is solid:** level with SDPA at N = 4096. Clearly ahead at N = 512 (44–50%) and N = 1024 causal (31%), where the margins are far larger than the noise.

**Worth saying in the article:** SDPA's FlashAttention-2 kernel is general-purpose (dropout, variable lengths, other head dimensions), and this comparison is forward only, at one head dimension.

**How to check:** run `final_bench.sh` with more repetitions, for example 10 interleaved runs, and report the spread as well as the median. Locking clocks (`sudo nvidia-smi -lgc <MHz>`) would remove most of the drift, but needs root.

---

## 4. Non-square causal attention is untested in the new loop

**Status: untested.**

**The article says** (Round 4, causal tile-skipping): "A few details matter for correctness, and the unchanged test file checks them", including clipping to `N_KEYS` for ragged lengths.

**Gap.** The unchanged test file skips causal tests when `Nq != Nk` ("keep causal tests square so the mask convention is unambiguous"). The two-loop split in v3–v5 computes `unmasked_stop` and `masked_stop` from both `q_start` and `N_KEYS`. I reasoned through `Nq > Nk` and `Nq < Nk`, but no test covers either.

**Convention.** v0–v5 use top-left alignment: row `q` sees keys `k <= q`. This matches `F.scaled_dot_product_attention(..., is_causal=True)`. The FlashAttention library itself uses bottom-right alignment when `Nq != Nk`, so "correct" depends on which convention you want.

**Also untested:** D = 16. `forward` accepts it, but no shape in the test file uses it, in any version.

**How to check:** write a separate test, leaving the unchanged test file alone. Compare v5 against an explicit top-left reference, `torch.ones(Nq, Nk).tril()` used as the mask, for shapes such as (1, 200, 96, 64), (1, 96, 200, 64), (1, 37, 53, 32) and (2, 100, 100, 16).

---

## 5. Why 128-key tiles win after the inner-loop changes

**Status: measured, explanation inferred.**

**The article says** (end of the `exp2` section; Round 5): after `exp2`, `O_acc = alpha * O_acc` is 12.7% of instructions and costs the same per iteration however many keys the iteration covers, so bigger key tiles amortise it. The re-sweep's preference for 64 × 128 tiles at large N is "what Round 4's profile predicted: four times fewer iterations means four times fewer rescales of `O_acc` and four times fewer barriers."

**Measured:**
- The re-sweep winners: 64 × 128 with 2 stages at N = 2048 and 4096 non-causal.
- The v4 Source page: the rescale line at 12.7%.
- Barrier stalls: 0.6 cycles per instruction for the final kernel's causal run, which used 64 × 128 tiles, against 1.8 to 3.4 for runs with 64 × 32 tiles.

**Inferred:** that the rescale and barriers are the cause. Plausible alternatives are general loop overhead and address arithmetic per iteration, how many tensor-core instructions the compiler can schedule back to back, and the 2-stage pipeline itself.

**A confound:** the barrier comparison in the final table puts a causal 64 × 128 run next to non-causal 64 × 32 runs, so the mask and the tile size both changed.

**How to check:** profile the same kernel and mask with both tile shapes, then compare per-line instruction counts on the Source page, barrier stalls and total instructions:

```bash
ncu --set full --import-source yes --nvtx --nvtx-include "ours N=4096 causal=False/" --launch-count 1 \
    -o results/ncu/v4_64x32 python profile_driver.py kernels/v4_exp2 --config 64,32,4,3 --iters 1 --sdpa 0
ncu --set full --import-source yes --nvtx --nvtx-include "ours N=4096 causal=False/" --launch-count 1 \
    -o results/ncu/v4_64x128 python profile_driver.py kernels/v4_exp2 --config 64,128,4,2 --iters 1 --sdpa 0
```

---

## 6. The suggested fixes for autotuner noise are untested

**Status: untested** (the problem itself is measured).

**Measured** (Round 3): in three fresh processes, v2's autotuner chose three different configurations for causal N = 4096: (64, 16, 4, 5), (64, 32, 4, 2) and (64, 32, 4, 3). Carefully timed, the candidates were within 2 to 8% of each other, and their order changed between repetitions. v2's median at that shape was 8% slower than v1's (2.342 ms against 2.164 ms), with a 9% spread across its three runs.

**Untested:** the three suggestions in the article.
- A shorter list.
- `cache_results=True`. It makes the choice sticky by construction, but a sticky first choice can still be the bad one.
- A longer `do_bench` (`do_bench=lambda fn, quantiles: triton.testing.do_bench(fn, warmup=100, rep=500, quantiles=quantiles)`). The lambda's signature matches how Triton 3.6's `Autotuner._bench` calls it (`self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))`, read from the source), but I have not run it, and I do not know whether 5× longer timing makes the choice stable.

**How to check:** run the choice 10 times with each setting and count distinct winners:

```bash
for i in $(seq 10); do PYTHONPATH=kernels/v2_autotune TRITON_PRINT_AUTOTUNING=1 python -c "
import torch
from flashattention_autograd_function_triton import FlashAttentionTriton as F
q, k, v = (torch.randn(32, 4096, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3))
F.apply(q, k, v, True)" 2>&1 | grep 'best config'; done | sort | uniq -c
```

Then repeat with the longer `do_bench` added to the decorator in a copy of `kernels/v2_autotune`.

---

## 7. How the baseline duplicates work

**Status: measured, explanation inferred.**

**The article says** (Round 0, step 6): with `warpsPerCTA = [1, 4]`, the 16 × 16 score tile is only two 16 × 8 MMA blocks wide, so warps 2 and 3 repeat the `Q Kᵀ` work of warps 0 and 1. The `P V` product (16 × 64, eight blocks wide) is not repeated, which gives 2 + 1 = 1.5×.

**Measured:** 206.2 G tensor operations against the algorithm's 137.4 G, exactly 1.5×, and the `warpsPerCTA = [1, 4]` layout in the TTGIR.

**Inferred:** which multiply is duplicated. The total is consistent with the explanation, but a different split could also give 1.5×.

**How to check:** both products have the same FLOPs (2 × 16 × 16 × 64), so if the explanation is right, the `Q Kᵀ` line executes twice as many tensor instructions as the `P V` line. In `ncu-ui`, on the Source page for `results/ncu/v0_N4096_nc.ncu-rep`, compare the tensor-instruction counts (instruction-mix column) for the two `tl.dot` lines. Alternatively, count `HMMA` instructions per loop iteration in the SASS (`python inspect_kernel.py kernels/v1_tiles 16,16,4,3 --sass --dump-sass v0.sass`) and attribute them to the two dots.

---

## 8. Where the barrier stalls come from

**Status: measured, explanation inferred.**

**The article says** (Round 2, step 4; Round 5): the remaining barrier stalls come from synchronisation around the asynchronous K and V copies, "one `BAR.SYNC` per loop iteration".

**Measured:** the barrier stall per issued instruction (2.1 for v1, 3.4 for the final kernel non-causal, 0.6 for SDPA), and one `BAR.SYNC` in the loop body of the 64 × 32 kernel's SASS.

**Inferred:** that the stall samples land on that `BAR.SYNC`, rather than on barriers elsewhere, such as in the reductions.

**How to check:** in `results/ncu/rounds_noncausal.ncu-rep`, open the final kernel's Source page with the SASS view. Set "Navigate By" to the warp-stall sampling metric and jump to the highest samples; the barrier stall reason should sit on the loop's `BAR.SYNC`.

---

## 9. P-cast accuracy comparison

**Status: measured on a small sample.**

**The article says** (Round 4, the P cast): casting before or after the row sum gives the same mean absolute error (4.52 × 10⁻⁵); TF32 gives 3.01 × 10⁻⁵. The error "is dominated by rounding the output to bf16".

**Measured:** error against a float64 reference, on 2 of the 32 heads, one random seed, N = 1024 and 4096 (`variants_experiment.py`). The maximum errors were identical across variants.

**Inferred:** that output rounding dominates. It is consistent with the identical maxima, but not isolated.

**How to check:** use more heads and seeds. Add a test-only variant that writes `O` in fp32, to separate the rounding of `P` from the rounding of the output.

---

## 10. Speed-ups at N = 512

**Status: within noise.**

**The article says:** v2 is 9% faster than v1 at N = 512 non-causal; the final kernel is at 144–150% of SDPA at N = 512.

**Measured:** these kernels take 37–49 µs, where launch overhead and clock state matter more, and the numbers are medians of three runs. The v2 figure in particular is inside v2's own spread (up to 9% across runs at some shapes). The 144–150% against SDPA is a much larger margin and less likely to be noise. I did not investigate why SDPA reaches only about 33 TFLOP/s there.

**How to check:** more runs, or `do_bench` with a longer `rep` for the small shapes.

---

## 11. "Within 1% of the best on every shape"

**Status: within noise.**

**The article says** (Round 5): the five-configuration autotune list is "within 1% of the best on every shape" (worst 99.3%).

**Measured:** this comes from a single sweep, one measurement per configuration and shape, while run-to-run noise is 2 to 3%. Choosing the list from the same data that scores it also overfits that noise, and the tuner's own in-process choice adds noise on top (item 6).

**How to check:** re-time only the shortlisted configurations, plus the per-shape winners, several times with `sweep.py`'s timing code. Compare against the per-shape best in each repetition.

---

## 12. Compiler behaviour the article relies on

**Status: scope.**

Everything was measured with Triton 3.6.0, PyTorch 2.11 (CUDA 13.0), driver 595.91, on one RTX 4070 SUPER. Several conclusions rest on choices this Triton version's compiler made. Any of them can change in another version:

- **The warp layout for each tile and warp count.** For 16 × 16 with 4 warps it is `[1, 4]` (along keys); for 64 × 32 with 8 warps it is `[8, 1]` (stacked along queries, duplicating rows); for 32 × 32 with 2 warps it is `[1, 2]`. The "never more warps than 16-row slices" rule is an empirical ceiling built on these layouts.
- **Q is kept in shared memory and reloaded every iteration**, rather than held in registers. This is part of the shared-memory formula.
- **`num_stages = s` gives `s - 1` buffers each for K and V.**
- **`tl.exp` lowers to `mul` + `ex2.approx.f32` (non-FTZ), and `tl.math.exp2` to `ex2.approx.ftz.f32`.** The `exp2` section's SASS story depends on this.
- **The P layout conversion is free when warps own whole rows**, and goes through shared memory when they do not.

**How to check after an upgrade:** re-run `inspect_kernel.py` on the configurations in the article (`--ttgir`, `--sass`) before reusing these conclusions.

---

## 13. Smaller interpretations

**Status: inferred.**

- **SDPA at 3.4 ms in the first capture** (Round 0; checklist item 5). I attributed it to the GPU clock still ramping up, because adding a one-second warm-up made it go away. That is a single observation. Nsight Systems can sample clocks with `--gpu-metrics-devices=0` to confirm it.
- **The Nsight Systems "Latency" field** (Round 0, step 1). I read "Latency: ←14.1 ms" as the time between the launch API call and the kernel starting. Check this against the Nsight Systems documentation.
- **The extra bytes in the fp32, D = 128 failure** (Round 1). The failure needed 107,520 bytes against the formula's 98,304. The TTGIR shows a buffer for `P` (8,192 bytes); I did not identify the remaining 1,024.
- **Reversing the causal program order** (Round 4). The "no gain" verdict rests on one measurement (1.079 ms against 1.072 ms, on v4), and the explanation offered is a guess.
- **4 and 5 stages being slower than 3** (Round 1). The explanation (fewer programs per SM, copies already early enough) is hedged in the article and not tested.
- **Hardware numbers from documentation rather than measurement** (The setup):
  - Measured or confirmed on this card: 102,400 bytes of shared memory per SM and 101,376 per block (device properties), the 1 KB driver reservation and the 24-block limit (Nsight Compute's launch statistics and occupancy sections).
  - From documentation only: 128 KB of L1 plus shared memory per SM, and the A100 (164 KB) and H100 (228 KB) shared-memory figures.
