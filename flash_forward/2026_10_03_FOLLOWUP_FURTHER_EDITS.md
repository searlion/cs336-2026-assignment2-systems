# Further edits for 2026_10_03_FOLLOWUP.md

I ran an experiment for each of the 13 open questions in `2026_10_03_FOLLOWUP_OPEN_QUESTIONS.md`. That meant:

- the unchanged harness 10 times on every version, instead of 3;
- Nsight Compute and Nsight Systems captures, including Nsight Systems' GPU-clock sampling;
- a matmul in six data types;
- a non-square correctness check;
- ablations of the inner loop;
- 140 fresh autotuning processes.

The evidence (scripts, raw results, summaries) is in `2026_10_03_followup_open_questions/`. Its `README.md` gives one line per answer and says how to re-run everything.

This document has three parts:

1. **Verdicts**: what each open question turned out to be, plus findings that no open question asked about.
2. **Edits, in article order**: for each, the current text, the text I would put in its place, and why.
3. **Claims that held up**: what was checked and needs no change.

Line numbers refer to `2026_10_03_FOLLOWUP.md` as of commit `f3a3430`.

---

## 1. Verdicts

| # | Open question | Verdict | Edit |
|---|---|---|---|
| 1 | The 72 TFLOP/s ceiling | **Corrected.** 72.2 TFLOP/s is the peak at Nsight Compute's locked 2.52 GHz. The harness's N = 4096 kernels run at 2.61–2.68 GHz, where the peak is 75–77 TFLOP/s. At those clocks SDPA reaches 89% (not 95%) and the final kernel 92% (non-causal). These match Nsight Compute's "% of peak", which is the same at any clock setting. | E2, E3, E5, E6, E31 |
| 2 | Tensor-pipe row at half | **Explanation confirmed and sharpened.** A matmul in six data types shows the pipe counter at exactly half the ops counter for bf16, fp16 and TF32 inputs with fp32 accumulation, and equal to it for fp16 accumulation, int8 and fp8. An NVIDIA moderator calls it "a defect in the metric". | E16, E34 |
| 3 | "Faster than SDPA" at N = 4096 | **Holds, with a caveat.** The final kernel was ahead in 10 of 10 paired runs: 103% (non-causal), 104% (causal). Under sustained power-limited load, our kernel's clock drops more than SDPA's and the lead shrinks to +1.7% and −0.7%. | E1, E36 |
| 4 | Non-square causal | **Correct.** All 2,748 cases pass for every version against an explicit top-left mask, the same convention as SDPA; 1,194 of them are non-square causal. D = 16 passes too. | E21 |
| 5 | Why 128-key tiles win | **Refined.** Comparing 64 × 32 × 4 × 3 with 64 × 128 × 4 × 2: the work done once per iteration shrinks 4×, the per-score work does not change, and the K/V copy instructions (which grow with the tile) shrink about 2×. Of the 35% fewer instructions, the `O_acc` rescale is 27%, the K/V copies and their barriers 22%, loop bookkeeping 21%, and the row max/sum/alpha updates 26%. At equal stage counts the drop is 26% (2 stages) or 31% (3 stages). Barrier stalls fall from 3.4 to 0.3 cycles per instruction. Removing the rescale alone recovers only 2.7 of the 4.5%. | E26 |
| 6 | Autotuner-noise fixes | **Different problem than described, and the advice changes.** The careful timings put the causal N = 4096 candidates within 1.9% of each other, not 2–8%. The big error is a *bias*: after the GPU has been idle, the candidates timed first look 5–6% slow while the clock rises. That made the harness pick the slowest-but-one candidate at N = 512 in 9 of 10 runs. A second of warm-up fixes it: 10 of 10 right after a 20 s idle, against 3 of 10 without. The longer `do_bench` was also right 10 of 10 there, but it still times the first candidate in the list 4.7% slow, so it helps only when the best candidate is not first. `cache_results` freezes the first choice. A shorter list made things worse. | E18, E19, E32 |
| 7 | 1.5× duplication | **Confirmed.** 4 HMMA per warp per iteration on the `Q Kᵀ` line against 2 on the `P V` line. | E10 |
| 8 | Barrier stalls | **Attribution confirmed, count corrected.** 99.5% of the barrier samples sit on the K/V copy pipeline's barriers, but there are **two** `BAR.SYNC`s per iteration, not one. | E17, E28, E34 |
| 9 | P-cast accuracy | **Order claim confirmed; the error-source claim is wrong for bf16.** Both orders give 4.51e-5 over 32 heads and 3 seeds. Rounding `P` and rounding the output contribute about equally. Only for TF32 is the error at the output-rounding floor. | E24 |
| 10 | Speed-ups at N = 512 | **v2's "9% faster" is 2% over 10 runs, and even that is an artefact.** N = 512 non-causal is the harness's first shape. Autotuned versions warm the GPU up while tuning; fixed ones are timed cold. SDPA, timed in the same processes, is 3.7% faster in v2's and v5's runs at that row. On a warm GPU, v2's usual choice is 2.4% *slower* than v1's (items 3 and 6). **The final kernel's 141–149% of SDPA held in every run, but is mostly a cold-cache effect.** SDPA's grid is only 1.14 waves (SMs busy 78% of the time), and it loses more than our kernel to the cold L2 that `do_bench` leaves. In the same experiment, the lead is 30% (non-causal) and 35% (causal) with `do_bench`, and 7% and 12% with a warm cache. | E15, E18, E36 |
| 11 | "Within 1% on every shape" | **Not quite.** Within 1% on 7 of 8 shapes; 2.3% behind at N = 1024 causal. Round 1's winner is within 7.3%, not 5%. | E27 |
| 12 | Compiler behaviour | **All 19 checks pass on Triton 3.6.0.** `check_compiler.py` can be re-run after an upgrade. | none (optional E8) |
| 13 | Smaller interpretations | See the list below. | E7, E9, E11–E14, E22 |

Item 13, interpretation by interpretation:

- **SDPA's 3.4 ms:** not reproduced. From cold, the clock sits at 2.52 GHz instead of 2.7 GHz, an 8% effect, and dips like it also occur mid-run.
- **The "Latency" field:** confirmed, 14.101 ms from the start of the launch call.
- **The extra 1,024 bytes:** the `tl.sum` scratch buffer.
- **Reversed causal order:** about 1%, but longest-first *across heads* gains 4–10%.
- **4 and 5 stages slower than 3:** occupancy explains about half of the 5-stage loss (2.0 of 3.5 points) but only a third of the 4-stage loss (0.4 of 1.1). The rest is the deeper pipeline itself.
- **Hardware numbers:** confirmed in NVIDIA's tuning guides.

### Findings that no open question asked about

- **The final profile describes a configuration the harness never ran.** The article's "final profile" (non-causal) is 64 × 32 × 4 × 3, at 90.2%, 264 M instructions and 3.4 barrier cycles. In 10 of 10 harness runs the autotuner picked 64 × 128 × 4 × 2 for that shape, at 92.3%, 171 M instructions and 0.3 barrier cycles. For causal N = 4096 it picked 64 × 32 × 4 × 4 in 6 runs and 64 × 128 × 4 × 2 in 4, at the same speed within 1%. (E28)
- **Longest-first launch order for causal attention is a real gain the article missed:** 4–5% at N = 4096 and 7–10% at N = 2048, checked against SDPA at those two shapes. It still needs the test file and the non-square check before it goes into a kernel. (E22, E38)
- **`--clock-control base` locks this card at 1.98 GHz.** The default, `boost`, is the 2.52 GHz the article quotes. Nsight Compute's % of peak is the same at 1.98, 2.52 and 2.71 GHz (unlocked). (E6)
- **Nsight Compute reports barrier stalls at the first shared-memory or `MUFU` instruction after a `BAR.SYNC`, never on the `BAR.SYNC` itself.** That is worth a checklist line. (E17, E34)
- **The baseline has nine barriers per loop iteration.** Two come from the K/V copy pipeline, five from the cross-warp `tl.max` and `tl.sum`, and two from `P`'s round trip through shared memory. A sentence in Round 0 could say so. (E10)
- **The harness's first row is timed on a cold GPU for fixed-configuration versions but a warm one for autotuned versions.** The harness starts with N = 512 non-causal after a 20-second pause, and `do_bench` warms up for only 25 ms. v2 and v5 first spend about 0.6 s timing their candidates. SDPA, timed in the same processes, is 3.7% faster at that row in v2's and v5's runs than in the others', and within about 1% at every later row. So at N = 512 non-causal, autotuned versions look about 4% better against fixed ones than they are. The harness stays unchanged by design, so this needs a caveat, not a fix. (E18, E36)
- **Most of the article's harness numbers reproduce within 1% over 10 runs, and almost all within 2%.** The exceptions are v2's two autotuner-driven cells and the short kernels at N = 512 causal (spreads of 7–22%). (E30, E36)

---

## 2. Edits, in article order

Each edit gives **where**, **now** (the current text), **change to** (proposed text in the article's voice) and **why** (with the evidence). Optional edits are marked as such.

### E1. Headline paragraph (lines 35 and 42)

**Now (line 35):** "The short version, at N = 4096 with the harness's own numbers (medians of three runs):"

**Change to:** "The short version, at N = 4096 with the harness's own numbers (medians of ten runs):". This applies only if the harness tables are switched to the 10-run medians (E30, E36). The numbers in this table barely move: the previous kernel's causal cell becomes 6.36 ms and 10.8 TFLOP/s instead of 6.43 and 10.7, and SDPA causal 61.5 TFLOP/s instead of 61.3.

**Now (line 42):** "`exp2` and autotuning were worth a few percent each. The kernel is now within 10% of the tensor cores' peak, as is SDPA."

**Change to:** "`exp2` and autotuning were worth a few percent each. Nsight Compute puts the kernel at 92% of the tensor cores' peak at N = 4096, and SDPA at 89%. The lead over SDPA is small, 3 to 4%, but it held in each of ten paired runs."

**Why:** SDPA at 89.2% is 11% below peak, so "as is SDPA" overstates it. The final kernel's 92.3% is for the configuration the harness actually uses (64 × 128 × 4 × 2; see E28). Evidence: `03_harness_repeats/summary.txt` (paired % of SDPA, 10/10 runs) and `05_key_tile/summary.txt`.

### E2. The GPU table (lines 80 and 81)

**Now:** "| Tensor-core peak, bf16 inputs with fp32 accumulation | 512 FLOP per clock per SM, so 72.2 TFLOP/s at the 2.52 GHz Nsight Compute ran at |"

**Change to:** "| Tensor-core peak, bf16 inputs with fp32 accumulation | 512 FLOP per clock per SM: 72.2 TFLOP/s at the 2.52 GHz Nsight Compute locks to, 75 to 77 TFLOP/s at the 2.61 to 2.68 GHz the card runs the harness's N = 4096 kernels at (measured) |"

**Now:** "| cuBLAS bf16 matmul, 8192 × 8192 × 8192 (measured) | 69.4 TFLOP/s |"

**Change to:** "| cuBLAS bf16 matmul, 8192 × 8192 × 8192 (measured) | 69 to 71 TFLOP/s, 93% of the peak at the 2.66 GHz it holds |"

**Why:** see E3. Evidence: `01_clock/harness_n4096_clocks.txt` (per-kernel clocks from Nsight Systems GPU metrics during a harness run) and `01_clock/results/sustained.json` (cuBLAS at 2,655 MHz, 70.9 TFLOP/s).

### E3. "First, the tensor-core peak" (line 86)

**Now:** "First, the tensor-core peak. On GeForce Ada cards, matrix multiplies with fp32 accumulation run at half the rate of those with fp16 accumulation. Nsight Compute lists the peak for "bf16 in, fp32 out" as 512 operations per clock per SM against 1,024 for "fp16 in, fp16 out". Attention needs fp32 accumulation, so the ceiling for this whole article is about 72 TFLOP/s, and cuBLAS reaching 69.4 TFLOP/s on a large matmul says that ceiling is real. In the previous article SDPA reached 68.4 TFLOP/s at N = 4096, which is about 95% of it. That is the bar."

**Change to:** "First, the tensor-core peak. On GeForce Ada cards, matrix multiplies with fp32 accumulation run at half the rate of those with fp16 accumulation. Nsight Compute lists the peak for "bf16 in, fp32 out" as 512 operations per clock per SM against 1,024 for "fp16 in, fp16 out", and the difference is real: the same Triton matmul runs at 72 TFLOP/s with fp32 accumulation and 110 TFLOP/s with fp16 accumulation. Attention needs fp32 accumulation, so the ceiling for this article is 512 operations per clock per SM. In TFLOP/s that ceiling moves with the clock, and this card does not hold its clock still. It is 72.2 TFLOP/s at the 2.52 GHz Nsight Compute locks the clock to, and 75 to 77 TFLOP/s at the 2.61 to 2.68 GHz at which the card ran the harness's N = 4096 kernels. So I compare kernels by Nsight Compute's "% of peak", which counts operations per clock cycle and comes out the same whatever the clock. A plain Triton matmul reaches 97.6% of it, so the ceiling is real, and SDPA reaches 89% at N = 4096. That is the bar."

**Why:** the 72 TFLOP/s used in the article is the peak at the profiler's locked clock, while the harness runs at a higher clock. Per-kernel clocks from Nsight Systems' GPU-metrics sampling during one harness run (`01_clock/harness_n4096_clocks.txt`):

| N = 4096, in one harness run | Clock (Nsight Systems GPU metrics) | Peak at that clock | Harness TFLOP/s | % of it | Nsight Compute % of peak |
|---|---|---|---|---|---|
| Final kernel, non-causal (64 × 128 × 4 × 2) | 2,609 MHz | 74.8 | 68.8 | 92.0% | 92.3% |
| SDPA, non-causal | 2,624 MHz | 75.2 | 66.7 | 88.6% | 89.2% |
| Final kernel, causal (64 × 32 × 4 × 4) | 2,609 MHz | 74.8 | 62.1 | 83.0% | 86.0% |
| SDPA, causal | 2,684 MHz | 76.9 | 60.9 | 79.1% | 82.2% |

(Only the launches the harness timed are used, not the autotuner's; the configuration is identified from each launch's registers and shared memory. The harness was a little slower under Nsight Systems than without it, 68.8 against 70.3 TFLOP/s, which is why the table uses that run's own TFLOP/s.) The non-causal rows agree with Nsight Compute's percentages to within a point. The causal rows come out about 3 points lower because the two count differently. The harness divides by the 68.7 G operations causal attention needs, while Nsight Compute counts the operations executed, including the masked halves of the diagonal tiles: 70.9 G for SDPA. 82.2% × 68.7 / 70.9 = 79.7%, against the 79.1% measured. The same percentages come out at 1.98, 2.52 and 2.71 GHz (E6). The fp16-accumulation matmul figure is from `02_tensor_pipe/matmul_timing.txt`. Evidence: `01_clock/`.

### E4. Second paragraph: A100 and H100 (line 88). Optional

No correction needed. NVIDIA's tuning guides confirm 164 KB of shared memory per SM for A100 and 228 KB for H100. The Ada guide confirms 100 KB per SM, 99 KB per block, 128 KB of combined L1 and shared memory, and "CUDA reserves 1 KB of shared memory per thread block". If the article wants citations, these are the sources:

- https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html

### E5. The baseline, re-measured (line 122)

**Now:** "The card is power-limited under sustained tensor-core load: it drew 218 W of its 220 W limit during the cuBLAS run above, so the clock it holds depends a little on temperature. I treat differences under about 3% between two runs as noise, and run the harness twice whenever a decision depends on a small difference."

**Change to:** "The card is power-limited under sustained tensor-core load: it drew 218 W of its 220 W limit during the cuBLAS run above. So the clock it holds depends on temperature and on the kernel itself. Run back to back for 12 seconds, the kernels in this article held between 2.55 GHz (this article's final kernel, causal) and 2.67 GHz (SDPA, causal). Most of the variation between runs is short dips of 5 to 10% that hit whatever is running at the time, SDPA included. So I compare each kernel with SDPA measured in the same run, treat differences under about 3% as noise, and report medians of ten interleaved runs."

**Why:** `01_clock/results/sustained.json` (per-workload clocks, all at the 220 W cap). In `03_harness_repeats/runs/`, runs 2, 3 and 10 show the dips hitting v1, v4 and SDPA alike. The 200 ms `nvidia-smi` samples show no matching clock drop, so the cause of the dips is still unknown. Use "ten interleaved runs" only if the tables are switched (E30, E36); otherwise keep "run the harness twice".

### E6. "Nsight Compute changes the conditions it measures" (line 186)

**Now:** "**Nsight Compute changes the conditions it measures.** It locks the clocks (here at 2.52 GHz), flushes the caches before every replay pass, and runs the kernel about 45 times. Its durations are close to, but not the same as, the harness's. I use Nsight Compute for counters and ratios, and `do_bench` for time."

**Change to:** "**Nsight Compute changes the conditions it measures.** By default (`--clock-control boost`) it locks the clocks, here at 2.52 GHz, below the 2.6 to 2.7 GHz the card runs the harness at. It also flushes the caches before every replay pass and runs the kernel about 45 times. For the final kernel and SDPA its durations are 6 to 7% longer than the harness's. Its percentages are not affected: the final kernel reaches 92.3% of the tensor-core peak with the default lock, 92.4% with `--clock-control base` (1.98 GHz on this card), and 92.3% with `--clock-control none`, where it ran at 2.71 GHz. I use Nsight Compute for counters and ratios, and `do_bench` for time."

**Why:** `01_clock/reports/clock_control_{base,boost,none}.ncu-rep`. In the same order (default, base, none), SDPA is 89.2%, 89.1% and 89.0%. The 6 to 7% compares the default-lock report with the 10-run harness medians (5.7% for the final kernel non-causal, 6.6% for SDPA non-causal, 7.4% for both causal). The baseline goes the other way: 6.27 ms under Nsight Compute against 6.39 ms in the harness.

### E7. Round 0, step 1: launch latency (line 227). No change needed

The 14.1 ms checks out. In `round0.nsys-rep`, the second SDPA kernel starts 14.101 ms after its launch call begins (14.096 ms after the call returns). NVIDIA's Nsight Systems blog defines launch latency "from the beginning of the launch API call to the beginning of the kernel execution", which matches the article's wording. Evidence: `13_smaller/nsys_latency_round0.txt`, from `13_smaller/nsys_latency.py`.

### E8. What the compiler did (around line 207). Optional

A sentence such as: "These observations are specific to Triton 3.6; `check_compiler.py` (in the evidence folder) re-checks each one after an upgrade." All 19 checks pass today (`12_compiler_behaviour/check_compiler.txt`).

### E9. Round 0, step 1: the 3.4 ms warning (line 233)

**Now:** "One warning about durations in a timeline. The three runs of my kernel above differ by up to 11%, and my first capture of this timeline, without the one-second warm-up in the driver, showed SDPA at 3.4 ms because the GPU's clock was still ramping up. A single kernel in a timeline is not a benchmark."

**Change to:** "One warning about durations in a timeline. The three runs of my kernel above differ by up to 11%, and my first capture of this timeline, without the one-second warm-up in the driver, showed SDPA at 3.4 ms. I could not reproduce that number later, but the warm-up does matter. Nsight Systems can sample the GPU's clock alongside the kernels (`nsys profile --gpu-metrics-devices=0`). After a few seconds idle, the GPU ran at 2.52 GHz for the whole of a short capture, and SDPA took 2.13 ms. After a second of work, it ran at 2.7 GHz and SDPA took 1.97 ms, until the clock dipped to 2.54 GHz for the last two calls and SDPA took 2.11 ms. A single kernel in a timeline is not a benchmark."

**Why:** the clock-ramp explanation predicts 8%, not 70%, and the cold captures did not reproduce 3.4 ms. A cold start gave 2.13 ms, and so did a capture containing SDPA's first-ever call in the process. Evidence: `13_smaller/reports/cold_capture_*.sqlite`, analysed with `01_clock/kernel_clock_from_nsys.py`.

### E10. Round 0, step 6: the 1.5× (line 301)

**Insert after** "...so it is not duplicated: 2 + 1 = 1.5 times the work.":

"The Source page confirms the split: per warp and per loop iteration, the `Q Kᵀ` line executes 4 `HMMA` instructions and the `P V` line 2, although the two products need the same number of operations."

**Optionally**, after "...which are the `STS`, `LDS` and barrier stalls above.":

"In all, the loop has nine `BAR.SYNC`s per iteration: two for the asynchronous K and V copies, five inside `tl.max` and `tl.sum`, and two around `P`'s trip through shared memory."

**Why:** `07_baseline_duplication/hmma_by_line.txt` (4.0 against 2.0 HMMA per warp-iteration; 8 warps doubles both) and `08_barrier_stalls/rounds_noncausal_attribution.txt` (the nine barriers).

### E11. The shared-memory budget (line 362)

**Now:** "The exceptions are the ones where Triton spreads the warps along the key axis, and there the difference is the buffer for `P`."

**Change to:** "The exceptions are the ones where Triton spreads the warps along the key axis. There the difference is the buffer for `P`, plus, when the row reductions cannot reuse other space, a small scratch buffer for them (1 KB for fp32 at D = 128, below)."

**Why:** `13_smaller/smem_fp32_d128.txt`, which lists the shared-memory offsets after Triton's allocation pass.

### E12. Round 1: why 4 and 5 stages are slower (line 421)

**Now:** "The 4-warp row is the only good one. Along it, going from 1 stage (no pipelining) to 3 is worth 10% (57.9 to 63.5 TFLOP/s); 4 and 5 stages are slightly slower again. The extra buffers cut the number of programs per SM from 4 to 3 and then 2, and by then the copies are apparently already early enough that more of them do not help."

**Change to:** "The 4-warp row is the only good one. Along it, going from 1 stage (no pipelining) to 3 is worth 10% (57.9 to 63.5 TFLOP/s); 4 and 5 stages are slightly slower again. Each extra stage adds 8 KB of shared memory, cutting the programs per SM from 4 to 3 and then 2, and that explains part of it. Triton passes the shared-memory size to the launch separately from the compiled code, so the 3-stage kernel can be launched with its shared memory padded to the size of a deeper pipeline, with the code unchanged. Padded to the 5-stage size (2 programs per SM), it loses 2.0 of the 3.5% that 5 stages lose. Padded to the 4-stage size (3 per SM), it loses only 0.4 of their 1.1%. The rest is the deeper pipeline itself."

**Why:** `13_smaller/stages_timing.csv`, produced by `stages_occupancy.py` with 10 interleaved rounds at N = 4096. The padding works by copying the `CompiledKernel` and replacing `metadata.shared` and `packed_metadata`.

### E13. Round 1: 4 stages at N = 512 (line 427)

**Now:** "The 64 × 32 tile wins at every N. Only the stage count moves: 4 stages are best at N = 512."

**Change to:** "The 64 × 32 tile wins at every N. Only the stage count moves: 4 stages are best at N = 512. That is mostly occupancy again, not pipelining. N = 512 has only 256 programs. With 3 stages, 4 fit per SM, 224 at a time, so the grid is 1.14 waves and the second wave leaves most SMs idle. With 4 stages, 3 fit per SM, and the grid is 1.5 waves. Padding the 3-stage kernel's shared memory so that only 3 fit per SM recovers 1.6 of the 2.3 points."

**Why:** `13_smaller/stages_timing.csv` (N = 512 rows).

### E14. Round 1: the fp32, D = 128 failure (line 470)

**Now:** "And with fp32 inputs `tl.dot` uses TF32 instructions, for which Triton lays the warps out along the key axis and adds a buffer for `P`. Three stages need 107,520 bytes, 6 KB over the limit; two stages need 74,752."

**Change to:** "And with fp32 inputs `tl.dot` uses TF32 instructions, for which Triton lays the warps out along the key axis. That adds an 8 KB buffer for `P`, and 1 KB of scratch for the row sum, which now has to combine partial sums from different warps. Three stages need 98,304 + 8,192 + 1,024 = 107,520 bytes, 6 KB over the limit; two stages need 74,752."

**Why:** after the allocation pass, the IR shows Q at offset 0, K at 32,768 and V at 65,536. The `tl.max` scratch and `P` share offset 98,304, and the `tl.sum` scratch (64 rows × 4 warps × 4 bytes) sits at 106,496. 106,496 + 1,024 = 107,520. Evidence: `13_smaller/smem_fp32_d128.txt`; the full dump is in `reports/`.

### E15. Round 1, end: SDPA at N = 512 (line 487)

**Now:** "...and ahead at N = 512, where SDPA itself only reaches 33 TFLOP/s."

**Change to:** "...and ahead at N = 512, where SDPA itself only reaches 33 TFLOP/s. Two things hold SDPA back there. Its 128 × 128 tiles make only 128 programs at this size, 1.14 waves of the 112 that fit on the GPU at once, so Nsight Compute shows its SMs busy for only 78% of the kernel. And it loses more than this kernel to the cold cache that `do_bench` starts every call with (Results)."

**Why:** `10_small_n/summary.txt` (waves, SM activity) and `10_small_n/flush_effect.txt`.

### E16. Round 2, step 2: why the pipe row reads half (line 507)

**Now:** "My best explanation is that the pipe-utilisation counter is scaled to the rate of fp16 inputs with fp16 accumulation (1,024 operations per clock per SM), twice what bf16 with fp32 accumulation can reach on this card (512). Whatever the cause, the consequence is practical:"

**Change to:** "A direct experiment shows where the factor of two comes from. Here is the same Triton matmul (8192³) with different input and accumulator types, with both counters:

| Inputs → accumulator | Operations, % of that type's peak | Tensor pipe % |
|---|---|---|
| bf16 → fp32 | 97.6 | 48.8 |
| fp16 → fp32 | 97.5 | 48.7 |
| tf32 → fp32 | 93.0 | 46.5 |
| fp16 → fp16 | 84.1 | 84.1 |
| int8 → int32 | 64.0 | 64.0 |

For bf16, fp16 and TF32 inputs with fp32 accumulation, the pipe counter reads exactly half; with fp16 accumulation, and for int8 inputs, the two agree. An NVIDIA moderator on the developer forums describes this as a known limitation of the counter on GeForce cards: it tops out at 50% for `HMMA` (fp16, bf16, TF32) with fp32 accumulation, "a defect in the metric that cannot be fixed". So the consequence is practical:"

**Why:** `02_tensor_pipe/pipe_table.txt`, from `matmul_variants.py`. The forum post: https://forums.developer.nvidia.com/t/maximum-tensor-core-utilization/326341/3. The `_v2` version of the counter reads the same. The fp8 path (`sm__ops_path_tensor_src_fp8`) also agrees with the pipe counter, but its peak does not distinguish fp16 from fp32 accumulation, so its "% of peak" tops out at 50% instead. That is a footnote at most.

### E17. Round 2, step 4: barrier stalls (line 519)

**Now:** "Barrier stalls remain at 2.1 cycles, from the synchronisation around the asynchronous K and V copies."

**Change to:** "Barrier stalls remain at 2.1 cycles. The Source page says where. The loop has two `BAR.SYNC`s per iteration, both generated for the `K_j = tl.load(...)` line, which guard the shared-memory buffers of the asynchronous K and V copies. They account for 99.5% of the barrier samples. One detail makes this easy to misread: Nsight Compute reports a barrier stall at the first shared-memory or `MUFU` instruction after a `BAR.SYNC`, not at the `BAR.SYNC` itself, which always shows zero."

**Why:** `08_barrier_stalls/rounds_noncausal_attribution.txt`. Each loop `BAR.SYNC` executes 1.05 M times, once per warp per iteration. SDPA also has two per iteration but four times fewer iterations: 5.6% of its stall samples are barrier stalls, against 17% for v1.

### E18. Round 3: the v2 table and what the autotuner chose (lines 594–607)

**Replace the table** with the 10-run version (from `03_harness_repeats/tables_for_article.md`):

| N | causal | v1, fixed (ms) | v2, autotuned (ms) | v2 TFLOP/s | v2 % of SDPA |
|---|---|---|---|---|---|
| 512 | no | 0.049 | 0.048 | 44.6 | 131% |
| 1024 | no | 0.149 | 0.150 | 57.1 | 101% |
| 2048 | no | 0.534 | 0.537 | 64.0 | 100% |
| 4096 | no | 2.121 | 2.129 | 64.5 | 95% |
| 512 | yes | 0.050 | 0.046 | 23.3 | 120% |
| 1024 | yes | 0.153 | 0.153 | 28.0 | 79% |
| 2048 | yes | 0.545 | 0.545 | 31.5 | 61% |
| 4096 | yes | 2.161 | 2.154 | 31.9 | 52% |

**Now (line 605):** "Mostly the same speed as the hard-coded winner, as the sweep predicted: 9% faster at N = 512 non-causal, where the 4-stage configuration wins, and equal within noise from 1024 to 2048. But at N = 4096 causal it was 8% *slower* (2.34 ms against 2.16), and its three runs disagreed with each other by 9%."

**Change to:** "Mostly the same speed as the hard-coded winner, as the sweep predicted: within 1% from N = 1024 up, with either mask. The interesting row is N = 512, non-causal. One of the candidates there, (64, 32, 4, 4), is 3.6% faster than v1's (64, 32, 4, 3) when timed carefully, yet v2 was only 2% faster than v1, and even that 2% is an artefact of the harness. N = 512 non-causal is the first shape it runs, right after a 20-second pause. v2 first spends about 0.6 s timing its candidates, which warms the GPU up, while v1 is timed on a cold GPU. SDPA, timed in the same processes, is 3.7% faster in v2's runs than in v1's at that row, and within about 1% at every other row."

**Now (line 607):** "Asking the autotuner what it chose explains it. With `TRITON_PRINT_AUTOTUNING=1`, three fresh processes picked three different configurations for that shape: (64, 16, 4, 5), then (64, 32, 4, 2), then (64, 32, 4, 3). Timed carefully, the four candidates are within 2 to 8% of each other at that shape, and their order changes from one repetition to the next. The autotuner times each candidate once, for about 100 ms, on a power-limited card whose speed drifts by a few percent. Among near-equal candidates its choice is close to a coin toss, and sometimes the coin lands on the slowest one. An autotuner can only be as precise as its measurements."

**Change to:** "Asking the autotuner what it chose explains the rest. With `TRITON_PRINT_AUTOTUNING=1`, it picked (64, 16, 4, 5) at N = 512 in 9 runs out of 10. That is the last candidate in the list: 6% slower than (64, 32, 4, 4), and 2.4% slower than v1's own configuration. It is not bad luck. N = 512 is the first shape the harness runs, right after a 20-second pause, and an idle GPU starts at a lower clock (2.52 GHz here) that takes up to a second of work to rise to 2.7 GHz. The autotuner times the candidates in list order, so the first ones are timed on a slower GPU: their timings came out 5 to 6% too high, and the last candidate won. With one second of GPU work before tuning, the autotuner picked (64, 32, 4, 4) in 10 runs out of 10.

At the shapes that come later the clock has settled, and the choice is noise, not bias. At N = 4096 causal, ten runs picked three different configurations. Timed carefully, though, the four candidates are within 2% of each other there, so the choice hardly matters. An autotuner can only be as precise as its measurements, and the first measurements after an idle GPU are the least precise of all."

**Why:**

- `03_harness_repeats/summary.txt` gives the choices per run and v1 against v2 paired by run. At causal N = 4096 the median is −0.2%, and v1 itself spreads 9% across runs, so the article's 3-run "8% slower" was inside v1's own noise.
- The warm-GPU artefact comes from SDPA's median at N = 512 non-causal, by the process it was timed in. It is 0.0655 ms in the v0, v1, v3 and v4 runs and 0.0631 ms in the v2 and v5 runs. At N = 512 causal, the next row, it is 0.0553 ms in all of them.
- `06_autotune_noise/results/ground_truth.csv`, `picks.jsonl`, `picks_warm.jsonl` and `summary.txt` contain:
  - the careful timings: 0.0%, +0.7%, +1.0% and +1.9% at causal N = 4096; at N = 512 non-causal, (64, 32, 4, 4) is fastest, (64, 32, 4, 3) is +3.6%, (64, 16, 4, 5) +6.2% and (64, 32, 4, 2) +35%;
  - the autotuner's own timing of each candidate by list position: +4.7% and +6.2% for the first two after a 20-second idle, about 0 after warm-up;
  - 7 of 10 wrong without warm-up and 10 of 10 right with it, after the same 20-second idle.

### E19. Round 3: "Three things help" (lines 638–644)

**Now:** the three bullets (short list, `cache_results=True`, longer `do_bench`) and the closing paragraph.

**Change to:**

"Three things help, all available in Triton 3.6, and a fourth needs care. I measured each in fresh processes against careful timings of the candidates: the bias after a 20-second idle at N = 512 (10 processes per setting), the rest at N = 4096 causal (20 per setting).

- **Warm the GPU up before the first tuning.** A second of matrix multiplies removed the bias above completely: right 10 times out of 10 after the idle, against 3 out of 10 without.
- **Time more carefully.** `@triton.autotune(..., do_bench=lambda fn, quantiles: triton.testing.do_bench(fn, warmup=100, rep=500, quantiles=quantiles))` gives each candidate five times longer. After the idle it also picked right 10 times out of 10. But its 100 ms warm-up only shrinks the bias to the first candidate in the list, which was still timed 4.7% slow; here that candidate was not the best one. At N = 4096 causal its choices were 0.2% slower than the best on average (1.0% at worst), against 0.3% (1.9%) for the default. It costs about 1.7 s more per new key with four candidates.
- **`cache_results=True`.** All 19 processes after the first reused the first process's choice. That makes the speed reproducible, but whatever the first process chose stays chosen (here, a candidate 0.7% from the best), so tune once, on a warm GPU.
- **A shorter list needs care.** Dropping the two near-duplicates of (64, 32, 4, 3) made the choices worse here (0.8% from the best on average), because one of them, (64, 32, 4, 2), was the fastest candidate at this shape. Which entries earn their place is a question for a sweep.

So for this kernel, on this GPU, at these shapes, autotuning buys at most a few percent of speed and costs several seconds per process. Run on a cold GPU, it can also be wrong most of the time (9 runs in 10 at the harness's first shape), not just now and then. What it does buy is correctness across inputs the sweep never saw. The trade changes in Round 5, where the inner-loop changes make the best configuration depend on the shape and the mask."

**Why:** `06_autotune_noise/summary.txt`, from `results/picks.jsonl` (N = 4096 causal, back-to-back processes), `results/picks_warm.jsonl` (default autotuner after a 20 s idle, with and without warm-up) and `results/picks_long_cold.jsonl` (longer `do_bench` after a 20 s idle). The default autotuner took 0.74 s for 4 candidates; the 5× longer `do_bench` took 2.47 s.

### E20. Round 3: "What it costs". No change

The compile-cost measurements were not part of the open questions, and nothing here contradicts them.

### E21. Round 4: correctness of the two loops (line 734)

**Now:** "A few details matter for correctness, and the unchanged test file checks them. The diagonal range starts at..."

**Change to:** "A few details matter for correctness. The diagonal range starts at..." Keep the rest of the paragraph, then add at its end:

"The unchanged test file checks the square cases. It keeps causal tests square, so I checked non-square causal attention separately (`check_nonsquare_causal.py` in the evidence folder), against an explicit top-left mask, the convention SDPA's `is_causal=True` uses. The check covers:

- N_q larger and smaller than N_k, including ragged lengths;
- D from 16 to 128, in bf16 and fp32;
- key tiles smaller than, equal to and larger than the query tile.

All 2,748 cases pass in every version (the 132 explicit configurations that do not fit in shared memory for fp32 at D = 128 are skipped). Since version 2.1 the FlashAttention library aligns the causal mask to the bottom right instead, so the two libraries give different answers whenever N_q ≠ N_k."

**Why:** `04_nonsquare_causal/summary.txt`. SDPA matches the top-left reference to 1e-15 and differs from bottom-right by about 3. The FlashAttention README states the bottom-right alignment since v2.1.

### E22. Round 4: the reversed launch order (line 750)

**Now:** "One idea that did not survive measurement: in causal attention the last query tiles of each head have the most work, and the GPU starts them last, which can leave a tail of busy SMs at the end of the kernel. Reversing the order (`query_tile_index = tl.num_programs(0) - 1 - tl.program_id(0)`) is a common fix. On the kernel with all three changes in this round it moved the causal time at N = 4096 from 1.079 ms to 1.072 ms, which is inside the noise, so it is not in the kernel. My guess is that with 2,048 programs per launch and 224 running at a time, the tail is a small part of the total."

**Change to:**

"One idea works only in the right form. In causal attention the last query tiles of each head have the most work, and the GPU starts them last, which can leave a tail of busy SMs at the end of the kernel. Reversing the order within each head (`query_tile_index = tl.num_programs(0) - 1 - tl.program_id(0)`) is a common fix, but here it gains only about 1% (1.099 ms to 1.087 ms at N = 4096, over ten interleaved rounds).

The likely reason is the launch order. The GPU starts programs roughly in order of `program_id(0)` first, then `program_id(1)`, so the heads are launched one after another, and the longest tiles of the last heads still start last. Swapping the grid axes, so that `program_id(0)` is the head, and reversing the query tiles along the other axis starts every head's longest tiles first:

```python
    batch_index = tl.program_id(0)
    query_tile_index = tl.num_programs(1) - 1 - tl.program_id(1)
    # launched with grid = (B, triton.cdiv(N_q, Q_TILE_SIZE))
```

That is 4 to 5% faster at N = 4096 and 7 to 10% at N = 2048, for 64 × 32, 64 × 64 and 64 × 128 tiles alike (checked against SDPA at both shapes; it still needs the test file before it goes into a kernel). I found it after the measurements above, so it is not in the final kernel; it is the first thing I would add."

**Why:**

- `13_smaller/reverse_order.csv` (10 rounds per configuration; both variants checked against SDPA).
- `13_smaller/schedule_sim.txt`: a list-scheduling model predicts the same pattern, at most 2% for the reversal and up to 11% for longest-first, as upper bounds.

If a v6 is added instead (see section 3), this paragraph becomes a short Round 4 subsection with its own harness table.

### E23. Round 4, `exp2`: the rescale line (line 831). No change needed

The observation stands. E26 changes how Round 5 uses it.

### E24. Round 4: the P cast (lines 872–880)

**Table (optional):** with all 32 heads and three seeds, the mean absolute errors are 4.51, 4.51 and 3.00 × 10⁻⁵, against 4.52, 4.52 and 3.01 in the article. The footnote could say "all 32 heads, three seeds".

**Now (line 878):** "Casting first changes nothing: the error is dominated by rounding the output to bf16, and the extra conversion back to fp32 for the sum costs a little. The order in the kernel stays."

**Change to:** "Casting first changes nothing: the two orders give the same error to three digits, and the extra conversion back to fp32 for the sum costs a little. The order in the kernel stays. The error itself has two parts of about the same size. Rounding the exact output to bf16 alone gives 2.9 × 10⁻⁵. The kernel writing its output in fp32 (a test-only change) gives 3.2 × 10⁻⁵, which is the cost of rounding `P`. Neither dominates; with both, the error is 4.5 × 10⁻⁵."

**Now (line 880, part):** "It is more accurate (TF32 keeps 10 mantissa bits to bf16's 7), but bf16 is already well inside the test tolerance."

**Change to:** "It is more accurate (TF32 keeps 10 mantissa bits to bf16's 7): its bf16 output is within 4% of the best a bf16 output can be. But bf16 is already well inside the test tolerance."

**Why:** `09_pcast_accuracy/summary.txt` (N = 4096, non-causal; the other shapes look the same). 40% of the bf16-`P` kernel's output elements differ from the correctly rounded exact output, against 13% for TF32.

### E25. Round 5: the re-sweep's opening sentence (line 886). No change needed

### E26. Round 5: why 128-key tiles win (line 903)

**Now:** "Long non-causal sequences now prefer 128-key tiles, which is what Round 4's profile predicted: four times fewer iterations means four times fewer rescales of `O_acc` and four times fewer barriers for the same work."

**Change to:** "Long non-causal sequences now prefer 128-key tiles, which is what Round 4's profile predicted: four times fewer iterations means four times less of everything the loop does once per iteration. Profiling the same kernel and mask with 64 × 32 tiles and 3 stages (Round 1's choice) and 64 × 128 tiles and 2 stages (this round's) shows how much. Executed instructions fall by 35%, from 264 M to 171 M (26% or 31% if the stage count is held at 2 or 3). Of the drop:

- the rescale of `O_acc` is 27%;
- the K and V copies and their barriers are 22%;
- the loop's own bookkeeping is 21%;
- the per-row max, sum and `alpha` updates are 26%.

The lines that run once per iteration fall about 4×: the rescale, the bookkeeping, the `alpha`, `m` and `l` updates, and the cross-thread part of `tl.max` and `tl.sum`. The copies fall about 2×, because their instructions grow with the tile. The per-score work barely changes: the `exp2`, the cast and the multiply by `V` not at all, and the in-thread part of the reductions by 8%. Barrier stalls fall from 3.4 to 0.3 cycles per instruction, and the tensor cores go from 90.2% to 92.3% of their peak. Removing the rescale outright, which gives wrong answers, makes the 32-key kernel only 2.7% faster, so the rescale is part of the story, not all of it."

**Why:** `05_key_tile/summary.txt` (10-round timing with and without the rescale; one Nsight Compute launch per configuration), `05_key_tile/per_line_categories.txt` and `05_key_tile/per_line_diff_noncausal.txt`. The open question's confound, causal against non-causal, is gone: these comparisons use the same mask. With the causal mask, barrier stalls fall from 1.8 to 0.6.

### E27. Round 5: the shortlist (line 905, and the code comment at lines 908–909)

**Now:** "Round 1's winner is still within about 5% everywhere (97.5% of the best on average), but a short list can do better. Picking configurations greedily to raise the worst shape gives five that are within 1% of the best on every shape, one of which is only there so that fp32 at D = 128 has a configuration that fits:"

**Change to:** "Round 1's winner is still within about 7% everywhere (97% of the best on average), but a short list can do better. Picking configurations greedily to raise the worst shape gives five. Re-timed five times against the strongest contenders, they come within 1% of the best on seven of the eight shapes and within 2.3% on the eighth (N = 1024 causal, where 64 × 64 tiles with 3 stages win). One of the five is only there so that fp32 at D = 128 has a configuration that fits:"

**Code comment, now:** "# From the second sweep (after tile-skipping and exp2), chosen so that every benchmark shape / # is within 1% of its best configuration: ..."

**Change to:** "...chosen so that every benchmark shape / # is within about 2% of its best configuration: ...". Make the same change in `2026_10_03_followup/kernels/v5_final/flashattention_autograd_function_triton.py` so the article and the file agree.

**Why:** `11_shortlist_retiming/summary.txt`. The 1% came from a single sweep measurement per configuration, with the list chosen from the same data.

### E28. Round 5: the final profile (lines 943–960)

The article's final profile, non-causal, is of 64 × 32 tiles with 3 stages. In each of the ten harness runs the autotuner picked 64 × 128 tiles with 2 stages for that shape, so the profile should show that configuration. Two ways to fix it:

- **Better:** re-profile with 64 × 128 × 4 × 2 and retake the two screenshots. v5's body is v4's, so `python profile_driver.py kernels/v4_exp2 --config 64,128,4,2 --iters 1` under the article's `ncu` command gives that kernel. It is also the fifth launch in `05_key_tile/reports/v4_key_tiles.ncu-rep`.
- **Minimal:** keep the screenshots, update the text and table below, and say in the captions that they show the Round 1 configuration.

**Now (line 943):** "Both kernels sit on the roof. In the profiled process the autotuner picked 64 × 32 tiles with 3 stages for this shape, which reached 90.2% of the tensor-core peak against SDPA's 89.2%, with the same 137.4 G operations."

**Change to:** "Both kernels sit on the roof. In all ten harness runs the autotuner picked 64 × 128 tiles with 2 stages for this shape, which reaches 92.3% of the tensor-core peak against SDPA's 89.2%, with the same 137.4 G operations."

**Table, change to** (non-causal column from `05_key_tile/summary.txt`; the other columns unchanged):

| N = 4096 | final, non-causal | SDPA, non-causal | final, causal | SDPA, causal |
|---|---|---|---|---|
| Duration under Nsight Compute | 2.06 ms | 2.14 ms | 1.15 ms | 1.19 ms |
| Tensor-core operations | 137.4 G | 137.4 G | 70.9 G | 70.9 G |
| Tensor-core % of peak | 92.3% | 89.2% | 85.6% | 82.2% |
| Instructions executed | 171 M | 220 M | 96 M | 114 M |
| Registers per thread | 255 | 255 | 255 | 255 |
| Shared memory per program | 41.0 KB | 49.2 KB | 41.0 KB | 49.2 KB |
| Longest stall | math pipe throttle (9.1) | math pipe throttle (6.8) | math pipe throttle (8.1) | math pipe throttle (6.6) |
| Barrier stall | 0.3 | 0.6 | 0.6 | 0.9 |

**Now (line 958):** "(For the causal shape the autotuner picked 64 × 128 tiles with 2 stages, hence the 255 registers and 41 KB.)"

**Change to:** "(Both columns use 64 × 128 tiles with 2 stages, hence the 255 registers and 41 KB. For the causal shape the harness's autotuner picked that configuration in 4 runs out of 10 and 64 × 32 tiles with 4 stages in the other 6, at the same speed within 1%.)"

**Now (line 960):** "What is left is small, and the profile says where it is. Both kernels spend their time waiting for the tensor cores, at about 90% of the peak that bf16 inputs with fp32 accumulation allow on this card. In the non-causal kernel, my warps also wait at barriers for 3.4 cycles per instruction against SDPA's 0.6. The SASS has one `BAR.SYNC` per loop iteration, guarding the shared-memory buffers of the asynchronous copies, and with 32-key tiles each program runs four times as many iterations as SDPA's 128-key ones. That is the same per-iteration overhead that made 128-key tiles win in the re-sweep. The causal run, where the autotuner did pick 128-key tiles, shows it: its barrier stall is 0.6 cycles, the same as SDPA's. That is the next place I would look."

**Change to:** "What is left is small. Both kernels spend their time waiting for the tensor cores, at about 90% of the peak that bf16 inputs with fp32 accumulation allow on this card. With 128-key tiles, my kernel's barrier stalls (0.3 cycles per instruction) are no longer above SDPA's (0.6). With Round 1's 64 × 32 tiles they were 3.4. The loop has two `BAR.SYNC`s per iteration, guarding the shared-memory buffers of the asynchronous copies, and 32-key tiles need four times as many iterations. The next thing I would look at is the order in which causal programs start (the end of the tile-skipping section), which is worth 4 to 5% at N = 4096."

**Why:** `03_harness_repeats/summary.txt` (autotuner choices) and `05_key_tile/summary.txt` (the 64 × 128 × 4 × 2 profile). The article's 64 × 32 numbers are reproduced exactly there too: 90.2%, 264 M and 3.38.

### E29. Round 5: the final harness table (lines 926–935). Optional

The 10-run version is in `03_harness_repeats/tables_for_article.md`. At N = 4096 the final kernel's cells are unchanged; the baseline's causal cell moves from 6.427 to 6.363 ms and SDPA's from 61.3 to 61.5 TFLOP/s. The cells that move most:

- N = 512 non-causal: the final kernel is 0.045 ms, 47.9 TFLOP/s and 141%, instead of 0.044, 48.7 and 144%. Part of that row is the warm-GPU artefact in E18: the final kernel autotunes and the baseline does not.
- N = 512 causal: the baseline is 0.113 instead of 0.106 ms.

Everything else moves by under 1%.

### E30. Footnote under the v1 table (line 485)

**Now:** "(These, and every harness table from here on, are medians of three runs of the unchanged harness, with the versions interleaved and a 20-second pause before each run.)"

**Change to (if the tables are switched):** "(These, and every harness table from here on, are medians of ten runs of the unchanged harness, with the versions interleaved and a 20-second pause before each run. "% of SDPA" is the median of each run's own ratio, since SDPA is measured in the same process.)"

The v1 table's cells move by about 1% or less, with one exception: N = 512 causal, where v1 is 0.050 ms (112% of SDPA) instead of 0.047 (115%), and the baseline 0.113 instead of 0.106. The short kernels there vary by up to 11% between runs. Every row is in `tables_for_article.md`.

### E31. Checklist item 2 (line 975)

**Now:** "Decide what "100%" means. For this kernel it is the tensor cores' bf16-with-fp32-accumulation peak (72 TFLOP/s here, half the fp16-accumulation peak on GeForce Ada), checked against a large cuBLAS matmul (69 TFLOP/s)."

**Change to:** "Decide what "100%" means. For this kernel it is the tensor cores' bf16-with-fp32-accumulation peak: 512 operations per clock per SM here, half the fp16-accumulation rate on GeForce Ada, checked against a large matmul (93 to 98% of it). In TFLOP/s it moves with the clock: 72 at Nsight Compute's 2.52 GHz, 75 to 77 at the 2.61 to 2.68 GHz the card actually runs the harness's N = 4096 kernels at. So compare Nsight Compute's "% of peak", not TFLOP/s against a fixed number."

### E32. Checklist: new item after item 2

"3. Autotune on a warm GPU. After an idle period the clock starts low and takes up to a second to rise, so the configurations an autotuner times first look slower than they are."

(Renumber the following items.)

### E33. Checklist item 5 (line 981)

**Now:** "Compare the kernel with a reference doing the same work, on the same timeline. Warm the GPU up for a second first, and do not quote single launches: one capture had SDPA at 3.4 ms instead of 2.0."

**Change to:** "Compare the kernel with a reference doing the same work, on the same timeline. Warm the GPU up for a second first: from idle it runs at 2.52 GHz instead of 2.7, about 8% slower. Do not quote single launches. `nsys profile --gpu-metrics-devices=0` shows the clock next to each kernel."

### E34. Checklist item 7 (line 986), and a new item after item 10 (line 989)

**Item 7, now:** "On GeForce Ada with bf16 inputs and fp32 accumulation, the tensor-pipe row tops out around 50%, and the "Latency Issue" rule fires on kernels that are compute-bound."

**Change to:** "On GeForce Ada, the tensor-pipe row tops out at 50% for matrix multiplies with bf16, fp16 or TF32 inputs and fp32 accumulation; with fp16 accumulation it can reach 100%. So the "Latency Issue" rule fires on kernels that are compute-bound."

**New item after 10:** "Barrier stalls are reported at the first shared-memory or `MUFU` instruction after a `BAR.SYNC`, not at the `BAR.SYNC` itself. Count the `BAR.SYNC`s in the loop and give each one the samples just below it."

### E35. Checklist item 12 (line 991). No change needed

Optionally add: "Occupancy also decides how a small grid splits into waves: 256 programs at 4 per SM is 1.14 waves, at 3 per SM 1.5."

### E36. Results (lines 1008–1033)

**Table:** replace with the 10-run version from `03_harness_repeats/tables_for_article.md`. These cells change by more than 2%:

- v2 at N = 512 non-causal: 0.048 instead of 0.045.
- v2 at N = 4096 causal: 2.154 ms (31.9 TFLOP/s, 52%) instead of 2.342 (29.3, 48%).
- v0, v1 and v4 at N = 512 causal: 0.113, 0.050 and 0.040 instead of 0.106, 0.047 and 0.037. These short kernels vary by 7 to 22% between runs.

Everything else moves by less than 2%: v5 at N = 512 non-causal by 1.7%, v3 and v4 at N = 1024 causal by 1.1%, and the rest by 1% or less.

**"What each round was worth", change to:**

| Change | Non-causal | Causal | Found by |
|---|---|---|---|
| Tiles, warps and stages (v0 → v1) | 3.0× | 2.9× | Speed of Light breakdown, roofline operation count, TTGIR layout |
| Autotuning the v1 configurations (v1 → v2) | none | none (the choice varies between runs) | the harness, then `TRITON_PRINT_AUTOTUNING` |
| Causal tile-skipping (v1 → v3) | 3% | 2.0× | roofline operation count, timeline |
| `exp2` (v3 → v4) | 1.5% | 2% | Source page, PTX, SASS |
| Re-tuning with autotune (v4 → v5) | 4% | under 1% | a second sweep, motivated by the Source page |

**Add after the tables:**

"One caveat on the harness itself: its first row, N = 512 non-causal, follows a 20-second pause, so the versions that autotune (v2 and v5) are timed on a GPU that tuning has warmed up, and the others on a cold one. SDPA, timed in the same processes, is 3.7% faster in v2's and v5's runs at that row, and within about 1% everywhere else. So at that row, compare each version with SDPA in the same run rather than with the other versions; even that is only approximate, because the harness times this kernel before SDPA.

Two caveats on the comparison with SDPA. At N = 4096 the lead is small, 3 to 4%. It held in all ten harness runs, where both kernels ran at nearly the same clock (2.61 GHz for mine, 2.62 GHz for SDPA's non-causal). The harness pauses between calls to flush the cache, so the card spends less time at its 220 W power cap. Run back to back for 12 seconds, at the cap all the time, my kernel holds 2.55 to 2.60 GHz and SDPA 2.64 to 2.67 GHz: my kernel does more work per clock, so it draws more power per clock. There the lead shrinks to 2% non-causal and disappears for causal. At N = 512 the lead is large but mostly measures how each kernel copes with a cold cache. `do_bench` flushes L2 before every call, and SDPA loses more to that than this kernel does. In one experiment with both timings, the lead is 30% (non-causal) and 35% (causal) under `do_bench`, and 7% and 12% when the calls are timed back to back with a warm cache. And SDPA's FlashAttention-2 kernel is general-purpose (dropout, variable lengths, other head dimensions, a backward pass), while this comparison is forward only, at one head dimension."

**Why:**

- `01_clock/results/sustained.json`: v5 at 66.1 against SDPA at 65.0 TFLOP/s non-causal, and 59.2 against 59.6 causal.
- `10_small_n/flush_effect.txt`: at N = 512 non-causal, ours goes from 0.048 to 0.042 ms without the flush, and SDPA from 0.063 to 0.045.
- `03_harness_repeats/summary.txt`: paired ratios over 10 of 10 runs.

### E37. Software and files (line 108). Optional

Add: "The experiments behind the open questions, and their evidence, are in `2026_10_03_followup_open_questions/`."

### E38. Future articles (lines 1039–1043)

**Now (line 1041):** "The remaining 10%: fewer barriers per tile, and whether a different pipelining structure gets closer to the tensor-core peak on Ada."

**Change to:** "The remaining 8%: whether a different pipelining structure gets closer to the tensor-core peak on Ada. With 128-key tiles the barriers are no longer the obvious culprit."

**Add:** "Longest-first launch order for causal attention: swapping the grid axes and reversing the query tiles was 4 to 5% faster at N = 4096 and 7 to 10% at N = 2048 in a quick test (Round 4)."

---

## 3. Claims that held up

These were checked and need no change:

- **The 1.5× duplication mechanism (Round 0).** Measured per source line; the 8-warp case doubles both lines, so 2×.
- **The barrier stalls come from the K/V copy pipeline (Rounds 2 and 5).** 99.5% of the barrier samples land there; only the count per iteration was wrong.
- **Every compiler behaviour the article relies on (Rounds 0, 1 and 4):**
  - the `warpsPerCTA` layouts;
  - Q kept in shared memory and re-read every iteration;
  - `s - 1` buffers each for K and V;
  - the shared-memory formula, exactly, for 2 to 5 stages;
  - `tl.exp` → `ex2.approx.f32` and `tl.math.exp2` → `ex2.approx.ftz.f32`;
  - the free `P` conversion when warps own whole rows, and the shared-memory round trip when they do not.

  All 19 checks pass on Triton 3.6.0.
- **The two-loop causal kernel is correct for non-square shapes and D = 16 (Round 4).**
- **Cast-before-sum and sum-before-cast give the same accuracy (Round 4).**
- **The "Latency" tooltip value and its meaning (Round 0).**
- **The hardware numbers in the setup table:** 100 KB of shared memory per SM, 99 KB per block, 1 KB reserved per block, 128 KB of L1 and shared memory, A100's 164 KB, H100's 228 KB, and 512 operations per clock per SM for bf16 with fp32 accumulation. The Ada whitepaper's RTX 4090 figures give 512 operations per clock per SM too.
- **The roofline and operation-count numbers (Rounds 0, 2 and 5).** 137.4 G operations and 2.16 GB from L2 come out again in the new captures. 206.2 G, 274.9 G and 8.6 GB were re-read from the article's own reports, and the per-line `HMMA` counts of item 7 account for 206.2 G and 274.9 G exactly. The tensor-core percentages do not depend on the clock setting. Evidence: `05_key_tile/l2_bytes_and_ops.txt`.
- **Almost every harness number:** 10-run medians agree with the article's 3-run medians to within 2%, most within 1%, apart from the cells listed in E36.

### If you add a v6

Longest-first ordering is a two-line change to v5 plus a different launch grid. To add it as a final round instead of a note:

1. Copy `kernels/v5_final` to `kernels/v6_longest_first`.
2. Apply the change from E22: `batch_index = tl.program_id(0)`, `query_tile_index = tl.num_programs(1) - 1 - tl.program_id(1)`, and `grid = lambda meta: (B, triton.cdiv(N_q, meta["Q_TILE_SIZE"]))`.
3. Run the unchanged test file and `check_nonsquare_causal.py`.
4. Run the harness ten times alongside v5 (`03_harness_repeats/run_harness_repeats.sh` with `v6_longest_first` added to its version list).
5. Re-profile.

I measured it only for causal shapes. For non-causal shapes all programs are equal, so the order should not matter, but heads now vary fastest, which changes which programs share L2. Measure that too.
