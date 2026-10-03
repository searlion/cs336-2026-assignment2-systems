# Evidence for the open questions in 2026_10_03_FOLLOWUP_OPEN_QUESTIONS.md

Each numbered folder answers the item with the same number in `../2026_10_03_FOLLOWUP_OPEN_QUESTIONS.md`. Item 10 has its own folder (`10_small_n`); most of its data comes from `03_harness_repeats`. The edits these findings call for are in `../2026_10_03_FOLLOWUP_FURTHER_EDITS.md`.

Everything was measured on the same machine as the article: RTX 4070 SUPER, driver 595.91, PyTorch 2.11 (CUDA 13.0), Triton 3.6.0, Nsight Systems 2026.1.3, Nsight Compute 2026.2.1. The kernels and the unchanged harness come from `../2026_10_03_followup/`.

## Answers at a glance

| # | Question | Answer |
|---|---|---|
| 1 | Which clock does the 72 TFLOP/s ceiling apply to? | 72.2 TFLOP/s is the peak at Nsight Compute's locked 2.52 GHz. In the harness the N = 4096 kernels run at 2.61–2.68 GHz, measured per kernel with Nsight Systems GPU metrics, so the peak there is 75–77 TFLOP/s. At those clocks the final kernel reaches 92.0% and SDPA 88.6% (non-causal), close to Nsight Compute's own "% of peak" (92.2–92.3% and 89.1–89.2%), which does not change with the clock setting. |
| 2 | Why does the tensor-pipe counter read half? | Measured with a matmul in six data types. The counter reads exactly half the ops counter for bf16, fp16 and TF32 inputs with fp32 accumulation. It matches the ops counter for fp16 accumulation, int8 and fp8. An NVIDIA moderator calls this "a defect in the metric" on GeForce. |
| 3 | Is the final kernel really faster than SDPA at N = 4096? | In the harness, yes, by a small margin that held every time: 103% (non-causal) and 104% (causal) of SDPA, faster in 10 of 10 paired runs. Under sustained back-to-back load the card's power limit lowers our kernel's clock more than SDPA's, and the lead shrinks to +1.7% (non-causal) and −0.7% (causal). |
| 4 | Is non-square causal attention correct in the two-loop kernel? | Yes. All 2,748 cases pass against an explicit top-left mask (1,194 of them non-square causal), for every version, D = 16, and key tiles both smaller and larger than the query tile. SDPA's `is_causal` is top-left. |
| 5 | Why do 128-key tiles win after Round 4? | Comparing 64 × 32 × 4 × 3 with 64 × 128 × 4 × 2: the once-per-iteration work shrinks about 4× (the copies about 2×), the per-score work stays the same, and instructions fall 35% (26–31% at equal stage counts). The rescale of `O_acc` accounts for 27% of that drop, the K/V copies and their barriers 22%, loop bookkeeping 21%, and the row max, sum and alpha updates 25%. Barrier stalls fall from 3.4 to 0.3 cycles per instruction. Removing the rescale alone recovers only 2.7% of the 4.5%. |
| 6 | Do the suggested fixes for autotuner noise work? | The careful timings put the causal N = 4096 candidates within 1.9% of each other, so a wrong choice costs little there. The real problem is a bias: after the GPU has been idle, the candidates timed first look 5–6% slow while the clock rises. That made the harness pick the last candidate, 6.2% slower, in 9 of 10 runs at N = 512. One second of warm-up fixes it. The 5× longer `do_bench` picked right too, but it still times the first candidate 4.7% slow. `cache_results` makes the first choice permanent. A shorter list made it worse here, because it dropped the best candidate. |
| 7 | Which multiply does the baseline duplicate? | `Q Kᵀ`: 4 HMMA per warp per iteration against 2 for `P V`, so 1.5×. Confirmed. |
| 8 | Where do the barrier stalls come from? | 99.5% fall on the two `BAR.SYNC`s per loop iteration of the K/V copy pipeline. The article says one per iteration; there are two. |
| 9 | P-cast accuracy | Cast-then-sum and sum-then-cast give the same error (4.51e-5 at N = 4096, all 32 heads, 3 seeds). Rounding `P` and rounding the output contribute about equally, so "dominated by output rounding" is true only for TF32. |
| 10 | Speed-ups at N = 512 | v2 is 2% faster than v1, not 9%, and even that is an artefact. N = 512 non-causal is the harness's first shape, timed on a cold GPU unless the version autotunes first; SDPA, timed in the same processes, is 3.7% faster in v2's and v5's runs at that row. On a warm GPU, v2's usual choice (item 6's bias) is 2.4% slower than v1's. The final kernel's median is 141% (non-causal) and 149% (causal) of SDPA, and it is ahead in every run (at least 132%). Two reasons: SDPA's grid is 1.14 waves (its SMs are idle 22% of the time), and SDPA loses more to the cache flush that `do_bench` runs before every call (+41% against +13% for the final kernel, non-causal). Without the flush (`do_bench_cudagraph`, warm GPU) the lead is 13% (non-causal) and 12% (causal). |
| 11 | Is the five-config list within 1% on every shape? | Within 1% on 7 of 8 shapes. On N = 1024 causal it is 2.3% behind (64 × 64 × 4 × 3 wins there). Round 1's winner is within 7.3% everywhere (2.8% on average). |
| 12 | Compiler behaviour | All 19 checks pass on Triton 3.6.0. Re-run `12_compiler_behaviour/check_compiler.py` after an upgrade. |
| 13 | Smaller interpretations | Details below. |

Item 13 in detail:

- **SDPA's 3.4 ms in the first capture:** not reproduced. From cold, the clock sits at 2.52 GHz for about a second, an 8% effect, not 70%.
- **The "Latency" field:** 14.101 ms measured from the start of the launch call, consistent with the tooltip.
- **The extra 1,024 bytes for fp32 at D = 128:** the `tl.sum` cross-warp scratch buffer.
- **Reversed causal order:** about 1%. Longest-first *across heads* gains 4–5% at N = 4096 and 7–10% at N = 2048.
- **Why 4 and 5 stages are slower than 3:** occupancy explains 2.0 of 5 stages' 3.5%, but only 0.4 of 4 stages' 1.1%. The deeper pipeline accounts for the rest.
- **Hardware numbers:** confirmed in NVIDIA's tuning guides.

## Folders

| Folder | What is in it |
|---|---|
| `01_clock/` | `sustained_clock.py`: 12 s per workload with `nvidia-smi` sampling (`results/sustained.json`). `harness_clock.py`: clocks during the 60-run harness batch. `kernel_clock_from_nsys.py` and `harness_n4096_clocks.py`: per-kernel clock from a harness run under `nsys --gpu-metrics-devices=0` (`harness_kernel_clocks.txt`, and `harness_n4096_clocks.txt`, which uses only the launches the harness timed). Nsight Compute captures with `--clock-control base`, `boost` and `none` are in `reports/`. |
| `02_tensor_pipe/` | `matmul_variants.py`: one Triton matmul in six data types. `matmul_timing.txt`: TFLOP/s for each. `pipe_table.txt`: pipe counters against ops counters for the matmuls and the attention kernels. |
| `03_harness_repeats/` | `run_harness_repeats.sh`: `final_bench.sh` with 10 runs instead of 3. Raw outputs, including autotuner choices, are in `runs/`. `summary.txt`: medians, ranges, paired % of SDPA, version-to-version changes and autotuner choices. `compare_with_article.txt`: the article's 3-run medians next to these. `tables_for_article.py` writes every harness table in the article from these runs (`tables_for_article.md`). `clock_ramp.py`: the SM clock in the first seconds of each run, from the 200 ms `nvidia-smi` samples (`clock_ramp.txt`). |
| `04_nonsquare_causal/` | `check_nonsquare_causal.py`: a separate test; the unchanged test file is left alone. `results.csv` has every case; `summary.txt` the totals and SDPA's convention. |
| `05_key_tile/` | `key_tile_driver.py`: timing with a no-rescale ablation (`timing.csv`), and one Nsight Compute launch per configuration (`reports/v4_key_tiles.ncu-rep`). `summary.txt`: timing and profile table. `per_line_categories.txt` and `per_line_diff_noncausal.txt`: where the instructions go, by category and line by line. `l2_bytes_and_ops.txt`: L2 traffic and tensor operations per kernel. |
| `06_autotune_noise/` | `make_variants.py` creates three decorator variants in `kernels/`. `ground_truth.py`: careful timing of v2's candidates. `run_picks.sh`: 20 fresh processes per setting. `run_picks_warm.sh` and `run_picks_long_cold.sh`: the clock-ramp test with harness-like cool-downs, for the default and the longer `do_bench`. `summary.txt`: choices, costs, and the autotuner's per-position timing bias. |
| `07_baseline_duplication/` | `hmma_by_line.py`: HMMA instructions per source line, from Nsight Compute's Source page (`hmma_by_line.txt`; exports in `source_*.csv`). |
| `08_barrier_stalls/` | `barrier_attribution.py`: assigns barrier stall samples to the `BAR.SYNC` they wait at (`rounds_noncausal_attribution.txt`; exports in `source_*.csv`). |
| `09_pcast_accuracy/` | `pcast_accuracy.py`: three variants, bf16 and fp32 output, 32 heads, 3 seeds (`results.json`, `summary.txt`). |
| `10_small_n/` | `analyze.py`: SDPA against the final kernel at N = 512 under Nsight Compute: waves, SM activity, occupancy (`summary.txt`). `flush_effect_warm.py`: the effect of `do_bench`'s cache flush at N = 512, on a warm GPU, against `do_bench_cudagraph` (no flush, no CPU gaps), for v5's and v1's configurations (`flush_effect_warm.txt`; the article's numbers). The earlier `flush_effect.py` and `flush_effect_v1.py` are superseded: they time our kernel first in a fresh process, while the GPU is still cold, and their column C (flush, wait, then time) includes launch time, because the GPU is idle when each call is issued. |
| `11_shortlist_retiming/` | `retime_shortlist.py`: v5's list against the sweep's top 5 per shape, 5 rounds (`retime.csv`, `summary.txt`). |
| `12_compiler_behaviour/` | `check_compiler.py`: the five compiler behaviours the article relies on, as PASS/FAIL checks (`check_compiler.txt`). |
| `13_smaller/` | `cold_capture.py`: cold-start captures with GPU metrics (`reports/`). `nsys_latency.py`: the "Latency" field (`nsys_latency_round0.txt`). `smem_fp32_d128.py`: shared-memory allocation offsets (`smem_fp32_d128.txt`). `reverse_order.py` and `schedule_sim.py`: causal launch order. `stages_occupancy.py`: 3-stage kernels launched with padded shared memory, to separate occupancy from pipeline depth. `plot_cold_capture.py`: draws the article's Round 0 clock figure (`../2026_10_03_followup/images/nsys_clock_cold_warm.png`) from the two cold-capture exports. |
| `14_review_checks/` | Checks run while reviewing the edited article: `layout_and_copy.py` (`layout_and_copy.txt`) compiles the kernel for several dtypes, head dimensions and tiles and prints where Triton puts the warps (the query axis when the query tile is at least as tall as the head dimension, the key axis otherwise), and measures device-to-device copy bandwidth. |
| `common.py`, `source_tools.py` | Shared helpers: load a version, launch it with an explicit config, a float64 reference, and parsing of Nsight Compute's Source page (`ncu --page source --print-source cuda,sass --csv`). |

## Re-running

From this folder, with the repository's virtual environment:

```bash
PY=../../.venv/bin/python ./03_harness_repeats/run_harness_repeats.sh   # about 30 minutes
PY=../../.venv/bin/python ./run_experiments.sh                          # about 30 minutes, every other GPU experiment
PY=../../.venv/bin/python ./run_followups.sh                            # about 15 minutes
```

Run them one at a time: each one times or profiles the GPU. The article's final profile (`final_noncausal`, `final_causal` and `final_causal_64x32x4x4` in `../2026_10_03_followup/results/ncu/`) comes from the last three commands of `../2026_10_03_followup/profile_all.sh`. Analysis scripts that only read results (`*/analyze.py`, `08_barrier_stalls/barrier_attribution.py`, `07_baseline_duplication/hmma_by_line.py`) can run at any time.

Binary reports (`.ncu-rep`, `.nsys-rep`, `.sqlite`, the MLIR dump) are in each folder's `reports/`. That is about 380 MB, git-ignored and regenerated by the scripts above. Items 7 and 8 read the article's own Nsight Compute reports in `../2026_10_03_followup/results/ncu/`, which are also git-ignored and regenerated by `profile_all.sh`.

## Measurement notes

- Nsight Compute's raw CSV reports times in `ms` or `us` and frequencies in `Ghz`; check the units row.
- Nsight Systems stores "GPC Clock Frequency" in Hz in a signed 32-bit field, so clocks above 2.147 GHz come back negative. `kernel_clock_from_nsys.py` adds 2³² to recover them.
- `--clock-control base` locks this card at 1.98 GHz. The default, `boost`, locks it at 2.52 GHz.
- A barrier stall is reported at the first shared-memory or `MUFU` instruction after a `BAR.SYNC.DEFER_BLOCKING`, never on the `BAR.SYNC` itself.
- The harness prints milliseconds to 3 decimals, a 2% step at N = 512. `03_harness_repeats/analyze.py` recovers time from the TFLOP/s columns instead.
