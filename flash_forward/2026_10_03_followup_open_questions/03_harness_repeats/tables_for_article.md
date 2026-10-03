### Round 3 table (v1 against v2)

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

### Results table (every version)

| N | causal | v0 baseline | v1 tiles | v2 autotune | v3 causal skip | v4 exp2 | v5 final | SDPA |
|---|---|---|---|---|---|---|---|---|
| 512 | no | 0.114 | 0.049 | 0.048 | 0.049 | 0.048 | 0.045 | 0.065 |
| 1024 | no | 0.386 | 0.149 | 0.150 | 0.147 | 0.144 | 0.144 | 0.151 |
| 2048 | no | 1.537 | 0.534 | 0.537 | 0.523 | 0.514 | 0.507 | 0.538 |
| 4096 | no | 6.393 (21.5, 32%) | 2.121 (64.8, 95%) | 2.129 (64.5, 95%) | 2.057 (66.8, 97%) | 2.027 (67.8, 99%) | 1.954 (70.3, 103%) | 2.009 (68.4) |
| 512 | yes | 0.113 | 0.050 | 0.046 | 0.038 | 0.040 | 0.037 | 0.055 |
| 1024 | yes | 0.383 | 0.153 | 0.153 | 0.095 | 0.091 | 0.091 | 0.120 |
| 2048 | yes | 1.534 | 0.545 | 0.545 | 0.303 | 0.295 | 0.294 | 0.330 |
| 4096 | yes | 6.363 (10.8, 18%) | 2.161 (31.8, 52%) | 2.154 (31.9, 52%) | 1.100 (62.4, 102%) | 1.082 (63.5, 103%) | 1.076 (63.8, 104%) | 1.117 (61.5) |

### What each round was worth, N = 4096 (ratio of 10-run medians; runs in which the newer version was faster)

- v0_baseline -> v1_tiles: non-causal 3.01x (10/10 runs faster), causal 2.94x (10/10 runs faster)
- v1_tiles -> v2_autotune: non-causal -0.4% (4/10 runs faster), causal +0.3% (5/10 runs faster)
- v1_tiles -> v3_causal_skip: non-causal +3.1% (10/10 runs faster), causal 1.96x (10/10 runs faster)
- v3_causal_skip -> v4_exp2: non-causal +1.5% (7/10 runs faster), causal +1.7% (10/10 runs faster)
- v4_exp2 -> v5_final: non-causal +3.8% (10/10 runs faster), causal +0.6% (9/10 runs faster)

### Final kernel against the baseline (Round 5 table)

| N | causal | baseline (ms) | final (ms) | final TFLOP/s | SDPA TFLOP/s | final % of SDPA |
|---|---|---|---|---|---|---|
| 512 | no | 0.114 | 0.045 | 47.9 | 32.9 | 141% |
| 1024 | no | 0.386 | 0.144 | 59.8 | 56.7 | 105% |
| 2048 | no | 1.537 | 0.507 | 67.8 | 63.9 | 106% |
| 4096 | no | 6.393 | 1.954 | 70.3 | 68.4 | 103% |
| 512 | yes | 0.113 | 0.037 | 29.1 | 19.4 | 149% |
| 1024 | yes | 0.383 | 0.091 | 47.1 | 35.8 | 132% |
| 2048 | yes | 1.534 | 0.294 | 58.5 | 52.1 | 112% |
| 4096 | yes | 6.363 | 1.076 | 63.8 | 61.5 | 104% |
