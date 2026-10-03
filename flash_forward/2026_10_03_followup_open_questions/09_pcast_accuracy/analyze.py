# analyze.py
# Item 9: mean / max absolute error against float64 for the three P-cast variants, with bf16 and
# fp32 output buffers, next to the floor (rounding the exact output to bf16). Averages over the
# 3 seeds; all 32 heads.
#   python analyze.py > summary.txt
import collections
import json
import statistics

R = json.load(open("results.json"))
variants = [k for k in R[0] if k not in ("N", "causal", "seed", "floor_bf16_rounding_of_exact_output")]
groups = collections.defaultdict(list)
for r in R:
    groups[(r["N"], r["causal"])].append(r)
print("mean |error| (x1e-5), averaged over 3 seeds; [max |error| (x1e-3), worst seed]")
for (N, causal), rs in sorted(groups.items()):
    floor = statistics.mean(r["floor_bf16_rounding_of_exact_output"]["mean"] for r in rs)
    fmax = max(r["floor_bf16_rounding_of_exact_output"]["max"] for r in rs)
    print(f"N={N} causal={causal}: floor (bf16 rounding of the exact output) {floor * 1e5:.3f} [{fmax * 1e3:.3f}]")
    for v in variants:
        b = statistics.mean(r[v]["bf16_output"]["mean"] for r in rs)
        bm = max(r[v]["bf16_output"]["max"] for r in rs)
        f = statistics.mean(r[v]["fp32_output"]["mean"] for r in rs)
        fm = max(r[v]["fp32_output"]["max"] for r in rs)
        neq = statistics.mean(r[v]["frac_elements_not_equal_to_rounded_exact"] for r in rs)
        print(f"   {v:34s} bf16 out {b * 1e5:.3f} [{bm * 1e3:.3f}]   fp32 out {f * 1e5:.3f} [{fm * 1e3:.3f}]   "
              f"bf16 out != bf16(exact): {100 * neq:.1f}% of elements")
