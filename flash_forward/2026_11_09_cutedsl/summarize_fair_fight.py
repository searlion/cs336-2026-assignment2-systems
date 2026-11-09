# summarize_fair_fight.py
# One compact table from results/fair_fight_N{512,1024,2048}.json and results/fair_fight.json
# (N = 4096): medians over the 3 rounds of each kernel's time, SM clock and power, the share of
# samples at the power cap, the highest temperature, and any thermal or hardware slowdown.
#
#   python summarize_fair_fight.py > results/fair_fight_summary.md
import json
import pathlib
import statistics

KERNELS = ("cute_v5", "triton_v5", "sdpa")
FILES = {512: "results/fair_fight_N512.json", 1024: "results/fair_fight_N1024.json",
         2048: "results/fair_fight_N2048.json", 4096: "results/fair_fight.json"}
SLOW = ("sw_thermal", "hw_thermal", "hw_slowdown", "hw_power_brake")

print("| N, mask | time, ms (CuTe v5 / Triton v5 / SDPA) | median SM clock, MHz | median power, W | "
      "share at power cap | max °C | CuTe vs Triton |")
print("|---|---|---|---|---|---|---|")
for N, path in FILES.items():
    if not pathlib.Path(path).exists():
        continue
    d = json.load(open(path))
    for causal in (False, True):
        ws = {k: [w for w in d["windows"] if w["kernel"] == k and w["causal"] == causal] for k in KERNELS}
        med = lambda k, f: statistics.median(w[f] for w in ws[k])  # noqa: E731
        t = {k: med(k, "ms_median") for k in KERNELS}
        clk = {k: med(k, "sm_MHz_median") for k in KERNELS}
        cap = {k: statistics.mean(w["reasons"].get("sw_power_cap", 0) for w in ws[k]) for k in KERNELS}
        slow = [k for k in KERNELS for w in ws[k] if any(r in w["reasons"] for r in SLOW)]
        # Matched: CuTe and Triton clocks within 2% and both at, or both away from, the power cap.
        matched = abs(clk["cute_v5"] / clk["triton_v5"] - 1) <= 0.02 and abs(cap["cute_v5"] - cap["triton_v5"]) <= 0.25
        gap = 100 * (t["triton_v5"] / t["cute_v5"] - 1)
        verdict = (f"{gap:+.0f}%" if matched else f"{gap:+.0f}% (clocks not matched)") + (" THERMAL" if slow else "")
        print(f"| {N}, {'causal' if causal else 'non-causal'} | "
              + " / ".join(f"{t[k]:.3f}" for k in KERNELS) + " | "
              + " / ".join(f"{clk[k]:.0f}" for k in KERNELS) + " | "
              + " / ".join(f"{med(k, 'power_W_median'):.0f}" for k in KERNELS) + " | "
              + " / ".join(f"{100 * cap[k]:.0f}%" for k in KERNELS) + " | "
              + f"{max(w['temp_C_max'] for k in KERNELS for w in ws[k])} | {verdict} |")
print("\n(CuTe vs Triton: how much longer Triton v5 took, + means CuTe v5 is faster.)")
