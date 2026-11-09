# summarize_round6.py
# The numbers the article's Round 6 uses, from the files they come from:
#   results/round6/fair_fight_round6.json   sustained, interleaved run at N = 4096 (fair_fight.py)
#   results/ncu/rounds_noncausal.txt        CuTe v5 under Nsight Compute (clock fixed at 2.52 GHz)
#   results/round6/wave_probe_ncu.txt       the same at B*H = 21 (whole waves) and 32
#   results/round6/wave_probe_time.txt      wall-clock throughput at B*H = 21 and 32
#
#   python summarize_round6.py > results/round6/summary.txt
import json
import re
import statistics

ff = json.load(open("results/round6/fair_fight_round6.json"))
names = list(dict.fromkeys(w["kernel"] for w in ff["windows"]))


def med(k, c, f):
    return statistics.median(w[f] for w in ff["windows"] if w["kernel"] == k and w["causal"] == c)


def cap(k, c):
    ws = [w for w in ff["windows"] if w["kernel"] == k and w["causal"] == c]
    return statistics.mean(w["reasons"].get("sw_power_cap", 0) for w in ws)


SLOW = ("sw_thermal", "hw_thermal", "hw_slowdown", "hw_power_brake")
print(f"# Fair fight, N = {ff['N']}, B*H = 32, D = 64: {len(ff['windows']) // (2 * len(names))} rounds, "
      f"{ff['cooldown_s']:.0f} s cool-down, {ff['seconds_per_window']:.0f} s per window (medians over rounds)")
print("Triton autotuner picks:", ff.get("triton_picks"))
print("| kernel | mask | ms | vs CuTe v5 | median SM MHz | vs CuTe v5 clock | median W | mJ per call | "
      "share at power cap | max °C | thermal/HW slowdown |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for c in (False, True):
    t0, f0 = med("cute_v5", c, "ms_median"), med("cute_v5", c, "sm_MHz_median")
    for k in names:
        t, f = med(k, c, "ms_median"), med(k, c, "sm_MHz_median")
        slow = sorted({r for w in ff["windows"] if w["kernel"] == k and w["causal"] == c
                       for r in w["reasons"] if r in SLOW})
        tmax = max(w["temp_C_max"] for w in ff["windows"] if w["kernel"] == k and w["causal"] == c)
        print(f"| {k} | {'causal' if c else 'non-causal'} | {t:.3f} | {100 * (t / t0 - 1):+.1f}% | {f:.0f} | "
              f"{100 * (f / f0 - 1):+.1f}% | {med(k, c, 'power_W_median'):.0f} | "
              f"{med(k, c, 'energy_mJ_per_call'):.1f} | {100 * cap(k, c):.0f}% | {tmax} | {', '.join(slow) or 'none'} |")
print("(vs CuTe v5: + means slower than CuTe v5. Clocks are matched when within 1% and the power-cap shares agree.)")

# Headroom. Peak rate per SM from CuTe v5's Nsight Compute capture: tensor TOP/s / (% of peak).
nc = open("results/ncu/rounds_noncausal.txt").read()
row = lambda title: re.search(rf"^{re.escape(title)}\s+(.*)$", nc, re.M).group(1).split()  # noqa: E731
top_s, pct, ghz = float(row("tensor TOP/s")[5]), float(row("tensor % of peak")[5]), float(row("SM clock (GHz)")[5])
per_clk_sm = top_s * 1e12 / (pct / 100) / (ghz * 1e9) / 56
print(f"\n# Headroom\npeak bf16 tensor rate: {top_s} TOP/s at {pct}% of peak and {ghz} GHz -> "
      f"{per_clk_sm:.0f} FLOP per clock per SM (56 SMs), consistent with the hardware's 512 (dense bf16 MMA, "
      f"fp32 accumulate, Ada); the floors use 512")
per_clk_sm = 512
for c, ops in ((False, 137.4e9), (True, 69.8e9)):
    f = med("cute_v5", c, "sm_MHz_median") * 1e6
    floor = ops / (per_clk_sm * 56 * f) * 1e3
    t = med("cute_v5", c, "ms_median")
    print(f"{'causal' if c else 'non-causal'}: {ops / 1e9:.1f} G tensor ops at CuTe v5's median {f / 1e6:.0f} MHz -> "
          f"floor {floor:.3f} ms; measured {t:.3f} ms = {100 * (t / floor - 1):.1f}% above the floor")
wp = open("results/round6/wave_probe_ncu.txt").read()
pcts = re.search(r"^tensor % of peak\s+(.*)$", wp, re.M).group(1).split()
print(f"Nsight Compute, CuTe v5 non-causal: {pcts[0]}% of tensor peak at B*H = 21 (8 whole waves), "
      f"{pcts[1]}% at B*H = 32 (12.19 waves); causal {pcts[2]}% / {pcts[3]}%")
print("Wall clock (wave_probe_time.txt):")
print(open("results/round6/wave_probe_time.txt").read().rstrip())
