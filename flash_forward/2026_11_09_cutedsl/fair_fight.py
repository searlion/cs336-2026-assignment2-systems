# fair_fight.py
# Was it a fair fight? NVML refreshes power, clocks, temperature and throttle reasons only every
# ~0.5 s on this card, longer than one harness do_bench call. So at one N (default 4096; both masks) each
# kernel runs the harness's own measurement (triton.testing.do_bench, median) back to back for
# about 4 s, while power_sampler reads NVML every 10 ms. Kernels are interleaved, with a cool-down
# before each window, over several rounds.
#
#   python fair_fight.py results/fair_fight.json [rounds] [cooldown_s] [N] [name=kernel_dir[@config] ...]
# With no kernel list it runs CuTe v5, Triton v5 and SDPA (the article's run). A config pins a CuTe
# version (its CONFIG tuple) or an autotuned Triton kernel (bq,bk,warps,stages); "sdpa" is SDPA.
# Added for round 6: the kernel list, the Triton autotuner's pick in the JSON, and energy per call.
import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import triton

from power_sampler import Sampler
from kload import load_module

ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
COOLDOWN = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
SECONDS = 4.0
N = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
B, H, D = 4, 8, 64

SPECS = sys.argv[5:] or ["cute_v5=kernels/v5_tuned", "triton_v5=../2026_10_03_followup/kernels/v5_final", "sdpa=sdpa"]
q, k, v = (torch.randn(B * H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
q4, k4, v4 = (t.view(B, H, N, D) for t in (q, k, v))
KERNELS, MODULES = {}, {}
for spec in SPECS:
    name, path = spec.split("=", 1)
    if path == "sdpa":
        KERNELS[name] = lambda c: F.scaled_dot_product_attention(q4, k4, v4, is_causal=c)
        continue
    path, _, cfg = path.partition("@")
    m = load_module(path)
    if cfg:
        t = tuple(int(x) for x in cfg.split(","))
        if hasattr(m, "flash_fwd_kernel"):
            m.flash_fwd_kernel.configs = [triton.Config({"Q_TILE_SIZE": t[0], "K_TILE_SIZE": t[1]},
                                                        num_warps=t[2], num_stages=t[3])]
        else:
            m.FlashAttentionCuTe.CONFIG = t
    MODULES[name] = m
    KERNELS[name] = (lambda fa: lambda c: fa.apply(q, k, v, c))(m.FlashAttentionTriton)
for fn in KERNELS.values():           # compile and autotune before anything is measured
    for c in (False, True):
        fn(c)
torch.cuda.synchronize()

sampler = Sampler().start()
out = {"device": sampler.info(), "N": N, "seconds_per_window": SECONDS, "cooldown_s": COOLDOWN,
       "kernels": SPECS, "windows": []}
# What each autotuned Triton kernel picked (its cache, keyed by shape and mask).
out["triton_picks"] = {name: {str(key): str(cfg) for key, cfg in m.flash_fwd_kernel.cache.items()}
                       for name, m in MODULES.items() if hasattr(getattr(m, "flash_fwd_kernel", None), "cache")}
print("Triton autotuner picks:", out["triton_picks"], flush=True)
for r in range(ROUNDS):
    for causal in (False, True):
        order = list(KERNELS) if r % 2 == 0 else list(reversed(KERNELS))   # alternate who goes first
        for name in order:
            time.sleep(COOLDOWN)
            fn = KERNELS[name]
            t0 = time.perf_counter()
            ms = []
            while time.perf_counter() - t0 < SECONDS:
                ms.append(triton.testing.do_bench(lambda: fn(causal), return_mode="median"))
            # The first reading in a window can still be the driver's value from the idle cool-down
            # (it refreshes every ~0.5 s), so samples from the first 0.6 s are left out.
            t1 = time.perf_counter()
            w = sampler.window(t0 + 0.6, t1)
            w["raw"] = [x for x in sampler.samples if t0 <= x[0] <= t1]
            w["temp_C_start"] = w["raw"][0][2]          # temperature as the window began
            w.update(kernel=name, causal=causal, round=r, do_bench_calls=len(ms), ms_median=statistics.median(ms))
            w["energy_mJ_per_call"] = round(w["power_W_median"] * w["ms_median"], 1)   # W x ms = mJ
            out["windows"].append(w)
            print(f"round {r} causal={causal!s:5} {name:12s} {w['ms_median']:.3f} ms  "
                  f"{w['power_W_median']:5.1f} W  {w['temp_C_start']}->{w['temp_C_max']} C  "
                  f"SM {w['sm_MHz_median']} MHz (min {w['sm_MHz_min']})  {w['reasons']}", flush=True)
sampler.stop()
json.dump(out, open(sys.argv[1], "w"), indent=1)

print(f"\nmedian over rounds (N = {N}):")
print(f"{'kernel':12s} {'causal':6s} {'ms':>7s} {'mJ/call':>8s} {'power W':>8s} {'max W':>6s} {'temp start':>10s} {'max temp':>8s} "
      f"{'SM MHz':>7s} {'min SM':>7s} {'mem MHz':>8s}  reasons (fraction of samples)")
for causal in (False, True):
    for name in KERNELS:
        ws = [w for w in out["windows"] if w["kernel"] == name and w["causal"] == causal]
        reasons = {}
        for w in ws:
            for kk, vv in w["reasons"].items():
                reasons[kk] = reasons.get(kk, 0) + vv / len(ws)
        print(f"{name:12s} {causal!s:6s} {statistics.median(w['ms_median'] for w in ws):7.3f} "
              f"{statistics.median(w['energy_mJ_per_call'] for w in ws):8.1f} "
              f"{statistics.median(w['power_W_median'] for w in ws):8.1f} {max(w['power_W_max'] for w in ws):6.1f} "
              f"{statistics.median(w['temp_C_start'] for w in ws):10.0f} {max(w['temp_C_max'] for w in ws):8d} "
              f"{statistics.median(w['sm_MHz_median'] for w in ws):7.0f} {min(w['sm_MHz_min'] for w in ws):7d} "
              f"{statistics.median(w['mem_MHz_median'] for w in ws):8.0f}  "
              + ", ".join(f"{kk} {vv:.2f}" for kk, vv in sorted(reasons.items())))
