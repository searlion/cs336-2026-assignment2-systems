# bench_with_power.py
# Runs the unchanged harness's bench() (bench_flashattention.py, same shapes, same order, same
# printout) while power_sampler.Sampler reads NVML every 10 ms, and saves, for every shape and
# for each of the two do_bench calls in it (the kernel under test, then SDPA), the power,
# temperature, clocks and clock-event reasons seen during that call.
#
#   PYTHONPATH=kernels/v5_tuned python bench_with_power.py results/final/power/v5_tuned_run1.json
import json
import sys

from power_sampler import Sampler, record_do_bench

windows = []
record_do_bench(windows)                 # before the harness imports triton.testing.do_bench
import bench_flashattention as harness   # noqa: E402  (unchanged file)

sampler = Sampler().start()
out = {"device": sampler.info(), "shapes": []}
print(f"{'N':>6} {'causal':>7} {'ours ms':>9} {'sdpa ms':>9} {'ours TF/s':>10} {'sdpa TF/s':>10} {'% sdpa':>7}")
for N in (512, 1024, 2048, 4096):
    for causal in (False, True):
        windows.clear()
        a, b, c, d = harness.bench(B=4, H=8, N=N, D=64, is_causal=causal)
        print(f"{N:>6} {str(causal):>7} {a:>9.3f} {b:>9.3f} {c:>10.1f} {d:>10.1f} {100 * b / a:>6.0f}%", flush=True)
        (o0, o1), (s0, s1) = windows
        out["shapes"].append({"N": N, "causal": causal, "ours_ms": a, "sdpa_ms": b,
                              "ours": sampler.window(o0, o1), "sdpa": sampler.window(s0, s1)})
sampler.stop()
json.dump(out, open(sys.argv[1], "w"), indent=1)
