# analyze.py
# Item 10: why does SDPA reach only about 33 TFLOP/s at N = 512 (B = 4, H = 8, D = 64), where the
# final kernel reaches about 48? With so little work, how many programs fit on the GPU at once and
# how many "waves" the grid needs matters: a grid of 1.14 waves leaves most SMs idle in the second.
#   python analyze.py reports/n512.ncu-rep > summary.txt
import csv
import io
import subprocess
import sys

NCU = "/opt/nvidia/nsight-compute/2026.2.1/ncu"
out = subprocess.run([NCU, "--import", sys.argv[1], "--page", "raw", "--csv"], capture_output=True, text=True).stdout
table = list(csv.reader(io.StringIO(out)))
header, units = table[0], table[1]
TIME = {"nsecond": 1e-6, "usecond": 1e-3, "msecond": 1.0, "second": 1e3, "ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}


def val(d, k):
    try:
        return float(d[k].replace(",", ""))
    except (KeyError, ValueError):
        return float("nan")


for row in table[2:]:
    d = dict(zip(header, row))
    u = dict(zip(header, units))
    name = d.get("Kernel Name", "?")
    label = next((v.split("<default domain>:", 1)[1].split(":none")[0] for k, v in d.items()
                  if "Push/Pop_Range" in k and "<default domain>:" in v), name)[:60]
    ms = val(d, "gpu__time_duration.sum") * TIME.get(u.get("gpu__time_duration.sum", "nsecond"), 1e-6)
    grid = d.get("Grid Size", "?")
    print(f"== {label}\n   kernel {name[:90]}")
    print(f"   duration {ms:.4f} ms (under Nsight Compute), grid {grid}, block {d.get('Block Size', '?')}")
    for k, title in [("launch__registers_per_thread", "registers/thread"),
                     ("launch__shared_mem_per_block_dynamic", "dynamic shared KB/block"),
                     ("launch__shared_mem_per_block_static", "static shared KB/block"),
                     ("launch__occupancy_limit_registers", "blocks/SM limit (registers)"),
                     ("launch__occupancy_limit_shared_mem", "blocks/SM limit (shared memory)"),
                     ("launch__occupancy_limit_warps", "blocks/SM limit (warps)"),
                     ("launch__waves_per_multiprocessor", "waves per SM"),
                     ("sm__maximum_warps_per_active_cycle_pct", "theoretical occupancy %"),
                     ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved occupancy %"),
                     ("sm__cycles_active.avg", "SM active cycles (avg)"),
                     ("sm__cycles_elapsed.avg", "elapsed cycles (avg)"),
                     ("sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed",
                      "tensor % of peak (elapsed)"),
                     ("sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum", "tensor ops")]:
        print(f"   {title:34s} {val(d, k):,.4g}")
    active = val(d, "sm__cycles_active.avg") / val(d, "sm__cycles_elapsed.avg")
    print(f"   {'SMs busy (active / elapsed cycles)':34s} {100 * active:.1f}%")
