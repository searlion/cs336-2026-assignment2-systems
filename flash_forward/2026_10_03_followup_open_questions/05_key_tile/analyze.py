# analyze.py
# Items 5 and 8: summarise the key-tile experiment. (ncu raw units: ms and KB per block.)
#   1. Timing (timing.csv): median over 10 rounds per kernel, config and mask, and what removing
#      the O_acc rescale saves at each tile shape.
#   2. Profiles (reports/v4_key_tiles.ncu-rep, one launch per config and mask): instructions,
#      the rescale line's share of them, barriers per iteration and the barrier share of the stall
#      samples (attributed to the BAR.SYNC before them, as in 08_barrier_stalls), stalls per
#      instruction, and tensor-core % of peak.
#   python analyze.py > summary.txt
import collections
import csv
import io
import pathlib
import statistics
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from source_tools import NCU, export, opcode  # noqa: E402

# ---- 1. timing
rows = list(csv.DictReader(open(HERE / "timing.csv")))
ms = collections.defaultdict(list)
for r in rows:
    ms[(r["kernel"], (int(r["q_tile"]), int(r["k_tile"]), int(r["num_warps"]), int(r["num_stages"])),
        r["causal"] == "True")].append(float(r["ms"]))
print("== timing, N = 4096: median ms over 10 rounds [min-max]; TFLOP/s; what removing the rescale saves")
configs = sorted({k[1] for k in ms}, key=lambda c: (c[1], c[3]))
for causal in (False, True):
    flops = 4 * 32 * 4096 * 4096 * 64 / (2 if causal else 1)
    best = min(statistics.median(ms[("v4", c, causal)]) for c in configs)
    print(f"causal={causal}")
    for c in configs:
        a, b = ms[("v4", c, causal)], ms[("v4_no_rescale", c, causal)]
        ma, mb = statistics.median(a), statistics.median(b)
        print(f"   {'x'.join(map(str, c)):10s} v4 {ma:.4f} [{min(a):.4f}-{max(a):.4f}] {flops / ma * 1e-9:5.1f} TF/s "
              f"({100 * best / ma:5.1f}% of best)   no rescale {mb:.4f} ({100 * (mb / ma - 1):+5.1f}%)")

# ---- 2. profiles
report = HERE / "reports" / "v4_key_tiles.ncu-rep"
out = subprocess.run([NCU, "--import", str(report), "--page", "raw", "--csv"], capture_output=True, text=True).stdout
table = list(csv.reader(io.StringIO(out)))
header = table[0]
raw = [dict(zip(header, r)) for r in table[2:]]


def f(d, k):
    try:
        return float(d[k].replace(",", ""))
    except (KeyError, ValueError):
        return float("nan")


def label(d):
    # The NVTX push/pop range is in a column named "thread Domain:Push/Pop_Range:...", with a value
    # like '183740  "<default domain>:64x32x4x3 causal=False:none:none:...'.
    for k, v in d.items():
        if "Push/Pop_Range" in k and isinstance(v, str) and "<default domain>:" in v:
            return v.split("<default domain>:", 1)[1].split(":none")[0]
    return str(d.get("Kernel Name", "?"))[:26]


print("\n== profiles (one launch each, N = 4096, Nsight Compute at its locked clock)")
cols = ["config", "ms", "tensor%", "instr M", "rescale%", "BAR/iter", "barrier%samp", "barrier/inst",
        "math/inst", "issue%", "regs", "smem KB"]
print(" ".join(f"{c:>12s}" for c in cols))
for skip, d in enumerate(raw):
    insts = export(report, skip, csv_out=HERE / "reports" / f"source_{skip}.csv")
    total_exec = sum(i["Instructions Executed"] for i in insts)
    rescale = sum(i["Instructions Executed"] for i in insts if i["source"].startswith("O_acc = alpha * O_acc"))
    hmma = sum(i["Instructions Executed"] for i in insts if opcode(i["sass"]).startswith("HMMA"))
    bars = [i for i in insts if opcode(i["sass"]).startswith("BAR")]
    loop_bar_exec = sum(i["Instructions Executed"] for i in bars if i["Instructions Executed"] > 0.01 * max(
        b["Instructions Executed"] for b in bars))
    # HMMA per warp per iteration: Q_TILE/num_warps/16 row blocks x (K_TILE/8 x D/16 + D/8 x K_TILE/16)
    q, k, nw = (int(x) for x in label(d).split()[0].split("x")[:3])
    hmma_per_iter = (q // nw // 16) * (k // 8 * 64 // 16 + 64 // 8 * k // 16)
    iters = hmma / hmma_per_iter
    samples = sum(i["Warp Stall Sampling (All Samples)"] for i in insts)
    barrier = sum(i["stall_barrier"] for i in insts)
    vals = [label(d), f"{f(d, 'gpu__time_duration.sum'):.3f}",
            f"{f(d, 'sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed'):.1f}",
            f"{total_exec / 1e6:.0f}", f"{100 * rescale / total_exec:.1f}", f"{loop_bar_exec / iters:.2f}",
            f"{100 * barrier / samples:.1f}",
            f"{f(d, 'smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio'):.2f}",
            f"{f(d, 'smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio'):.2f}",
            f"{f(d, 'sm__inst_issued.avg.pct_of_peak_sustained_elapsed'):.1f}",
            f"{f(d, 'launch__registers_per_thread'):.0f}", f"{f(d, 'launch__shared_mem_per_block_dynamic'):.1f}"]
    print(" ".join(f"{v:>12s}" for v in vals))
print("\nunits: ms under Nsight Compute; tensor% = bf16->fp32 tensor ops, % of peak; rescale% = share of executed "
      "instructions on the `O_acc = alpha * O_acc` line; BAR/iter = loop BAR.SYNCs per warp per key tile; "
      "barrier%samp = barrier share of all warp-stall samples; */inst = stall cycles per issued instruction")
