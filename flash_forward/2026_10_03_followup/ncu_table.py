# ncu_table.py
# Print the handful of Nsight Compute counters this article uses, one column per profiled
# kernel, from one or more .ncu-rep files.
#
#   python ncu_table.py results/ncu/rounds_nc.ncu-rep
import csv
import io
import re
import subprocess
import sys

NCU = "/opt/nvidia/nsight-compute/2026.2.1/ncu"
SCALE = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
         "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1, "Ghz": 1e9, "Mhz": 1e6}


def results(path):
    out = subprocess.run([NCU, "--import", path, "--page", "raw", "--csv"], capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    header, units = rows[0], rows[1]
    for row in rows[2:]:
        d = {}
        for k, u, v in zip(header, units, row):
            try:
                d[k] = float(v.replace(",", "")) * SCALE.get(u.split("/")[0], 1)
            except ValueError:
                d[k] = v
        yield d


def label(d):
    for k, v in d.items():
        if "nvtx" in k.lower() and isinstance(v, str) and v:
            return v.split("/")[-1].strip("<>").split(",")[-1]
    return d.get("Kernel Name", "?")[:30]


STALLS = re.compile(r"smsp__average_warps_issue_stalled_(.*)_per_issue_active\.ratio$")
ROWS = [
    ("duration (ms)", lambda d: d["gpu__time_duration.sum"] * 1e3, "{:.3f}"),
    ("SM clock (GHz)", lambda d: d["sm__cycles_elapsed.avg.per_second"] / 1e9, "{:.2f}"),
    ("tensor ops (G)", lambda d: d["sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum"] / 1e9, "{:.1f}"),
    ("tensor TOP/s", lambda d: d["sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum.per_second"] / 1e12, "{:.1f}"),
    ("tensor % of peak", lambda d: d["sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed"], "{:.1f}"),
    ("LSU pipe (%)", lambda d: d["sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"], "{:.1f}"),
    ("issue slots busy (%)", lambda d: d["sm__inst_issued.avg.pct_of_peak_sustained_elapsed"], "{:.1f}"),
    ("instructions (M)", lambda d: d["smsp__inst_executed.sum"] / 1e6, "{:.0f}"),
    ("registers/thread", lambda d: d["launch__registers_per_thread"], "{:.0f}"),
    ("shared mem/block (KB)", lambda d: d["launch__shared_mem_per_block_dynamic"] / 1e3, "{:.1f}"),
    ("theoretical occupancy (%)", lambda d: d["sm__maximum_warps_per_active_cycle_pct"], "{:.1f}"),
    ("achieved occupancy (%)", lambda d: d["sm__warps_active.avg.pct_of_peak_sustained_active"], "{:.1f}"),
    ("L2->L1 bytes (GB)", lambda d: d["l1tex__m_xbar2l1tex_read_bytes.sum"] / 1e9, "{:.2f}"),
    ("L2 hit rate (%)", lambda d: d["lts__t_sector_hit_rate.pct"], "{:.1f}"),
    ("DRAM bytes (MB)", lambda d: (d["dram__bytes_read.sum"] + d["dram__bytes_write.sum"]) / 1e6, "{:.0f}"),
    ("shared st+ld wavefronts (M)", lambda d: d["l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"] / 1e6, "{:.0f}"),
    ("warp cycles / issued inst", lambda d: d["smsp__average_warp_latency_per_inst_issued.ratio"], "{:.2f}"),
]


def main():
    cols = [d for path in sys.argv[1:] for d in results(path)]
    names = [label(d) for d in cols]
    w = max(len(n) for n in names) + 2
    print(f"{'':30s}" + "".join(f"{n:>{w}s}" for n in names))
    for title, f, fmt in ROWS:
        vals = []
        for d in cols:
            try:
                vals.append(fmt.format(f(d)))
            except (KeyError, TypeError, ValueError):
                vals.append("n/a")
        print(f"{title:30s}" + "".join(f"{v:>{w}s}" for v in vals))
    print("top stalls (cycles per issued instruction):")
    for d, n in zip(cols, names):
        st = sorted(((m.group(1), v) for k, v in d.items() if (m := STALLS.match(k)) and isinstance(v, float)),
                    key=lambda x: -x[1])[:4]
        print(f"  {n:{w}s}" + ", ".join(f"{s} {v:.2f}" for s, v in st))


if __name__ == "__main__":
    main()
