# pipe_table.py
# Item 2: one row per profiled kernel: the tensor-pipe utilisation counters next to the
# operation counter of the kernel's own data path (% of its peak), and the measured peak rates
# (operations per clock per SM) Nsight Compute uses for each path.
#   python pipe_table.py report.ncu-rep [more reports...]
import csv
import io
import subprocess
import sys

NCU = "/opt/nvidia/nsight-compute/2026.2.1/ncu"
PATHS = ["bf16_dst_fp32", "fp16_dst_fp32", "fp16_dst_fp16", "tf32_dst_fp32", "int8", "fp8"]


def rows(report):
    out = subprocess.run([NCU, "--import", report, "--page", "raw", "--csv"], capture_output=True, text=True).stdout
    table = list(csv.reader(io.StringIO(out)))
    header, units = table[0], table[1]
    scale = {"nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1.0, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0,
             "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "cycle/nsecond": 1e9, "cycle/usecond": 1e6,
             "cycle/msecond": 1e3, "cycle/second": 1.0}
    for r in table[2:]:
        d = {}
        for k, u, v in zip(header, units, r):
            try:
                # times in seconds, frequencies in Hz; everything else as printed
                d[k] = float(v.replace(",", "")) * scale.get(u.lower(), 1.0)
            except ValueError:
                d[k] = v
        yield d


def label(d):
    # The NVTX push/pop range is in a column named "thread Domain:Push/Pop_Range:...", with a value
    # like '183740  "<default domain>:64x32x4x3 causal=False:none:none:...'.
    for k, v in d.items():
        if "Push/Pop_Range" in k and isinstance(v, str) and "<default domain>:" in v:
            return v.split("<default domain>:", 1)[1].split(":none")[0]
    return str(d.get("Kernel Name", "?"))[:26]


def get(d, name):
    v = d.get(name)
    return v if isinstance(v, float) else None


print(f"{'kernel':26s} {'path (ops>0)':14s} {'ops % peak':>10s} {'pipe %':>7s} {'pipe_v2 %':>9s} {'hmma pipe %':>11s} "
      f"{'imma pipe %':>11s} {'hmma inst %':>11s} {'pipe/ops':>8s} {'peak ops/clk/SM':>15s} {'clock GHz':>9s} {'TOP/s':>7s}")
for report in sys.argv[1:]:
    for d in rows(report):
        clk = get(d, "sm__cycles_elapsed.avg.per_second")
        for p in PATHS:
            base = f"sm__ops_path_tensor_src_{p}_sparsity_off.sum"
            ops = get(d, base)
            if not ops:
                continue
            pct = get(d, base + ".pct_of_peak_sustained_elapsed")
            peak = get(d, base + ".peak_sustained")
            pipe = get(d, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")
            pipe2 = get(d, "sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_elapsed")
            hpipe = get(d, "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed")
            ipipe = get(d, "sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_elapsed")
            hinst = get(d, "sm__inst_executed_pipe_tensor_op_hmma_v2.avg.pct_of_peak_sustained_elapsed")
            dur = get(d, "gpu__time_duration.sum")
            f = lambda x, s="{:.1f}": "n/a" if x is None else s.format(x)  # noqa: E731
            print(f"{label(d):26s} {p:14s} {f(pct):>10s} {f(pipe):>7s} {f(pipe2):>9s} {f(hpipe):>11s} {f(ipipe):>11s} "
                  f"{f(hinst):>11s} {f(pipe / pct if pipe and pct else None, '{:.2f}'):>8s} "
                  f"{f(peak / 56 if peak else None, '{:.0f}'):>15s} {f(clk / 1e9 if clk else None, '{:.2f}'):>9s} "
                  f"{f(ops / dur * 1e-12 if dur else None):>7s}")
