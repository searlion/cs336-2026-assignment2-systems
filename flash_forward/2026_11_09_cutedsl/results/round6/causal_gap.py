# causal_gap.py: where do CuTe v5 (64 x 64, 2 stages) and Triton v5 pinned to 64 x 64 (3 stages)
# spend their executed instructions and warp-stall samples at N = 4096 causal? Reads the per-SASS
# "Instructions Executed" and stall-sample columns of Nsight Compute's source page, and groups
# consecutive instructions with the same execution count into regions (prologue/epilogue run once
# per warp; loop bodies run once per iteration).
#   python results/round6/causal_gap.py
import csv, io, subprocess, collections
NCU = "/opt/nvidia/nsight-compute/2026.2.1/ncu"
REPS = [("CuTe v5 causal", "results/round6/ncu_v5_v6.ncu-rep", "v5_tuned", 1),
        ("Triton 64x64x3 causal", "results/round6/triton_64x64x4x3_causal.ncu-rep", "flash_fwd", 0)]


def kernels(path):
    out = subprocess.run([NCU, "-i", path, "--page", "source", "--print-source", "sass", "--csv"],
                         capture_output=True, text=True).stdout
    blocks, cur = [], None
    for line in out.splitlines():
        if line.startswith('"Kernel Name"'):
            cur = [line.split(",")[1].strip('"'), []]
            blocks.append(cur)
        elif cur is not None:
            cur[1].append(line)
    for name, lines in blocks:
        rows = list(csv.reader(io.StringIO("\n".join(lines))))
        yield name, rows[0], rows[1:]


for label, path, key, which in REPS:
    ks = [k for k in kernels(path) if key in k[0]]
    name, hdr, rows = ks[which]
    ie, samp = hdr.index("Instructions Executed"), hdr.index("Warp Stall Sampling (All Samples)")
    stall_cols = [i for i, h in enumerate(hdr) if h.startswith("stall_") or "(stall" in h.lower()]
    data = [(r[1].strip(), int(r[ie] or 0), int(r[samp] or 0)) for r in rows if len(r) > samp]
    total_i = sum(d[1] for d in data)
    total_s = sum(d[2] for d in data)
    print(f"== {label}: {total_i / 1e6:.1f} M warp instructions executed, {total_s} stall samples")
    regions = []
    for src, n, s in data:
        if regions and regions[-1][0] == n:
            regions[-1][1] += 1; regions[-1][2] += s; regions[-1][3].append(src)
        else:
            regions.append([n, 1, s, [src]])
    by = collections.defaultdict(lambda: [0, 0, 0])      # exec count -> static instrs, executed, samples
    for n, cnt, s, srcs in regions:
        by[n][0] += cnt; by[n][1] += n * cnt; by[n][2] += s
    for n, (cnt, ex, s) in sorted(by.items(), key=lambda x: -x[1][1])[:6]:
        print(f"  executed {n:>8d} times/instr: {cnt:4d} static instrs, {ex / 1e6:6.1f} M executed "
              f"({100 * ex / total_i:4.1f}%), {100 * s / max(total_s, 1):4.1f}% of stall samples")
    hm = sum(n for src, n, s in data if src.startswith("HMMA"))
    print(f"  HMMA executed {hm / 1e6:.2f} M; non-HMMA instructions per HMMA: {(total_i - hm) / hm:.2f}")

# Opcode mix of the unmasked loop body (the instructions executed 258,048 times = once per iteration).
print("\nUnmasked loop body, opcodes per iteration (CuTe / Triton):")
mix = {}
for label, path, key, which in REPS:
    name, hdr, rows = [k for k in kernels(path) if key in k[0]][which]
    ie = hdr.index("Instructions Executed")
    c = collections.Counter()
    for r in rows:
        if len(r) > ie and r[ie] == "258048":
            c[r[1].strip().split()[0].split(".")[0] if not r[1].strip().startswith("@") else r[1].strip().split()[1].split(".")[0]] += 1
    mix[label] = c
a, b = mix.values()
for op in sorted(set(a) | set(b), key=lambda o: -(a[o] + b[o])):
    if a[op] != b[op]:
        print(f"  {op:10s} {a[op]:4d} {b[op]:4d}  ({b[op] - a[op]:+d})")
print(f"  total      {sum(a.values()):4d} {sum(b.values()):4d}")
