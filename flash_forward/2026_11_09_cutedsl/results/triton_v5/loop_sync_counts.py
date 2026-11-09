# loop_sync_counts.py: per-loop counts of the synchronisation instructions in Triton v5's SASS
# (nvdisasm output with labels), to compare with the CuTe kernels' sass_counts.py rows.
#   python results/triton_v5/loop_sync_counts.py results/triton_v5/v5_64x128x4x2_noncausal.sass
import collections
import re
import sys

lines = open(sys.argv[1]).read().splitlines()
ins, labels, pending = [], {}, []
for l in lines:
    m = re.match(r"\.(L_x_\d+):", l.strip())
    if m:
        pending.append(m.group(1))
        continue
    m = re.search(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.]+)([^;]*);", l)
    if m and (m.group(2) or "").strip() != "@!PT":
        a = int(m.group(1), 16)
        for p in pending:
            labels[p] = a
        pending = []
        ins.append((a, m.group(3), m.group(4)))
loops = []
for a, op, rest in ins:
    m = re.search(r"\.(L_x_\d+)", rest)
    if op.startswith("BRA") and m and labels.get(m.group(1), 1 << 30) < a:
        loops.append((labels[m.group(1)], a))
OPS = ["HMMA", "LDSM", "LDGSTS", "LDGDEPBAR", "DEPBAR", "BAR.SYNC", "MUFU.EX2", "LDL", "STL"]
print(f"{sys.argv[1]}")
print(f"{'':26s}" + "".join(f"{o:>10s}" for o in OPS) + f"{'total':>8s}")
for lo, hi in loops:
    seq = [op for a, op, _ in ins if lo <= a <= hi]
    if not any(o.startswith("HMMA") for o in seq):
        continue
    c = collections.Counter()
    for op in seq:
        for o in OPS:
            if op == o or op.startswith(o + ".") or (o == "DEPBAR" and op.startswith("DEPBAR")):
                c[o] += 1
                break
    print(f"loop {lo:#07x}-{hi:#07x}    " + "".join(f"{c[o]:>10d}" for o in OPS) + f"{len(seq):>8d}")
print("DEPBAR operands in loops:", sorted({r.strip() for a, op, r in ins if op.startswith("DEPBAR")
                                          and any(lo <= a <= hi for lo, hi in loops)}))
