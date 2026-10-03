# hmma_by_line.py
# Item 7: which multiply does the baseline duplicate? Both products of one loop iteration have the
# same FLOPs (2 x 16 x 16 x 64), so without duplication each would execute the same number of
# tensor-core instructions. Counts the HMMA instructions (static, and executed) on each source line.
#   python hmma_by_line.py REPORT SKIP:NAME [SKIP:NAME ...]
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from source_tools import export, opcode  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
report = sys.argv[1]
for spec in sys.argv[2:]:
    skip, name = spec.split(":", 1)
    insts = export(report, int(skip), csv_out=HERE / f"source_{name}.csv")
    by_line = collections.defaultdict(lambda: [0, 0.0, set(), ""])
    for i in insts:
        op = opcode(i["sass"])
        if op.startswith("HMMA"):
            e = by_line[(pathlib.Path(i["file"]).name, i["line"])]
            e[0] += 1
            e[1] += i["Instructions Executed"]
            e[2].add(op)
            e[3] = i["source"]
    total = sum(e[1] for e in by_line.values())
    print(f"\n== {name}: HMMA instructions by source line")
    print(f"{'line':>40s} {'static':>6s} {'executed':>12s} {'share':>6s} {'per warp-iteration':>18s}  opcode / source")
    loop_execs = max(i["Instructions Executed"] for i in insts if opcode(i["sass"]).startswith("HMMA"))
    for (f, ln), (n, ex, ops, src) in sorted(by_line.items(), key=lambda kv: kv[0][1]):
        print(f"{f + ':' + str(ln):>40s} {n:6d} {ex:12.4g} {100 * ex / total:5.1f}% {ex / loop_execs:18.1f}  "
              f"{','.join(sorted(ops))}  {src[:60]}")
    # Each m16n8k16 HMMA is 2*16*8*16 = 4096 FLOPs.
    print(f"total executed {total:.4g} HMMA x 4096 = {total * 4096 / 1e9:.1f} G tensor operations")
