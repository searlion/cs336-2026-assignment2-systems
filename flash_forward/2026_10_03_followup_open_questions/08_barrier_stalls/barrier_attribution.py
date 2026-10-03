# barrier_attribution.py
# Item 8: which barrier do the "barrier" stall samples belong to?
#
# BAR.SYNC.DEFER_BLOCKING lets a warp keep issuing after the barrier until its first instruction
# that goes through the MIO queue (shared-memory loads and stores, LDSM, MUFU); the warp-state
# sampler then reports the warp as stalled on "barrier" at *that* instruction, not at the BAR.SYNC
# (whose own barrier count is always 0). So every sample is attributed here to the nearest
# BAR.SYNC before it in address order, and each BAR.SYNC is named by the source line it came from.
#
#   python barrier_attribution.py REPORT SKIP:NAME [SKIP:NAME ...]
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from source_tools import export, opcode  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
report = sys.argv[1]
for spec in sys.argv[2:]:
    skip, name = spec.split(":", 1)
    insts = export(report, int(skip), csv_out=HERE / f"source_{name}.csv")
    insts.sort(key=lambda i: int(i["address"], 16))
    total = sum(i["Warp Stall Sampling (All Samples)"] for i in insts)
    bar_total = sum(i["stall_barrier"] for i in insts)
    hmma = sum(i["Instructions Executed"] for i in insts if opcode(i["sass"]).startswith("HMMA"))
    per_bar, current = {}, None
    for i in insts:
        if opcode(i["sass"]).startswith("BAR"):
            current = i["address"]
            per_bar[current] = dict(inst=i, samples=0.0)
        elif i["stall_barrier"] and current:
            per_bar[current]["samples"] += i["stall_barrier"]
    print(f"\n== {name}: {len(insts)} SASS instructions, {total:.0f} stall samples, barrier {bar_total:.0f} "
          f"({100 * bar_total / max(total, 1):.1f}% of samples); HMMA executed {hmma:.3g}")
    print(f"   {'address':>8s} {'executed':>10s} {'per HMMA':>8s} {'barrier samples':>15s} {'share':>6s}  source line")
    for addr, b in per_bar.items():
        i = b["inst"]
        where = f"{pathlib.Path(i['file']).name}:{i['line']}  {i['source'][:70]}" if i["file"] else "(no line info)"
        print(f"   {addr[-6:]:>8s} {i['Instructions Executed']:10.3g} {i['Instructions Executed'] / max(hmma, 1):8.4f} "
              f"{b['samples']:15.0f} {100 * b['samples'] / max(bar_total, 1):5.1f}%  {where}")
