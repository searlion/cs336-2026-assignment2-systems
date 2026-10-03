# per_line_diff.py
# Item 5: where do the instructions go when the key tile grows from 32 to 128 (same kernel, same
# mask)? Executed instructions per source line, from the Source-page exports analyze.py saved
# (reports/source_<k>.csv, k = position of the launch in the report).
#   python per_line_diff.py 0 4 > per_line_diff_noncausal.txt   (64x32x4x3 vs 64x128x4x2, non-causal)
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from source_tools import export  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
report = HERE / "reports" / "v4_key_tiles.ncu-rep"
a, b = (int(x) for x in sys.argv[1:3])


def per_line(skip):
    out = collections.Counter()
    src = {}
    for i in export(report, skip):
        key = (pathlib.Path(i["file"]).name, i["line"])
        out[key] += i["Instructions Executed"]
        src[key] = i["source"]
    return out, src


ca, sa = per_line(a)
cb, sb = per_line(b)
ta, tb = sum(ca.values()), sum(cb.values())
print(f"total executed (warp-level) instructions: {ta / 1e6:.1f} M -> {tb / 1e6:.1f} M ({100 * (tb / ta - 1):+.1f}%)")
print(f"{'file:line':38s} {'A (M)':>8s} {'B (M)':>8s} {'B - A (M)':>10s} {'share of drop':>13s}  source")
for key in sorted(set(ca) | set(cb), key=lambda k: cb[k] - ca[k]):
    d = cb[key] - ca[key]
    if abs(d) < 0.005 * ta:
        continue
    print(f"{key[0] + ':' + str(key[1]):38s} {ca[key] / 1e6:8.1f} {cb[key] / 1e6:8.1f} {d / 1e6:10.1f} "
          f"{100 * d / (tb - ta):12.1f}%  {(sa.get(key) or sb.get(key) or '')[:60]}")
