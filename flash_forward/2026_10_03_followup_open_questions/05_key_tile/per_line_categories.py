# per_line_categories.py
# Item 5: the instruction drop from 64x32x4x3 to 64x128x4x2 (non-causal, same kernel and mask),
# grouped by what each source line does: per-iteration work shrinks, per-score work does not.
#   python per_line_categories.py > per_line_categories.txt
import collections
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from source_tools import export  # noqa: E402

K = "flashattention_autograd_function_triton.py"
CATEGORIES = {  # v4_exp2 line numbers; standard.py lines are tl.max / tl.sum internals
    "rescale O_acc (line 49)": [(K, 49)],
    "K/V loads: async copies and their barriers (27, 28)": [(K, 27), (K, 28)],
    "loop bookkeeping (22)": [(K, 22)],
    "row max, sum, alpha, m, l (42, 45, 46, tl.max/tl.sum)": [(K, 42), (K, 45), (K, 46), ("standard.py", 191),
                                                             ("standard.py", 293), ("standard.py", 170), ("standard.py", 263)],
    "Q re-read from shared memory (109)": [(K, 109)],
    "Q K^T dot (30)": [(K, 30)],
    "exp2 (44), per score": [(K, 44)],
    "P cast (48), per score": [(K, 48)],
    "P V dot (50)": [(K, 50)],
}


def per_line(skip):
    c = collections.Counter()
    for i in export(HERE / "reports" / "v4_key_tiles.ncu-rep", skip):
        c[(pathlib.Path(i["file"]).name, i["line"])] += i["Instructions Executed"]
    return c


a, b = per_line(0), per_line(4)  # 64x32x4x3 and 64x128x4x2, non-causal
drop = sum(b.values()) - sum(a.values())
seen = set()
print(f"executed instructions: {sum(a.values()) / 1e6:.1f} M (64x32x4x3) -> {sum(b.values()) / 1e6:.1f} M (64x128x4x2)")
for name, keys in CATEGORIES.items():
    d = sum(b[k] - a[k] for k in keys)
    seen |= set(keys)
    print(f"{name:56s} {d / 1e6:7.1f} M  {100 * d / drop:5.1f}% of the drop")
rest = sum(b[k] - a[k] for k in set(a) | set(b) if k not in seen)
print(f"{'everything else':56s} {rest / 1e6:7.1f} M  {100 * rest / drop:5.1f}%")
