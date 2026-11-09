# sass_timeline.py: one character per SASS instruction of a loop, to see how tensor-core work
# (H = HMMA) is spread among the softmax (E = MUFU.EX2, s = SHFL, x = FMNMX, f = FFMA/FMUL/FADD,
# c = F2FP), shared loads (L = LDSM), copies (G = LDGSTS), barriers (B = BAR, D = DEPBAR) and the rest (.)
#   python sass_timeline.py results/cubin/v7_swpipe/D64_noncausal_64x64x4.sass [loop index]
import re
import sys

sass = open(sys.argv[1]).read()
LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.]+)([^;]*);")
ins = [(int(m.group(1), 16), m.group(3), m.group(4)) for m in map(LINE.search, sass.splitlines())
       if m and (m.group(2) or "").strip() != "@!PT"]
loops = [(int(re.search(r"0x([0-9a-f]+)", r).group(1), 16), a) for a, op, r in ins
         if op.startswith("BRA") and re.search(r"0x([0-9a-f]+)", r) and int(re.search(r"0x([0-9a-f]+)", r).group(1), 16) < a]
loops = [l for l in loops if sum(1 for a, op, _ in ins if l[0] <= a <= l[1] and op.startswith("HMMA"))]
CODE = [("HMMA", "H"), ("MUFU.EX2", "E"), ("SHFL", "s"), ("FMNMX", "x"), ("FFMA", "f"), ("FMUL", "f"),
        ("FADD", "f"), ("F2FP", "c"), ("LDSM", "L"), ("LDGSTS", "G"), ("BAR", "B"), ("DEPBAR", "D"), ("LDGDEPBAR", "D")]
for li in ([int(sys.argv[2])] if len(sys.argv) > 2 else range(len(loops))):
    lo, hi = loops[li]
    s = "".join(next((c for p, c in CODE if op.startswith(p)), ".") for a, op, _ in ins if lo <= a <= hi)
    print(f"loop {li} ({lo:#x}-{hi:#x}), {len(s)} instructions:")
    for i in range(0, len(s), 100):
        print("  " + s[i:i + 100])
