# plot_results_svg.py
# images/results_N4096.svg: TFLOP/s of every version at N = 4096 (median of the harness runs in
# results/final/summary.csv), non-causal and causal side by side, with the Triton kernel, NVIDIA's
# example and SDPA as reference rows. Plain SVG, no external fonts.
import csv
import glob
import pathlib
import re

FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COLOR = {"cute": "#2a78d6", "triton": "#eb6834", "nvidia": "#1baf7a", "sdpa": "#eda100"}
ROWS = [("v0_naive", "v0 naive port", "cute"), ("v1_swizzle", "v1 swizzle", "cute"),
        ("v2_ldmatrix", "v2 ldmatrix", "cute"), ("v3_p_regs", "v3 P in registers", "cute"),
        ("v4_cpasync", "v4 cp.async", "cute"), ("v5_tuned", "v5 tuned (final)", "cute"),
        ("triton_v5", "Triton v5 (last article)", "triton"), ("nvidia_example", "NVIDIA CuTe example", "nvidia"),
        ("sdpa", "PyTorch SDPA", "sdpa")]

data = {}
for r in csv.DictReader(open("results/final/summary.csv")):
    if int(r["N"]) == 4096:
        data[(r["version"], r["causal"] == "1")] = float(r["tflops"])

# min-max over the individual runs, for the whiskers
ROW = re.compile(r"^\s*4096\s+(True|False)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)")
NV = re.compile(r"^\s*4096\s+(True|False)\s+64x64\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)")
runs = {}
for path in glob.glob("results/final/*_run*.txt"):
    version = pathlib.Path(path).stem.rsplit("_run", 1)[0]
    for line in open(path):
        m = (NV if version == "nvidia_example" else ROW).match(line)
        if m:
            c = m.group(1) == "True"
            runs.setdefault((version, c), []).append(float(m.group(2)))
            if version != "nvidia_example":
                runs.setdefault(("sdpa", c), []).append(float(m.group(3)))

W, row_h, bar_h, top = 980, 30, 18, 96
label_w, panel_w, gap, xmax = 190, 330, 50, 80.0
H = top + len(ROWS) * row_h + 74
b = [f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     f'<text x="20" y="30" font-size="16" font-weight="600" fill="{INK}">N = 4096, B = 4, H = 8, D = 64, bf16: '
     f'TFLOP/s (median of the harness runs)</text>']
for p, causal in enumerate((False, True)):
    x0 = label_w + p * (panel_w + gap)
    b.append(f'<text x="{x0}" y="66" font-size="14" font-weight="600" fill="{INK}">{"causal" if causal else "non-causal"}</text>')
    for t in range(0, int(xmax) + 1, 20):
        x = x0 + panel_w * t / xmax
        b.append(f'<line x1="{x}" y1="{top - 8}" x2="{x}" y2="{top + len(ROWS) * row_h}" stroke="{GRID}" stroke-width="1"/>')
        b.append(f'<text x="{x}" y="{top + len(ROWS) * row_h + 16}" font-size="11" fill="{MUTED}" text-anchor="middle">{t}</text>')
    for i, (key, name, kind) in enumerate(ROWS):
        y = top + i * row_h
        if p == 0:
            b.append(f'<text x="{label_w - 12}" y="{y + 14}" font-size="12.5" fill="{INK2}" text-anchor="end">{name}</text>')
        v = data.get((key, causal))
        if v is None:
            continue
        w = panel_w * v / xmax
        b.append(f'<rect x="{x0}" y="{y + 1}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{COLOR[kind]}"/>')
        lo, hi = min(runs[(key, causal)]), max(runs[(key, causal)])
        xl, xh, ym = x0 + panel_w * lo / xmax, x0 + panel_w * hi / xmax, y + 1 + bar_h / 2
        b.append(f'<line x1="{xl:.1f}" y1="{ym}" x2="{xh:.1f}" y2="{ym}" stroke="{INK}" stroke-width="1.5"/>')
        for xx in (xl, xh):
            b.append(f'<line x1="{xx:.1f}" y1="{ym - 5}" x2="{xx:.1f}" y2="{ym + 5}" stroke="{INK}" stroke-width="1.5"/>')
        b.append(f'<text x="{max(x0 + w, xh) + 6:.1f}" y="{y + 14}" font-size="12" fill="{INK}">{v:.1f}</text>')
    if p == 0:
        b.append(f'<line x1="{x0}" y1="{top + 6 * row_h - 4}" x2="{x0 + 2 * panel_w + gap}" y2="{top + 6 * row_h - 4}" stroke="{MUTED}" stroke-dasharray="3 3"/>')
b.append(f'<text x="20" y="{H - 32}" font-size="11.5" fill="{MUTED}">Bars: median; black lines: lowest to highest run (3 runs, 10 for v5 and Triton v5, 35 for SDPA). '
         f'Below the dashed line: the references.</text>')
b.append(f'<text x="20" y="{H - 14}" font-size="11.5" fill="{MUTED}">v4 and v5 run the same kernel non-causal: the gap between them is run-to-run noise. '
         f'Causal TFLOP/s count half of the non-causal work, as the harness does.</text>')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">\n'
       + "\n".join(b) + "\n</svg>\n")
open("images/results_N4096.svg", "w").write(svg)
print("wrote images/results_N4096.svg")
