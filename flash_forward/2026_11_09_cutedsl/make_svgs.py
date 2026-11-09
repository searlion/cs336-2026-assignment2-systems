# make_svgs.py
# Writes the five explanatory diagrams of the article as plain SVG (no external fonts):
#   images/mma_ownership.svg   who holds which value of a 64 x 64 score tile
#   images/swizzle_banks.svg   which banks 8 rows of a tile land in, row-major vs swizzled
#   images/pipeline.svg        synchronous copies (v3) against the cp.async schedule (v4)
#   images/gpu_hierarchy.svg   GPU -> SM -> block -> warp -> lane, and where each kind of memory lives
#   images/layouts.svg         the two example layouts, (64, 64):(64, 1) and ((2, 2), 8):((1, 16), 2)
# The results chart is drawn by plot_results_svg.py from the benchmark files.
import pathlib

OUT = pathlib.Path(__file__).parent / "images"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#d6d5d0"
BLUE, BLUE_L, ORANGE, ORANGE_L, AQUA, AQUA_L = "#2a78d6", "#d4e4f7", "#eb6834", "#fbe0d5", "#1baf7a", "#d1efe4"
SURF = "#ffffff"


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="{FONT}">\n<rect width="{w}" height="{h}" fill="{SURF}"/>\n' + "\n".join(body) + "\n</svg>\n")


def text(x, y, s, size=13, fill=INK, anchor="start", weight="normal", family=None):
    fam = f' font-family="{family}"' if family else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{fam}>{s}</text>')


def rect(x, y, w, h, fill, stroke="none", sw=1, rx=0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>'


def line(x1, y1, x2, y2, stroke=INK2, sw=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


# ---------------------------------------------------------------------------------------------
def mma_ownership():
    b = []
    b.append(text(20, 30, "Who holds which score: a 64 × 64 tile of S = Q Kᵀ, 4 warps, 16 × 8 MMA tiles", 16, weight="600"))
    # left: the whole tile, 4 px per element
    x0, y0, px = 90, 70, 4
    b.append(text(x0, y0 - 12, "The whole tile (64 rows × 64 keys)", 13, INK2))
    fills = ["#eef4fc", "#f6f6f4", "#eef4fc", "#f6f6f4"]
    for w in range(4):
        b.append(rect(x0, y0 + w * 64, 256, 64, fills[w], GRID))
        b.append(text(x0 - 10, y0 + w * 64 + 37, f"warp {w}", 12, INK2, "end"))
    for a in range(8):  # warp 0's eight 16 x 8 MMA tiles
        b.append(rect(x0 + a * 32, y0, 32, 64, "none", "#9fb8d9", 1))
    for r in (0, 8):     # rows 0 and 8 of the tile
        b.append(rect(x0, y0 + r * px, 256, px, BLUE))
    b.append(rect(x0, y0, 32, 64, "none", INK, 2))
    b.append(text(x0 - 50, y0 + 256 + 22, "Warp 0 owns rows 0–15. Its strip is 8 MMA tiles of 16 rows × 8 keys.", 12, INK2))
    b.append(text(x0 - 50, y0 + 256 + 40, "Blue: rows 0 and 8, which lanes 0–3 (one quad) hold.", 12, INK2))
    # right: zoom of one MMA tile, lane per element
    zx, zy, cw, ch = 440, 70, 40, 15
    b.append(text(zx, zy - 12, "One 16 × 8 MMA tile, zoomed: the lane that holds each value", 13, INK2))
    b.append(line(x0 + 32, y0, zx, zy, MUTED, 1, "4 3"))
    b.append(line(x0 + 32, y0 + 64, zx, zy + 16 * ch, MUTED, 1, "4 3"))
    for r in range(16):
        for c in range(8):
            lane = (r % 8) * 4 + c // 2
            quad0 = lane < 4
            b.append(rect(zx + c * cw, zy + r * ch, cw, ch, BLUE if quad0 else "#f6f6f4", SURF, 1))
            b.append(text(zx + c * cw + cw / 2, zy + r * ch + 11, f"T{lane}", 9.5, SURF if quad0 else INK2,
                          "middle", family=MONO))
    for c in range(8):
        b.append(text(zx + c * cw + cw / 2, zy + 16 * ch + 14, str(c), 10, MUTED, "middle"))
    for r in (0, 8, 15):
        b.append(text(zx - 6, zy + r * ch + 11, str(r), 10, MUTED, "end"))
    b.append(text(zx + 4 * cw, zy + 16 * ch + 30, "key within the tile", 11, MUTED, "middle"))
    nx = zx + 8 * cw + 22
    notes = ["Each lane holds 2 neighbouring keys", "in 2 rows (r and r + 8) of every tile.",
             "", "The 4 lanes of a quad share those rows:",
             "lanes 0–3 hold all of rows 0 and 8.", "",
             "So the maximum (or sum) of a row is",
             "1. a max over the lane's own 16 values,", "    no communication, then",
             "2. two shuffles inside the quad:", "    with lane xor 2, then with lane xor 1."]
    for i, s in enumerate(notes):
        b.append(text(nx, zy + 14 + i * 19, s, 12.5, INK if i in (3, 6) else INK2))
    return svg(1050, 400, b)


# ---------------------------------------------------------------------------------------------
def swizzle_banks():
    b = []
    b.append(text(20, 30, "8 rows of a 64-wide bf16 tile in shared memory: one row = 128 bytes = all 32 banks", 16, weight="600"))
    cw, ch = 46, 26

    def panel(x0, y0, title, swz, caption):
        b.append(text(x0, y0 - 34, title, 14, INK, weight="600"))
        for p in range(8):
            b.append(text(x0 + p * cw + cw / 2, y0 - 10, f"{4 * p}–{4 * p + 3}", 10, MUTED, "middle"))
        b.append(text(x0 - 8, y0 - 10, "banks", 10, MUTED, "end"))
        for r in range(8):
            b.append(text(x0 - 8, y0 + r * ch + 17, f"row {r}", 11, INK2, "end"))
            for p in range(8):
                c = p ^ r if swz else p            # logical 16-byte chunk stored at position p
                hit = c == 0
                b.append(rect(x0 + p * cw, y0 + r * ch, cw, ch, BLUE if hit else "#f6f6f4", SURF, 2))
                b.append(text(x0 + p * cw + cw / 2, y0 + r * ch + 17, f"c{c}", 11, SURF if hit else INK2,
                              "middle", family=MONO))
        for i, s in enumerate(caption):
            b.append(text(x0, y0 + 8 * ch + 24 + i * 18, s, 12.5, INK if i == 0 else INK2,
                          weight="600" if i == 0 else "normal"))

    panel(70, 100, "Row-major (v0)", False,
          ["Chunk c0 of rows 0–7: all in banks 0–3.", "A warp reading it needs 8 passes", "(an 8-way bank conflict)."])
    panel(530, 100, "Swizzled: chunk c of row r stored at c xor r (v1)", True,
          ["Chunk c0 of rows 0–7: one per bank group.", "The same read takes 1 pass.",
           "Rows stay intact; only the 16-byte chunks move."])
    b.append(text(20, 405, "cN = the N-th 16-byte chunk (8 bf16 values) of a row, as the program sees it. Blue: the chunk that one MMA operand load reads from each row.", 12, MUTED))
    return svg(960, 420, b)


# ---------------------------------------------------------------------------------------------
def pipeline():
    b = []
    b.append(text(20, 30, "One key block, two ways (schematic, not to scale)", 16, weight="600"))
    x0, unit = 150, 26

    def bar(x, y, w, label, fill, ink=INK, h=26):
        b.append(rect(x0 + x * unit, y, w * unit - 2, h, fill, rx=4))
        b.append(text(x0 + x * unit + (w * unit - 2) / 2, y + 17, label, 11.5, ink, "middle"))

    def barrier(x, y, h):
        b.append(line(x0 + x * unit - 1, y - 6, x0 + x * unit - 1, y + h + 6, INK, 2))

    # v3: synchronous copies
    y = 80
    b.append(text(20, y - 16, "v3: copies through registers", 14, weight="600"))
    b.append(text(20, y + 17, "warp", 12, INK2))
    b.append(text(20, y + 51, "copies in flight", 12, INK2))
    barrier(0.08, y, 60)
    bar(0, y, 6, "load K_j, V_j", ORANGE_L)
    barrier(6, y, 60)
    bar(6, y, 5, "S = Q K_jᵀ", BLUE_L)
    bar(11, y, 6, "softmax", AQUA_L)
    bar(17, y, 5, "O += P V_j", BLUE_L)
    barrier(22, y, 60)
    bar(22, y, 4, "load K_j+1 …", ORANGE_L)
    bar(0, y + 34, 6, "K_j, V_j", ORANGE, SURF)
    bar(22, y + 34, 4, "K_j+1, V_j+1", ORANGE, SURF)
    b.append(text(x0, y + 86, "Each load goes global memory -> registers -> shared memory, and the warp waits for it before any maths.", 12, INK2))
    b.append(text(x0, y + 104, "Two barriers (black lines) per key block: before the load, so no warp still reads the old K, V, and after it.", 12, INK2))

    # v4: cp.async, FlashAttention-2 order
    y = 250
    b.append(text(20, y - 16, "v4: cp.async in the FlashAttention-2 order", 14, weight="600"))
    b.append(text(20, y + 17, "warp", 12, INK2))
    b.append(text(20, y + 51, "copies in flight", 12, INK2))
    bar(0, y, 1, "", "#f0efec")
    barrier(1, y, 60)
    bar(1, y, 5, "S = Q K_jᵀ", BLUE_L)
    bar(6, y, 1, "", "#f0efec")
    barrier(7, y, 60)
    bar(7, y, 6, "softmax", AQUA_L)
    bar(13, y, 5, "O += P V_j", BLUE_L)
    bar(18, y, 1, "", "#f0efec")
    barrier(19, y, 60)
    bar(19, y, 5, "S = Q K_j+1ᵀ", BLUE_L)
    bar(1, y + 34, 5, "V_j", ORANGE, SURF)
    bar(7, y + 34, 5, "K_j+1", ORANGE, SURF)
    bar(19, y + 34, 5, "V_j+1", ORANGE, SURF)
    b.append(text(x0, y + 86, "Each copy is started before the maths that does not need it, and waited for", 12, INK2))
    b.append(text(x0, y + 104, "only where its data is used. Grey: wait for the copy, then a barrier: still two per key block.", 12, INK2))
    return svg(860, 380, b)

def gpu_hierarchy():
    b = []
    b.append(text(20, 30, "Where the threads and memory live: GPU → SM → block → warp → lane", 16, weight="600"))
    top, h = 56, 290
    GREY_L = "#f6f6f4"

    def panel(x, w, title, sub, border):
        b.append(rect(x, top, w, h, SURF, border, 1.5, 8))
        b.append(text(x + 12, top + 22, title, 14, weight="600"))
        for i, s in enumerate(sub):
            b.append(text(x + 12, top + 40 + 15 * i, s, 11.5, INK2))

    def arrow(x):
        y = top + h / 2
        b.append(line(x + 6, y, x + 22, y, INK2, 1.5))
        b.append(f'<path d="M{x + 17},{y - 5} L{x + 23},{y} L{x + 17},{y + 5}" fill="none" stroke="{INK2}" stroke-width="1.5"/>')

    def bar(x, y, w, label, fill=GREY_L, size=11):
        b.append(rect(x, y, w, 24, fill, GRID, 1, 4))
        b.append(text(x + w / 2, y + 16, label, size, INK2, "middle"))

    # 1. the GPU: 56 SMs and global memory
    x = 20
    panel(x, 230, "The GPU", ["56 SMs (streaming multiprocessors)"], GRID)
    for i in range(56):
        r, c = divmod(i, 8)
        cx, cy = x + 19 + c * 24, top + 56 + r * 24
        b.append(rect(cx, cy, 20, 20, BLUE_L if i == 0 else GREY_L, BLUE if i == 0 else GRID, 1.5 if i == 0 else 1, 3))
    bar(x + 12, top + 232, 206, "Global memory 12 GB (+ L2 cache 48 MB)", size=10)
    arrow(x + 230)

    # 2. one SM: 3 of our blocks, and the resources they split
    x = 280
    panel(x, 250, "One SM", ["holds 3 of our blocks at once"], BLUE)
    for i in range(3):
        y = top + 54 + i * 40
        b.append(rect(x + 12, y, 226, 32, ORANGE_L if i == 0 else GREY_L, ORANGE if i == 0 else GRID, 1.5 if i == 0 else 1, 4))
        b.append(text(x + 125, y + 21, f"block {i}", 12, INK, "middle"))
    bar(x + 12, top + 178, 226, "Shared memory: 100 KB, split by blocks")
    bar(x + 12, top + 206, 226, "Registers: 65,536, split by threads")
    bar(x + 12, top + 234, 226, "Tensor cores: small matrix multiplies")
    arrow(x + 250)

    # 3. one block: 4 warps, each with 16 rows of the 64-row tile
    x = 560
    panel(x, 250, "One block (CTA)", ["128 threads = 4 warps", "computes 64 query rows"], ORANGE)
    for i in range(4):
        y = top + 70 + i * 38
        b.append(rect(x + 12, y, 226, 30, AQUA_L if i == 0 else GREY_L, AQUA if i == 0 else GRID, 1.5 if i == 0 else 1, 4))
        b.append(text(x + 125, y + 20, f"warp {i}: rows {16 * i}–{16 * i + 15}", 12, INK, "middle"))
    bar(x + 12, top + 234, 226, "its shared memory: Q, K, V tiles (24 KB)")
    arrow(x + 250)

    # 4. one warp: 32 lanes in 8 quads
    x = 840
    panel(x, 200, "One warp", ["32 threads (lanes) that run", "the same instruction together"], AQUA)
    for lane in range(32):
        r, c = divmod(lane, 4)
        cx, cy = x + 14 + c * 44, top + 70 + r * 20
        b.append(rect(cx, cy, 40, 16, GREY_L, GRID, 1, 3))
        b.append(text(cx + 20, cy + 12, f"lane {lane}", 9.5, INK2, "middle"))
    b.append(rect(x + 11, top + 67, 178, 22, "none", INK, 1.5, 4))
    b.append(text(x + 14, top + 244, "Each row is a quad (4 lanes).", 11, INK2))
    b.append(text(x + 14, top + 260, "Each lane has its own registers.", 11, INK2))

    b.append(text(20, top + h + 28, "Fastest to slowest: registers (one lane) → shared memory (one block) → L2 cache → global memory (whole GPU).", 12.5, INK2))
    return svg(1060, top + h + 44, b)


def layouts():
    b = []
    GREY_L = "#f6f6f4"
    cw, ch = 46, 28

    def cell(x, y, s, fill=SURF, ink=INK, size=12):
        b.append(rect(x, y, cw, ch, fill, GRID, 1))
        b.append(text(x + cw / 2, y + 18, s, size, ink, "middle"))

    def arrow(x1, y1, x2, y2, label, lx, ly, anchor="middle"):
        b.append(line(x1, y1, x2, y2, INK2, 1.5))
        if x1 == x2:
            b.append(f'<path d="M{x2 - 5},{y2 - 6} L{x2},{y2} L{x2 + 5},{y2 - 6}" fill="none" stroke="{INK2}" stroke-width="1.5"/>')
        else:
            b.append(f'<path d="M{x2 - 6},{y2 - 5} L{x2},{y2} L{x2 - 6},{y2 + 5}" fill="none" stroke="{INK2}" stroke-width="1.5"/>')
        b.append(text(lx, ly, label, 12, INK2, anchor))

    # left: (64, 64):(64, 1), row-major; show rows 0-2 and 63, columns 0-3 and 63
    x0, y0 = 70, 110
    b.append(text(20, 30, "(64, 64):(64, 1)", 15, weight="600", family=MONO))
    b.append(text(20, 50, "64 rows × 64 columns, row-major (corner shown)", 12.5, INK2))
    b.append(text(20, 68, "offset = 64·r + 1·c", 12.5, INK, family=MONO))
    cols, rows = [0, 1, 2, 3, None, 63], [0, 1, 2, None, 63]
    for j, c in enumerate(cols):
        b.append(text(x0 + j * cw + cw / 2, y0 - 8, "…" if c is None else f"c={c}", 11.5, MUTED, "middle"))
    for i, r in enumerate(rows):
        y = y0 + i * ch
        b.append(text(x0 - 8, y + 18, "⋮" if r is None else f"r={r}", 11.5, MUTED, "end"))
        for j, c in enumerate(cols):
            if r is None or c is None:
                b.append(text(x0 + j * cw + cw / 2, y + 18, "⋮" if r is None and c is not None else "…" if c is None and r is not None else "", 12, MUTED, "middle"))
            else:
                cell(x0 + j * cw, y, str(64 * r + c))
    gy = y0 + len(rows) * ch
    arrow(x0, gy + 18, x0 + 3 * cw, gy + 18, "next column: +1", x0 + 1.5 * cw, gy + 36)
    arrow(x0 + 6 * cw + 16, y0 + 4, x0 + 6 * cw + 16, y0 + 2 * ch + 24, "next row:", x0 + 6 * cw + 24, y0 + ch + 10, "start")
    b.append(text(x0 + 6 * cw + 24, y0 + ch + 26, "+64", 12, INK2))

    # right: ((2, 2), 8):((1, 16), 2), the row index is itself two modes (i0, i1)
    x1 = 540
    b.append(text(x1 - 60, 30, "((2, 2), 8):((1, 16), 2)", 15, weight="600", family=MONO))
    b.append(text(x1 - 60, 50, "4 rows × 8 columns; each row index is a pair (i0, i1)", 12.5, INK2))
    b.append(text(x1 - 60, 68, "offset = 1·i0 + 16·i1 + 2·c", 12.5, INK, family=MONO))
    for c in range(8):
        b.append(text(x1 + c * cw + cw / 2, y0 - 8, f"c={c}", 11.5, MUTED, "middle"))
    for i in range(4):
        i0, i1 = i % 2, i // 2
        y = y0 + i * ch
        b.append(text(x1 - 8, y + 18, f"({i0}, {i1})", 11.5, MUTED, "end"))
        for c in range(8):
            cell(x1 + c * cw, y, str(i0 + 16 * i1 + 2 * c), ORANGE_L if i1 == 0 else AQUA_L)
    gy = y0 + 4 * ch
    arrow(x1, gy + 18, x1 + 3 * cw, gy + 18, "next column: +2", x1 + 1.5 * cw, gy + 36)
    b.append(text(x1 + 4 * cw, gy + 23, "next i0: +1", 12, INK2))
    b.append(text(x1 + 4 * cw, gy + 39, "next i1: +16", 12, INK2))
    b.append(text(x1 - 60, gy + 64, "Orange rows (i1 = 0) take offsets 0–15, green rows (i1 = 1) take 16–31;", 12, INK2))
    b.append(text(x1 - 60, gy + 80, "within each colour, one row takes the even offsets and the other the odd.", 12, INK2))
    return svg(920, gy + 96, b)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "mma_ownership.svg").write_text(mma_ownership())
    (OUT / "swizzle_banks.svg").write_text(swizzle_banks())
    (OUT / "pipeline.svg").write_text(pipeline())
    (OUT / "gpu_hierarchy.svg").write_text(gpu_hierarchy())
    (OUT / "layouts.svg").write_text(layouts())
    print("wrote", *sorted(p.name for p in OUT.glob("*.svg")))
