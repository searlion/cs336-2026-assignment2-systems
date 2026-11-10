# make_svgs.py
# Writes the eleven explanatory diagrams of the article as plain SVG (no external fonts):
#   images/mma_ownership.svg   who holds which value of a 64 x 64 score tile
#   images/swizzle_banks.svg   which banks 8 rows of a tile land in, row-major vs swizzled
#   images/pipeline.svg        synchronous copies (v3) against the cp.async schedule (v4)
#   images/gpu_hierarchy.svg   GPU -> SM -> block -> warp -> lane, and where each kind of memory lives
#   images/layouts.svg         the two example layouts, (64, 64):(64, 1) and ((2, 2), 8):((1, 16), 2)
#   images/tiled_mma.svg       how a 64 x 64 score tile is split into warps, MMA instructions and pairs
#   images/ldmatrix_pairs.svg  why permutation_mnk N = 16 halves the loads of K and V
#   images/warp_shuffle.svg    a quad's row maximum through shared memory against two warp shuffles
#   images/tiled_copy.svg      the layout of a tiled copy: each make_layout line on a 64 x 64 tile of K
#   images/copy_atoms.svg      the atom of a tiled copy: plain copies against cp.async and ldmatrix
#   images/memory_hierarchy.svg registers, L1 / shared memory, L2 and global memory, fastest to slowest
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


def tiled_mma():
    b = []
    WARP = [BLUE_L, "#f6f6f4", BLUE_L, "#f6f6f4"]
    b.append(text(20, 30, "How a 64 × 64 score tile S = Q Kᵀ is split into MMA instructions", 16, weight="600"))
    x0, y0, px = 150, 84, 5                     # 5 px per element: the tile is 320 x 320
    b.append(text(x0 + 160, y0 - 12, "64 keys (one key block)", 12.5, INK2, "middle"))
    for w in range(4):
        y = y0 + 80 * w
        b.append(rect(x0, y, 320, 80, WARP[w], GRID, 1))
        b.append(text(x0 - 10, y + 36, f"warp {w}", 12.5, INK, "end"))
        b.append(text(x0 - 10, y + 52, f"rows {16 * w}–{16 * w + 15}", 11.5, MUTED, "end"))
        for c in range(1, 8):
            b.append(line(x0 + 40 * c, y, x0 + 40 * c, y + 80, INK2 if c % 2 == 0 else GRID, 1))
    b.append(rect(x0, y0, 320, 320, "none", INK2, 1))
    # one instruction (warp 0, last piece) and one pair (warp 2, last two pieces)
    b.append(rect(x0 + 280, y0, 40, 80, "none", INK, 2.5))
    b.append(rect(x0 + 240, y0 + 160, 80, 80, "none", ORANGE, 2.5))

    tx = x0 + 350
    lines1 = [("Black: one MMA instruction, (16, 8, 16)", INK, "600"),
              ("16 rows × 8 keys of S. One instruction adds up", INK2, "normal"),
              ("16 of the 64 head dimensions, so 4 in a row", INK2, "normal"),
              ("finish the piece.", INK2, "normal")]
    for i, (t, c, wgt) in enumerate(lines1):
        b.append(text(tx, y0 + 18 + 17 * i, t, 12.5, c, weight=wgt))
    lines2 = [("A warp's strip is 8 pieces: 8 × 4 = 32", INK, "600"),
              ("instructions per key block, one after another.", INK2, "normal"),
              ("The 4 warps run at the same time.", INK2, "normal")]
    for i, (t, c, wgt) in enumerate(lines2):
        b.append(text(tx, y0 + 100 + 17 * i, t, 12.5, c, weight=wgt))
    lines3 = [("Orange: a pair, 16 rows × 16 keys", ORANGE, "600"),
              ("Both pieces need the same 16 × 16 piece of K,", INK2, "normal"),
              ("so one ldmatrix.x4 loads it once for both.", INK2, "normal"),
              ("CuTe works through each strip a pair at a time.", INK2, "normal")]
    for i, (t, c, wgt) in enumerate(lines3):
        b.append(text(tx, y0 + 178 + 17 * i, t, 12.5, c, weight=wgt))
    b.append(text(20, y0 + 350, "64 × 64 is the kernel's tile size (64 query rows × 64 keys), not the instruction's. "
                  "16 × 8 is the instruction's: it takes 4 × 8 = 32 pieces to cover the tile.", 12.5, INK2))
    return svg(870, y0 + 366, b)

def ldmatrix_pairs():
    b = []
    t = 56                                      # one 8 x 8 tile
    b.append(text(20, 30, "Loading a 16-key × 16-dim piece of K into registers", 15, weight="600"))

    def piece(x, y, title, sub):
        b.append(text(x, y - 44, title, 13.5, weight="600"))
        b.append(text(x, y - 26, sub, 12, INK2))
        b.append(text(x + t, y - 8, "head dims 0–15", 11, MUTED, "middle"))
        b.append(text(x - 8, y + t / 2 + 4, "keys 0–7", 11, MUTED, "end"))
        b.append(text(x - 8, y + 1.5 * t + 4, "keys 8–15", 11, MUTED, "end"))
        for i in range(2):
            for j in range(2):
                b.append(rect(x + j * t, y + i * t, t, t, "#f6f6f4", GRID, 1))
                b.append(text(x + j * t + t / 2, y + i * t + t / 2 + 4, "8 × 8", 11, INK2, "middle"))

    # default: two loads of two tiles, one per MMA
    x, y = 110, 110
    piece(x, y, "Default (step of 8 keys)", "each MMA gets its own load")
    b.append(rect(x + 3, y + 3, 2 * t - 6, t - 6, "none", ORANGE, 2.5, 4))
    b.append(rect(x + 3, y + t + 3, 2 * t - 6, t - 6, "none", AQUA, 2.5, 4))
    b.append(text(x + 2 * t + 14, y + t / 2 + 4, "load 1 → MMA 1", 12, ORANGE))
    b.append(text(x + 2 * t + 14, y + 1.5 * t + 4, "load 2 → MMA 2", 12, AQUA))
    b.append(text(x, y + 2 * t + 26, "2 load instructions", 13, INK, weight="600"))

    # (64, 16, 16): one load of four tiles, shared by the pair
    x = 470
    piece(x, y, "permutation_mnk = (64, 16, 16)", "the pair shares one load")
    b.append(rect(x + 3, y + 3, 2 * t - 6, 2 * t - 6, "none", BLUE, 2.5, 4))
    b.append(text(x + 2 * t + 14, y + t + 4, "1 load → MMA 1 and 2", 12, BLUE))
    b.append(text(x, y + 2 * t + 26, "1 load instruction", 13, INK, weight="600"))

    b.append(text(20, y + 2 * t + 64, "Same data either way. One load instruction (ldmatrix) can carry up to four 8 × 8 tiles; with a step of", 12.5, INK2))
    b.append(text(20, y + 2 * t + 82, "8 keys it only ever carries two. For the whole 64 × 64 tile of K: 32 loads per warp instead of 16; the same for V.", 12.5, INK2))
    return svg(780, y + 2 * t + 98, b)


def warp_shuffle():
    b = []
    b.append(f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')
    b.append(text(20, 30, "The maximum of one row, held by the 4 lanes of a quad (values 3, 7, 2, 5)", 15, weight="600"))
    w, h, gap = 56, 30, 78

    def lanes(x, y, vals, fill="#f6f6f4", label=True):
        for i, v in enumerate(vals):
            b.append(rect(x + i * gap, y, w, h, fill, GRID, 1, 4))
            b.append(text(x + i * gap + w / 2, y + 20, str(v), 13, INK, "middle"))
            if label:
                b.append(text(x + i * gap + w / 2, y - 6, f"lane {i}", 11, MUTED, "middle"))

    def arrow(x1, y1, x2, y2, color=INK2):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.3" marker-end="url(#ah)"/>')

    # without shuffles: through shared memory
    x = 40
    b.append(text(x, 64, "Without shuffles: through shared memory", 13.5, ORANGE, weight="600"))
    lanes(x, 96, [3, 7, 2, 5])
    sy = 170
    b.append(rect(x - 6, sy - 4, 3 * gap + w + 12, h + 8, ORANGE_L, "none", 1, 6))
    for i, v in enumerate([3, 7, 2, 5]):
        b.append(rect(x + i * gap, sy, w, h, SURF, GRID, 1, 2))
        b.append(text(x + i * gap + w / 2, sy + 20, str(v), 13, INK, "middle"))
        arrow(x + i * gap + w / 2, 96 + h + 2, x + i * gap + w / 2, sy - 6)
    b.append(text(x + 4 * gap - 6, sy + 20, "shared memory", 11.5, ORANGE))
    b.append(text(x + 4 * gap - 6, 96 + h + 30, "1. write", 11.5, INK2))
    b.append(text(x + 4 * gap - 6, sy + h + 22, "2. wait for all 4", 11.5, INK2))
    ry = 254
    for i in range(4):
        for j in range(4):
            if j != i or True:
                b.append(f'<line x1="{x + j * gap + w / 2}" y1="{sy + h + 4}" x2="{x + i * gap + w / 2}" y2="{ry - 2}" '
                         f'stroke="{GRID}" stroke-width="1"/>')
    lanes(x, ry, [7, 7, 7, 7], ORANGE_L, label=False)
    b.append(text(x + 4 * gap - 6, ry + 20, "3. read all 4, max", 11.5, INK2))
    b.append(text(x, ry + 58, "Per lane: 1 store, 1 wait, 3 loads, 3 max.", 12.5, INK, weight="600"))
    b.append(text(x, ry + 76, "Shared memory and a wait, on every row, every key block.", 12, INK2))

    # with shuffles: register to register
    x = 520
    b.append(text(x, 64, "With shuffles: lane to lane, in registers", 13.5, BLUE, weight="600"))
    lanes(x, 96, [3, 7, 2, 5])
    y2 = 175
    for i in range(4):
        arrow(x + i * gap + w / 2, 96 + h + 2, x + (i ^ 2) * gap + w / 2, y2 - 3, BLUE)
    lanes(x, y2, [3, 7, 3, 7], BLUE_L, label=False)
    b.append(text(x + 4 * gap - 6, 96 + h + 30, "1. swap with", 11.5, INK2))
    b.append(text(x + 4 * gap - 6, 96 + h + 45, "lane xor 2, max", 11.5, INK2))
    for i in range(4):
        arrow(x + i * gap + w / 2, y2 + h + 2, x + (i ^ 1) * gap + w / 2, ry - 3, BLUE)
    lanes(x, ry, [7, 7, 7, 7], BLUE_L, label=False)
    b.append(text(x + 4 * gap - 6, y2 + h + 30, "2. swap with", 11.5, INK2))
    b.append(text(x + 4 * gap - 6, y2 + h + 45, "lane xor 1, max", 11.5, INK2))
    b.append(text(x, ry + 58, "Per lane: 2 shuffles, 2 max.", 12.5, INK, weight="600"))
    b.append(text(x, ry + 76, "No shared memory, no separate wait: every lane ends with 7.", 12, INK2))
    return svg(980, ry + 92, b)


def tiled_copy():
    """The layout half of a tiled copy: each make_layout line, and what it means on a 64 x 64 tile of K."""
    b = []
    GREY_L = "#f6f6f4"
    b.append(text(20, 30, "The layout of a tiled copy: which thread copies which 16 bytes of a 64 × 64 tile of K", 15, weight="600"))

    # code line 1 and the 16 x 8 thread grid it describes
    b.append(text(20, 64, "make_layout((16, 8), stride=(8, 1))", 13, BLUE, weight="600", family=MONO))
    b.append(text(20, 84, "The block's 128 threads (4 warps × 32 threads) as a 16 × 8 grid: thread = 8 · row + 1 · column", 12, INK2))
    x0, y0, cw, ch = 74, 124, 40, 18
    for c in range(8):
        b.append(text(x0 + c * cw + cw / 2, y0 - 6, f"col {c}", 10.5, MUTED, "middle"))
    for r in range(16):
        b.append(text(x0 - 6, y0 + r * ch + 13, f"row {r}", 10.5, MUTED, "end"))
        for c in range(8):
            hi = (r, c) == (2, 3)
            base = GREY_L if (r // 4) % 2 == 0 else "#eef4fc"
            b.append(rect(x0 + c * cw, y0 + r * ch, cw, ch, ORANGE_L if hi else base, ORANGE if hi else GRID, 2 if hi else 1))
            b.append(text(x0 + c * cw + cw / 2, y0 + r * ch + 13, f"t{8 * r + c}", 9.5, INK, "middle"))
    gx, gy = x0 + 8 * cw, y0 + 16 * ch
    for w in range(4):
        wy = y0 + 4 * w * ch
        b.append(line(gx + 6, wy + 2, gx + 6, wy + 4 * ch - 2, INK2, 1.5))
        b.append(text(gx + 12, wy + 2 * ch + 4, f"warp {w}", 11, INK2))
    b.append(text(x0, gy + 22, "Each grid row covers one row of K: 64 values in 8 pieces, one piece per thread.", 11.5, INK2))

    # callout for thread 19
    X = 490
    cy = y0 + 2 * ch + ch / 2
    b.append(text(X, cy - 4, "row 2, column 3  →  8 · 2 + 3 = thread 19", 12.5, ORANGE, weight="600"))
    b.append(text(X, cy + 14, "so thread 19 copies the 4th piece of row 2", 12, INK2))

    # code line 2 and the 8 values of one piece
    y2 = 218
    b.append(text(X, y2, "make_layout((1, 8))", 13, BLUE, weight="600", family=MONO))
    b.append(text(X, y2 + 20, "each thread's piece: 1 row × 8 values", 12, INK2))
    for v in range(8):
        b.append(rect(X + v * 46, y2 + 34, 46, 26, ORANGE_L, ORANGE if v in (0, 7) else GRID, 1))
        b.append(text(X + v * 46 + 23, y2 + 51, f"{24 + v}", 11, INK, "middle"))
    b.append(text(X, y2 + 78, "thread 19's piece: values 24–31 of row 2", 12, INK2))
    b.append(text(X, y2 + 94, "8 bf16 values × 2 bytes = 16 bytes, one copy instruction", 12, INK2))

    # together
    y3 = 352
    b.append(text(X, y3, "Together", 13, INK, weight="600"))
    b.append(text(X, y3 + 20, "One step: 16 rows × (8 threads × 8 values) = 16 rows × 64 values.", 12, INK2))
    b.append(text(X, y3 + 38, "The tile has 64 rows, so CuTe repeats the step 4 times:", 12, INK2))
    b.append(text(X, y3 + 56, "thread 19 also copies values 24–31 of rows 18, 34 and 50.", 12, INK2))
    return svg(900, gy + 40, b)


def copy_atoms():
    """The atom half of a tiled copy: the same pieces, carried by different instructions."""
    b = []
    b.append(f'<defs><marker id="ah3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')
    b.append(text(20, 30, "The atom of a tiled copy: the instruction that carries each piece", 15, weight="600"))
    b.append(text(20, 50, "Who copies what stays the same; the instruction changes, and with it the speed.", 12.5, INK2))
    G, R, S = "#fdf3d8", BLUE_L, AQUA_L

    def path(x, y, nodes, dash=False):
        for j, (lab, fill) in enumerate(nodes):
            b.append(rect(x, y, 84, 26, fill, GRID, 1, 6))
            b.append(text(x + 42, y + 17, lab, 11, INK, "middle"))
            if j + 1 < len(nodes):
                d = ' stroke-dasharray="5 4"' if dash else ""
                b.append(f'<line x1="{x + 86}" y1="{y + 13}" x2="{x + 112}" y2="{y + 13}" stroke="{INK2}" '
                         f'stroke-width="1.5"{d} marker-end="url(#ah3)"/>')
            x += 114

    def card(x, y, title, rounds, nodes, dash, pros, cons, good):
        b.append(rect(x, y, 420, 168, SURF, BLUE if good else GRID, 1.5 if good else 1, 8))
        b.append(text(x + 14, y + 24, title, 12.5, INK, weight="600"))
        b.append(text(x + 406, y + 24, rounds, 11, MUTED, "end"))
        path(x + 14, y + 38, nodes, dash)
        yy = y + 90
        for items, sign, col in ((pros, "+ ", AQUA), (cons, "− ", ORANGE)):
            for t in items:
                cont = t.startswith("  ")
                b.append(text(x + (26 if cont else 14), yy, t.strip() if cont else sign + t, 11.5, col))
                yy += 17

    y = 76
    b.append(text(20, y, "Copy 1: K and V tiles, global memory → shared memory, with the layout above", 13, BLUE, weight="600"))
    card(20, y + 12, "CopyUniversalOp: plain load, then store", "Rounds 0–3",
         [("global", G), ("registers", R), ("shared", S)], False,
         ["simple: nothing else to manage"],
         ["the thread stalls until the data arrives,", "  then stores it; no maths meanwhile"], False)
    card(460, y + 12, "CopyG2SOp: cp.async", "Rounds 4–5",
         [("global", G), ("shared", S)], True,
         ["skips registers; the thread starts the copy", "  and goes back to maths at once"],
         ["you must commit, wait and sync yourself;", "  a wrong order reads stale data"], True)

    y = 290
    b.append(text(20, y, "Copy 2: operands, shared memory → registers, with a layout taken from the tiled MMA", 13, BLUE, weight="600"))
    card(20, y + 12, "plain loads", "Rounds 0–1",
         [("shared", S), ("registers", R)], False,
         ["works for any layout"],
         ["224 small loads per key block per thread;", "  the load/store unit becomes the busiest unit"], False)
    card(460, y + 12, "LdMatrix8x8x16bOp: ldmatrix", "Rounds 2–5",
         [("shared", S), ("registers", R)], False,
         ["40 loads: each carries four 8 × 8 tiles and gives", "  each lane exactly the values the multiply needs"],
         ["16-bit values in 8 × 8 tiles only"], True)
    b.append(text(20, y + 210, "Blue border: the choice the final kernel uses, and the one Triton makes by itself.", 12, INK2))
    return svg(900, y + 226, b)


def memory_hierarchy():
    """Registers, L1 / shared memory, L2 and global memory: where each sits and who can see it."""
    b = []
    GREY_L = "#f6f6f4"
    b.append(f'<defs><marker id="ah4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
             f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')
    b.append(text(20, 30, "Where data lives on the GPU: small and fast at the top, big and slow at the bottom", 16, weight="600"))

    def arrow2(x, y1, y2):                     # double-headed: data moves both ways
        b.append(f'<line x1="{x}" y1="{y1 + 3}" x2="{x}" y2="{y2 - 3}" stroke="{INK2}" stroke-width="1.4" '
                 f'marker-start="url(#ah4)" marker-end="url(#ah4)"/>')

    top = 62

    def sm(x, name):
        b.append(rect(x, top, 250, 196, SURF, BLUE, 1.5, 8))
        b.append(text(x + 12, top + 22, name, 13.5, weight="600"))
        for i, t in enumerate(["t0", "t1", "t2", "t3", "…"]):
            bx = x + 12 + i * 46
            if t == "…":
                b.append(text(bx + 18, top + 58, "…", 14, INK2, "middle"))
            else:
                b.append(rect(bx, top + 36, 40, 32, BLUE_L, BLUE, 1, 4))
                b.append(text(bx + 20, top + 57, t, 11.5, INK, "middle", family=MONO))
        b.append(text(x + 12, top + 86, "Registers: each thread (t) has its own", 11.5, INK2))
        y = top + 128
        b.append(rect(x + 12, y, 108, 54, GREY_L, GRID, 1, 4))
        b.append(text(x + 66, y + 23, "L1 cache", 12.5, INK, "middle", "600"))
        b.append(text(x + 66, y + 40, "fills itself", 11, MUTED, "middle"))
        b.append(rect(x + 126, y, 112, 54, AQUA_L, AQUA, 1, 4))
        b.append(text(x + 182, y + 23, "Shared memory", 12.5, INK, "middle", "600"))
        b.append(text(x + 182, y + 40, "your code fills it", 11, MUTED, "middle"))
        arrow2(x + 66, top + 92, y)
        arrow2(x + 182, top + 92, y)
        return x + 66, x + 182

    cols = sm(40, "SM 0") + sm(340, "SM 55")
    b.append(text(315, top + 104, "…", 18, INK2, "middle"))
    b.append(text(315, top + 124, "56 SMs", 11, MUTED, "middle"))

    l2y, gy = 296, 368
    for cx in cols:
        arrow2(cx, top + 196, l2y)
    b.append(rect(40, l2y, 550, 40, ORANGE_L, ORANGE, 1, 6))
    b.append(text(315, l2y + 25, "L2 cache: 48 MB, one for the whole GPU, fills itself", 12.5, INK, "middle", "600"))
    for cx in (130, 315, 500):
        arrow2(cx, l2y + 40, gy)
    b.append(rect(40, gy, 550, 44, GREY_L, INK2, 1, 6))
    b.append(text(315, gy + 27, "Global memory (DRAM): 12 GB, the GPU's main memory", 12.5, INK, "middle", "600"))

    # right: one note per level, level with it
    nx = 630
    notes = [
        (top + 44, "Registers", ["Fastest. Only that one thread can use them.", "65,536 per SM, split among its threads."]),
        (top + 146, "L1 cache and shared memory", ["Very fast. Only threads on this SM see them.", "One 128 KB store per SM, split between the",
                                                   "two; shared memory gets up to 100 KB."]),
        (l2y + 14, "L2 cache", ["Slower. Every SM sees it."]),
        (gy + 16, "Global memory", ["Slowest, but holds everything: Q, K, V, O."]),
    ]
    for y, head, lines in notes:
        b.append(text(nx, y, head, 13, INK, weight="600"))
        for i, s in enumerate(lines):
            b.append(text(nx, y + 18 + 16 * i, s, 12, INK2))

    # speed / size scale on the far right
    ax = 975
    b.append(text(ax, top + 10, "faster", 11.5, INK2, "middle"))
    b.append(text(ax, top + 24, "smaller", 11.5, INK2, "middle"))
    arrow2(ax, top + 32, gy + 10)
    b.append(text(ax, gy + 28, "slower", 11.5, INK2, "middle"))
    b.append(text(ax, gy + 42, "bigger", 11.5, INK2, "middle"))

    b.append(text(20, gy + 76, "Arrows: the paths data takes. Caches (L1, L2) keep recently used data by themselves; "
                  "shared memory and registers hold only what the program puts there.", 12, INK2))
    return svg(1010, gy + 92, b)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "mma_ownership.svg").write_text(mma_ownership())
    (OUT / "swizzle_banks.svg").write_text(swizzle_banks())
    (OUT / "pipeline.svg").write_text(pipeline())
    (OUT / "gpu_hierarchy.svg").write_text(gpu_hierarchy())
    (OUT / "layouts.svg").write_text(layouts())
    (OUT / "tiled_mma.svg").write_text(tiled_mma())
    (OUT / "ldmatrix_pairs.svg").write_text(ldmatrix_pairs())
    (OUT / "warp_shuffle.svg").write_text(warp_shuffle())
    (OUT / "tiled_copy.svg").write_text(tiled_copy())
    (OUT / "copy_atoms.svg").write_text(copy_atoms())
    (OUT / "memory_hierarchy.svg").write_text(memory_hierarchy())
    print("wrote", *sorted(p.name for p in OUT.glob("*.svg")))
