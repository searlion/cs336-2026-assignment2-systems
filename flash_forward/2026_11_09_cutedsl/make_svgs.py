# make_svgs.py
# Writes the sixteen explanatory diagrams of the article as plain SVG (no external fonts):
#   images/mma_ownership.svg   who holds which value of a 64 x 64 score tile
#   images/swizzle_banks.svg   which banks 8 rows of a tile land in, row-major vs swizzled
#   images/pipeline.svg        synchronous copies (v3) against the cp.async schedule (v4)
#   images/gpu_hierarchy.svg   GPU -> SM -> block -> warp -> lane, and where each kind of memory lives
#   images/layouts.svg         the two example layouts, (64, 64):(64, 1) and ((2, 2), 8):((1, 16), 2)
#   images/tiled_mma.svg       what (16, 8, 16), (4, 1, 1) and permutation (64, 16, 16) each cover of S = Q K^T
#   images/ldmatrix_pairs.svg  why permutation_mnk N = 16 halves the loads of K and V
#   images/warp_shuffle.svg    a quad's row maximum through shared memory against two warp shuffles
#   images/warp_split.svg      where row 0 of S lives under (4, 1, 1) and (2, 2, 1), and its maximum's cost
#   images/tiled_copy.svg      the layout of a tiled copy: each make_layout line on a 64 x 64 tile of K
#   images/copy_atoms.svg      the atom of a tiled copy: plain copies against cp.async and ldmatrix
#   images/memory_hierarchy.svg registers, L1 / shared memory, L2 and global memory, fastest to slowest
#   images/block_grid.svg      how 2,048 blocks cover 32 (sequence, head) problems of 4,096 query rows
#   images/warp_lanes.svg      a warp's lanes and quads, who holds each row of an MMA result, warps -> block
#   images/ldmatrix_x4.svg     one ldmatrix.x4: a 16 x 16 piece of K, its four 8 x 8 tiles, who receives each value
#   images/qkv_tiles.svg       the 64 x 64 tiles: Q, K, V in global memory, shared memory, and S / O in registers
#   images/cp_async.svg        cp.async: global -> shared without registers, commit_group / wait_group(n), then a barrier
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
    # What thread 0 gets from each partition line of the code, checked on the GPU with the real
    # tiled MMA: Q rows 0 and 8, K rows 0, 8, ..., 56, S rows 0 and 8; head dims / keys with c % 8 in {0, 1}.
    px, n = 4, 64
    side = px * n
    xs, y0 = (40, 360, 680), 126
    W = 1040
    b = [text(20, 30, "What thread 0 gets from each line of the code", 16, weight="600")]
    b.append(text(20, 52, "Blue: the values thread 0 works on. Every one of the 128 threads gets its own share "
                          "the same way.", 12.5, INK2))
    mine = lambda r, c, keys: (r % 8 == 0 if keys else r in (0, 8)) and c % 8 in (0, 1)
    panels = [
        ("sQ", "Q tile: 64 rows × 64 head dims", "tSsQ = partition_A(sQ)", "32 values: rows 0 and 8", False, True),
        ("sK", "K tile: 64 keys × 64 head dims", "tSsK = partition_B(sK)", "128 values: keys 0, 8, 16, …, 56", True, True),
        ("acc_S", "S tile: 64 rows × 64 keys", "acc_S = make_rmem_tensor(…)", "32 values: rows 0 and 8, in registers",
         False, False),
    ]
    for x0, (name, cap, code, note, keys, kband) in zip(xs, panels):
        b.append(text(x0, y0 - 48, name, 14, INK, weight="600", family=MONO))
        b.append(text(x0, y0 - 30, cap, 12.5, INK2))
        b.append(rect(x0, y0, side, side, "#f6f6f4", GRID))
        if kband:   # the four 16-wide bands the loop walks through, one per step k
            for kk in range(4):
                if kk:
                    b.append(line(x0 + kk * 16 * px, y0 - 4, x0 + kk * 16 * px, y0 + side, MUTED, 1, "3 3"))
                b.append(text(x0 + (kk + 0.5) * 16 * px, y0 - 9, f"k = {kk}", 11, MUTED, "middle"))
        for r in range(n):
            for c in range(n):
                if mine(r, c, keys):
                    b.append(rect(x0 + c * px, y0 + r * px, px, px, BLUE))
                elif name == "acc_S" and r in (0, 8):
                    b.append(rect(x0 + c * px, y0 + r * px, px, px, BLUE_L))
        b.append(text(x0, y0 + side + 24, code, 12.5, INK, family=MONO))
        b.append(text(x0, y0 + side + 43, note, 12.5, INK2))
    # S panel: say who holds the light blue
    sx = xs[2]
    b.append(line(sx + side + 4, y0 + 2, sx + side + 14, y0 + 2, MUTED))
    b.append(line(sx + side + 4, y0 + 8 * px + 2, sx + side + 14, y0 + 8 * px + 2, MUTED))
    b.append(line(sx + side + 14, y0 + 2, sx + side + 14, y0 + 8 * px + 2, MUTED))
    b.append(text(sx + side + 18, y0 + 4 * px + 2, "light: lanes 1–3", 11, INK2))
    b.append(text(sx + side + 18, y0 + 4 * px + 16, "(same quad)", 11, INK2))
    # bottom: the fragments and the loop
    by = y0 + side + 82
    b.append(line(20, by - 20, W - 20, by - 20, GRID))
    b.append(text(20, by, "make_fragment_A/B", 12.5, INK, family=MONO))
    b.append(text(200, by, "allocate registers for the blue values of Q and K: 32 and 128 values.", 12.5, INK2))
    b.append(text(20, by + 22, "loop step k", 12.5, INK, family=MONO))
    b.append(text(200, by + 22, "[None, None, k] picks band k: copy those Q and K values into registers, "
                                "then gemm adds Q·Kᵀ into acc_S.", 12.5, INK2))
    return svg(W, by + 40, b)


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
    cw, ch = 46, 28
    C_A, C_B, C_C = "#2a78d6", "#c4501f", "#0f7f58"     # text colours: first row counter, second row counter, column
    tsp = lambda s, c: f'<tspan fill="{c}" font-weight="700">{s}</tspan>'

    def cell(x, y, s, fill=SURF, w=cw):
        b.append(rect(x, y, w, ch, fill, GRID, 1))
        b.append(text(x + w / 2, y + 18, s, 12, INK, "middle"))

    def counters(x, y, rows):
        # a small table: counter | values it takes | what one step adds to the offset
        b.append(text(x, y, "counter", 11.5, MUTED))
        b.append(text(x + 120, y, "counts", 11.5, MUTED))
        b.append(text(x + 230, y, "each step adds", 11.5, MUTED))
        b.append(line(x, y + 7, x + 340, y + 7, GRID))
        for k, (name, color, vals, step) in enumerate(rows):
            yy = y + 26 + 22 * k
            b.append(text(x, yy, name, 12.5, color, weight="600"))
            b.append(text(x + 120, yy, vals, 12.5, INK))
            b.append(text(x + 230, yy, step, 12.5, color, weight="600", family=MONO))

    b.append(text(20, 26, "Every number in a grid is an offset: where that element sits in memory, counting from 0.", 13, INK2))
    b.append(text(20, 46, "Before the colon is the shape (how many of each); after it is the stride (how far one step moves). "
                  "Read them in pairs, by position.", 13, INK2))

    # ---- example 1: (64, 64):(64, 1), an ordinary row-major table
    y = 92
    b.append(text(20, y, "Example 1", 15, weight="600"))
    b.append(text(110, y, f"({tsp(64, C_A)}, {tsp(64, C_C)}):({tsp(64, C_A)}, {tsp(1, C_C)})", 15, INK, family=MONO))
    b.append(text(20, y + 22, "An ordinary 64 × 64 table, stored one row after another.", 12.5, INK2))
    counters(20, y + 56, [("r  (row)", C_A, "0 to 63", "+64"), ("c  (column)", C_C, "0 to 63", "+1")])
    b.append(text(20, y + 136, f"offset = {tsp('64', C_A)}·r + {tsp('1', C_C)}·c", 13, INK, family=MONO))

    gx, gy = 520, y + 10
    cols, rows = [0, 1, 2, 3, None, 63], [0, 1, 2, None, 63]
    for j, c in enumerate(cols):
        b.append(text(gx + j * cw + cw / 2, gy - 8, "…" if c is None else f"c={c}", 11.5, C_C, "middle"))
    for i, r in enumerate(rows):
        yy = gy + i * ch
        b.append(text(gx - 8, yy + 18, "⋮" if r is None else f"r={r}", 11.5, C_A, "end"))
        for j, c in enumerate(cols):
            if r is None or c is None:
                s = "⋮" if r is None and c is not None else "…" if c is None and r is not None else ""
                b.append(text(gx + j * cw + cw / 2, yy + 18, s, 12, MUTED, "middle"))
            else:
                cell(gx + j * cw, yy, str(64 * r + c))
    b.append(rect(gx + 3 * cw, gy + 2 * ch, cw, ch, "none", INK, 2.5))
    b.append(text(gx, gy + 5 * ch + 22, f"Outlined: row 2, column 3  →  {tsp(64, C_A)}·2 + {tsp(1, C_C)}·3 = 131", 12.5, INK))

    b.append(line(20, 296, 900, 296, GRID))

    # ---- example 2: ((2, 2), 8):((1, 16), 2), the row number is split into two counters
    y = 332
    b.append(text(20, y, "Example 2", 15, weight="600"))
    b.append(text(110, y, f"(({tsp(2, C_A)}, {tsp(2, C_B)}), {tsp(8, C_C)}):(({tsp(1, C_A)}, {tsp(16, C_B)}), {tsp(2, C_C)})",
                  15, INK, family=MONO))
    b.append(text(20, y + 22, "4 rows × 8 columns. The inner (2, 2) means the row number is", 12.5, INK2))
    b.append(text(20, y + 40, "counted by two counters, i0 and i1, each 0 or 1, like the two", 12.5, INK2))
    b.append(text(20, y + 58, "digits of a number: row = i0 + 2·i1. The inner (1, 16) gives", 12.5, INK2))
    b.append(text(20, y + 76, "each counter its own step.", 12.5, INK2))
    counters(20, y + 110, [("i0  (row, inner)", C_A, "0, 1", "+1"), ("i1  (row, outer)", C_B, "0, 1", "+16"),
                          ("c  (column)", C_C, "0 to 7", "+2")])
    b.append(text(20, y + 212, f"offset = {tsp('1', C_A)}·i0 + {tsp('16', C_B)}·i1 + {tsp('2', C_C)}·c", 13, INK, family=MONO))

    gx, gy, lw = 548, y + 46, 30
    b.append(text(gx - 1.5 * lw, gy - 8, "i0", 11.5, C_A, "middle", "600"))
    b.append(text(gx - 0.5 * lw, gy - 8, "i1", 11.5, C_B, "middle", "600"))
    for c in range(8):
        b.append(text(gx + c * 44 + 22, gy - 8, f"c={c}", 11.5, C_C, "middle"))
    for r in range(4):
        i0, i1 = r % 2, r // 2
        yy = gy + r * ch
        b.append(text(gx - 2 * lw - 12, yy + 18, f"row {r}", 12, INK, "end"))
        b.append(text(gx - 1.5 * lw, yy + 18, str(i0), 12, C_A, "middle", "600"))
        b.append(text(gx - 0.5 * lw, yy + 18, str(i1), 12, C_B, "middle", "600"))
        for c in range(8):
            cell(gx + c * 44, yy, str(i0 + 16 * i1 + 2 * c), ORANGE_L if i1 else SURF, 44)
    b.append(rect(gx + 2 * 44, gy + 3 * ch, 44, ch, "none", INK, 2.5))
    b.append(text(gx - 2 * lw - 20, gy + 4 * ch + 22,
                  f"Outlined: row 3, column 2  →  i0 = 1, i1 = 1, c = 2", 12.5, INK))
    b.append(text(gx - 2 * lw - 20, gy + 4 * ch + 40,
                  f"offset = {tsp(1, C_A)}·1 + {tsp(16, C_B)}·1 + {tsp(2, C_C)}·2 = 21", 12.5, INK))
    b.append(text(gx - 2 * lw - 20, gy + 4 * ch + 72, "Rows 0 and 1 interleave (even and odd offsets 0–15);", 12.5, INK2))
    b.append(text(gx - 2 * lw - 20, gy + 4 * ch + 90, "the shaded rows 2 and 3 do the same, 16 further on.", 12.5, INK2))
    b.append(text(gx - 2 * lw - 20, gy + 4 * ch + 108, "A single row stride could not describe this.", 12.5, INK2))
    return svg(920, gy + 4 * ch + 126, b)


def tiled_mma():
    """How the three arguments of make_tiled_mma build up, from one instruction to the 64 x 64 x 64 tile.
    Each panel draws S = Q Kt as a matrix multiply: Kt above S, Q to its left, so M, N and K are all visible."""
    b = []
    W, P = 160, 2.5                             # 64 elements -> 160 px
    FAINT, GREY_L = "#ecebe7", "#f6f6f4"
    b.append(text(20, 30, "What each argument of make_tiled_mma covers of S = Q Kᵀ", 16, weight="600"))
    b.append(text(20, 50, "M = query rows of S, N = keys (columns of S), K = head dims summed over. "
                  "Each panel adds one argument to the one before.", 12.5, INK2))

    def rot(x, y, s, size=11, fill=MUTED):
        return (f'<text transform="translate({x},{y}) rotate(-90)" font-size="{size}" fill="{fill}" '
                f'text-anchor="middle">{s}</text>')

    def panel(px0, py0, title, code, caption, table, draw):
        b.append(text(px0, py0, title, 14, weight="600"))
        is_code = "(" in code
        b.append(text(px0, py0 + 20, code, 12.5, INK if is_code else INK2, family=MONO if is_code else None))
        for i, c in enumerate(caption):
            b.append(text(px0, py0 + 40 + 16 * i, c, 12.5, INK2))
        qx, ktx, kty = px0 + 40, px0 + 40 + W + 12, py0 + 88
        sy = kty + W + 12
        mats = {"Q": (qx, sy, 16, 16), "Kt": (ktx, kty, 16, 8), "S": (ktx, sy, 16, 8)}   # origin, grid rows/cols
        for name, (x, y, gr, gc) in mats.items():
            b.append(rect(x, y, W, W, SURF, GRID, 1))
            for k in range(gr, 64, gr):
                b.append(line(x, y + k * P, x + W, y + k * P, FAINT, 1))
            for k in range(gc, 64, gc):
                b.append(line(x + k * P, y, x + k * P, y + W, FAINT, 1))

        def hl(name, r0, c0, nr, nc, fill, stroke, sw=1.5):
            x, y, _, _ = mats[name]
            b.append(rect(x + c0 * P, y + r0 * P, nc * P, nr * P, fill, stroke, sw))

        def note(name, r, c, s, fill=INK2):  # text at element (r, c) of a matrix
            x, y, _, _ = mats[name]
            b.append(text(x + c * P, y + r * P, s, 11, fill))

        draw(hl, note)
        for name, (x, y, _, _) in mats.items():
            b.append(text(x + W - 6, y + W - 7, "Kᵀ" if name == "Kt" else name, 13, MUTED, "end", "600"))
        b.append(text(ktx + W / 2, kty - 7, "keys 0–63 (N)", 11, MUTED, "middle"))
        b.append(rot(ktx + W + 14, kty + W / 2, "head dims (K)"))
        b.append(rot(qx - 10, sy + W / 2, "query rows 0–63 (M)"))
        b.append(text(qx + W / 2, sy + W + 16, "head dims 0–63 (K)", 11, MUTED, "middle"))
        b.append(text(ktx + W / 2, sy + W + 16, "keys (N)", 11, MUTED, "middle"))
        for i, (letter, expr, col, wgt) in enumerate(table):
            y = kty + 22 + 22 * i
            if letter:
                b.append(text(px0, y, letter, 13.5, col, weight="600"))
            b.append(text(px0 + 22, y, expr, 13, col, weight=wgt))

    L, R, T1, T2 = 20, 520, 92, 552

    # 1. the instruction
    def d1(hl, note):
        hl("Q", 0, 0, 16, 16, BLUE_L, BLUE)
        hl("Kt", 0, 0, 16, 8, BLUE_L, BLUE)
        hl("S", 0, 0, 16, 8, BLUE_L, BLUE)
        note("S", 10, 11, "warp 0")
    panel(L, T1, "1. The instruction", "MmaF16BF16Op(..., (16, 8, 16))",
          ["One warp runs one mma: a 16 × 8 piece of S. It sums only 16",
           "of the 64 head dims: 16 × 16 of Q times 16 × 8 of Kᵀ."],
          [("M", "16", INK, "normal"), ("N", "8", INK, "normal"), ("K", "16", INK, "normal"),
           ("", "(M, N, K) = (16, 8, 16)", INK2, "normal")], d1)

    # 2. four warps along M
    def d2(hl, note):
        for w in range(4):
            hl("Q", 16 * w, 0, 16, 16, BLUE_L, BLUE)
            hl("S", 16 * w, 0, 16, 8, BLUE_L, BLUE)
            note("S", 16 * w + 10, 11, f"warp {w}")
        hl("Kt", 0, 0, 16, 8, BLUE_L, BLUE)
        note("Kt", 7, 11, "the same piece")
        note("Kt", 13, 11, "for all 4 warps")
    panel(R, T1, "2. The warps", "make_tiled_mma(op, (4, 1, 1), ...)",
          ["Warps along (M, N, K): 4 × 1 × 1. The 4 warps stack along M,",
           "each doing step 1 on its own 16 rows: a 64 × 8 strip of S."],
          [("M", "16 × 4 warps = 64", BLUE, "600"), ("N", "8 × 1 = 8", INK, "normal"),
           ("K", "16 × 1 = 16", INK, "normal"), ("", "default step: (64, 8, 16)", INK2, "normal")], d2)

    # 3. permutation_mnk widens N to 16: pairs
    def d3(hl, note):
        for w in range(4):
            hl("Q", 16 * w, 0, 16, 16, BLUE_L, BLUE)
            hl("S", 16 * w, 0, 16, 8, ORANGE_L, ORANGE)
            hl("S", 16 * w, 8, 16, 8, ORANGE_L, ORANGE)
            note("S", 16 * w + 10, 19, f"warp {w}: 2 mma")
        for i in range(2):
            for j in range(2):
                hl("Kt", 8 * i, 8 * j, 8, 8, ORANGE_L, ORANGE, 1)
        hl("Kt", 0, 0, 16, 16, "none", ORANGE, 2.5)
        note("Kt", 7, 19, "four 8 × 8 tiles =")
        note("Kt", 13, 19, "one ldmatrix.x4")
    panel(L, T2, "3. The step", "permutation_mnk=(64, 16, 16)",
          ["Sets the step to 64 rows × 16 keys × 16 dims. Only N changes:",
           "each warp now runs 2 mmas side by side that share one load of K."],
          [("M", "64  (as in step 2)", INK, "normal"), ("N", "8 × 2 mma = 16", ORANGE, "600"),
           ("K", "16  (as in step 2)", INK, "normal"), ("", "step: (64, 16, 16)", INK2, "normal")], d3)

    # 4. the step repeated over the kernel's 64 x 64 x 64 tile
    def d4(hl, note):
        for r in range(0, 64, 16):
            for c in range(0, 64, 16):
                hl("Q", r, c, 16, 16, GREY_L, INK2, 1)
                hl("S", r, c, 16, 16, GREY_L, INK2, 1)
                hl("Kt", r, c, 16, 16, GREY_L, INK2, 1)
        hl("Q", 0, 0, 64, 16, ORANGE_L, ORANGE, 2.5)
        hl("Kt", 0, 0, 16, 16, ORANGE_L, ORANGE, 2.5)
        hl("S", 0, 0, 64, 16, ORANGE_L, ORANGE, 2.5)
    panel(R, T2, "4. The whole tile", "no argument: the kernel's tile is 64 rows × 64 keys × 64 dims",
          ["CuTe repeats the step (orange) 4 times along keys and 4 times",
           "along head dims, adding each into the same piece of S."],
          [("M", "64 ÷ 64 = 1 step", INK, "normal"), ("N", "64 ÷ 16 = 4 steps", INK, "normal"),
           ("K", "64 ÷ 16 = 4 steps", INK, "normal"), ("", "16 steps × 2 mma", INK2, "normal"),
           ("", "= 32 mma per warp", INK, "600")], d4)
    return svg(1010, T2 + 88 + 2 * W + 12 + 36, b)

def ldmatrix_pairs():
    """Drawn as K transposed (head dims down, keys across), like the tiled_mma figure above it,
    so each MMA's 8 keys are a column."""
    b = []
    t = 56                                      # one 8 x 8 tile
    b.append(text(20, 30, "Loading the 16 × 16 piece of Kᵀ a pair needs into registers", 15, weight="600"))

    def piece(x, y, title, sub):
        b.append(text(x, y - 52, title, 13.5, weight="600"))
        b.append(text(x, y - 34, sub, 12, INK2))
        b.append(text(x + t / 2, y - 8, "keys 0–7", 11, MUTED, "middle"))
        b.append(text(x + 1.5 * t, y - 8, "keys 8–15", 11, MUTED, "middle"))
        b.append(text(x - 8, y + t / 2 + 4, "dims 0–7", 11, MUTED, "end"))
        b.append(text(x - 8, y + 1.5 * t + 4, "dims 8–15", 11, MUTED, "end"))
        for i in range(2):
            for j in range(2):
                b.append(rect(x + j * t, y + i * t, t, t, "#f6f6f4", GRID, 1))
                b.append(text(x + j * t + t / 2, y + i * t + t / 2 + 4, "8 × 8", 11, INK2, "middle"))

    # default: two loads of two tiles, one per MMA (one column of 8 keys each)
    x, y = 110, 118
    piece(x, y, "Default (step of 8 keys)", "each MMA gets its own load")
    b.append(rect(x + 3, y + 3, t - 6, 2 * t - 6, "none", ORANGE, 2.5, 4))
    b.append(rect(x + t + 3, y + 3, t - 6, 2 * t - 6, "none", AQUA, 2.5, 4))
    b.append(text(x + 2 * t + 14, y + t / 2 + 4, "load 1 → MMA 1 (keys 0–7)", 12, ORANGE))
    b.append(text(x + 2 * t + 14, y + 1.5 * t + 4, "load 2 → MMA 2 (keys 8–15)", 12, AQUA))
    b.append(text(x, y + 2 * t + 26, "2 load instructions", 13, INK, weight="600"))

    # (64, 16, 16): one load of four tiles, shared by the pair
    x = 470
    piece(x, y, "permutation_mnk = (64, 16, 16)", "the pair shares one load")
    b.append(rect(x + 3, y + 3, 2 * t - 6, 2 * t - 6, "none", BLUE, 2.5, 4))
    b.append(text(x + 2 * t + 14, y + t + 4, "1 load → MMA 1 and 2", 12, BLUE))
    b.append(text(x, y + 2 * t + 26, "1 load instruction", 13, INK, weight="600"))

    b.append(text(20, y + 2 * t + 64, "Drawn as Kᵀ, as in the figure above: each MMA's 8 keys are a column. In shared memory K is stored with keys as rows,", 12.5, INK2))
    b.append(text(20, y + 2 * t + 82, "so there each MMA's piece is a row of two tiles instead; the tiles and the loads are the same.", 12.5, INK2))
    b.append(text(20, y + 2 * t + 108, "One load instruction (ldmatrix) can carry up to four 8 × 8 tiles; with a step of 8 keys it only ever carries two.", 12.5, INK2))
    b.append(text(20, y + 2 * t + 126, "For the whole 64 × 64 tile of K: 32 loads per warp instead of 16; the same for V.", 12.5, INK2))
    return svg(800, y + 2 * t + 142, b)


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


def warp_split():
    # Where row 0 of S lives under (4, 1, 1) and (2, 2, 1), and what its maximum costs. Ownership checked on
    # the GPU: under (2, 2, 1) warps 0 and 2 take turns every 8 keys of rows 0-15 and 32-47.
    b = [f'<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>']
    b.append(text(20, 30, "Where row 0 of S lives, and what its maximum costs", 16, weight="600"))
    fills = [BLUE_L, ORANGE_L, AQUA_L, "#ece6f6"]
    strong = [BLUE, ORANGE, AQUA, "#8a6fc4"]
    px, y0 = 4, 112

    def arrow(x1, y1, x2, y2):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.3" '
                 f'marker-end="url(#ah2)"/>')

    def box(x, y, w, s, fill):
        b.append(rect(x, y, w, 30, fill, GRID, 1, 4))
        b.append(text(x + w / 2, y + 19.5, s, 12, INK, "middle"))

    def grid(x0, owner, head, sub):
        b.append(text(x0, 64, head, 14, INK, weight="600", family=MONO))
        b.append(text(x0 + 110, 64, sub, 13, INK2))
        b.append(text(x0, 84, "64 × 64 tile of S, coloured by the warp that holds each score", 12, MUTED))
        for br in range(8):
            for bc in range(8):
                w = owner(br, bc)
                b.append(rect(x0 + bc * 32, y0 + br * 32, 32, 32, fills[w], SURF, 1))
                b.append(text(x0 + bc * 32 + 16, y0 + br * 32 + 20, str(w), 11, MUTED, "middle"))
        for bc in range(8):   # row 0 itself, in the strong colour of the warp that holds each 8 keys
            b.append(rect(x0 + bc * 32, y0, 32, px, strong[owner(0, bc)]))
        b.append(text(x0 - 8, y0 + 5, "row 0 →", 11.5, INK, "end"))

    # left: what the kernel uses
    L, R, W = 70, 560, 260
    grid(L, lambda br, bc: br // 2, "(4, 1, 1)", "what the kernel uses")
    grid(R, lambda br, bc: (br // 2) % 2 + 2 * (bc % 2), "(2, 2, 1)", "the counterexample")
    ty = y0 + 256 + 26
    b.append(text(L, ty, "Row 0 is all in warp 0, in lanes 0–3.", 12.5, INK))
    b.append(text(R, ty, "Row 0 is split: warps 0 and 2 take turns every 8 keys.", 12.5, INK))

    fy = ty + 22
    box(L, fy, W, "1. two shuffles inside warp 0", BLUE_L)
    b.append(text(L, fy + 56, "Done: registers only, no waiting.", 12.5, BLUE, weight="600"))

    hw = (W - 10) / 2
    box(R, fy, hw, "1. shuffles in warp 0", BLUE_L)
    box(R + hw + 10, fy, hw, "1. shuffles in warp 2", BLUE_L)
    steps = ["2. both write their max to shared memory", "3. barrier: all 128 threads wait",
             "4. each reads the other's max"]
    for i, st in enumerate(steps):
        yy = fy + 48 + i * 48
        if i == 0:
            arrow(R + hw / 2, fy + 32, R + hw / 2, yy - 3)
            arrow(R + hw * 1.5 + 10, fy + 32, R + hw * 1.5 + 10, yy - 3)
        else:
            arrow(R + W / 2, yy - 16, R + W / 2, yy - 3)
        box(R, yy, W, st, ORANGE_L)
    b.append(text(R, fy + 48 * 3 + 54, "Steps 2–4 again on every key block.", 12.5, ORANGE, weight="600"))
    return svg(920, fy + 48 * 3 + 72, b)


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


def block_grid():
    """How the 2,048 blocks cover the work at N = 4096: 32 (sequence, head) problems x 64 slices of Q."""
    b = []
    GREY_L = "#f6f6f4"
    b.append(f'<defs><marker id="ah5" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')
    b.append(text(20, 30, "How 2,048 blocks cover the work (N = 4,096 rows, 4 sequences × 8 heads)", 16, weight="600"))
    top, bot = 116, 332

    # 1. 32 separate problems
    b.append(text(20, 66, "1. 32 separate problems", 13.5, weight="600"))
    b.append(text(20, 84, "each (sequence, head) pair has its own Q, K, V", 11.5, INK2))
    x0, cw, ch = 64, 30, 48
    for h in range(8):
        b.append(text(x0 + h * 32 + cw / 2, top - 8, f"h{h}", 10.5, MUTED, "middle"))
    for sq in range(4):
        y = top + sq * 56
        b.append(text(x0 - 8, y + 28, f"seq {sq}", 10.5, MUTED, "end"))
        for h in range(8):
            hit = (sq, h) == (0, 7)
            b.append(rect(x0 + h * 32, y, cw, ch, BLUE_L if hit else GREY_L, BLUE if hit else GRID, 1.5 if hit else 1, 3))

    # 2. one problem's Q, cut into 64 slices
    qx, qw = 400, 90
    b.append(text(380, 66, "2. One problem's Q, cut into 64 slices", 13.5, weight="600"))
    b.append(text(380, 84, "sequence 0, head 7: 4,096 rows × 64 values", 11.5, INK2))
    b.append(line(x0 + 7 * 32 + cw, top, qx, top, MUTED, 1, "4 3"))
    b.append(line(x0 + 7 * 32 + cw, top + ch, qx, bot, MUTED, 1, "4 3"))
    b.append(rect(qx, top, qw, bot - top, BLUE_L, BLUE, 1.5))
    sh = 24
    slices = [(top, "block 0: rows 0–63"), (top + sh, "block 1: rows 64–127"),
              (top + 2 * sh, "block 2: rows 128–191"), (bot - sh, "block 63: rows 4032–4095")]
    for i, (y, lab) in enumerate(slices):
        hit = i == 1
        b.append(rect(qx, y, qw, sh, ORANGE_L if hit else SURF, ORANGE if hit else BLUE, 2 if hit else 1))
        b.append(text(qx + qw + 10, y + 16, lab, 11.5, ORANGE if hit else INK2, weight="600" if hit else "normal"))
    b.append(text(qx + qw / 2, (top + 3 * sh + bot - sh) / 2 + 6, "⋮", 18, INK2, "middle"))
    b.append(text(qx + qw + 10, (top + 3 * sh + bot - sh) / 2 + 4, "one block per 64 rows", 11.5, MUTED))

    # 3. what block 1 reads and writes
    X = 700
    b.append(text(X, 66, "3. What block 1 reads and writes", 13.5, weight="600"))
    b.append(text(X, 84, "only head 7 of sequence 0, nothing else", 11.5, INK2))
    b.append(line(qx + qw + 150, top + sh + sh / 2, X - 6, top + 30, ORANGE, 1, "4 3"))
    b.append(rect(X, top + 16, 64, 28, ORANGE_L, ORANGE, 2))
    b.append(text(X + 32, top + 34, "Q", 12, INK, "middle", "600"))
    b.append(text(X, top + 62, "reads its own", 11, INK2))
    b.append(text(X, top + 76, "64 rows of Q", 11, INK2))
    b.append(rect(X, bot - 72, 64, 28, ORANGE_L, ORANGE, 2))
    b.append(text(X + 32, bot - 54, "O", 12, INK, "middle", "600"))
    b.append(text(X, bot - 26, "writes its own", 11, INK2))
    b.append(text(X, bot - 12, "64 rows of O", 11, INK2))
    for i, name in enumerate(("K", "V")):
        kx = 800 + i * 66
        b.append(text(kx + 25, top - 8, name, 12, INK, "middle", "600"))
        b.append(rect(kx, top, 50, bot - top, GREY_L, INK2, 1))
        for j in range(1, 9):
            b.append(line(kx, top + j * 24, kx + 50, top + j * 24, GRID, 1))
        b.append(rect(kx, top + 24, 50, 24, AQUA_L, AQUA, 2))
    ax = 940
    b.append(f'<line x1="{ax}" y1="{top + 4}" x2="{ax}" y2="{bot - 4}" stroke="{INK2}" stroke-width="1.4" marker-end="url(#ah5)"/>')
    b.append(text(ax + 8, top + 90, "reads all", 11, INK2))
    b.append(text(ax + 8, top + 105, "4,096 rows,", 11, INK2))
    b.append(text(ax + 8, top + 120, "64 at a time", 11, INK2))
    b.append(text(ax + 8, top + 135, "(green: the", 11, MUTED))
    b.append(text(ax + 8, top + 150, "current step)", 11, MUTED))

    b.append(text(20, bot + 34, "64 slices per problem × 32 problems = 2,048 blocks. Different blocks of the same head share its K and V; "
                  "blocks of different heads share nothing.", 12.5, INK2))
    b.append(text(20, bot + 54, "Why share? A query row gets one score per key of its head (softmax turns the scores into weights). "
                  "Its output adds up every value", 12.5, INK2))
    b.append(text(20, bot + 72, "of that head, each multiplied by its key's weight. So all 64 blocks read the head's whole K and V "
                  "(causal: only up to their rows).", 12.5, INK2))
    return svg(1030, bot + 88, b)


def warp_lanes():
    """A warp's 32 lanes in 8 quads, which lanes hold which row of an MMA result, and how warps make a block."""
    b = []
    GREY_L = "#f6f6f4"
    b.append(f'<defs><marker id="ah6" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')
    b.append(text(20, 30, "A warp, its lanes and its quads", 16, weight="600"))
    y0 = 104

    # 1. one warp: 32 lanes, 8 quads
    b.append(text(20, 62, "1. One warp = 32 lanes (threads)", 13.5, weight="600"))
    b.append(text(20, 80, "all 32 run the same instruction at the same time", 11.5, INK2))
    x0, cw, ch = 84, 58, 24
    for q in range(8):
        y = y0 + q * 28
        hit = q == 0
        b.append(text(x0 - 10, y + 16, f"quad {q}", 11, BLUE if hit else MUTED, "end", "600" if hit else "normal"))
        for c in range(4):
            b.append(rect(x0 + c * (cw + 4), y, cw, ch, BLUE_L if hit else GREY_L, BLUE if hit else GRID, 1.5 if hit else 1, 4))
            b.append(text(x0 + c * (cw + 4) + cw / 2, y + 16, f"lane {4 * q + c}", 10.5, INK if hit else INK2, "middle"))
    b.append(text(20, y0 + 8 * 28 + 18, "A quad = 4 neighbouring lanes. Not a piece of hardware,", 11.5, INK2))
    b.append(text(20, y0 + 8 * 28 + 34, "just how the multiply instruction hands out its results.", 11.5, INK2))

    # 2. who holds which value of one 16 x 8 MMA result
    X = 400
    b.append(text(X, 62, "2. Who holds each value of one multiply's 16 × 8 result", 13.5, weight="600"))
    b.append(text(X, 80, "rows 0–7 shown; rows 8–15 go to the same lanes again", 11.5, INK2))
    gx, gw = X + 46, 50
    for c in range(8):
        b.append(text(gx + c * gw + gw / 2, y0 - 6, f"col {c}", 10, MUTED, "middle"))
    for r in range(8):
        y = y0 + r * 28
        hit = r == 0
        b.append(text(gx - 8, y + 16, f"row {r}", 11, BLUE if hit else MUTED, "end", "600" if hit else "normal"))
        for c in range(8):
            lane = 4 * r + c // 2
            b.append(rect(gx + c * gw, y, gw - 2, ch, BLUE if hit else GREY_L, SURF if hit else GRID, 1, 3))
            b.append(text(gx + c * gw + gw / 2 - 1, y + 16, f"lane {lane}", 9.5, SURF if hit else INK2, "middle"))
    b.append(text(X, y0 + 8 * 28 + 18, "Row 0 is held only by lanes 0–3 (quad 0), 2 values each. So a row's max or sum", 11.5, INK2))
    b.append(text(X, y0 + 8 * 28 + 34, "needs just those 4 lanes to swap values (shuffles), not the whole warp.", 11.5, INK2))

    # 3. bigger: warps make a block, blocks run on SMs
    y = y0 + 8 * 28 + 70
    b.append(text(20, y, "3. Bigger units", 13.5, weight="600"))
    bx = 20
    b.append(rect(bx, y + 14, 4 * 74 + 14, 52, SURF, ORANGE, 1.5, 6))
    b.append(text(bx + 8, y + 30, "a block = 4 warps = 128 threads", 11, ORANGE, weight="600"))
    for w in range(4):
        b.append(rect(bx + 8 + w * 74, y + 36, 68, 22, GREY_L, GRID, 1, 4))
        b.append(text(bx + 8 + w * 74 + 34, y + 51, f"warp {w}", 11, INK2, "middle"))
    ax = bx + 4 * 74 + 14
    b.append(f'<line x1="{ax + 6}" y1="{y + 40}" x2="{ax + 40}" y2="{y + 40}" stroke="{INK2}" stroke-width="1.4" marker-end="url(#ah6)"/>')
    b.append(text(ax + 50, y + 36, "runs on one SM (streaming multiprocessor), next to other blocks.", 11.5, INK2))
    b.append(text(ax + 50, y + 52, "This GPU has 56 SMs, each with its own registers, shared memory and tensor cores.", 11.5, INK2))
    return svg(870, y + 80, b)


def ldmatrix_x4():
    """One ldmatrix.x4: a 16 x 16 piece of the K tile, its four 8 x 8 tiles, who receives each value."""
    b = []
    YEL_L = "#fdf3d8"
    TILE = [(BLUE_L, BLUE), (ORANGE_L, ORANGE), (AQUA_L, AQUA), (YEL_L, "#c9a227")]
    GREY_L = "#f6f6f4"
    b.append(text(20, 30, "One ldmatrix.x4: four 8 × 8 tiles from shared memory, handed out to the warp's 32 lanes", 16, weight="600"))
    y0 = 116

    # 1. the K tile in shared memory, one 16 x 16 piece of it, and that piece's four 8 x 8 tiles
    b.append(text(20, 64, "1. In shared memory: the block's K tile", 13.5, weight="600"))
    b.append(text(20, 82, "64 keys × 64 head dims; one 16 × 16 piece = four 8 × 8 tiles", 11.5, INK2))
    kx, ps = 40, 32
    b.append(text(kx + 64, y0 - 8, "64 head dims", 10.5, MUTED, "middle"))
    b.append(f'<text x="{kx - 8}" y="{y0 + 64}" font-size="10.5" fill="{MUTED}" text-anchor="middle" '
             f'transform="rotate(-90 {kx - 8} {y0 + 64})">64 keys</text>')
    for r in range(4):
        for c in range(4):
            hit = (r, c) == (0, 3)
            b.append(rect(kx + c * ps, y0 + r * ps, ps, ps, BLUE_L if hit else GREY_L, BLUE if hit else GRID, 2 if hit else 1))
    zx, cs, ts = 196, 10, 80
    b.append(line(kx + 4 * ps, y0, zx, y0, MUTED, 1, "4 3"))
    b.append(line(kx + 4 * ps, y0 + ps, zx, y0 + 2 * ts + 6, MUTED, 1, "4 3"))
    for t in range(4):
        tx, ty = zx + (t % 2) * (ts + 6), y0 + (t // 2) * (ts + 6)
        fill, edge = TILE[t]
        b.append(rect(tx, ty, ts, ts, fill, edge, 1.5 if t == 0 else 1))
        for i in range(1, 8):
            b.append(line(tx + i * cs, ty, tx + i * cs, ty + ts, SURF, 1))
            b.append(line(tx, ty + i * cs, tx + ts, ty + i * cs, SURF, 1))
        b.append(rect(tx + 18, ty + 29, 44, 22, SURF, edge, 1, 4))
        b.append(text(tx + 40, ty + 44, f"tile {t}", 11, INK, "middle", "600"))
    b.append(text(zx + ts + 3, y0 + 2 * ts + 22, "16 keys × 16 head dims", 10.5, MUTED, "middle"))
    b.append(text(20, y0 + 2 * ts + 46, "16 × 16 is what a pair of multiplies needs from K", 11.5, INK2))
    b.append(text(20, y0 + 2 * ts + 62, "(each takes 8 keys × 16 head dims). One instruction", 11.5, INK2))
    b.append(text(20, y0 + 2 * ts + 78, "reads all four tiles; each value is 16 bits (bf16).", 11.5, INK2))

    # 2. tile 0 zoomed: which lane receives each value
    X = 400
    b.append(text(X, 64, "2. Tile 0, zoomed: which lane receives each value", 13.5, weight="600"))
    b.append(text(X, 82, "row r goes to quad r (lanes 4r to 4r + 3), 2 values each", 11.5, INK2))
    gx, cw = 452, 40
    for r in range(8):
        y = y0 + r * 26
        hit = r == 0
        b.append(text(gx - 8, y + 16, f"row {r}", 11, BLUE if hit else MUTED, "end", "600" if hit else "normal"))
        for c in range(8):
            lane = 4 * r + c // 2
            b.append(rect(gx + c * cw, y, cw - 2, 24, BLUE if hit else BLUE_L, SURF, 1, 3))
            b.append(text(gx + c * cw + cw / 2 - 1, y + 16, f"lane {lane}", 9, SURF if hit else INK2, "middle"))
    b.append(text(gx, y0 + 8 * 26 + 18, "64 values ÷ 32 lanes = 2 values per lane per tile.", 11.5, INK2))

    # 3. what lane 0 ends with
    X3 = 800
    b.append(text(X3, 64, "3. Lane 0 ends with", 13.5, weight="600"))
    b.append(text(X3, 82, "4 registers, 2 values in each", 11.5, INK2))
    for t in range(4):
        fill, edge = TILE[t]
        y = y0 + t * 40
        b.append(rect(X3, y, 200, 32, fill, edge, 1, 4))
        b.append(text(X3 + 10, y + 20, f"reg {t}: tile {t}, row 0, values 0–1", 11, INK))
    b.append(text(X3, y0 + 4 * 40 + 14, "Every other lane gets its own", 11.5, INK2))
    b.append(text(X3, y0 + 4 * 40 + 30, "2 values from each tile the", 11.5, INK2))
    b.append(text(X3, y0 + 4 * 40 + 46, "same way: 8 values in all.", 11.5, INK2))

    yb = y0 + 2 * ts + 112
    b.append(text(20, yb, "After one instruction, every lane holds exactly the values the multiply (MMA) wants from it, with no extra shuffling.", 12.5, INK2))
    b.append(text(20, yb + 18, "ldmatrix.trans does the same with each tile's rows and columns swapped, which is what V needs.", 12.5, INK2))
    return svg(1020, yb + 34, b)


def qkv_tiles():
    """Where the 64 x 64 tiles come from: one head's Q, K, V in global memory, the three tiles in shared memory,
    and the S and O tiles a step computes."""
    b = []
    GREY_L = "#f6f6f4"
    b.append(f'<defs><marker id="ah7" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')
    b.append(text(20, 30, "The 64 × 64 tiles: what one block takes from Q, K and V (one head, N = 4,096, D = 64)", 16, weight="600"))

    def arrow(x1, y1, x2, y2):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.4" marker-end="url(#ah7)"/>')

    # 1. global memory: one head's Q, K, V
    top, h, w = 124, 288, 56
    b.append(text(20, 64, "1. Global memory: one head", 13.5, weight="600"))
    b.append(text(20, 82, "4,096 rows × 64 head dims each (not to scale)", 11.5, INK2))
    for i, name in enumerate("QKV"):
        x = 36 + i * 84
        b.append(text(x + w / 2, top - 10, name, 13, INK, "middle", "600"))
        b.append(rect(x, top, w, h, GREY_L, INK2, 1))
        for j in range(1, 12):
            b.append(line(x, top + j * 24, x + w, top + j * 24, GRID, 1))
        if name == "Q":
            b.append(rect(x, top + 48, w, 24, ORANGE_L, ORANGE, 2))
        else:
            b.append(rect(x, top + 96, w, 24, AQUA_L, AQUA, 2))
    arrow(296, top + 4, 296, top + h - 4)
    b.append(f'<text x="312" y="{top + h / 2}" font-size="11" fill="{INK2}" text-anchor="middle" '
             f'transform="rotate(90 312 {top + h / 2})">64 steps, one per key block</text>')
    b.append(text(20, top + h + 22, "Orange: this block's 64 query rows.", 11.5, ORANGE))
    b.append(text(20, top + h + 38, "Green: the 64 keys of the current step.", 11.5, AQUA))
    b.append(text(20, top + h + 54, "The block steps down K and V, 64 rows at a time.", 11.5, INK2))

    # 2. shared memory: three 64 x 64 tiles
    X, ts = 350, 84
    b.append(text(X, 64, "2. Shared memory: 3 tiles = 24,576 bytes", 13.5, weight="600"))
    b.append(text(X, 82, "each 64 × 64 bf16 values = 8,192 bytes", 11.5, INK2))
    tiles = [("Q tile", "64 query rows × 64 dims", "loaded once", ORANGE_L, ORANGE),
             ("K tile", "64 keys × 64 dims", "replaced every step", AQUA_L, AQUA),
             ("V tile", "64 keys × 64 dims", "replaced every step", AQUA_L, AQUA)]
    for i, (name, dims, when, fill, edge) in enumerate(tiles):
        y = 104 + i * (ts + 22)
        b.append(rect(X, y, ts, ts, fill, edge, 1.5))
        if i == 0:
            for wv in range(1, 4):
                b.append(line(X, y + wv * ts / 4, X + ts, y + wv * ts / 4, edge, 1, "3 2"))
        b.append(text(X + ts + 12, y + 26, name, 12.5, INK, weight="600"))
        b.append(text(X + ts + 12, y + 44, dims, 11.5, INK2))
        b.append(text(X + ts + 12, y + 60, when, 11.5, MUTED))
    b.append(text(X, 104 + 3 * (ts + 22) + 8, "Q's dashed lines: the 4 warps' 16-row strips.", 11.5, INK2))

    # 3. registers: warp 0's share of one step, S = Q K^T then O += P V
    X3, ms = 660, 72
    b.append(text(X3, 64, "3. What each step computes (warp 0 shown)", 13.5, weight="600"))
    b.append(text(X3, 82, "warp 0 owns rows 0–15; results stay in its registers", 11.5, INK2))
    xs = [X3 + 10, X3 + 136, X3 + 262]

    def mat(x, y, name, sub, fill, edge, strip):
        b.append(text(x + ms / 2, y - 8, name, 12, INK, "middle", "600"))
        if strip:
            b.append(rect(x, y, ms, ms, GREY_L, GRID, 1))
            for wv in range(1, 4):
                b.append(line(x, y + wv * ms / 4, x + ms, y + wv * ms / 4, GRID, 1, "3 2"))
            b.append(rect(x, y, ms, ms / 4, fill, edge, 1.5))
        else:
            b.append(rect(x, y, ms, ms, fill, edge, 1.5))
        for i, part in enumerate(sub.split("|")):
            b.append(text(x + ms / 2, y + ms + 15 + 13 * i, part, 10.5, INK2, "middle"))

    def ops(y, o1, o2):
        b.append(text(xs[0] + ms + 27, y + ms / 2 + 6, o1, 18, INK2, "middle"))
        b.append(text(xs[1] + ms + 27, y + ms / 2 + 6, o2, 18, INK2, "middle"))

    r1 = 122
    mat(xs[0], r1, "Q tile", "rows 0–15|× 64 dims", ORANGE_L, ORANGE, True)
    mat(xs[1], r1, "Kᵀ (K tile, turned)", "64 dims|× all 64 keys", AQUA_L, AQUA, False)
    mat(xs[2], r1, "S = Q Kᵀ", "rows 0–15|× 64 keys", BLUE_L, BLUE, True)
    ops(r1, "×", "=")

    # S -> P: softmax along each row
    r2 = 300
    sx, px, yl = xs[2] + ms / 2, xs[0] + ms / 2, r1 + ms + 46
    b.append(f'<path d="M{sx},{r1 + ms + 34} L{sx},{yl} L{px},{yl} L{px},{r2 - 24}" fill="none" '
             f'stroke="{INK2}" stroke-width="1.4" marker-end="url(#ah7)"/>')
    b.append(rect((sx + px) / 2 - 92, yl - 10, 184, 20, SURF))
    b.append(text((sx + px) / 2, yl + 4, "P = softmax(S), along each row", 11.5, INK, "middle", "600"))

    mat(xs[0], r2, "P", "rows 0–15|× 64 keys", BLUE_L, BLUE, True)
    mat(xs[1], r2, "V tile", "all 64 keys|× 64 dims", AQUA_L, AQUA, False)
    mat(xs[2], r2, "O", "rows 0–15|× 64 dims", ORANGE_L, ORANGE, True)
    ops(r2, "×", "+=")

    notes = ["Each score in S says how well a query row matches a key.",
             "Softmax turns a row of scores into weights; O adds up",
             "the V rows with those weights. The other 3 warps do the",
             "same for their 16 rows at the same time, reading the",
             "same K and V tiles. O builds up over all 64 steps.",
             "The division by each row's sum happens once, at the end."]
    for i, t in enumerate(notes):
        b.append(text(X3, r2 + ms + 60 + 16 * i, t, 11.5, INK2))

    yb = top + h + 154
    b.append(text(20, yb, "\"64 query rows × 64 keys\" is the tile size: each block owns 64 rows of Q, and each step brings in 64 rows of K and V. "
                  "D = 64 makes every tile 64 × 64.", 12.5, INK2))
    return svg(1060, yb + 18, b)


def cp_async():
    """What cp.async does, in three parts: one copy (global -> shared, no registers), groups (start K and V
    together, wait only for K with commit_group / wait_group(n)), and the barrier that shares the tile."""
    b = []
    G, GREY_L, PAUSE = "#fdf3d8", "#f6f6f4", "#e4e3df"
    STEP = ["", ORANGE, BLUE, AQUA]
    b.append(f'<defs><marker id="ah8" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')
    b.append(text(20, 30, "cp.async: how one thread loads its part of the K and V tiles", 16, weight="600"))

    def arrow(x1, y1, x2, y2):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.5" marker-end="url(#ah8)"/>')

    def badge(x, y, n):
        b.append(f'<circle cx="{x}" cy="{y}" r="11" fill="{STEP[n]}"/>')
        b.append(text(x, y + 4.5, str(n), 12.5, SURF, "middle", "700"))

    def heading(y, n, s, subs):
        badge(31, y - 5, n)
        b.append(text(50, y, s, 14, weight="600"))
        for i, sub in enumerate(subs):
            b.append(text(50, y + 19 + 16 * i, sub, 12, INK2))

    def box(x, y, w, label, fill, edge=GRID, ink=INK, dash=False, h=30, size=12, family=None):
        d = ' stroke-dasharray="4 3"' if dash else ""
        b.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{edge}" stroke-width="1"{d} rx="6"/>')
        b.append(text(x + w / 2, y + h / 2 + 4.5, label, size, ink, "middle", family=family))

    # 0. the order in the code; the badge under each instruction says which part explains it
    y = 58
    b.append(text(20, y, "The order in the code. The number under each step is the part of this figure that explains it.", 12.5, INK2))
    items = [("cp.async × 4 (K)", 1), ("commit_group", 2), ("cp.async × 4 (V)", 1), ("commit_group", 2),
             ("wait_group(1)", 2), ("barrier", 3), ("S = Q Kᵀ", 0)]
    bw, gap, x0 = 114, 13, 22
    for i, (s, n) in enumerate(items):
        x = x0 + i * (bw + gap)
        box(x, y + 14, bw, s, BLUE_L if n == 0 else GREY_L, BLUE if n == 0 else GRID, size=10.5,
            family=MONO if s.startswith(("cp.", "commit", "wait")) else None)
        if i + 1 < len(items):
            arrow(x + bw + 2, y + 29, x + bw + gap - 2, y + 29)
        if n:
            badge(x + bw / 2, y + 62, n)
        else:
            b.append(text(x + bw / 2, y + 66, "uses K", 11, BLUE, "middle", "600"))

    # 1. one copy's path
    y = 162
    heading(y, 1, "One copy: cp.async goes straight to shared memory",
            ["The K tile is 8,192 bytes = 512 pieces of 16 bytes. Each of the block's 128 threads copies 4 pieces of K,",
             "and 4 of V, with one cp.async per piece."])
    bx = [190, 350, 510]
    for row, (name, skip, note1, note2) in enumerate([
            ("Normal copy", False, "The thread waits for the data,", "then stores it. No maths meanwhile."),
            ("cp.async", True, "The thread only starts the copy", "and moves on at once.")]):
        yy = y + 52 + row * 48
        b.append(text(50, yy + 20, name, 12.5, INK, weight="600"))
        box(bx[0], yy, 120, "global memory", G)
        box(bx[1], yy, 120, "registers", GREY_L if skip else BLUE_L, MUTED if skip else GRID, MUTED if skip else INK, skip)
        box(bx[2], yy, 120, "shared memory", AQUA_L)
        if skip:
            b.append(f'<path d="M{bx[0] + 60},{yy + 31} C{bx[0] + 100},{yy + 54} {bx[2] + 20},{yy + 54} {bx[2] + 58},{yy + 34}" '
                     f'fill="none" stroke="{AQUA}" stroke-width="2" marker-end="url(#ah8)"/>')
            b.append(text(bx[1] + 60, yy + 62, "straight to shared memory", 11, AQUA, "middle", "600"))
        else:
            arrow(bx[0] + 122, yy + 15, bx[1] - 4, yy + 15)
            arrow(bx[1] + 122, yy + 15, bx[2] - 4, yy + 15)
        b.append(text(670, yy + 12, note1, 12, INK2))
        b.append(text(670, yy + 28, note2, 12, INK2))

    # 2. groups: start K and V together, wait only for K
    y = 360
    heading(y, 2, "Groups: start K, then V straight after, and wait only for K",
            ["A thread starts its instructions one at a time, in order, but does not wait for one to finish before starting",
             "the next. So it starts K's copies, then V's right after, and both move at the same time. commit_group puts the",
             "copies started so far into one group. wait_group(n) pauses the thread until at most n of its groups are still moving."])
    cx = [50, 320, 420, 540]
    hy = y + 78
    for x, h in zip(cx, ["What the thread does", "K group", "V group", "What happens"]):
        b.append(text(x, hy, h, 11.5, MUTED, weight="600"))
    # each row is one moment; the K and V columns say where that group's 4 copies are at that moment
    rows = [
        ("start 4 K copies, commit_group", "moving", "not started", False,
         ["K's 4 copies become group 1.", "The thread does not wait."]),
        ("start 4 V copies, commit_group", "moving", "moving", False,
         ["V's 4 copies become group 2. Both groups", "now move at once; still no waiting."]),
        ("wait_group(1)", "moving", "moving", True,
         ["2 groups moving: it waits until only 1 is left.", "The THREAD pauses; K and V keep moving."]),
        ("wakes up: wait_group(1) is done", "landed", "moving", False,
         ["K lands, so only 1 group (V) is still moving.", "The thread goes on to the next instruction."]),
        ("barrier, then S = Q Kᵀ", "calculating", "landed", False,
         ["The barrier waits for threads, not copies, so V", "keeps moving, and lands while the thread uses K."]),
        ("wait_group(0), barrier, then O += P V", "done", "calculating", False,
         ["V has landed, so wait_group(0) lets it calculate", "with V. K is done; its space can take the next K."]),
    ]
    rh = 44
    for i, (does, k_state, v_state, paused, note) in enumerate(rows):
        ry = hy + 10 + i * rh
        b.append(rect(32, ry, 856, rh - 4, "#f0efec" if paused else SURF, GRID, 1, 6))
        if paused:
            b.append(rect(32, ry, 6, rh - 4, INK2, rx=2))
        b.append(text(cx[0], ry + 25, does, 12 if len(does) < 34 else 11, MUTED if does.startswith("(") else INK,
                      family=MONO if does.startswith(("start", "wait")) else None))
        for gx, st in ((cx[1], k_state), (cx[2], v_state)):
            if st == "moving":
                box(gx, ry + 7, 88, "moving", ORANGE, ORANGE, SURF, h=26, size=11)
            elif st == "calculating":
                box(gx, ry + 7, 88, "calculating", BLUE_L, BLUE, INK, h=26, size=11)
            elif st == "done":
                box(gx, ry + 7, 88, "done", GREY_L, MUTED, INK2, h=26, size=11)
            elif st == "landed":
                box(gx, ry + 7, 88, "landed ✓", SURF, AQUA, AQUA, h=26, size=11)
            else:
                box(gx, ry + 7, 88, "not started", SURF, GRID, MUTED, True, h=26, size=11)
        for k, t in enumerate(note):
            yk = ry + (25 if len(note) == 1 else 17 + 15 * k)
            b.append(text(cx[3], yk, t, 11.5, INK if paused else INK2, weight="600" if paused and k == 1 else "normal"))
    yy = hy + 10 + len(rows) * rh + 18
    b.append(text(50, yy, "K and V are the same size, so each moves for 3 steps; V is one step behind because it was started one step later.",
                  12, INK2))
    yy += 22
    b.append(text(50, yy, "n is how many of the newest groups may still be moving; every older group must be done. "
                  "So wait_group(1) means \"K done,", 12, INK2))
    b.append(text(50, yy + 16, "V may still be moving\". wait_group(0) would wait for every group. "
                  "So the group the thread needs first must be the", 12, INK2))
    b.append(text(50, yy + 32, "oldest one: that is why K is started first. The three timelines below compare the choices.",
                  12, INK2))

    # why two groups, and why K first: three small timelines
    y2 = yy + 66
    b.append(text(50, y2, "Why two groups, and why K first? Three ways to load K and V", 13, weight="600"))
    u = 19

    def timeline(x, y, title, call, mode, caption, best=False):
        b.append(text(x, y, title, 12, BLUE if best else INK, weight="600"))
        b.append(text(x, y + 16, call, 11, INK2, family=MONO))
        ry = y + 30
        b.append(text(x, ry + 17, "moving", 11, MUTED))
        b.append(text(x, ry + 51, "thread", 11, MUTED))
        tx = x + 52
        # K and V are the same size, so each takes the same time (4 units); the one started second starts and
        # lands 1.5 units later (it queues behind the first)
        dur, lag = 4, 1.5
        if mode == "one":
            b.append(rect(tx, ry, (dur + lag) * u - 2, 26, ORANGE, rx=4))
            b.append(text(tx + (dur + lag) / 2 * u, ry + 17.5, "K and V", 10.5, SURF, "middle"))
            land = dur + lag
        else:
            first, second = ("K", "V") if mode == "kfirst" else ("V", "K")
            land = dur if mode == "kfirst" else dur + lag
            for row, g in enumerate((first, second)):
                x0g = row * lag
                b.append(rect(tx + x0g * u, ry + 14 * row, dur * u - 2, 12, ORANGE, rx=3))
                end = x0g + dur
                lx = (land + end) / 2 if x0g < land < end else x0g + dur / 2   # keep the label off the dashed line
                b.append(text(tx + lx * u, ry + 14 * row + 10, g, 9.5, SURF, "middle", "600"))
        b.append(rect(tx, ry + 34, land * u - 2, 26, PAUSE, rx=4))
        b.append(text(tx + land * u / 2, ry + 51.5, "paused", 11, INK2, "middle"))
        b.append(rect(tx + land * u, ry + 34, 4 * u - 2, 26, BLUE_L, rx=4))
        b.append(text(tx + land * u + 2 * u, ry + 51.5, "S = Q Kᵀ", 10.5, INK, "middle"))
        b.append(line(tx + land * u - 1, ry - 4, tx + land * u - 1, ry + 64, INK, 1.5, "4 3"))
        for i, c in enumerate(caption):
            b.append(text(x, ry + 84 + 16 * i, c, 11.5, INK2))

    timeline(50, y2 + 26, "One group for K and V", "wait_group(0)", "one",
             ["S = Q Kᵀ waits for V too,", "although it does not use V."])
    timeline(340, y2 + 26, "V first, then K", "wait_group(0)", "vfirst",
             ["Now K is the newest group, so it", "lands last, and the only way to wait", "for it is wait_group(0). No better",
              "than one group."])
    timeline(630, y2 + 26, "K first, then V (as above)", "wait_group(1)", "kfirst",
             ["S = Q Kᵀ starts as soon as K", "has landed; V finishes while", "the thread computes."], best=True)
    b.append(text(50, y2 + 230, "K and V are the same size (8 KB per tile), so their bars are the same length. "
                  "The one started second lands a little later, because its", 11.5, INK2))
    b.append(text(50, y2 + 246, "copies queue behind the first one's. That gap is the time the K-first order saves: "
                  "small when memory is idle, larger when it is busy.", 11.5, INK2))

    # 3. waiting for everyone's copies
    y = y2 + 290
    heading(y, 3, "Barrier: wait for every thread's copies",
            ["wait_group only knows about this thread's own copies. The other 127 threads copied the rest of the tile."])
    ty, tw, sh = y + 44, 150, 22
    owners = ["thread 0", "thread 1", "thread 2", "thread 3"]

    def tile(x, mine_only, title, cap1, cap2):
        for i, who in enumerate(owners):
            ok = (not mine_only) or i == 0
            b.append(rect(x, ty + i * sh, tw, sh, AQUA_L if ok else GREY_L, AQUA if ok else GRID, 1))
            b.append(text(x + tw / 2, ty + i * sh + 15, f"{who}'s K pieces" if ok else f"{who}'s K pieces: ready?",
                          11, INK if ok else MUTED, "middle"))
        b.append(text(x + tw / 2, ty + 4 * sh + 20, title, 12, INK, "middle", "600"))
        b.append(text(x + tw / 2, ty + 4 * sh + 36, cap1, 11.5, INK2, "middle"))
        b.append(text(x + tw / 2, ty + 4 * sh + 52, cap2, 11.5, INK2, "middle"))

    tile(190, True, "After wait_group (thread 0's view)", "its own K pieces have landed;", "the others' may not have")
    bx0 = 470
    b.append(rect(bx0 - 4, ty - 8, 8, 4 * sh + 16, INK, rx=2))
    arrow(190 + tw + 10, ty + 2 * sh, bx0 - 10, ty + 2 * sh)
    arrow(bx0 + 10, ty + 2 * sh, 640 - 10, ty + 2 * sh)
    b.append(text(bx0, ty + 4 * sh + 20, "barrier (sync_threads)", 12, INK, "middle", "600"))
    b.append(text(bx0, ty + 4 * sh + 36, "every thread pauses here until", 11.5, INK2, "middle"))
    b.append(text(bx0, ty + 4 * sh + 52, "all 128 threads have arrived", 11.5, INK2, "middle"))
    tile(640, False, "After the barrier", "every thread can read", "the whole K tile")
    b.append(text(50, ty + 4 * sh + 82, "4 threads shown. Every thread did its own wait_group before reaching the barrier, "
                  "so all 512 pieces of K have landed.", 12, INK2))
    return svg(920, ty + 4 * sh + 98, b)


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
    (OUT / "warp_split.svg").write_text(warp_split())
    (OUT / "tiled_copy.svg").write_text(tiled_copy())
    (OUT / "copy_atoms.svg").write_text(copy_atoms())
    (OUT / "memory_hierarchy.svg").write_text(memory_hierarchy())
    (OUT / "block_grid.svg").write_text(block_grid())
    (OUT / "warp_lanes.svg").write_text(warp_lanes())
    (OUT / "ldmatrix_x4.svg").write_text(ldmatrix_x4())
    (OUT / "qkv_tiles.svg").write_text(qkv_tiles())
    (OUT / "cp_async.svg").write_text(cp_async())
    print("wrote", *sorted(p.name for p in OUT.glob("*.svg")))
