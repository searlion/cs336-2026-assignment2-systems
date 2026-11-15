# make_svgs.py
# Writes the twenty explanatory diagrams of the article as plain SVG (no external fonts):
#   images/mma_ownership.svg   thread 0 through the partition code and the k loop, line by line
#   images/bank_conflicts.svg  what a bank is, a conflict-free read, an 8-way conflict, and its cost
#   images/swizzle_banks.svg   which banks 8 rows of a tile land in, row-major vs swizzled
#   images/swizzle_bms.svg     Swizzle<3,3,3> step by step: slot, binary, XOR, why, the code, B/M/S
#   images/smem_layout.svg     every line of smem_layout: width, plain block, swizzle bits, atom, tiling
#   images/pipeline.svg        Round 4 line by line: v4's six steps, on a time line with sK / sV, against v3
#   images/gpu_hierarchy.svg   GPU -> SM -> block -> warp -> lane, and where each kind of memory lives
#   images/layouts.svg         the two example layouts, (64, 64):(64, 1) and ((2, 2), 8):((1, 16), 2)
#   images/mnk_shape.svg       (16, 8, 16) as (M, N, K): three 2-D matrices, and the 16 x 8 x 16 box of work
#   images/tiled_mma.svg       what (16, 8, 16), (4, 1, 1) and permutation (64, 16, 16) each cover of S = Q K^T
#   images/ldmatrix_pairs.svg  why permutation_mnk N = 16 halves the loads of K and V
#   images/warp_shuffle.svg    a quad's row maximum through shared memory against two warp shuffles
#   images/shuffle_args.svg    each argument of shuffle_sync_bfly, then both lines of quad_reduce traced on quad 0
#   images/warp_split.svg      where row 0 of S lives under (4, 1, 1) and (2, 2, 1), and its maximum's cost
#   images/tiled_copy.svg      every line of the copy of K into shared memory, followed for thread 19
#   images/copy_atoms.svg      the atom of a tiled copy: plain copies against cp.async and ldmatrix
#   images/memory_hierarchy.svg registers, L1 / shared memory, L2 and global memory, fastest to slowest
#   images/block_grid.svg      how 2,048 blocks cover 32 (sequence, head) problems of 4,096 query rows
#   images/warp_lanes.svg      block -> 4 warps -> warp 0's lanes and quads, who holds each row of an MMA result
#   images/ldmatrix_x4.svg     one ldmatrix.x4: a 16 x 16 piece of K, its four 8 x 8 tiles, who receives each value
#   images/qkv_tiles.svg       the 64 x 64 tiles: Q, K, V in global memory, shared memory, and S / O in registers
#   images/key_block_v0_1.svg  one key block of v0, lines 1-4 (barriers, copies of K and V), following thread 0
#   images/key_block_v0_2.svg  one key block of v0, lines 5-9 (the k loop, loads, multiply, the rest)
#   images/p_roundtrip.svg    Round 3 before/after: P through shared memory (v2) against as_a_operand (v3)
#   images/a_operand_view.svg as_a_operand line by line: thread 0's accumulator registers re-read as A operands
#   images/cp_async.svg        cp.async: global -> shared without registers, commit_group / wait_group(n), then a barrier
#   images/ldsm_count.svg    why 224 LDS (+ 64 PRMT) become 40 LDSM: one 16 x 16 piece, then the whole pass
#   images/ldsm_copies.svg     Round 2's ldmatrix copies, line by line: the atom, s2r_A / s2r_K / s2r_V, and one copy of Q
# The results chart is drawn by plot_results_svg.py from the benchmark files.
import pathlib

OUT = pathlib.Path(__file__).parent / "images"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace"
# One palette for every figure (and for plot_results_svg.py): warm neutrals plus four accents.
# Neutrals, darkest to lightest: text, secondary text, muted labels, strong lines, grid lines,
# mid-grey fill, light-grey fill, panel background, page.
INK, INK2, MUTED, DIM_LINE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#b9b8b2", "#d6d5d0"
GREY_M, FILL, PANEL, SURF = "#e4e3df", "#ecebe7", "#f6f6f4", "#ffffff"
# Accents: strong / light (plus a few extra shades). Used in this order when four things need colours.
BLUE, BLUE_L, BLUE_XL = "#2a78d6", "#d4e4f7", "#eef4fc"
ORANGE, ORANGE_L, ORANGE_M, ORANGE_D = "#eb6834", "#fbe0d5", "#f4b89f", "#c4501f"   # _D: orange text
AQUA, AQUA_L, AQUA_D = "#1baf7a", "#d1efe4", "#0f7f58"                            # _D: green text
GOLD, GOLD_L, GOLD_EDGE = "#c9a227", "#fdf3d8", "#e6d39a"


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
    # Follows thread 0 through the partition code and the k loop, one section per code line.
    # Checked on the GPU with CuTe DSL (thread 0, row-major 64 x 64 tiles):
    #   tSsQ ((2,2,2),1,4):((1,512,8),0,16)   tSsK ((2,2),8,4):((1,8),512,16)   acc_S ((2,2),1,8)
    # Lane layout of mma.m16n8k16 (PTX ISA), lane L = 4 * g + t:
    #   Q (A): rows g, g + 8; dims 2t, 2t + 1, 2t + 8, 2t + 9     (4 registers of 2 bf16)
    #   K (B): key g; dims 2t, 2t + 1, 2t + 8, 2t + 9              (2 registers of 2 bf16)
    #   S (C): rows g, g + 8; keys 2t, 2t + 1                      (4 fp32 registers)
    import textwrap
    W, TX, TW = 1040, 640, 58
    FAINT, PIECE = FILL, BLUE_XL
    b = [text(20, 30, "Following thread 0 through the code", 17, weight="600"),
         text(20, 52, "Blue: values thread 0 holds. Light blue: lanes 1–3, the other three threads of its "
                      "quad. Every thread gets its own share the same way.", 12.5, INK2)]

    def para(y, s, size=12.5, fill=INK2, lh=18, n=TW, weight="normal"):
        for ln in textwrap.wrap(s, n):
            b.append(text(TX, y, ln, size, fill, weight=weight))
            y += lh
        return y + 8

    def header(y, n, title, code):
        b.append(line(20, y - 26, W - 20, y - 26, GRID))
        b.append(f'<circle cx="31" cy="{y - 5}" r="11" fill="{INK}"/>')
        b.append(text(31, y, str(n), 12.5, SURF, "middle", "700"))
        b.append(text(50, y, title, 15, weight="600"))
        y += 14
        h = 21 * len(code) + 12
        b.append(rect(20, y, W - 40, h, PANEL, GRID, rx=4))
        for i, c in enumerate(code):
            code_part, _, comment = c.partition("  #")
            yy = y + 21 + 21 * i
            ind = len(code_part) - len(code_part.lstrip(" "))
            b.append(text(34 + 7.53 * ind, yy, code_part.lstrip(" "), 12.5, INK, family=MONO))
            if comment:
                b.append(text(34 + 7.53 * (len(code_part) + 2), yy, "#" + comment, 12.5, MUTED, family=MONO))
        return y + h + 34

    def lane_cell(x, y, cs, lane, label=True):
        fill = BLUE if lane == 0 else BLUE_L
        b.append(rect(x + 1, y + 1, cs - 2, cs - 2, fill, rx=1.5))
        if label:
            b.append(text(x + cs / 2, y + cs - 3.5, str(lane), 9, SURF if lane == 0 else INK, "middle", "600"))

    def blank(x, y, rows, cols, cs):
        for r in range(rows):
            for c in range(cols):
                b.append(rect(x + c * cs + 1, y + r * cs + 1, cs - 2, cs - 2, FAINT, rx=1.5))

    # ---- 1: get_slice picks thread 0 = lane 0 of warp 0, and with it lane 0's spots in each piece --
    y = header(98, 1, "Pick a thread: thread 0 and its spots",
               ["thr_mma = tiled_mma.get_slice(tidx)  # tidx = 0: thread 0, one of the block's 128"])
    # warp 0's lanes in quads: this is what get_slice(0) picks, and it doubles as the colour key
    lw, lx, ly = 16, 56, y + 16
    b.append(text(lx, ly - 10, "Warp 0's 32 lanes. tidx = 0 is lane 0.", 12, INK, weight="600"))
    for L in range(32):
        x = lx + L * lw + (L // 4) * 6
        b.append(rect(x + 1, ly + 1, lw - 2, lw - 2, BLUE if L == 0 else BLUE_L if L < 4 else FAINT, rx=2))
        b.append(text(x + lw / 2, ly + lw - 4.5, str(L), 9, SURF if L == 0 else INK if L < 4 else MUTED,
                      "middle", "600" if L < 4 else "normal"))
    b.append(line(lx + 1, ly + lw + 6, lx + 4 * lw - 1, ly + lw + 6, INK2))
    b.append(text(lx + 2 * lw, ly + lw + 20, "quad 0", 10.5, INK, "middle", "600"))
    b.append(text(lx + 4 * lw + 6 + 1, ly + lw + 20, "grey: lanes 4–31, the same pattern on other rows and keys",
                  10.5, MUTED))
    # lane 0's spots in one instruction's pieces; lanes 1-3 fill the rest of the same rows
    cs = 11
    gy = ly + lw + 112
    b.append(text(lx, gy - 50, "Lane 0's spots in one instruction, (16, 8, 16)", 12.5, INK, weight="600"))
    pieces = [(56, "Q piece", "16 rows × 16 dims", 16, 16, "dim", ("row 0", "row 8"), (0, 8),
               "rows 0, 8 × dims 0, 1, 8, 9", 8),
              (282, "K piece", "8 keys × 16 dims", 8, 16, "dim", ("key 0",), (0,),
               "key 0 × dims 0, 1, 8, 9", 4),
              (508, "S piece", "16 rows × 8 keys", 16, 8, "key", ("row 0", "row 8"), (0, 8),
               "rows 0, 8 × keys 0, 1", 4)]
    cap_y = gy + 16 * cs + 36
    for x0, name, size, rows, cols, axis, rlabels, owned, spots, nval in pieces:
        b.append(text(x0, gy - 22, name, 12, INK, weight="600"))
        b.append(text(x0, gy - 8, size, 11, INK2))
        blank(x0, gy, rows, cols, cs)
        for r, lab in zip(owned, rlabels):
            for c in range(cols):
                lane_cell(x0 + c * cs, gy + r * cs, cs, (c % 8) // 2)
            b.append(text(x0 - 5, gy + r * cs + 9, lab, 9.5, INK2, "end"))
        ay = gy + rows * cs + 12
        b.append(text(x0 - 5, ay, axis, 9.5, INK2, "end"))
        for c in (0, 8, cols - 1) if cols == 16 else (0, cols - 1):
            b.append(text(x0 + c * cs + cs / 2, ay, str(c), 9, MUTED, "middle"))
        b.append(text(x0, cap_y, f"thread 0: {nval} values", 12, BLUE, weight="600"))
        b.append(text(x0, cap_y + 16, spots, 11, INK2))
    pic_bottom = cap_y + 24
    ty = para(y, "The code starts by picking one thread, tidx, of the block's 128. Thread 0 is lane 0 of "
                 "warp 0, and warp 0 has rows 0–15 of the tile (idea 2).")
    ty = para(ty, "get_slice returns a rule, not data: which spots of each instruction's Q, K and S pieces "
                  "this lane holds. Choosing the instruction in idea 2 fixed that table. NVIDIA defines it "
                  "and CuTe has it built in; no line of our code spells it out.")
    ty = para(ty, "Lanes 1–3, the rest of quad 0, fill in rows 0 and 8 (key 0 of K), two neighbouring "
                  "columns each: a 32-bit register holds two bf16 values. Quad 1 does rows 1 and 9, and so "
                  "on, as in “What is a warp, a lane and a quad?”.")
    ty = para(ty, "Nothing is loaded yet. The next lines apply this rule to the whole 64 × 64 tiles.")
    y = max(pic_bottom, ty) + 46

    # ---- 2, 3, 4: each partition line: the tile on the left, the tensor it returns on the right ------
    # The tensor is drawn in its shape (values per piece, pieces down, pieces across): one small block
    # ("chip") per piece holding thread 0's values of that piece, as they sit in the piece.
    px, side, cx = 5, 320, 440       # tile cell size, tile side, x of the returned tensor
    PIECE_LINE = DIM_LINE

    def tile(x0, y0, name, cap, row_step, col_step, owner, light=None, warp_rows=False):
        b.append(text(x0, y0 - 28, name, 12.5, INK, weight="600", family=MONO))
        b.append(text(x0 + 8 * len(name) + 10, y0 - 28, cap, 12, INK2))
        b.append(rect(x0, y0, side, side, PANEL, GRID))
        if warp_rows:
            b.append(rect(x0, y0, side, 16 * px, PIECE))
            for w in range(4):
                b.append(text(x0 - 6, y0 + (16 * w + 8) * px + 4, f"warp {w}", 10.5,
                              INK if w == 0 else MUTED, "end", "600" if w == 0 else "normal"))
        for c in range(col_step, 64, col_step):
            b.append(line(x0 + c * px, y0, x0 + c * px, y0 + side, PIECE_LINE, 1))
        for r in range(row_step, 64, row_step):
            b.append(line(x0, y0 + r * px, x0 + side, y0 + r * px, PIECE_LINE, 1))
        for r in range(64):
            for c in range(64):
                if owner(r, c):
                    b.append(rect(x0 + c * px, y0 + r * px, px, px, BLUE))
                elif light and light(r, c):
                    b.append(rect(x0 + c * px, y0 + r * px, px, px, BLUE_L))

    def k_labels(x0, y0):              # the 4 bands of 16 head dims, which the loop walks with k
        for kk in range(4):
            b.append(text(x0 + (16 * kk + 8) * px, y0 - 8, f"k = {kk}", 10.5, ORANGE, "middle"))

    def chips(x0, y0, down, across, cr, cc, cs, pitch_x, pitch_y, hl=None):
        for i in range(down):
            for j in range(across):
                for r in range(cr):
                    for c in range(cc):
                        b.append(rect(x0 + j * pitch_x + c * cs + 0.5, y0 + i * pitch_y + r * cs + 0.5,
                                      cs - 1, cs - 1, BLUE))
        if hl is not None:             # orange frame around piece column hl
            b.append(rect(x0 + hl * pitch_x - 3.5, y0 - 3.5, cc * cs + 7, (down - 1) * pitch_y + cr * cs + 7,
                          "none", ORANGE, 1.6, rx=2))

    def arrow(x1, x2, yy, label=None):
        b.append(line(x1, yy, x2 - 7, yy, INK2, 1.3))
        b.append(f'<path d="M{x2},{yy} l-8,-4.5 v9 z" fill="{INK2}"/>')
        if label:
            b.append(text((x1 + x2) / 2, yy - 7, label, 12, INK, "middle", "600"))

    def shape_lines(x, yy, shape, rows, total):
        b.append(text(x, yy, f"shape {shape}", 12.5, INK, weight="600"))
        yy += 21
        for num, what, col in rows:
            b.append(text(x + 10, yy, num, 13, col, "end", "700"))
            b.append(text(x + 20, yy, what, 12, INK2))
            yy += 19
        b.append(text(x + 20, yy, total, 12, INK, weight="600"))

    def returned(name, yy, sub):
        b.append(text(cx, yy, name, 12.5, INK, weight="600", family=MONO))
        b.append(text(cx + 8 * len(name) + 10, yy, sub, 11.5, INK2))

    # 2: Q
    y = header(y, 2, "My spots in the Q tile",
               ["tSsQ = thr_mma.partition_A(sQ)  # sQ: the 64 x 64 Q tile in shared memory"])
    gy = y + 40
    tile(56, gy, "sQ", "64 rows × 64 head dims", 16, 16, lambda r, c: r in (0, 8) and c % 8 in (0, 1),
         warp_rows=True)
    k_labels(56, gy)
    arrow(56 + side + 8, cx - 8, gy + 8 * px)
    returned("tSsQ", gy + 14, "one block per piece")
    chips(cx, gy + 32, 1, 4, 2, 4, 8, 42, 0)
    for kk in range(4):
        b.append(text(cx + kk * 42 + 16, gy + 32 + 16 + 14, f"k = {kk}", 10.5, ORANGE, "middle"))
    shape_lines(cx, gy + 104, "(8, 1, 4)", [("8", "values per piece (step 1)", BLUE),
                                             ("1", "piece down: warp 0's 16 rows", INK),
                                             ("4", "pieces across: 64 dims ÷ 16", ORANGE)], "= 32 values")
    ty = para(y, "partition_A cuts the Q tile into the instruction's 16 × 16 pieces and keeps thread 0's 8 "
                 "spots in each: rows 0 and 8, dims +0, +1, +8, +9, as in step 1. Only warp 0's rows count, "
                 "so that is 1 piece down and 4 across.")
    ty = para(ty, "It copies nothing. tSsQ only records where these 32 values sit in shared memory.")
    ty = para(ty, "CuTe prints the shape as ((2, 2, 2), 1, 4): the 8 split into pair × row 0 or 8 × dims +0 "
                  "or +8.", 11.5, MUTED, 17)
    y = max(gy + side + 10, ty) + 46

    # 3: K
    y = header(y, 3, "My spots in the K tile",
               ["tSsK = thr_mma.partition_B(sK)  # sK: one 64 x 64 block of K in shared memory"])
    gy = y + 40
    tile(56, gy, "sK", "64 keys × 64 head dims", 8, 16, lambda r, c: r % 8 == 0 and c % 8 in (0, 1))
    k_labels(56, gy)
    b.append(text(50, gy - 8, "key", 10.5, INK2, "end"))
    for n in range(8):
        b.append(text(50, gy + 8 * n * px + 8, str(8 * n), 10, MUTED, "end"))
    arrow(56 + side + 8, cx - 8, gy + 36)
    returned("tSsK", gy + 14, "one block per piece")
    chips(cx, gy + 32, 8, 4, 1, 4, 8, 42, 16)
    for kk in range(4):
        b.append(text(cx + kk * 42 + 16, gy + 32 + 7 * 16 + 8 + 14, f"k = {kk}", 10.5, ORANGE, "middle"))
    shape_lines(cx, gy + 204, "(4, 8, 4)", [("4", "values per piece (step 1)", BLUE),
                                             ("8", "pieces down: 64 keys ÷ 8", INK),
                                             ("4", "pieces across: 64 dims ÷ 16", ORANGE)], "= 128 values")
    ty = para(y, "partition_B does the same for K. A K piece is 8 keys × 16 dims, so the tile is 8 pieces "
                 "down and 4 across. In each, thread 0 has the first key, dims +0, +1, +8, +9.")
    ty = para(ty, "Why 128 values, against 32 of Q? Every warp scores its own rows against all 64 keys, so K "
                  "is not split between warps.")
    y = max(gy + side + 10, ty) + 46

    # 4: registers
    y = header(y, 4, "Registers to hold them, and my part of S",
               ["tSrQ = thr_mma.make_fragment_A(tSsQ)  # registers, same shape as tSsQ: (8, 1, 4)",
                "tSrK = thr_mma.make_fragment_B(tSsK)  # registers, same shape as tSsK: (4, 8, 4)",
                "acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((64, 64)), cutlass.Float32)"])
    gy = y + 40
    tile(56, gy, "S", "64 rows × 64 keys: the scores, in registers", 16, 8,
         lambda r, c: r in (0, 8) and c % 8 in (0, 1), lambda r, c: r in (0, 8), warp_rows=True)
    for n in range(8):
        b.append(text(56 + (8 * n + 4) * px, gy - 8, str(n), 10.5, MUTED, "middle"))
    arrow(56 + side + 8, cx - 8, gy + 8 * px)
    returned("acc_S", gy + 14, "one block per S piece")
    chips(cx, gy + 32, 1, 8, 2, 2, 8, 22, 0)
    for n in range(8):
        b.append(text(cx + n * 22 + 8, gy + 32 + 16 + 14, str(n), 10.5, MUTED, "middle"))
    shape_lines(cx, gy + 104, "(4, 1, 8)", [("4", "values per S piece (step 1)", BLUE),
                                             ("1", "piece down: warp 0's 16 rows", INK),
                                             ("8", "pieces across: 64 keys ÷ 8", INK)], "= 32 values")
    ty = para(y, "make_fragment_A and make_fragment_B set aside registers in the shapes of tSsQ and tSsK. "
                 "They stay empty until the loop.")
    ty = para(ty, "acc_S is thread 0's part of the score tile S: rows 0 and 8, keys +0 and +1 of each 16 × 8 "
                  "S piece. The kernel zeroes it before each key block.")
    ty = para(ty, "Light blue: lanes 1–3 hold the rest of rows 0 and 8. A row of S is spread over these 4 "
                  "threads, which the softmax will have to combine.")
    y = max(gy + side + 10, ty) + 46

    # ---- 5: the loop, shown at k = 1 -------------------------------------------------------------
    y = header(y, 5, "The loop: 16 head dims per step",
               ["for k in range(4):  # 4 = the last number of both shapes",
                "    cute.autovec_copy(tSsQ[None, None, k], tSrQ[None, None, k])  # ① Q: shared -> registers",
                "    cute.autovec_copy(tSsK[None, None, k], tSrK[None, None, k])  # ② K: shared -> registers",
                "    cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)  # ③"])
    cs, pq, pk = 8, 42, 16           # cell size, pitch across, pitch down for K
    x0, gy = 56, y + 44
    ky = gy + 70
    rx, mx, sx = 300, 352, 432       # registers: tSrQ / tSrK, the mma labels, acc_S
    # shared memory: tSsQ and tSsK, piece column k = 1 framed
    b.append(text(x0, gy - 14, "tSsQ", 12.5, INK, weight="600", family=MONO))
    b.append(text(x0 + 42, gy - 14, "shared memory", 11.5, INK2))
    chips(x0, gy, 1, 4, 2, 4, cs, pq, 0, hl=1)
    for kk in range(4):
        b.append(text(x0 + kk * pq + 16, gy + 30, f"k = {kk}", 10.5, ORANGE if kk == 1 else MUTED, "middle",
                      "600" if kk == 1 else "normal"))
    b.append(text(x0, ky - 14, "tSsK", 12.5, INK, weight="600", family=MONO))
    chips(x0, ky, 8, 4, 1, 4, cs, pq, pk, hl=1)
    for kk in range(4):
        b.append(text(x0 + kk * pq + 16, ky + 7 * pk + 8 + 16, f"k = {kk}", 10.5, ORANGE if kk == 1 else MUTED,
                      "middle", "600" if kk == 1 else "normal"))
    by = ky + 7 * pk + 8 + 44
    b.append(text(x0, by, "orange: [None, None, 1]", 12, ORANGE, weight="600"))
    b.append(text(x0, by + 18, "Q: all 8 × 1 = 8 values", 11.5, INK2))
    b.append(text(x0, by + 35, "K: all 4 × 8 = 32 values", 11.5, INK2))
    # ① ② copy piece column 1 into the same places in registers
    arrow(x0 + 3 * pq + 4 * cs + 10, rx - 10, gy + 8, "①")
    arrow(x0 + 3 * pq + 4 * cs + 10, rx - 10, ky + 3 * pk + 4, "②")
    b.append(text(rx, gy - 14, "tSrQ", 12.5, INK, weight="600", family=MONO))
    b.append(text(rx + 42, gy - 14, "registers", 11.5, INK2))
    chips(rx, gy, 1, 1, 2, 4, cs, pq, 0)
    b.append(text(rx + 4 * cs + 10, gy + 12, "Q piece 1: used by all 8 mmas", 11, INK2))
    b.append(text(rx, ky - 14, "tSrK", 12.5, INK, weight="600", family=MONO))
    chips(rx, ky, 8, 1, 1, 4, cs, pq, pk)
    # ③ one mma per K piece n, adding into acc_S piece n
    b.append(text(mx, ky - 14, "③ gemm", 12.5, INK, weight="600"))
    b.append(text(sx, ky - 14, "acc_S", 12.5, INK, weight="600", family=MONO))
    for n in range(8):
        yy = ky + n * pk
        b.append(text(mx, yy + 8, f"mma {n}", 10.5, INK2))
        arrow(mx + 38, sx - 6, yy + 4)
        for r in range(2):
            for c in range(2):
                b.append(rect(sx + c * 7 + 0.5, yy - 2 + r * 7 + 0.5, 6, 6, BLUE))
        b.append(text(sx + 20, yy + 8, f"+= piece {n}", 10.5, INK2))
    # accumulation over the 4 steps
    ay = by
    b.append(text(rx, ay, "S[0, 0] after each step sums head dims:", 11.5, INK, weight="600"))
    for kk in range(4):
        yy = ay + 12 + 16 * kk
        b.append(text(rx, yy + 9, f"k = {kk}", 10, INK2))
        for d in range(4):
            fill = ORANGE if d == kk else ORANGE_L if d < kk else PANEL
            b.append(rect(rx + 36 + d * 40, yy, 40, 10, fill, GRID, 0.5))
        b.append(text(rx + 36 + 4 * 40 + 6, yy + 9, f"0–{16 * kk + 15}", 10, INK2))
    pic_bottom = ay + 12 + 16 * 4

    ty = para(y, "Step k takes piece column k of both shapes: head dims 16k to 16k + 15. In [None, None, k], "
                 "None keeps a whole mode, like : in NumPy, and k picks one piece across.")
    ty = para(ty, "① ② copy those values from shared memory into the same places in tSrQ and tSrK.")
    ty = para(ty, "③ gemm runs 8 instructions, one per K piece n: Q piece k with K piece n, added into S "
                  "piece n of acc_S. Each runs on the whole warp; thread 0 brings only its spots, and lanes "
                  "1–3 the rest of each row (step 1).")
    ty = para(ty, "acc_S is both input and output, so the steps add up: after k = 3 every score sums all 64 "
                  "head dims.")
    y = max(pic_bottom, ty) + 20
    return svg(W, y, b)


# ---------------------------------------------------------------------------------------------
def bank_conflicts():
    """What a shared-memory bank is, a conflict-free read, an 8-way conflict, and what conflicts cost."""
    W = 1000
    FAINT, GOLD = PANEL, GOLD_L
    b = [f'<defs><marker id="ahb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
         f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>',
         text(20, 30, "Shared-memory banks and bank conflicts", 17, weight="600"),
         text(20, 52, "Shared memory is the small, fast memory that a block's threads share. A warp is 32 threads, "
                      "called lanes, that load together.", 12.5, INK2)]

    def section(y, n, title):
        b.append(line(20, y - 26, W - 20, y - 26, GRID))
        b.append(f'<circle cx="31" cy="{y - 5}" r="11" fill="{INK}"/>')
        b.append(text(31, y, str(n), 12.5, SURF, "middle", "700"))
        b.append(text(50, y, title, 15, weight="600"))
        return y + 22

    def arrow(x1, y1, x2, y2, col=INK2):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="1.1" '
                 f'marker-end="url(#ahb)"/>')

    # ---- 1: what a bank is ---------------------------------------------------------------------
    y = section(98, 1, "A bank is one of 32 slices of shared memory")
    x0, cw, ch = 76, 28, 22
    gy = y + 30
    for k in range(32):
        b.append(text(x0 + k * cw + cw / 2, gy - 8, str(k), 9.5, MUTED, "middle"))
    b.append(text(x0 - 8, gy - 8, "bank", 10, MUTED, "end"))
    for r in range(4):
        b.append(text(x0 - 8, gy + r * ch + 15, f"bytes {128 * r}–{128 * r + 127}" if r < 2 else "…", 9.5,
                      MUTED, "end"))
        for k in range(32):
            w = 32 * r + k
            fill = BLUE_L if k == 0 else FAINT
            b.append(rect(x0 + k * cw, gy + r * ch, cw, ch, fill, SURF, 1.5))
            b.append(text(x0 + k * cw + cw / 2, gy + r * ch + 15, f"w{w}", 9, INK if k == 0 else INK2, "middle"))
    ty = gy + 4 * ch + 26
    b.append(text(x0, ty, "w = word: 4 bytes, two bf16 values. Word w lives in bank w mod 32, so the "
                  "banks repeat every 128 bytes (blue: bank 0 holds w0, w32, w64, w96, …).", 12, INK2))
    b.append(text(x0, ty + 20, "Each bank hands out one word per pass. A pass is one round of reading; "
                  "Nsight Compute calls it a wavefront.", 12, INK, weight="600"))
    y = ty + 20 + 62

    # ---- 2: no conflict -----------------------------------------------------------------------
    y = section(y, 2, "No conflict: every lane reads from a different bank, so 1 pass")
    ly, by = y + 20, y + 84
    for k in range(32):
        x = x0 + k * cw
        b.append(rect(x + 2, ly, cw - 4, 20, BLUE_L, rx=3))
        b.append(text(x + cw / 2, ly + 14, f"L{k}", 9, INK, "middle"))
        arrow(x + cw / 2, ly + 22, x + cw / 2, by - 3)
        b.append(rect(x, by, cw, ch, AQUA_L, SURF, 1.5))
        b.append(text(x + cw / 2, by + 15, f"w{k}", 9, INK, "middle"))
    b.append(text(x0 - 8, ly + 14, "lanes", 10, MUTED, "end"))
    b.append(text(x0 - 8, by + 15, "banks", 10, MUTED, "end"))
    b.append(text(x0, by + ch + 26, "Lane k reads word k: 32 lanes, 32 banks. All banks work at once, and the "
                  "whole warp gets its data in 1 pass.", 12, INK2))
    b.append(text(x0, by + ch + 44, "Lanes that read the same word also take 1 pass: the bank sends that word to "
                  "all of them.", 12, INK2))
    y = by + ch + 44 + 62

    # ---- 3: an 8-way conflict ------------------------------------------------------------------
    y = section(y, 3, "Bank conflict: 8 lanes read 8 different words in bank 0, so 8 passes")
    cw3, ch3 = 18, 18
    gy = y + 30
    for k in range(32):
        if k % 4 == 0:
            b.append(text(x0 + k * cw3 + cw3 / 2, gy - 8, str(k), 9.5, MUTED, "middle"))
    b.append(text(x0 - 8, gy - 8, "bank", 10, MUTED, "end"))
    for r in range(8):
        b.append(text(x0 - 8, gy + r * ch3 + 13, f"row {r}", 9.5, INK2, "end"))
        for k in range(32):
            hit = k == 0
            b.append(rect(x0 + k * cw3, gy + r * ch3, cw3, ch3, ORANGE if hit else FAINT, SURF, 1.2))
        b.append(text(x0 + cw3 / 2, gy + r * ch3 + 13, f"L{r}", 8.5, SURF, "middle", "700"))
    b.append(text(x0, gy + 8 * ch3 + 20, "8 rows of a 64-value bf16 tile: each row is 128 bytes = all 32 banks,",
                  11.5, INK2))
    b.append(text(x0, gy + 8 * ch3 + 36, "so the start of every row lands in bank 0.", 11.5, INK2))
    # bank 0 serving one lane per pass
    px0, pw = 700, 34
    b.append(text(px0, gy - 8, "bank 0 serves them one by one", 12, INK, weight="600"))
    for i in range(8):
        yy = gy + i * ch3
        b.append(text(px0, yy + 13, f"pass {i + 1}", 10.5, INK2))
        b.append(rect(px0 + 52 + i * 18, yy + 2, 18, ch3 - 4, ORANGE, SURF, 1, 2))
        b.append(text(px0 + 52 + i * 18 + 9, yy + 13, f"L{i}", 8.5, SURF, "middle", "700"))
    b.append(text(px0, gy + 8 * ch3 + 20, "Each lane waits its turn: an 8-way", 11.5, INK2))
    b.append(text(px0, gy + 8 * ch3 + 36, "conflict takes 8 passes instead of 1.", 11.5, INK2))
    y = gy + 8 * ch3 + 36 + 62

    # ---- 4: the cost ---------------------------------------------------------------------------
    y = section(y, 4, "Why it slows the kernel")
    tx0, unit, ty = 200, 52, y + 26
    for i, (lab, n, col) in enumerate([("no conflict", 1, AQUA), ("8-way conflict", 8, ORANGE)]):
        yy = ty + i * 40
        b.append(text(tx0 - 12, yy + 17, lab, 12, INK, "end", "600"))
        for p in range(n):
            b.append(rect(tx0 + p * unit, yy, unit - 3, 24, col, rx=3))
            b.append(text(tx0 + p * unit + (unit - 3) / 2, yy + 16, f"pass {p + 1}", 10, SURF, "middle", "600"))
        b.append(text(tx0 + n * unit + 8, yy + 17, "→ data ready, the warp goes on", 11.5, INK2))
    ax = ty + 2 * 40 + 4
    arrow(tx0, ax, tx0 + 8 * unit + 20, ax, MUTED)
    b.append(text(tx0 + 8 * unit + 26, ax + 4, "time", 10.5, MUTED))
    yy = ax + 34
    for s in ["The warp cannot use the data until its last pass is done, so it sits idle for the extra passes.",
              "Shared memory serves one pass at a time for all the block's warps, so their loads queue behind "
              "the extra passes too.",
              "In v0, 117 M shared loads took 671 M passes, about 6 per load. 554 M of those passes are the "
              "conflicts."]:
        b.append(f'<circle cx="{x0 + 4}" cy="{yy - 4}" r="2.5" fill="{INK2}"/>')
        b.append(text(x0 + 14, yy, s, 12, INK2))
        yy += 20
    return svg(W, yy + 10, b)


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
                b.append(rect(x0 + p * cw, y0 + r * ch, cw, ch, BLUE if hit else PANEL, SURF, 2))
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
def swizzle_bms():
    """Swizzle<3, 3, 3> from scratch, following one value (row 3, column 8 -> slot 200 -> 208)."""
    W = 900
    FAINT, GREY = PANEL, GREY_M
    b = [f'<defs><marker id="ahb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>',
         text(20, 30, "Swizzle&lt;3,3,3&gt; step by step, following one value", 17, weight="600"),
         text(20, 52, "The tile: 8 rows × 64 columns of bf16 values (2 bytes each) in shared memory. "
                      "The value we follow: row 3, column 8.", 12.5, INK2)]

    def header(y, n, title):
        b.append(line(20, y - 26, W - 20, y - 26, GRID))
        b.append(f'<circle cx="31" cy="{y - 5}" r="11" fill="{INK}"/>')
        b.append(text(31, y, str(n), 12.5, SURF, "middle", "700"))
        b.append(text(50, y, title, 15, weight="600"))
        return y + 24

    def para(y, lines, x=50, size=12.5, fill=INK2, lh=18):
        for s in lines:
            b.append(text(x, y, s, size, fill))
            y += lh
        return y

    def arrow(x1, y1, x2, y2):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.3" '
                 f'marker-end="url(#ahb)"/>')

    def bitrow(x, y, digits, fills, bw=30, bh=28, gap=12):
        """9 bits in 3 groups of 3; digits like '011001000', fills: 9 colours."""
        for k in range(9):
            g, i = divmod(k, 3)
            bx = x + g * (3 * bw + gap) + i * bw
            b.append(rect(bx, y, bw, bh, fills[k], SURF, 2))
            b.append(text(bx + bw / 2, y + 19, digits[k], 14, INK, "middle", "700", MONO))
        return x + 3 * (3 * bw + gap) - gap                      # right edge

    def gcx(x, g, bw=30, gap=12):                                # centre of group g
        return x + g * (3 * bw + gap) + 1.5 * bw

    R, C, P = AQUA_L, BLUE_L, GREY                               # row bits, chunk bits, position bits

    # ---- 1: slots ---------------------------------------------------------------------------------
    y = header(100, 1, "Slot: where a value sits in shared memory")
    y = para(y, ["Shared memory is one long line of numbered places, and the tile is stored in it row after row:",
                 "row 0 fills places 0–63, row 1 fills 64–127, and so on. A value's place in that line is its slot:",
                 "slot = row × 64 + column. Row 3, column 8 is slot 3 × 64 + 8 = 200."])
    y = para(y + 6, ["Each row is cut into 8 chunks of 8 values. 8 bf16 values are 16 bytes, the amount one thread loads",
                     "at a time, and the swizzle moves whole chunks. Column 8 is chunk 1, position 0 (its first value)."])
    x0, gy, cw, ch = 110, y + 30, 64, 24
    for c in range(8):
        b.append(text(x0 + c * cw + cw / 2, gy - 22, f"chunk {c}", 11, INK, "middle", "600"))
        b.append(text(x0 + c * cw + cw / 2, gy - 8, f"cols {8 * c}–{8 * c + 7}", 9.5, MUTED, "middle"))
    for r in range(8):
        yy = gy + r * ch
        b.append(text(x0 - 10, yy + 16, f"row {r}", 11, INK2, "end"))
        for c in range(8):
            hit = (r, c) == (3, 1)
            b.append(rect(x0 + c * cw, yy, cw, ch, BLUE_L if hit else FAINT, SURF, 2))
        b.append(text(x0 + 8 * cw + 12, yy + 16, f"slots {64 * r}–{64 * r + 63}", 11, MUTED))
    hx, hy = x0 + cw, gy + 3 * ch                                 # chunk 1 of row 3: show its 8 values
    for i in range(8):
        b.append(rect(hx + 2 + i * 7.5, hy + 5, 6, ch - 10, BLUE if i == 0 else SURF, BLUE, 0.8))
    b.append(text(x0, gy + 8 * ch + 22, "Light blue: chunk 1 of row 3, its 8 values. Dark blue: its first value, "
                                        "row 3, column 8 = slot 200.", 12, INK, weight="600"))
    y = gy + 8 * ch + 70

    # ---- 2: binary ---------------------------------------------------------------------------------
    y = header(y, 2, "Write the slot in binary: it splits into row, chunk and position")
    y = para(y, ["In binary each digit (bit) is worth twice the one to its right: 1, 2, 4, 8, ... Because a row is 64 = 2⁶ values",
                 "and a chunk 8 = 2³, the 9 bits of a slot fall into three groups of 3 that can be read off directly:"])
    x0, by = 230, y + 40
    vals = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    for k in range(9):
        g, i = divmod(k, 3)
        b.append(text(x0 + g * 102 + i * 30 + 15, by - 10, str(vals[k]), 10, MUTED, "middle"))
    b.append(text(x0 - 14, by - 10, "worth", 10, MUTED, "end"))
    right = bitrow(x0, by, "011001000", [R] * 3 + [C] * 3 + [P] * 3)
    b.append(text(x0 - 14, by + 19, "slot 200 =", 13, INK, "end", "600"))
    b.append(text(right + 16, by + 19, "128 + 64 + 8 = 200", 12, INK2))
    for g, (name, val, sub) in enumerate([("row", "011 = 3", "counts 64s"), ("chunk", "001 = 1", "counts 8s"),
                                          ("position", "000 = 0", "counts 1s")]):
        b.append(text(gcx(x0, g), by + 48, name, 12, INK, "middle", "700"))
        b.append(text(gcx(x0, g), by + 64, val, 11.5, INK2, "middle", family=MONO))
        b.append(text(gcx(x0, g), by + 79, sub, 10.5, MUTED, "middle"))
    b.append(text(right + 16, by + 50, "3 × 64 + 1 × 8 + 0 = 200:", 12, INK2))
    b.append(text(right + 16, by + 66, "the same sum, by group", 12, INK2))
    y = para(by + 112, ["Row mod 8: the row group has only 3 bits, enough for rows 0–7. The pattern covers 8 rows and then repeats,",
                        "so in a taller tile these bits hold the remainder of row ÷ 8, the row mod 8: row 11 reads as 3, row 8 as 0."])
    y += 38

    # ---- 3: XOR ------------------------------------------------------------------------------------
    y = header(y, 3, "The swizzle: XOR the row into the chunk")
    y = para(y, ["XOR compares two bits: the same gives 0, different gives 1. Put the row bits over the chunk bits and",
                 "XOR each column. Row and position are left as they are; only the chunk number changes."])
    x0, ty, bw = 230, y + 30, 58
    rows = [("row 3", "011", R), ("chunk 1", "001", C), ("new chunk", "010", ORANGE_L)]
    for j, (lab, d, f) in enumerate(rows):
        yy = ty + j * 36 + (14 if j == 2 else 0)
        b.append(text(x0 - 14, yy + 20, lab, 12.5, INK, "end", "600"))
        for i in range(3):
            b.append(rect(x0 + i * bw, yy, bw, 30, f, SURF, 2))
            b.append(text(x0 + i * bw + bw / 2, yy + 20, d[i], 14, INK, "middle", "700", MONO))
    b.append(line(x0 - 4, ty + 73, x0 + 3 * bw + 4, ty + 73, INK, 1.5))
    b.append(text(x0 + 3 * bw + 10, ty + 77, "XOR", 11, ORANGE, "start", "700"))
    for i, s in enumerate(["same → 0", "differ → 1", "same → 0"]):
        b.append(text(x0 + i * bw + bw / 2, ty + 134, s, 10.5, MUTED, "middle"))
    nx = x0 + 3 * bw + 60
    para(ty + 14, ["1 XOR 3 = 2: chunk 1 of row 3 is stored at place 2, where",
                   "chunk 2 would be: slot 3 × 64 + 2 × 8 + 0 = 208, not 200.",
                   "",
                   "Row 0 XORs with 000 and never moves. In every other row",
                   "the chunks swap places in pairs (row 3: 0 ↔ 3, 1 ↔ 2, ...);",
                   "each row still holds all 8 of its own chunks, in a new order."], x=nx, fill=INK2)
    y = ty + 132 + 44

    # ---- 4: the whole row -------------------------------------------------------------------------
    y = header(y, 4, "Do the same for every chunk of row 3")
    y = para(y, ["Step 3 moved one chunk. The swizzle does the same XOR for all 8 chunks of the row, always with the same",
                 "row number, 011. Each column below is one chunk of row 3:"])
    x0, ty, cw, rh = 230, y + 30, 66, 30
    rws = [("chunk", lambda c: f"c{c}", C), ("in binary", lambda c: f"{c:03b}", C),
           ("XOR row 3", lambda c: "011", R), ("= place", lambda c: f"{c ^ 3:03b} = {c ^ 3}", ORANGE_L)]
    for j, (lab, fn, fill) in enumerate(rws):
        yy = ty + j * rh + (8 if j == 3 else 0)
        b.append(text(x0 - 14, yy + 18, lab, 12, INK, "end", "600"))
        for c in range(8):
            b.append(rect(x0 + c * cw, yy, cw - 4, rh - 4, fill, rx=3))
            b.append(text(x0 + c * cw + (cw - 4) / 2, yy + 18, fn(c), 12.5, INK, "middle", "700", MONO))
    b.append(line(x0, ty + 3 * rh + 2, x0 + 8 * cw - 4, ty + 3 * rh + 2, INK, 1.5))
    b.append(rect(x0 + cw - 3, ty - 3, cw + 2, 4 * rh + 10, "none", ORANGE, 2, 4))
    b.append(text(x0 + cw + (cw - 4) / 2, ty - 9, "step 3", 11, ORANGE, "middle", "700"))
    # the row as stored: sorted by place
    sy = ty + 4 * rh + 66
    b.append(text(x0, sy - 26, "Now sort by place: row 3 as stored in memory", 12.5, INK, weight="600"))
    for pl in range(8):
        b.append(text(x0 + pl * cw + (cw - 4) / 2, sy - 6, f"place {pl}", 10.5, INK2, "middle", "600"))
        b.append(rect(x0 + pl * cw, sy, cw - 4, rh - 4, BLUE if pl ^ 3 == 1 else ORANGE_L, rx=3))
        b.append(text(x0 + pl * cw + (cw - 4) / 2, sy + 18, f"c{pl ^ 3}", 12.5, SURF if pl ^ 3 == 1 else INK,
                      "middle", "700", MONO))
    b.append(text(x0 - 14, sy + 18, "row 3", 12, INK, "end", "600"))
    y = para(sy + rh + 22, ["This is row 3 of the right-hand grid in step 5. The chunks swap in pairs: c0 ↔ c3, c1 ↔ c2, "
                            "c4 ↔ c7, c5 ↔ c6. No two land",
                            "on the same place, because XOR can be undone: XOR the place with 011 again and you get the chunk back.",
                            "Doing the same with row 0 (000), row 1 (001), ... row 7 (111) fills the other 7 rows of that grid."])
    y += 34

    # ---- 4: why ----------------------------------------------------------------------------------
    y = header(y, 5, "Why: the same chunk of 8 rows now sits in 8 different groups of banks")
    y = para(y, ["The kernel's loads read the same chunk from 8 rows at the same time, for example chunk 1 of rows 0–7 (blue).",
                 "Each place across a row is served by its own group of 4 banks, and one group serves one read at a time.",
                 "Each grid below is the 8 rows as stored in memory; each cell says which chunk is stored at that place."])
    cw, ch, gy = 44, 24, y + 56

    def grid(x0, swz, title, cap):
        b.append(text(x0, gy - 46, title, 13, INK, weight="700"))
        for p in range(8):
            b.append(text(x0 + p * cw + cw / 2, gy - 22, f"place {p}", 10, INK2, "middle", "600"))
            b.append(text(x0 + p * cw + cw / 2, gy - 8, f"{4 * p}–{4 * p + 3}", 9.5, MUTED, "middle"))
        b.append(text(x0 - 8, gy - 8, "banks", 9.5, MUTED, "end"))
        for r in range(8):
            b.append(text(x0 - 8, gy + r * ch + 16, f"row {r}", 10.5, INK2, "end"))
            for p in range(8):
                c = p ^ r if swz else p
                hit = c == 1
                b.append(rect(x0 + p * cw, gy + r * ch, cw, ch, BLUE if hit else FAINT, SURF, 2))
                b.append(text(x0 + p * cw + cw / 2, gy + r * ch + 16, f"c{c}", 10.5, SURF if hit else INK2,
                              "middle", family=MONO))
        for i, s_ in enumerate(cap):
            b.append(text(x0, gy + 8 * ch + 24 + 17 * i, s_, 12, INK if i == 0 else INK2,
                          weight="600" if i == 0 else "normal"))

    grid(110, False, "Without the swizzle",
         ["Chunk 1 is at place 1 in every row.", "All 8 reads need banks 4–7:", "they queue up, 8 passes."])
    grid(530, True, "With the swizzle: chunk c of row r at place c XOR r",
         ["Chunk 1 of row r is at place 1 XOR r.", "Row 0 → place 1, row 1 → 0, row 2 → 3, row 3 → 2, ...",
          "8 reads, 8 different groups of banks: 1 pass.",
          "Orange: row 3, the value from step 3, now at place 2."])
    # row 3, the example of step 3: chunk 1 at place 2
    b.append(rect(530 + 2 * cw - 1, gy + 3 * ch - 1, cw + 2, ch + 2, "none", ORANGE, 2.5))
    y = para(gy + 8 * ch + 24 + 17 * 4 + 14,
             ["Follow chunk 0 instead and the blue cells form a diagonal (place r in row r, as in the smem_layout figure).",
              "Every chunk makes a different pattern, but always 8 different places, so every chunk reads in 1 pass."])
    y += 34

    # ---- 5: code ----------------------------------------------------------------------------------
    y = header(y, 6, "How the computer does it: from one slot number to the new slot number")
    y = para(y, ["Steps 3–5 describe the rule with a row and a chunk. But when the kernel asks for row 3, column 8, CuTe works",
                 "with its slot only: one number goes in (200) and one number must come out (208). There are two ways to do that."])

    def tag(x, yy, letter, title):
        b.append(rect(x, yy - 15, 22, 21, INK, rx=4))
        b.append(text(x + 11, yy, letter, 12.5, SURF, "middle", "700"))
        b.append(text(x + 32, yy, title, 13.5, INK, weight="700"))

    def codebox(yy, rows, cx=44):
        h = 22 * len(rows) + 14
        b.append(rect(50, yy, W - 100, h, FAINT, GRID, 1, 5))
        for i, (code, com) in enumerate(rows):
            ly = yy + 24 + 22 * i
            k = code.find(" = ")                                 # line up the = signs at their column
            if k > 0 and code[:k].endswith(" "):
                b.append(text(66, ly, code[:k].rstrip(), 12.5, INK, family=MONO))
                b.append(text(66 + 7.53 * (k + 1), ly, code[k + 1:], 12.5, INK, family=MONO))
            else:
                b.append(text(66, ly, code, 12.5, INK, family=MONO))
            if com:
                b.append(text(66 + 7.53 * cx, ly, "# " + com, 12.5, MUTED, family=MONO))
        return yy + h

    # A: the long way
    y += 18
    tag(50, y, "A", "The long way: take the slot apart, do step 3, put it back together")
    y = codebox(y + 14, [
        ("row      = slot // 64 % 8", "200 // 64 = 3           (step 2: the row)"),
        ("chunk    = slot // 8 % 8", "200 // 8 = 25, 25 % 8 = 1   (the chunk)"),
        ("position = slot % 8", "200 % 8 = 0             (the position)"),
        ("place    = chunk ^ row", "1 XOR 3 = 2             (step 3)"),
        ("new = (slot // 64) * 64 + place * 8 + position", "192 + 16 + 0 = 208")], cx=50)
    y = para(y + 22, ["(// is division rounded down, % the remainder.) This works, but it needs divisions and multiplications,",
                      "and it pulls the slot apart into three separate numbers only to glue them back together."])

    # B: the short way
    y += 22
    tag(50, y, "B", "The short way: change only the chunk bits, right inside the slot")
    y = para(y + 26, ["Step 2 showed that the slot's bits already are row | chunk | position, side by side. Only the 3 chunk bits must",
                      "change, and XOR with 0 changes nothing. So build a number that holds the row in the chunk's place and 0s",
                      "everywhere else (000 011 000), then XOR the whole slot with it. The row and position bits meet 0s and stay as",
                      "they are; the chunk bits meet the row bits, which is exactly step 3. In code, with S = 3 and M = 3:"])
    y = codebox(y - 4, [("new = slot ^ ((slot &gt;&gt; S) &amp; (0b111 &lt;&lt; M))", "")])
    y = para(y + 22, ["&gt;&gt; 3 slides every bit 3 places to the right; &lt;&lt; 3 slides 3 places to the left; "
                      "&amp; keeps a bit only where both numbers",
                      "have a 1; ^ is XOR; 0b111 is binary 111. Read it from the inside out (colours show where each bit came from):"])
    x0, rh = 230, 38
    steps = [("slot", "011001000", [R] * 3 + [C] * 3 + [P] * 3, "200", "row 3, chunk 1, position 0"),
             "① build 000 011 000: the row, moved into the chunk's place",
             ("slot &gt;&gt; 3", "000011001", [SURF] * 3 + [R] * 3 + [C] * 3, "25",
              "the row bits slide down under the chunk"),
             ("0b111 &lt;&lt; 3", "000111000", [SURF] * 3 + [GREY] * 3 + [SURF] * 3, "56",
              "a mask: 1s only over the chunk bits"),
             ("(slot &gt;&gt; 3) &amp; mask", "000011000", [SURF] * 3 + [R] * 3 + [SURF] * 3, "24",
              "keep only what is under the mask: the row"),
             "② XOR the whole slot with it",
             ("slot ^ that", "011010000", [R] * 3 + [ORANGE_L] * 3 + [P] * 3, "208",
              "chunk 1 → 2 (step 3); the rest met 0s and stayed")]
    yy = y + 14
    for st in steps:
        if isinstance(st, str):
            b.append(text(x0 - 150, yy + 14, st, 12, ORANGE, weight="700"))
            yy += 24
            continue
        lab, d, f, dec, what = st
        b.append(text(x0 - 14, yy + 19, lab, 12, INK, "end", "600", MONO))
        right = bitrow(x0, yy, d, f)
        for k in range(9):                                       # outline the empty bits
            g, i = divmod(k, 3)
            if f[k] == SURF:
                b.append(rect(x0 + g * 102 + i * 30 + 1, yy + 1, 28, 26, "none", GRID, 1))
        b.append(text(right + 14, yy + 19, f"= {dec}", 12.5, INK, weight="600", family=MONO))
        b.append(text(right + 70, yy + 19, what, 12, INK2))
        yy += rh
    y = para(yy + 18, ["Same answer both ways: 208. The short way is the long way squeezed into 3 bit operations: no division,",
                       "no multiplication, and the slot never has to be taken apart. This one line is what Swizzle&lt;3,3,3&gt; is."],
             fill=INK)
    y += 34

    # ---- 6: B, M, S --------------------------------------------------------------------------------
    y = header(y, 7, "So what the three numbers of Swizzle&lt;B, M, S&gt; count")
    x0, by = 230, y + 34
    right = bitrow(x0, by, "011001000", [R] * 3 + [C] * 3 + [P] * 3)
    b.append(text(x0 - 14, by + 19, "slot 200 =", 13, INK, "end", "600"))
    sx, tx = gcx(x0, 0), gcx(x0, 1)
    b.append(f'<path d="M{sx},{by - 4} C{sx},{by - 30} {tx},{by - 30} {tx},{by - 6}" fill="none" '
             f'stroke="{AQUA}" stroke-width="1.8" marker-end="url(#ahb)"/>')
    b.append(text((sx + tx) / 2, by - 30, "S = 3 places", 12, AQUA, "middle", "700"))
    bry = by + 36
    for g, lab, col in [(1, "B = 3 bits", BLUE), (2, "M = 3 bits", INK2)]:
        gx0 = x0 + g * 102
        b.append(f'<path d="M{gx0 + 2},{bry} V{bry + 6} H{gx0 + 88} V{bry}" fill="none" stroke="{col}" '
                 f'stroke-width="1.5"/>')
        b.append(text(gx0 + 45, bry + 22, lab, 12, col, "middle", "700"))
    ky = bry + 58
    for i, (lab, col, what) in enumerate([
            ("B = 3", BLUE, "how many bits get changed: the chunk number. 2³ = 8 chunks per row, and the pattern spans 8 rows."),
            ("M = 3", INK2, "how many low bits are never touched: the position. 2³ = 8 values, so a 16-byte chunk moves whole."),
            ("S = 3", AQUA, "how far above the chunk bits the row bits start: bit M + S = 6, since a row is 2⁶ = 64 values.")]):
        b.append(text(50, ky + 21 * i, lab, 12.5, col, weight="700", family=MONO))
        b.append(text(106, ky + 21 * i, what, 12, INK2))
    y = ky + 21 * 2 + 26
    return svg(W, y, b)


# ---------------------------------------------------------------------------------------------
def smem_layout_fig():
    """Every line of smem_layout (round 1), one section per line, on the 64 x 64 tile of Q."""
    # Checked with CuTe DSL: smem_layout(64, 64) = S<3,3,3> o 0 o ((8,8),(64,1)):((64,512),(1,0)).
    # Slot of (r, c) = 64 r + 8 ((c // 8) xor (r % 8)) + c % 8, e.g. (3, 8) -> 208 (plain: 200);
    # row 8 starts again at 512. smem_layout(64, 128) puts (0, 64) at 4096: rows are filled first.
    import textwrap
    W, TX, TW = 1040, 640, 58
    FAINT, HL, DIM = PANEL, GOLD_L, DIM_LINE
    CODE = ["def smem_layout(rows, cols):",
            "    width = 64 if cols % 64 == 0 else 32                   # Swizzle<3,3,3> for 128-byte rows",
            "    atom = cute.make_composed_layout(cute.make_swizzle(3 if width == 64 else 2, 3, 3), 0,",
            "                                     cute.make_layout((8, width), stride=(width, 1)))",
            "    return cute.tile_to_shape(atom, (rows, cols), (0, 1))"]
    b = [f'<defs><marker id="ahs" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>',
         text(20, 30, "Building the swizzled layout of shared memory, line by line", 17, weight="600"),
         text(20, 52, "A layout maps a spot (row, column) of a tile to a slot: a position in the buffer, counted "
                      "in values. Shared memory is cut into", 12.5, INK2),
         text(20, 70, "32 banks of 4 bytes; a 16-byte chunk of a row covers 4 banks. Blue: chunk c0, the chunk "
                      "one load reads from 8 rows at once.", 12.5, INK2)]

    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def para(y, s, size=12.5, fill=INK2, lh=18, n=TW):
        for ln in textwrap.wrap(s, n):
            b.append(text(TX, y, esc(ln), size, fill))
            y += lh
        return y + 8

    def header(y, n, title, code, hl=None):
        b.append(line(20, y - 26, W - 20, y - 26, GRID))
        b.append(f'<circle cx="31" cy="{y - 5}" r="11" fill="{INK}"/>')
        b.append(text(31, y, str(n), 12.5, SURF, "middle", "700"))
        b.append(text(50, y, title, 15, weight="600"))
        y += 14
        h = 21 * len(code) + 12
        b.append(rect(20, y, W - 40, h, FAINT, GRID, rx=4))
        for i, c in enumerate(code):
            yy = y + 21 + 21 * i
            on = hl is None or i in hl
            if hl is not None and on:
                b.append(rect(21, yy - 15, W - 42, 21, HL))
            k = c.find("#")
            code_part = c if k < 0 else c[:k].rstrip()
            ind = len(code_part) - len(code_part.lstrip(" "))
            b.append(text(34 + 7.53 * ind, yy, esc(code_part.lstrip(" ")), 12.5, INK if on else DIM, family=MONO))
            if k >= 0:
                b.append(text(34 + 7.53 * k, yy, esc(c[k:]), 12.5, MUTED if on else DIM, family=MONO))
        return y + h + 34

    def arrow(x1, y1, x2, y2, label=None, dy=-7):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.4" '
                 f'marker-end="url(#ahs)"/>')
        if label:
            b.append(text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, 11.5, INK, "middle", "600"))

    def chunk_grid(x0, y0, swz, cw=46, ch=22, banks=True):
        # 8 rows x 8 chunk positions; the label in each cell is the chunk stored there
        if banks:
            for p in range(8):
                b.append(text(x0 + p * cw + cw / 2, y0 - 8, f"{4 * p}–{4 * p + 3}", 10, MUTED, "middle"))
            b.append(text(x0 - 8, y0 - 8, "banks", 10, MUTED, "end"))
        for r in range(8):
            b.append(text(x0 - 8, y0 + r * ch + 15, f"row {r}", 10.5, INK2, "end"))
            for p in range(8):
                c = p ^ r if swz else p
                hit = c == 0
                b.append(rect(x0 + p * cw, y0 + r * ch, cw, ch, BLUE if hit else FAINT, SURF, 2))
                b.append(text(x0 + p * cw + cw / 2, y0 + r * ch + 15, f"c{c}", 10.5, SURF if hit else INK2,
                              "middle", family=MONO))

    def box(x, y, w, s, fill=FAINT, mono=True, col=INK):
        b.append(rect(x, y, w, 28, fill, GRID, 1, 5))
        b.append(text(x + w / 2, y + 18, s, 12, col, "middle", "600", MONO if mono else None))

    # ---- 1: width ------------------------------------------------------------------------------
    y = header(134, 1, "How wide is one row of the pattern?", CODE, hl={0, 1})
    x0, gy = 56, y + 40

    def bar(yy, nchunks, label, sub):
        b.append(text(x0, yy - 10, label, 12, INK, weight="600"))
        for p in range(nchunks):
            b.append(rect(x0 + p * 64, yy, 64, 24, BLUE_L if p == 0 else FAINT, GRID, 1))
            b.append(text(x0 + p * 64 + 32, yy + 16, f"c{p}", 11, INK, "middle", family=MONO))
            b.append(text(x0 + p * 64 + 32, yy + 38, f"banks {4 * p}–{4 * p + 3}", 9.5, MUTED, "middle"))
        b.append(text(x0, yy + 58, sub, 11.5, INK2))

    bar(gy, 8, "width = 64: one row of the tile", "64 values × 2 bytes = 128 bytes: all 32 banks, 8 chunks")
    bar(gy + 104, 4, "width = 32: a narrower tile", "32 values × 2 bytes = 64 bytes: half the banks, 4 chunks")
    pic_bottom = gy + 104 + 58
    ty = para(y, "rows and cols are the size of the tile to lay out. For Q that is bM × D = 64 × 64.")
    ty = para(ty, "width is how many values one row of the pattern holds. If cols is a multiple of 64, it is 64 "
                  "values. In bf16 (2 bytes per value) that is 128 bytes: exactly one sweep across all 32 banks.")
    ty = para(ty, "Otherwise it is 32 values, 64 bytes: for a tile only 32 wide, such as P when a block takes 32 "
                  "keys at a time.")
    ty = para(ty, "The swizzle moves whole chunks: 8 values, 16 bytes, 4 banks. A 64-wide row has 8 chunks, "
                  "c0 to c7.")
    y = max(pic_bottom, ty) + 52

    # ---- 2: the plain 8-row block --------------------------------------------------------------
    y = header(y, 2, "Start from a plain block of 8 rows", CODE, hl={3})
    x0, gy = 96, y + 46
    chunk_grid(x0, gy, False)
    b.append(text(x0, gy + 8 * 22 + 24, "c0 of all 8 rows is in banks 0–3:", 12, INK, weight="600"))
    b.append(text(x0, gy + 8 * 22 + 42, "reading it takes 8 passes (an 8-way bank conflict).", 12, INK2))
    pic_bottom = gy + 8 * 22 + 42
    ty = para(y, "make_layout((8, width), stride=(width, 1)) is a plain block of 8 rows, stored one after "
                 "another (row-major). Value (r, c) goes to slot 64 r + c.")
    ty = para(ty, "This is v0's layout. Every row is exactly 128 bytes, so chunk c0 lands in banks 0–3 in "
                  "every row, c1 in banks 4–7, and so on.")
    ty = para(ty, "The loads that feed the multiply read the same chunk of 8 rows at once. Here all 8 are in "
                  "the same banks, so a bank serves them one at a time: 8 passes (rounds of reading) instead of 1.")
    ty = para(ty, "Why 8 rows? 8 chunks per row are just enough to give each of those 8 rows its own banks.")
    y = max(pic_bottom, ty) + 52

    # ---- 3: make_swizzle -----------------------------------------------------------------------
    y = header(y, 3, "The swizzle: which bits of the slot to mix", CODE, hl={2})
    x0, gy, bw = 96, y + 66, 30
    groups = [("row mod 8", "bits 6–8", "011", INK), ("chunk", "bits 3–5", "001", BLUE),
              ("value in chunk", "bits 0–2", "000", MUTED)]
    b.append(text(x0, gy - 40, "Slot of row 3, value 8 (chunk c1): 200 = 011 001 000 in binary", 12, INK,
                  weight="600"))

    def bits(yy, digits, cols):
        for g, (name, where, _, col) in enumerate(groups):
            gx = x0 + g * (3 * bw + 18)
            for i in range(3):
                b.append(rect(gx + i * bw, yy, bw, 28, FAINT if cols[g] is None else cols[g], GRID, 1))
                b.append(text(gx + i * bw + bw / 2, yy + 19, digits[g][i], 13, INK, "middle", "700", MONO))

    bits(gy, [g[2] for g in groups], [None, BLUE_L, None])
    for g, (name, where, _, col) in enumerate(groups):
        gx = x0 + g * (3 * bw + 18)
        b.append(text(gx + 1.5 * bw, gy + 44, name, 11, INK, "middle", "600"))
        b.append(text(gx + 1.5 * bw, gy + 58, where, 10, MUTED, "middle"))
    # XOR arrow from the row bits onto the chunk bits
    rx, cx = x0 + 1.5 * bw, x0 + (3 * bw + 18) + 1.5 * bw
    b.append(f'<path d="M{rx},{gy - 2} C{rx},{gy - 18} {cx},{gy - 18} {cx},{gy - 4}" fill="none" '
             f'stroke="{ORANGE}" stroke-width="1.6" marker-end="url(#ahs)"/>')
    b.append(text((rx + cx) / 2, gy - 22, "XOR", 11, ORANGE, "middle", "700"))
    gy2 = gy + 96
    b.append(text(x0, gy2 - 12, "chunk 001 XOR row 011 = 010: chunk c1 of row 3 moves to position 2", 12, INK,
                  weight="600"))
    bits(gy2, ["011", "010", "000"], [None, ORANGE_L, None])
    b.append(text(x0 + 3 * (3 * bw + 18) + 4, gy2 + 19, "= slot 208", 13, INK, weight="600"))
    ky = gy2 + 62
    b.append(text(x0, ky, "make_swizzle(3, 3, 3): the three numbers", 12, INK, weight="600"))
    for i, (num, what) in enumerate([("3", "mix 3 bits: the chunk number, 0 to 7"),
                                      ("3", "leave the lowest 3 bits alone: a chunk's 8 values stay together"),
                                      ("3", "take the bits to mix in from 3 places higher: row mod 8")]):
        b.append(text(x0 + 10, ky + 20 + 18 * i, num, 12.5, ORANGE, "end", "700"))
        b.append(text(x0 + 20, ky + 20 + 18 * i, what, 11.5, INK2))
    pic_bottom = ky + 20 + 18 * 2
    ty = para(y, "make_swizzle(3, 3, 3) builds the rule called Swizzle<3,3,3>. It works on the slot number "
                 "written in binary.")
    ty = para(ty, "In a 64-wide row, a slot's bits split in three: the lowest 3 are the value inside its chunk, "
                  "the next 3 the chunk number, and the 3 above those the row mod 8 (a row is 64 = 2⁶ values).")
    ty = para(ty, "XOR (exclusive or) compares two bits: the same gives 0, different gives 1. The rule XORs the "
                  "row bits into the chunk bits, so each row reorders its chunks in its own way.")
    ty = para(ty, "For 32-wide rows the first number is 2: 2 bits, for 4 chunks.")
    y = max(pic_bottom, ty) + 52

    # ---- 4: make_composed_layout ---------------------------------------------------------------
    y = header(y, 4, "Join them: the atom, 8 swizzled rows", CODE, hl={2, 3})
    x0, fy = 56, y + 20
    box(x0, fy, 66, "(3, 8)")
    arrow(x0 + 70, fy + 14, x0 + 116, fy + 14)
    box(x0 + 120, fy, 92, "plain", BLUE_L, False)
    arrow(x0 + 216, fy + 14, x0 + 262, fy + 14)
    box(x0 + 266, fy, 46, "200")
    arrow(x0 + 316, fy + 14, x0 + 346, fy + 14, "+ 0")
    arrow(x0 + 350, fy + 14, x0 + 380, fy + 14)
    box(x0 + 384, fy, 80, "swizzle", ORANGE_L, False)
    arrow(x0 + 468, fy + 14, x0 + 498, fy + 14)
    box(x0 + 502, fy, 46, "208")
    gx, gy = 96, fy + 84
    chunk_grid(gx, gy, True)
    b.append(text(gx, gy + 8 * 22 + 24, "c0 of the 8 rows is now in 8 different groups of banks:", 12, INK,
                  weight="600"))
    b.append(text(gx, gy + 8 * 22 + 42, "the same read takes 1 pass.", 12, INK2))
    pic_bottom = gy + 8 * 22 + 42
    ty = para(y, "make_composed_layout(swizzle, 0, layout) chains three steps: the plain layout turns (row, "
                 "column) into a slot, 0 is added (an extra shift, unused here), and the swizzle moves the chunk.")
    ty = para(ty, "The result, atom, is the smallest piece of the pattern: 8 rows × 64 values.")
    ty = para(ty, "Each row keeps its own 8 chunks, only in a new order. Row 0 does not move: XOR with 0 "
                  "changes nothing.")
    y = max(pic_bottom, ty) + 52

    # ---- 5: tile_to_shape ----------------------------------------------------------------------
    y = header(y, 5, "Repeat the atom to fill the tile", CODE, hl={4})
    x0, gy, sw, sh = 96, y + 30, 8 * 30, 30
    b.append(text(x0, gy - 10, "the 64 × 64 tile of Q", 12, INK, weight="600"))
    for a in range(8):
        ay = gy + a * (sh + 4)
        b.append(rect(x0, ay, sw, sh, FAINT, GRID, 1))
        for r in range(8):
            b.append(rect(x0 + r * 30 + 1, ay + r * sh / 8 + 0.5, 28, sh / 8 - 1, BLUE))   # c0 sits at position r
        b.append(text(x0 + sw + 12, ay + 13, f"atom {a}: rows {8 * a}–{8 * a + 7}", 11.5, INK, weight="600"))
        b.append(text(x0 + sw + 12, ay + 26, f"slots {512 * a}–{512 * a + 511}", 10.5, MUTED))
    if True:
        b.append(f'<path d="M{x0 - 14},{gy + 4} V{gy + 8 * (sh + 4) - 8}" stroke="{ORANGE}" stroke-width="1.6" '
                 f'marker-end="url(#ahs)"/>')
        b.append(text(x0 - 20, gy + 4 * (sh + 4), "fill", 11, ORANGE, "end", "600"))
        b.append(text(x0 - 20, gy + 4 * (sh + 4) + 14, "down", 11, ORANGE, "end", "600"))
    pic_bottom = gy + 8 * (sh + 4)
    ty = para(y, "tile_to_shape copies the 8 × 64 atom until it covers rows × cols. The 64 × 64 tile of Q "
                 "takes 8 copies, stacked down.")
    ty = para(ty, "(0, 1) is the fill order, by dimension: dimension 0 (down the rows) first, then dimension 1 "
                  "(across the columns). A 64 × 128 tile would fill every row of its left half first, so its right half "
                  "starts at slot 4096.")
    ty = para(ty, "Each atom is 512 values, 1 KB, and repeats the same pattern. In blue, c0 of each row: one "
                  "per position, in every atom.")
    y = max(pic_bottom, ty) + 52

    # ---- 6: get_tensor -------------------------------------------------------------------------
    y = header(y, 6, "Use it for Q's buffer",
               ["sQ = storage.sQ.get_tensor(smem_layout(bM, D))             # was cute.make_layout((bM, D), "
                "stride=(D, 1))"])
    x0, fy = 56, y + 30
    for i, (lab, layout, slot, fill) in enumerate([("v0", "plain (bM, D):(D, 1)", "slot 200", BLUE_L),
                                                    ("v1", "smem_layout(bM, D)", "slot 208", ORANGE_L)]):
        yy = fy + i * 52
        b.append(text(x0, yy + 18, lab, 12.5, INK, weight="700"))
        box(x0 + 30, yy, 86, "sQ[3, 8]")
        arrow(x0 + 120, yy + 14, x0 + 156, yy + 14)
        box(x0 + 160, yy, 190, layout, fill)
        arrow(x0 + 354, yy + 14, x0 + 390, yy + 14)
        box(x0 + 394, yy, 90, slot)
    b.append(text(x0, fy + 52 + 56, "Same code, same value; only where it sits in shared memory changes.", 12, INK,
                  weight="600"))
    pic_bottom = fy + 52 + 56
    ty = para(y, "get_tensor views Q's buffer in shared memory through the new layout. The comment shows the "
                 "old one: plain row-major, (bM, D):(D, 1).")
    ty = para(ty, "The rest of the kernel does not change: it still asks for sQ[r, c], row r, column c. Only the "
                  "slot that request reaches changes.")
    ty = para(ty, "v1 does the same for K, V and P.")
    y = max(pic_bottom, ty) + 20
    return svg(W, y, b)


# ---------------------------------------------------------------------------------------------
def ldsm_count():
    """From one 16 x 16 piece to the loop's instruction count: v1's LDS (+ PRMT for V) against v2's LDSM.
    Loop 0 of the non-causal SASS (results/cubin/v1_swizzle, v2_ldmatrix), per thread = per warp:
    LDS 224 -> 0, PRMT 64 -> 0, LDSM 0 -> 40, IMAD -6, UIADD3 -2, MOV +2, IADD3 +2: 594 -> 342."""
    W = 1040
    FAINT = PANEL
    TILE = [BLUE_L, ORANGE_L, AQUA_L, GOLD_L]                 # the four 8 x 8 tiles of a piece
    b = [f'<defs><marker id="ahc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>',
         text(20, 30, "Why 224 loads become 40: one 16 × 16 piece at a time", 17, weight="600"),
         text(20, 52, "Every operand of the two multiplies is read from shared memory in 16 × 16 pieces of bf16. "
                      "Each of the warp's 32 lanes must end up", 12.5, INK2),
         text(20, 70, "holding 8 values of the piece, as 4 pairs packed into 4 registers (2 values per 32-bit "
                      "register). Counts are instructions per warp.", 12.5, INK2)]

    def header(y, n, title):
        b.append(line(20, y - 26, W - 20, y - 26, GRID))
        b.append(f'<circle cx="31" cy="{y - 5}" r="11" fill="{INK}"/>')
        b.append(text(31, y, str(n), 12.5, SURF, "middle", "700"))
        b.append(text(50, y, title, 15, weight="600"))
        return y + 24

    def para(x, y, lines, size=12.5, fill=INK2, lh=18):
        for s in lines:
            b.append(text(x, y, s, size, fill))
            y += lh
        return y

    def arrow(x1, y1, x2, y2):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.4" '
                 f'marker-end="url(#ahc)"/>')

    def piece(x, y, cs, dark=(), labels=None, rlab=None, clab=None):
        """16 x 16 cells; tiles coloured; `dark` = set of (row, col) cells to fill dark."""
        for r in range(16):
            for c in range(16):
                t = (r >= 8) + 2 * (c >= 8)
                hit = (r, c) in dark
                b.append(rect(x + c * cs, y + r * cs, cs, cs, INK if hit else TILE[t], SURF, 0.6))
        b.append(rect(x, y, 16 * cs, 16 * cs, "none", INK2, 1))
        b.append(line(x + 8 * cs, y, x + 8 * cs, y + 16 * cs, INK2, 1))
        b.append(line(x, y + 8 * cs, x + 16 * cs, y + 8 * cs, INK2, 1))
        if rlab:
            b.append(text(x - 6, y + 8 * cs, rlab, 10.5, MUTED, "end"))
        if clab:
            b.append(text(x + 8 * cs, y - 6, clab, 10.5, MUTED, "middle"))
        if labels:
            for (tr, tc), s in labels.items():
                b.append(text(x + tc * 8 * cs + 4 * cs, y + tr * 8 * cs + 4 * cs + 4, s, 11, INK, "middle", "700"))

    def instr(x, y, rows, w=330):
        for i, (op, what) in enumerate(rows):
            yy = y + i * 21
            b.append(text(x, yy, op, 12, INK, weight="700", family=MONO))
            b.append(text(x + 120, yy, what, 12, INK2))
        return y + len(rows) * 21

    cs = 11
    # ---- 1: v1, Q / K / P ----------------------------------------------------------------------
    y = header(118, 1, "v1, a piece of Q, K or P: one plain load per pair")
    px, py = 100, y + 26
    lane0 = {(0, 0), (0, 1), (8, 0), (8, 1), (0, 8), (0, 9), (8, 8), (8, 9)}
    piece(px, py, cs, lane0, rlab="16 rows", clab="16 columns")
    b.append(text(px + 8 * cs, py + 16 * cs + 18, "black: lane 0's 8 values", 11, INK2, "middle"))
    tx = 330
    ty = para(tx, y + 6, ["The 4 pairs a lane needs sit in 4 different places, one in each 8 × 8 tile. Each pair",
                          "is two neighbours in memory, so one 32-bit load (LDS) can fetch it. But an LDS",
                          "fetches one address per lane, so a lane needs 4 of them:"])
    ty = instr(tx + 10, ty + 8, [("LDS.32", "row 0, columns 0–1"), ("LDS.32", "row 8, columns 0–1"),
                                 ("LDS.32", "row 0, columns 8–9"), ("LDS.32", "row 8, columns 8–9")])
    b.append(text(tx, ty + 10, "= 4 instructions per piece. All 32 lanes run them together, each at its own "
                               "addresses.", 12.5, INK, weight="600"))
    y = max(py + 16 * cs + 18, ty + 10) + 52

    # ---- 2: v1, V --------------------------------------------------------------------------------
    y = header(y, 2, "v1, a piece of V: worse, because its pairs are not neighbours")
    py = y + 26
    vpairs = {(0, 0), (1, 0), (8, 0), (9, 0), (0, 8), (1, 8), (8, 8), (9, 8)}
    piece(px, py, cs, vpairs, rlab="16 keys", clab="16 head dims")
    b.append(text(px + 8 * cs, py + 16 * cs + 18, "black: lane 0's 8 values", 11, INK2, "middle"))
    b.append(f'<path d="M{px - 3},{py + cs / 2} C{px - 20},{py + cs / 2} {px - 20},{py + 1.5 * cs} '
             f'{px - 3},{py + 1.5 * cs}" fill="none" stroke="{ORANGE}" stroke-width="1.6"/>')
    b.append(text(px - 22, py + cs - 2, "128 bytes", 10.5, ORANGE, "end", "700"))
    b.append(text(px - 22, py + cs + 11, "apart", 10.5, ORANGE, "end", "700"))
    ty = para(tx, y + 6, ["O = P·V wants each pair to be 2 keys of the same head dim. V is stored one key per",
                          "row, so the two values of a pair are a whole row (128 bytes) apart. Each needs its",
                          "own 16-bit load, then a PRMT instruction packs the two halves into one register:"])
    ty = instr(tx + 10, ty + 8, [("LDS.U16 × 2", "key 0 and key 1 of dim 0, then PRMT to pack them"),
                                 ("LDS.U16 × 2", "keys 8, 9 of dim 0, then PRMT"),
                                 ("LDS.U16 × 2", "keys 0, 1 of dim 8, then PRMT"),
                                 ("LDS.U16 × 2", "keys 8, 9 of dim 8, then PRMT")])
    b.append(text(tx, ty + 10, "= 8 loads + 4 PRMT = 12 instructions per piece.", 12.5, INK, weight="600"))
    y = max(py + 16 * cs + 18, ty + 10) + 52

    # ---- 3: v2, ldmatrix -------------------------------------------------------------------------
    y = header(y, 3, "v2: one ldmatrix per piece, for every operand")
    py = y + 26
    piece(px, py, cs, labels={(0, 0): "lanes 0–7", (1, 0): "lanes 8–15", (0, 1): "lanes 16–23", (1, 1): "lanes 24–31"},
          rlab="16 rows", clab="the same piece")
    b.append(text(px + 8 * cs, py + 16 * cs + 18, "lanes that give each tile's 8 row addresses", 11, INK2, "middle"))
    ty = para(tx, y + 6, ["ldmatrix.x4 is one instruction for the whole warp. Each of the 32 lanes gives the address",
                          "of one 8-value row (8 rows × 4 tiles = 32 rows). The hardware reads all 4 tiles and",
                          "deals every lane its 4 pairs, already packed in 4 registers."])
    ty = instr(tx + 10, ty + 8, [("LDSM.16.M88.4", "Q, K or P: replaces 4 LDS.32"),
                                 ("LDSM.16.MT88.4", "V, rows and columns swapped on the way (.trans), so the"),
                                 ("", "pairs come out as 2 keys of one dim: replaces 8 LDS.U16 + 4 PRMT")])
    b.append(text(tx, ty + 10, "= 1 instruction per piece.", 12.5, INK, weight="600"))
    y = max(py + 16 * cs + 18, ty + 10) + 52

    # ---- 4: add it up ----------------------------------------------------------------------------
    y = header(y, 4, "Add it up over one pass of the loop (one key block)")
    y = para(50, y + 2, ["Each warp multiplies its own 16 rows of Q by the whole 64 × 64 tile of K, then its 16 rows "
                         "of P by the whole tile of V. So it reads:"])
    cols = [50, 330, 450, 720]
    hy = y + 14
    for c, s in zip(cols, ["operand (per warp)", "16 × 16 pieces", "v1", "v2"]):
        b.append(text(c, hy, s, 12, INK, weight="700"))
    b.append(line(50, hy + 8, W - 50, hy + 8, GRID))
    rows = [("Q: 16 rows × 64 dims", "4", "4 × 4 = 16 LDS", "4 LDSM"),
            ("K: 64 keys × 64 dims", "16", "16 × 4 = 64 LDS", "16 LDSM"),
            ("P: 16 rows × 64 keys", "4", "4 × 4 = 16 LDS", "4 LDSM"),
            ("V: 64 keys × 64 dims", "16", "16 × 8 = 128 LDS, + 16 × 4 = 64 PRMT", "16 LDSM (.trans)")]
    for i, r in enumerate(rows):
        yy = hy + 30 + 22 * i
        for c, s in zip(cols, r):
            b.append(text(c, yy, s, 12.5, INK2))
    ty = hy + 30 + 22 * 4
    b.append(line(50, ty - 12, W - 50, ty - 12, GRID))
    for c, s in zip(cols, ["total", "40", "224 LDS + 64 PRMT = 288", "40 LDSM"]):
        b.append(text(c, ty + 4, s, 12.5, INK, weight="700"))

    # stacked bars: the whole pass
    by, u = ty + 50, 1.25
    b.append(text(50, by - 14, "All the instructions in one pass of the loop", 12.5, INK, weight="700"))
    for j, (lab, segs, tot) in enumerate([
            ("v1", [(224, "224 LDS", ORANGE_L), (64, "64 PRMT", ORANGE_L), (306, "everything else: 306", FAINT)], 594),
            ("v2", [(40, "40 LDSM", BLUE_L), (302, "everything else: 302", FAINT)], 342)]):
        yy = by + j * 44
        b.append(text(90, yy + 20, lab, 13, INK, "end", "700"))
        x = 100
        for n, s, f in segs:
            b.append(rect(x, yy, n * u, 30, f, INK2, 0.8))
            b.append(text(x + n * u / 2, yy + 20, s, 11.5, INK, "middle", "600"))
            x += n * u
        b.append(text(x + 10, yy + 20, f"{tot}", 13, INK, weight="700"))
    y = para(50, by + 2 * 44 + 22,
             ["288 load-and-pack instructions become 40: 248 fewer. Everything else is the same work in both (the 64 "
              "multiplies, the softmax,",
              "the copies of the next K and V) and differs only by 4 address instructions, 306 → 302. Together: 594 → 342."],
             fill=INK2)
    y = para(50, y + 8,
             ["Every shared-memory instruction goes through the load/store pipe. With 248 fewer per pass, it falls from "
              "72.5% to 26.0% busy,",
              "stops holding up the multiplies, and the tensor cores rise from 71.3% to 86.9% of peak."],
             fill=INK)
    return svg(W, y + 14, b)


# ---------------------------------------------------------------------------------------------
def pipeline():
    """Round 4's loop body line by line. Part 1: the eight lines of v4's attend that the article shows, grouped
    into six numbered steps, each with one plain sentence. Part 2: the same steps on a time line for one warp
    (what the warp runs, what the shared buffers sK and sV hold), with v3's order on the same time axis."""
    import re
    W, FAINT, WAIT, OLD = 900, PANEL, GREY_M, FILL
    KIND = {"wait": INK2, "copy": ORANGE, "math": BLUE}
    b = [text(20, 30, "Round 4: one key block j, line by line", 17, weight="600"),
         text(20, 52, "sK and sV are the shared-memory buffers for one K tile and one V tile. v4 reuses each buffer "
                      "as soon as no warp reads it.", 12.5, INK2)]

    def sub(s):
        """K_j, V_j-1, K_j+1 ... -> K with a lowered j, j-1, j+1 (dy keeps it working in every browser)."""
        out, back = [], ""
        for i, part in enumerate(re.split(r"([KV])_(j(?:[+−]1)?)", s)):
            if i % 3 == 2:                              # the subscript
                out.append(f'<tspan dy="3.5" font-size="0.78em">{part}</tspan>')
                back = ' dy="-3.5"'
            elif part:
                out.append(f'<tspan{back}>{part}</tspan>' if back else part)
                back = ""
        return "".join(out)

    def badge(x, y, n, kind, r=10, faded=False):
        b.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{KIND[kind]}"{" opacity=\"0.45\"" if faded else ""}/>')
        b.append(text(x, y + 4.3, str(n), 12 if r >= 10 else 10.5, SURF, "middle", "700"))

    # ---- part 1: the code, one numbered step per line (step 5 spans three lines) ----------------------------
    steps = [
        (1, "wait", ["cute.arch.cp_async_wait_group(0); cute.arch.sync_threads()"],
         ["Wait until K_j has landed in sK, then wait for every warp. Now all of K_j is there,",
          "and every warp has finished with V_j−1, so sV is free."]),
        (2, "copy", ["self.start_load(kv.gV, kv.sV, kv, j); cute.arch.cp_async_commit_group()"],
         ["Start copying V_j into sV with cp.async. The warp does not wait for it."]),
        (3, "math", ["# ... S = Q K_j^T ..."],
         ["Uses K_j, not V_j, so it runs while V_j is still being copied."]),
        (4, "wait", ["cute.arch.cp_async_wait_group(0); cute.arch.sync_threads()"],
         ["Wait until V_j has landed in sV, then wait for every warp. Now every warp has finished",
          "step 3, so no warp still reads K_j: sK is free."]),
        (5, "copy", ["if j + 1 < kv.n_end:", "    self.start_load(kv.gK, kv.sK, kv, j + 1)",
                     "cute.arch.cp_async_commit_group()"],
         ["Start copying K_j+1 into sK, unless j is the last block. The commit is outside the if,",
          "so the last block commits an empty group, which does nothing."]),
        (6, "math", ["# ... mask, softmax, O += P V_j ..."],
         ["Uses V_j, not K_j+1, so it runs while K_j+1 is being copied. Then step 1 of block j+1 waits for K_j+1."]),
    ]
    y = 82
    code_h = 21 * sum(len(c) for _, _, c, _ in steps) + 18 * sum(len(e) for *_, e in steps) + 12 * len(steps) + 8
    b.append(rect(20, y, W - 40, code_h, FAINT, GRID, rx=4))
    y += 22
    for n, kind, code, eng in steps:
        badge(40, y - 4.5, n, kind)
        for c in code:
            ind = len(c) - len(c.lstrip(" "))
            esc = c.lstrip(" ").replace("&", "&amp;").replace("<", "&lt;")
            b.append(text(62 + 7.53 * ind, y, esc, 12.5, MUTED if c.startswith("#") else INK, family=MONO))
            y += 21
        for e in eng:
            b.append(text(62, y - 3, sub(e), 12.5, INK2))
            y += 18
        y += 12
    y += 18

    # ---- part 2: the same steps in time ---------------------------------------------------------------------
    b.append(text(20, y, "The same steps in time, for one warp (schematic, not to scale)", 15, weight="600"))
    y += 22
    lx = 20
    for kind, label in (("wait", "grey: waiting"), ("copy", "orange: copy in flight"), ("math", "blue: maths")):
        fill = {"wait": WAIT, "copy": ORANGE, "math": BLUE_L}[kind]
        b.append(rect(lx, y - 10, 22, 12, fill, rx=2))
        b.append(text(lx + 28, y, label, 12, INK2))
        lx += 44 + 6.6 * len(label)
    b.append(rect(lx, y - 10, 22, 12, SURF, BLUE, 1, 2))
    b.append(text(lx + 28, y, "outlined: landed, ready to read", 12, INK2))
    lx += 44 + 6.6 * len("outlined: landed, ready to read")
    b.append(line(lx + 4, y - 13, lx + 4, y + 4, INK, 2))
    b.append(text(lx + 12, y, "black line: barrier", 12, INK2))

    x0, u, bh = 160, 29.5, 26

    def X(t):
        return x0 + t * u

    def bar(t0, t1, yy, label, fill, ink=INK, edge="none", faded=False):
        op = ' opacity="0.5"' if faded else ""
        b.append(f'<g{op}>' + rect(X(t0), yy, (t1 - t0) * u - 2, bh, fill, edge, 1, 4)
                 + text((X(t0) + X(t1) - 2) / 2, yy + 17, sub(label), 11.5, ink, "middle") + '</g>')

    def barrier(t, y0, y1):
        b.append(line(X(t) - 1, y0, X(t) - 1, y1, INK, 2))

    # time of each step; step 2 and 5 only start copies, so they are narrow ticks
    T = {1: (0, 1.4), 2: (1.4, 1.8), 3: (1.8, 7.8), 4: (7.8, 9.4), 5: (9.4, 9.8), 6: (9.8, 19.6),
         "1n": (19.6, 20.8), "2n": (20.8, 21.2), "3n": (21.2, 24.0)}
    y += 30
    # which key block
    for (t0, t1, s) in ((0, 19.6, "key block j"), (19.6, 24.0, "block j+1")):
        b.append(line(X(t0) + 2, y + 6, X(t1) - 4, y + 6, MUTED))
        b.append(line(X(t0) + 2, y + 2, X(t0) + 2, y + 10, MUTED))
        b.append(line(X(t1) - 4, y + 2, X(t1) - 4, y + 10, MUTED))
        b.append(rect((X(t0) + X(t1)) / 2 - 3.6 * len(s) - 6, y - 4, 7.2 * len(s) + 12, 18, SURF))
        b.append(text((X(t0) + X(t1)) / 2, y + 10, sub(s), 12, INK2, "middle", "600"))
    y += 32
    lanes = y + 14
    for key, kind in ((1, "wait"), (2, "copy"), (3, "math"), (4, "wait"), (5, "copy"), (6, "math"),
                      ("1n", "wait"), ("2n", "copy"), ("3n", "math")):
        t0, t1 = T[key]
        badge((X(t0) + X(t1) - 2) / 2, y, int(str(key)[0]), kind, 9, faded=isinstance(key, str))
    y = lanes
    rows = {"warp": y, "sK": y + 40, "sV": y + 80}
    for name, yy in (("v4: the warp runs", rows["warp"]), ("sK holds", rows["sK"]), ("sV holds", rows["sV"])):
        b.append(text(20, yy + 17, name, 12.5, INK, weight="600" if name.startswith("v4") else "normal"))
    wy = rows["warp"]
    bar(*T[1], wy, "wait", WAIT, INK2)
    bar(*T[2], wy, "", ORANGE)
    bar(*T[3], wy, "S = Q K_jᵀ", BLUE_L)
    bar(*T[4], wy, "wait", WAIT, INK2)
    bar(*T[5], wy, "", ORANGE)
    bar(*T[6], wy, "mask, softmax, O += P V_j", BLUE_L)
    bar(*T["1n"], wy, "", WAIT, faded=True)
    bar(*T["2n"], wy, "", ORANGE, faded=True)
    bar(*T["3n"], wy, "S = Q K_j+1ᵀ", BLUE_L, faded=True)
    # the shared buffers
    ky, vy = rows["sK"], rows["sV"]
    # each copy bar starts under the tick of the step that starts it (2 or 5)
    bar(0, 1.0, ky, "K_j", ORANGE, SURF)
    bar(1.0, 9.4, ky, "K_j (step 3 reads it)", SURF, INK, BLUE)
    bar(T[5][0], 15.6, ky, "K_j+1 in flight", ORANGE, SURF)
    bar(15.6, T["1n"][1], ky, "K_j+1 landed", SURF, INK, BLUE)
    bar(T["1n"][1], 24.0, ky, "step 3 reads it", SURF, INK, BLUE, faded=True)
    bar(0, 1.4, vy, "V_j−1", OLD, MUTED)
    bar(T[2][0], 8.6, vy, "V_j in flight", ORANGE, SURF)
    bar(8.6, 19.6, vy, "V_j (step 6 reads it)", SURF, INK, BLUE)
    bar(T["2n"][0], 24.0, vy, "V_j+1 in flight", ORANGE, SURF, faded=True)
    for t in (T[1][1], T[4][1], T["1n"][1]):
        barrier(t, wy - 5, vy + bh + 5)
    y = vy + bh + 26
    for s in ("Each copy starts as soon as its buffer is free (steps 2 and 5) and is waited for just before the maths that",
              "reads it (steps 4 and 1). So the copies run during the maths, and the waits are short."):
        b.append(text(x0, y, s, 12.5, INK2))
        y += 18

    # ---- v3 on the same time axis ---------------------------------------------------------------------------
    y += 22
    b.append(line(20, y - 20, W - 20, y - 20, GRID))
    b.append(text(20, y + 17, "v3: the warp runs", 12.5, INK, weight="600"))
    bar(0, 7.0, y, "copy K_j and V_j (the warp waits)", ORANGE_L)
    bar(7.0, 13.0, y, "S = Q K_jᵀ", BLUE_L)
    bar(13.0, 22.8, y, "mask, softmax, O += P V_j", BLUE_L)
    for t in (0.07, 7.0, 22.8):
        barrier(t, y - 5, y + bh + 5)
    y += bh + 26
    for s in ("v3 copies K_j and V_j through registers, so the warp itself waits for every value, and no maths starts until",
              "both have landed. A barrier before the copy (no warp still reads the old tiles) and one after it (all copies landed)."):
        b.append(text(x0, y, sub(s), 12.5, INK2))
        y += 18
    return svg(W, y + 8, b)

def gpu_hierarchy():
    b = []
    b.append(text(20, 30, "Where the threads and memory live: GPU → SM → block → warp → lane", 16, weight="600"))
    top, h = 56, 290
    GREY_L = PANEL

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
    C_A, C_B, C_C = BLUE, ORANGE_D, AQUA_D     # text colours: first row counter, second row counter, column
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

    # ---- example 2: ((2, 2), 8):((1, 16), 2), told in four steps: pair them up, what (2, 2) means,
    # why it is needed, and the whole table with its memory order.
    ROW_T = [SURF, GREY_M, ORANGE_L, ORANGE_M]       # row tints: i1 picks grey or orange, i0 light or dark
    y = 332
    b.append(text(20, y, "Example 2", 15, weight="600"))
    b.append(text(110, y, f"(({tsp(2, C_A)}, {tsp(2, C_B)}), {tsp(8, C_C)}):(({tsp(1, C_A)}, {tsp(16, C_B)}), {tsp(2, C_C)})",
                  15, INK, family=MONO))
    b.append(text(20, y + 22, "A 4 × 8 table whose rows are not evenly spaced in memory. Four steps:", 12.5, INK2))
    px = 430                                               # left edge of the pictures

    def para(x, y, lines, size=12.5, fill=INK2, lh=18):
        for s in lines:
            b.append(text(x, y, s, size, fill))
            y += lh
        return y

    def step(y, title, lines):
        b.append(text(20, y, title, 13.5, INK, weight="600"))
        return para(20, y + 22, lines)

    # (1) pair shape and stride by position
    T = y + 70
    step(T, "1. Pair each number with the one in the same place", [
        "The brackets line up too: (2, 2) pairs with (1, 16),",
        "and 8 pairs with 2. That gives three pairs, and",
        "each pair is one counter: the shape number says",
        "how many values it counts, the stride number how",
        "far in memory each step of it moves."])
    cols = [(px + 130, C_A, BLUE_L, "2", "1", "i0", "0 or 1", "+1 a step"),
            (px + 210, C_B, ORANGE_L, "2", "16", "i1", "0 or 1", "+16 a step"),
            (px + 320, C_C, AQUA_L, "8", "2", "c", "0 to 7", "+2 a step")]
    for cx, col, tint, *_ in cols:
        b.append(rect(cx - 28, T + 14, 56, 70, tint, rx=5))
    for yy, label, k in ((T + 42, "shape", 3), (T + 74, "stride", 4)):
        b.append(text(px, yy - 2, label, 12.5, MUTED))
        for xx, s in ((82, "("), (94, "("), (168, ","), (246, ")"), (256, ","), (360, ")")):
            b.append(text(px + xx, yy, s, 18, INK2, "middle", family=MONO))
        for c in cols:
            b.append(text(c[0], yy, c[k], 18, c[1], "middle", "700", MONO))
    for x0, x1, s in ((px + 102, px + 238, "the rows: 2 × 2 = 4"), (px + 292, px + 348, "the columns: 8")):
        b.append(f'<path d="M{x0},{T + 9} V{T + 3} H{x1} V{T + 9}" fill="none" stroke="{INK2}" stroke-width="1"/>')
        b.append(text((x0 + x1) / 2, T - 4, s, 12, INK2, "middle"))
    for cx, col, _, _, _, name, vals, stp in cols:
        b.append(text(cx, T + 104, name, 13, col, "middle", "700"))
        b.append(text(cx, T + 122, vals, 12, INK2, "middle"))
        b.append(text(cx, T + 140, stp, 12, col, "middle", "600"))

    # (2) what (2, 2) means: the row number written with two digits
    T += 190
    step(T, "2. (2, 2): the row number is written with two digits", [
        "One counter going 0, 1, 2, 3 is replaced by two",
        "digits i0 and i1, each 0 or 1, as in binary:",
        "row = i0 + 2·i1. i0 changes every row, i1 every",
        "second row. Each digit moves by its own stride,",
        "so a row starts at 1·i0 + 16·i1."])
    hx = [px + 20, px + 75, px + 120, px + 165]
    for x, s, col in zip(hx, ["row", "i0", "i1", "row starts at  1·i0 + 16·i1"], [MUTED, C_A, C_B, MUTED]):
        b.append(text(x, T, s, 12, col, "middle" if s != hx[3] and x != hx[3] else "start", "600"))
    for r in range(4):
        i0, i1 = r % 2, r // 2
        yy = T + 12 + 28 * r
        b.append(rect(px, yy, 400, 26, ROW_T[r], GRID, 1))
        b.append(text(hx[0], yy + 18, str(r), 13, INK, "middle", "600"))
        b.append(text(hx[1], yy + 18, str(i0), 13, C_A, "middle", "700"))
        b.append(text(hx[2], yy + 18, str(i1), 13, C_B, "middle", "700"))
        b.append(text(hx[3], yy + 18, f"1·{tsp(i0, C_A)} + 16·{tsp(i1, C_B)} = {tsp(i0 + 16 * i1, INK)}",
                      13, INK2, family=MONO))

    # (3) why: the row starts 0, 1, 16, 17 are not evenly spaced
    T += 165
    step(T, "3. Why nest? No single stride gives 0, 1, 16, 17", [
        "A single stride s puts the rows at 0, s, 2s, 3s,",
        "evenly spaced. These rows start +1, +15, +1 apart.",
        "Two digits, each with its own stride, can say it."])
    sc, lx = 17, px + 20

    def numline(yy, label, pts, arcs):
        b.append(text(px, yy - 44, label, 12, MUTED))
        b.append(line(lx, yy, lx + 25 * sc, yy, GRID, 1.5))
        for v in range(26):
            b.append(line(lx + v * sc, yy - 3, lx + v * sc, yy + 3, GRID))
        for v, fill in pts:
            b.append(f'<circle cx="{lx + v * sc}" cy="{yy}" r="6" fill="{fill}" stroke="{INK2}" stroke-width="1"/>')
            b.append(text(lx + v * sc, yy + 20, str(v), 11.5, INK, "middle"))
        for a, z, s in arcs:
            b.append(f'<path d="M{lx + a * sc},{yy - 9} Q{lx + (a + z) / 2 * sc},{yy - 30} {lx + z * sc},{yy - 9}" '
                     f'fill="none" stroke="{INK2}" stroke-width="1"/>')
            b.append(text(lx + (a + z) / 2 * sc, yy - 24, s, 11.5, INK, "middle", "600"))
    numline(T + 62, "this layout: where rows 0–3 start", [(0, ROW_T[0]), (1, ROW_T[1]), (16, ROW_T[2]), (17, ROW_T[3])],
            [(1, 16, "+15")])
    b.append(text(lx + 0.5 * sc, T + 48, "+1", 11.5, INK, "middle", "600"))
    b.append(text(lx + 16.5 * sc, T + 48, "+1", 11.5, INK, "middle", "600"))
    numline(T + 142, "one stride (here 8): always the same gap", [(0, SURF), (8, SURF), (16, SURF), (24, SURF)],
            [(0, 8, "+8"), (8, 16, "+8"), (16, 24, "+8")])

    # (4) the whole table, and the order it sits in memory
    T += 200
    step(T, "4. Add the column: each step in c adds 2", [])
    b.append(text(20, T + 24, f"offset = {tsp('1', C_A)}·i0 + {tsp('16', C_B)}·i1 + {tsp('2', C_C)}·c", 13, INK, family=MONO))
    para(20, T + 50, [
        "Outlined: row 3 (i0 = 1, i1 = 1), column 2:",
        f"1·1 + 16·1 + 2·2 = 21."])
    gx, gy, cw2 = px + 50, T + 6, 44
    for c in range(8):
        b.append(text(gx + c * cw2 + cw2 / 2, gy - 4, f"c={c}", 11.5, C_C, "middle"))
    for r in range(4):
        i0, i1 = r % 2, r // 2
        yy = gy + 4 + r * ch
        b.append(text(gx - 8, yy + 18, f"row {r}", 12, INK, "end"))
        for c in range(8):
            cell(gx + c * cw2, yy, str(i0 + 16 * i1 + 2 * c), ROW_T[r], cw2)
    b.append(rect(gx + 2 * cw2, gy + 4 + 3 * ch, cw2, ch, "none", INK, 2.5))

    my, mw = gy + 4 + 4 * ch + 44, 25
    para(20, my + 4, [
        "In memory the rows interleave: rows 0 and 1 take",
        "turns in offsets 0–15, rows 2 and 3 in 16–31.",
        "Each cell is shaded by the row it belongs to."])
    for k in range(2):
        yy = my - 10 + k * 30
        b.append(text(px + 42, yy + 17, f"{16 * k}–{16 * k + 15}", 11.5, MUTED, "end"))
        for j in range(16):
            off = 16 * k + j
            r = 2 * (off // 16) + off % 2
            b.append(rect(px + 50 + j * mw, yy, mw, 24, ROW_T[r], GRID, 1))
            b.append(text(px + 50 + j * mw + mw / 2, yy + 16, str(off), 11, INK, "middle"))
    b.append(text(px + 50, my + 64, "memory, offset by offset", 11.5, MUTED))
    return svg(920, my + 80, b)

def tiled_mma():
    """How the three arguments of make_tiled_mma build up, from one instruction to the 64 x 64 x 64 tile.
    Each panel draws S = Q Kt as a matrix multiply: Kt above S, Q to its left, so M, N and K are all visible."""
    b = []
    W, P = 160, 2.5                             # 64 elements -> 160 px
    FAINT, GREY_L = FILL, PANEL
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

        def hl(name, r0, c0, nr, nc, fill, stroke, sw=1.5, dash=None):
            x, y, _, _ = mats[name]
            r = rect(x + c0 * P, y + r0 * P, nc * P, nr * P, fill, stroke, sw)
            b.append(r.replace("/>", f' stroke-dasharray="{dash}"/>') if dash else r)

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
            hl("S", 16 * w, 8, 16, 56, "none", BLUE, 1.2, "4 3")   # the rest of this warp's rows: later steps
            hl("S", 16 * w, 0, 16, 8, BLUE_L, BLUE)
            note("S", 16 * w + 10, 11, f"warp {w}: rows {16 * w}–{16 * w + 15}")
        hl("Kt", 0, 0, 16, 8, BLUE_L, BLUE)
        note("Kt", 7, 11, "the same piece")
        note("Kt", 13, 11, "for all 4 warps")
    panel(R, T1, "2. The warps", "make_tiled_mma(op, (4, 1, 1), ...)",
          ["4 warps along M, 1 along N: each warp alone owns 16 rows × 64 keys.",
           "One step does keys 0–7 (solid); later steps do the rest (dashed)."],
          [("M", "16 × 4 warps = 64", BLUE, "600"), ("N", "8 × 1 warp = 8", INK, "normal"),
           ("K", "16 × 1 = 16", INK, "normal"), ("", "4 warps: (64, 8, 16)", INK2, "normal")], d2)

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
        note("Kt", 7, 19, "K for both mmas =")
        note("Kt", 13, 19, "one ldmatrix.x4")
    panel(L, T2, "3. A wider step", "permutation_mnk=(64, 16, 16)",
          ["Makes each step 16 keys wide, not 8: each warp runs 2 mmas side by side.",
           "Same math; one load instruction now fetches K for both mmas, not two."],
          [("M", "64  (as in panel 2)", INK, "normal"), ("N", "8 × 2 mma = 16", ORANGE, "600"),
           ("K", "16  (as in panel 2)", INK, "normal"), ("", "one step: (64, 16, 16)", INK2, "normal")], d3)

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

def mnk_shape():
    """(16, 8, 16) is (M, N, K): three 2-D matrices that each use two of the numbers, and the
    16 x 8 x 16 box of multiply-adds that is the only 3-D thing."""
    b = []
    c = 11                                       # px per element
    CM, CN, CK = BLUE, ORANGE, AQUA              # one colour per letter
    b.append(text(20, 30, "(16, 8, 16) is (M, N, K): three sizes of one matrix multiply, not a 3-D array", 16, weight="600"))
    b.append(text(20, 50, "C = A · B with A of size M × K, B of size K × N, C of size M × N. "
                  "Every matrix is 2-D and uses two of the three numbers.", 12.5, INK2))

    def rot(x, y, s, fill, size=12):
        return (f'<text transform="translate({x},{y}) rotate(-90)" font-size="{size}" fill="{fill}" '
                f'text-anchor="middle" font-weight="600">{s}</text>')

    def grid(x, y, rows, cols, fill=SURF):
        b.append(rect(x, y, cols * c, rows * c, fill, GRID, 1))
        for r in range(1, rows):
            b.append(line(x, y + r * c, x + cols * c, y + r * c, FILL, 1))
        for k in range(1, cols):
            b.append(line(x + k * c, y, x + k * c, y + rows * c, FILL, 1))
        b.append(rect(x, y, cols * c, rows * c, "none", INK2, 1.2))

    def spans(x, y, parts, size=12.5, anchor="start"):   # parts: (text, colour, weight)
        t = "".join(f'<tspan fill="{f}" font-weight="{w}">{s}</tspan>' for s, f, w in parts)
        return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{t}</text>'

    # 1. the three matrices
    b.append(text(20, 88, "1. Three matrices, each 2-D", 14, weight="600"))
    top, r0, c0 = 140, 5, 5                      # the highlighted value of C is (r0, c0)
    mats = [(70, 16, 16, "A = piece of Q", "query rows × head dims", ("M", CM), ("K", CK)),
            (310, 16, 8, "B = piece of Kᵀ", "head dims × keys", ("K", CK), ("N", CN)),
            (460, 16, 8, "C = piece of S", "query rows × keys", ("M", CM), ("N", CN))]
    for x, rows, cols, name, what, (rl, rc), (cl, cc) in mats:
        grid(x, top, rows, cols)
        b.append(spans(x + cols * c / 2, top - 8, [(f"{cl} = {cols}", cc, "600")], 12, "middle"))
        b.append(rot(x - 9, top + rows * c / 2, f"{rl} = {rows}", rc))
        b.append(text(x + cols * c / 2, top + rows * c + 20, name, 12.5, INK, "middle", "600"))
        b.append(spans(x + cols * c / 2, top + rows * c + 37,
                       [(rl, rc, "600"), (" × ", INK2, "normal"), (cl, cc, "600"),
                        (f" = {rows} × {cols}", INK2, "normal")], 12, "middle"))
        b.append(text(x + cols * c / 2, top + rows * c + 53, what, 11, MUTED, "middle"))
    b.append(text(272, top + 8 * c + 8, "×", 22, INK2, "middle"))
    b.append(text(424, top + 8 * c + 8, "=", 22, INK2, "middle"))
    b.append(rect(70, top + r0 * c, 16 * c, c, AQUA_L, AQUA, 1.5))        # row r0 of A
    b.append(rect(310 + c0 * c, top, c, 16 * c, AQUA_L, AQUA, 1.5))       # column c0 of B
    b.append(rect(460 + c0 * c, top + r0 * c, c, c, AQUA, AQUA, 1.5))     # one value of C
    cy = top + 16 * c + 100
    for i, s in enumerate(["One value of C (filled) = a row of A times a column of B:",
                           "16 products added up. That length, K, is the third number.",
                           "It is the side A and B share, and it is gone from C."]):
        b.append(text(20, cy + 18 * i, s, 12.5, INK2))

    # 2. the box of work
    X0 = 640
    b.append(text(X0 - 20, 88, "2. What is 16 × 8 × 16: the work", 14, weight="600"))
    dx, dy = 5.5, -3.7                           # one step along K
    fx, fy = X0 + 20, top - 16 * dy              # front face = C, depth = K
    ex, ey = 16 * dx, 16 * dy
    fw, fh = 8 * c, 16 * c
    top_face = f"{fx},{fy} {fx + fw},{fy} {fx + fw + ex},{fy + ey} {fx + ex},{fy + ey}"
    side_face = f"{fx + fw},{fy} {fx + fw + ex},{fy + ey} {fx + fw + ex},{fy + fh + ey} {fx + fw},{fy + fh}"
    b.append(f'<polygon points="{top_face}" fill="{PANEL}" stroke="{INK2}" stroke-width="1.2"/>')
    b.append(f'<polygon points="{side_face}" fill="{FILL}" stroke="{INK2}" stroke-width="1.2"/>')
    for k in range(1, 16):
        b.append(line(fx + k * dx, fy + k * dy, fx + fw + k * dx, fy + k * dy, GRID, 0.8))
        b.append(line(fx + fw + k * dx, fy + k * dy, fx + fw + k * dx, fy + fh + k * dy, GRID, 0.8))
    px, py = fx + (c0 + 0.5) * c, fy + (r0 + 0.5) * c
    b.append(line(px, py, px + ex, py + ey, AQUA, 2))
    for k in range(16):
        b.append(f'<circle cx="{px + (k + 0.5) * dx:.1f}" cy="{py + (k + 0.5) * dy:.1f}" r="2" fill="{AQUA}"/>')
    grid(fx, fy, 16, 8, "none")                  # see-through, so the rod runs behind it
    b.append(rect(fx + c0 * c, fy + r0 * c, c, c, AQUA, AQUA, 1.5))
    b.append(text(fx + fw / 2, fy + fh + 18, "N = 8", 12, CN, "middle", "600"))
    b.append(rot(fx - 9, fy + fh / 2, "M = 16", CM))
    b.append(text(fx + ex / 2 - 6, fy + ey / 2 - 4, "K = 16", 12, CK, "end", "600"))
    b.append(text(fx + fw + ex + 8, fy + fh / 2 + ey, "front face = C", 11, MUTED))
    for i, s in enumerate(["Each value of C is a rod of 16 products along K (dots).",
                           "16 × 8 values × 16 products = 2,048 multiply-adds,",
                           "all done by one mma instruction. Adding up each",
                           "rod squashes the box flat into C. No stored array",
                           "has this shape; it only counts the work."]):
        b.append(text(X0 - 20, cy + 18 * i, s, 12.5, INK2))
    return svg(960, cy + 18 * 5 + 16, b)

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
                b.append(rect(x + j * t, y + i * t, t, t, PANEL, GRID, 1))
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

    def lanes(x, y, vals, fill=PANEL, label=True):
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


def shuffle_args():
    """One line of quad_reduce taken apart argument by argument, then both lines traced on quad 0.
    Semantics from the PTX ISA (shfl.sync): partner j = lane ^ b; c[4:0] = clamp, c[12:8] = segment mask;
    maxLane = (lane & segmask) | (clamp & ~segmask); if j > maxLane the lane reads its own value."""
    import textwrap
    b = []
    GREY_L = PANEL
    b.append(f'<defs><marker id="ah7" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
             f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{BLUE}"/></marker></defs>')
    b.append(text(20, 30, "One shuffle, argument by argument", 16, weight="600"))

    def badge(x, y, n, r=9):
        b.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{INK}"/>')
        b.append(text(x, y + 4, str(n), 11, SURF, "middle", "700"))

    # the code line: op has two arguments, my v (orange) and the whole shuffle call, which evaluates to my
    # partner's v (green); the pieces of the call are numbered for the cards (textLength pins the monospace width)
    fs, cw, cy = 15, 9.0, 90

    def mono(x, y, s, fill=INK):
        b.append(f'<text x="{x}" y="{y}" font-size="{fs}" fill="{fill}" font-family="{MONO}" xml:space="preserve" '
                 f'textLength="{cw * len(s)}" lengthAdjust="spacingAndGlyphs">{s}</text>')

    # (text, role, badge): role "mine" = op's 1st argument, "call" = op's 2nd argument
    segs = [("v = ", None, 0), ("op", None, 1), ("( ", None, 0), ("v", "mine", 0), (" , ", None, 0),
            ("cute.arch.", "call", 0), ("shuffle_sync_bfly", "call", 2), ("(", "call", 0), ("v", "call", 3),
            (", ", "call", 0), ("offset=2", "call", 4), (", ", "call", 0), ("mask=-1", "call", 5),
            (", ", "call", 0), ("mask_and_clamp=31", "call", 6), (")", "call", 0), (" )", None, 0)]
    x0 = 490 - cw * sum(len(s) for s, _, _ in segs) / 2
    pos, x = [], x0
    for s, role, n in segs:
        pos.append(x)
        x += cw * len(s)
    b.append(rect(x0 - 24, 42, x - x0 + 48, 150, GREY_L, GRID, 1, 6))
    span = {r: (min(p for p, (_, rr, _) in zip(pos, segs) if rr == r),
                max(p + cw * len(s) for p, (s, rr, _) in zip(pos, segs) if rr == r)) for r in ("mine", "call")}
    style = {"mine": (ORANGE_L, ORANGE, ORANGE_D), "call": (AQUA_L, AQUA, AQUA_D)}
    for r, (a, z) in span.items():                  # the two arguments of op, drawn under everything else
        b.append(rect(a - 6, cy - 21, z - a + 12, 32, style[r][0], style[r][1], 1.8, 5))
    for (s, role, n), p in zip(segs, pos):
        if n:
            b.append(rect(p - 2, cy - 15, cw * len(s) + 4, 21, BLUE_L, BLUE, 1, 3))
            badge(p + cw * len(s) / 2, cy - 31, n)
        mono(p, cy, s)
    (ma, mz), (ca, cz) = span["mine"], span["call"]
    b.append(text((ma + mz) / 2, cy + 30, "my v", 12.5, ORANGE_D, "middle", "700"))
    b.append(text((ma + mz) / 2, cy + 45, "op's 1st argument", 10.5, INK2, "middle"))
    b.append(text((ca + cz) / 2, cy + 30, "my partner's v: what this whole call returns", 12.5, AQUA_D, "middle", "700"))
    b.append(text((ca + cz) / 2, cy + 45, "op's 2nd argument. The v inside the call is my v again: the value I send.",
                  10.5, INK2, "middle"))

    # the same line with lane 0's numbers
    ey = cy + 82
    b.append(text(x0, ey, "On lane 0:", 12, MUTED))
    ex = x0 + 80
    for s, role in [("v = fmax( ", None), ("3", "mine"), (" , ", None), ("2", "call"), (" ) = 3", None)]:
        if role:
            b.append(rect(ex - 5, ey - 16, cw * len(s) + 10, 23, style[role][0], style[role][1], 1.8, 4))
        mono(ex, ey, s)
        ex += cw * len(s)
    b.append(text(ex + 20, ey, "lane 0 sends its 3 and, in the same instruction, lane 2 sends its 2", 12, INK2))

    # one card per argument
    CW, CH, X0, X1, Y0 = 460, 184, 20, 500, 206
    RH = [290, 312, 686]                             # taller rows for card 2 (bfly vs down), 4 (XOR), 5 (masks)

    def card(col, row, n, name, body, draw=None):
        x, y = (X0, X1)[col], Y0 + sum(RH[:row]) + row * 14
        b.append(rect(x, y, CW, RH[row], SURF, GRID, 1, 6))
        badge(x + 20, y + 22, n)
        b.append(text(x + 38, y + 27, name, 13.5, INK, weight="600", family=MONO))
        ty = y + 50
        for para in body:
            for ln in textwrap.wrap(para, 70):
                b.append(text(x + 16, ty, ln, 11.5, INK2))
                ty += 16
            ty += 4
        if draw:
            draw(x + 16, ty + 4)

    def lane_row(x, y, n, partner=None, cw_=40, quads=False):
        for i in range(n):
            b.append(rect(x + i * cw_, y, cw_ - 4, 20, BLUE_L if (i < 4) else GREY_L, GRID, 1, 3))
            b.append(text(x + i * cw_ + (cw_ - 4) / 2, y + 14, str(i), 10.5, INK2, "middle"))
        if partner:
            for i in range(n):
                j = partner(i)
                if j > i:
                    x1, x2 = x + i * cw_ + (cw_ - 4) / 2, x + j * cw_ + (cw_ - 4) / 2
                    h = 14 + 12 * (i % 2)               # 1-3 arcs above 0-2, so the arrows stay apart
                    b.append(f'<path d="M{x1},{y - 2} C{x1},{y - h} {x2},{y - h} {x2},{y - 2}" fill="none" '
                             f'stroke="{BLUE}" stroke-width="1.3" marker-start="url(#ah7)" marker-end="url(#ah7)"/>')

    def draw_op(x, y):
        # the two call sites in the kernel, op highlighted
        for k, (pre, op, lab) in enumerate([("quad_reduce(…, ", "cute.arch.fmax", "row max"),
                                            ("quad_reduce(…, ", "lambda x, y: x + y", "row sum")]):
            yy = y + k * 22
            b.append(text(x, yy + 14, lab, 11.5, MUTED))
            px, ccw = x + 60, 7.2                       # textLength pins the monospace width
            ox = px + ccw * len(pre)
            b.append(rect(ox - 2, yy, ccw * len(op) + 4, 19, BLUE_L, BLUE, 1, 3))
            for s, sx in [(pre, px), (op, ox), (")", ox + ccw * len(op))]:
                b.append(f'<text x="{sx}" y="{yy + 14}" font-size="12" fill="{INK}" font-family="{MONO}" '
                         f'xml:space="preserve" textLength="{ccw * len(s)}" lengthAdjust="spacingAndGlyphs">{s}</text>')
        y += 52
        b.append(text(x, y + 14, "lane 0:", 11.5, MUTED))
        b.append(text(x + 60, y + 14, "fmax(3, 2) = 3", 12.5, INK, family=MONO))
        b.append(text(x + 190, y + 14, "my v, then my partner's", 11.5, MUTED))

    def draw_bfly(x, y):
        # which lanes' values each lane of quad 0 has combined, after each line of quad_reduce
        groups = [("bfly", "lane ^ offset", [["0", "1", "2", "3"], ["0,2", "1,3", "0,2", "1,3"], ["0–3"] * 4],
                   "all 4 lanes hold the row max", AQUA_D),
                  ("down", "lane + offset", [["0", "1", "2", "3"], ["0,2", "1,3", "2", "3"], ["0–3", "1–3", "2,3", "3"]],
                   "only lane 0 holds the row max", ORANGE_D)]
        labels = ["start", "offset=2", "offset=1"]
        for r, lab in enumerate(labels):
            b.append(text(x, y + 34 + r * 24, lab, 11, BLUE if r else MUTED, weight="600" if r else "normal",
                          family=MONO if r else None))
        for g, (mode, rule, rows, verdict, vcol) in enumerate(groups):
            gx = x + 66 + g * 186
            b.append(text(gx, y + 10, mode, 12, INK, weight="700", family=MONO))
            b.append(text(gx + 38, y + 10, "partner = " + rule, 10.5, INK2))
            for r, cells in enumerate(rows):
                for i, c in enumerate(cells):
                    final = r == len(rows) - 1
                    fill, edge = GREY_L, GRID
                    if final:
                        fill, edge = (AQUA_L, AQUA) if c == "0–3" else (ORANGE_L, ORANGE)
                    cx = gx + i * 42
                    b.append(rect(cx, y + 20 + r * 24, 38, 20, fill, edge, 1.3 if final else 1, 3))
                    b.append(text(cx + 19, y + 34 + r * 24, c, 11, INK, "middle"))
            b.append(text(gx, y + 110, verdict, 11.5, vcol, weight="700"))
        b.append(text(x, y + 130, "Cells: the lanes whose values this lane has combined. down is shown cut into quads,",
                      10.5, MUTED))
        b.append(text(x, y + 144, "so a lane whose partner is past lane 3 keeps its own v.", 10.5, MUTED))

    def draw_v(x, y):
        b.append(text(x, y + 14, "sends", 11.5, MUTED))
        b.append(rect(x + 44, y, 40, 20, BLUE_L, BLUE, 1, 3))
        b.append(text(x + 64, y + 14, "v", 12, INK, "middle", family=MONO))
        b.append(f'<line x1="{x + 92}" y1="{y + 10}" x2="{x + 150}" y2="{y + 10}" stroke="{BLUE}" stroke-width="1.3" '
                 f'marker-start="url(#ah7)" marker-end="url(#ah7)"/>')
        b.append(text(x + 121, y + 32, "one instruction", 10, MUTED, "middle"))
        b.append(rect(x + 158, y, 120, 20, GREY_L, GRID, 1, 3))
        b.append(text(x + 218, y + 14, "partner's v", 11.5, INK2, "middle"))
        b.append(text(x + 290, y + 14, "returned", 11.5, MUTED))

    def draw_offset(x, y):
        # lane ^ offset for the 4 lanes of a quad, in binary, with the flipped bit highlighted
        def bits(v, flip=None):                      # 2-bit binary, bit `flip` in bold blue
            out = ""
            for k in (1, 0):
                d = str((v >> k) & 1)
                out += (f'<tspan fill="{BLUE}" font-weight="700">{d}</tspan>' if k == flip else d)
            return out

        def mono_t(xx, yy, inner, fill=INK):
            b.append(f'<text x="{xx}" y="{yy}" font-size="12" fill="{fill}" font-family="{MONO}">{inner}</text>')

        cols = [x, x + 54, x + 140, x + 290]
        b.append(text(cols[0], y + 12, "lane", 10.5, MUTED))
        b.append(text(cols[1], y + 12, "binary", 10.5, MUTED))
        for c, off, lab in [(cols[2], 2, "line 1: lane ^ 0b10"), (cols[3], 1, "line 2: lane ^ 0b01")]:
            mono_t(c, y + 12, f"offset={off}", BLUE)
            b.append(text(c + 64, y + 12, lab.split(": ")[1], 10.5, MUTED))
        b.append(line(x, y + 18, x + 420, y + 18, GRID))
        for i in range(4):
            ry = y + 36 + i * 20
            mono_t(cols[0] + 10, ry, str(i))
            mono_t(cols[1], ry, bits(i))
            for c, off, k in [(cols[2], 2, 1), (cols[3], 1, 0)]:
                mono_t(c, ry, f"{bits(i ^ off, k)} = lane {i ^ off}")
        b.append(text(x, y + 124, "Quad 1 (lanes 4–7) works the same: 5 ^ 2 = 7, 6 ^ 1 = 7. Bits 2–4 of a lane say which",
                      10.5, MUTED))
        b.append(text(x, y + 138, "quad it is in, and never flip, so a partner is always in the same quad.", 10.5, MUTED))

    def draw_mask(x, y):
        # masks as 32 squares, lane 0 on the left; filled = that lane is listed
        y += 8                                       # room for the two-line quad labels
        rows = [("-1", 0xFFFFFFFF, "all 32 ✓", AQUA_D), ("15", 0xF, "quad 0 only ✗", ORANGE_D),
                ("286331153", 0x11111111, "0, 4, …, 28 ✗", ORANGE_D),
                ("63281", 0xF731, "0, 4–5, 8–10,\n12–15 ✗", ORANGE_D), ("1", 1, "lane 0 only ✗", ORANGE_D),
                ("2", 2, "lane 1 only ✗", ORANGE_D), ("0", 0, "no lane ✗", ORANGE_D)]
        sx, st = x + 72, 8
        b.append(text(x, y - 2, "mask: hex", 10, MUTED))
        b.append(text(x, y + 10, "= decimal", 10, MUTED))
        for q in range(8):                           # one label per quad: quad 0 = lanes 0-3, the quad traced above
            qx = sx + q * (4 * st + 3) + (4 * st - 2) / 2
            on = q == 0
            b.append(text(qx, y - 2, f"quad {q}", 9.5, BLUE if on else MUTED, "middle", "700" if on else "normal"))
            b.append(text(qx, y + 10, f"{4 * q}–{4 * q + 3}", 9.5, BLUE if on else MUTED, "middle"))
        RW = 30                                      # row pitch: hex on one line, its decimal under it
        for r, (dec, val, res, col) in enumerate(rows):
            ry = y + 18 + r * RW
            b.append(text(x, ry + 11, f"0x{val:X}", 10, INK, family=MONO))
            b.append(text(x, ry + 24, f"= {dec}", 9, MUTED, family=MONO))
            for i in range(32):
                on = (val >> i) & 1
                b.append(rect(sx + i * st + (i // 4) * 3, ry + 4, 5, 15, BLUE if on else SURF, BLUE if on else GRID, 1, 1))
            lines = res.split("\n")                 # a long result takes two lines
            for k, ln in enumerate(lines):
                b.append(text(sx + 32 * st + 7 * 3 + 8, ry + 16 + (k - (len(lines) - 1) / 2) * 13, ln, 11, col,
                              weight="700"))
        ny = y + 18 + len(rows) * RW + 12
        notes = [("One bit is one lane: 1 = 0b0001 lists lane 0 alone, not quad 0. Quad 0 takes four bits, "
                  "0xF = 0b1111. Each hex digit is one quad, so 0x11111111 puts 0001 in every quad: the first "
                  "lane of each (0, 4, …, 28). 0xF731 builds a mask quad by quad: 1 = 0001 (lane 0), "
                  "3 = 0011 (lanes 4, 5), 7 = 0111 (lanes 8–10), F = 1111 (lanes 12–15). Binary is written lane 31 … lane 0; the squares run lane 0 → 31. "
                  "0xFFFFFFFF read as an unsigned number is 4294967295; as a signed 32-bit int it is -1, "
                  "which is what the code passes.", INK2),
                 ("✗: all 32 lanes run the shuffle, but some are not listed, and lanes read unlisted partners. "
                  "The hardware promises nothing: v may come back as garbage.", MUTED)]
        for para, col in notes:
            for ln in textwrap.wrap(para, 86):
                b.append(text(x, ny, ln, 10.5, col))
                ny += 14
            ny += 6

    def draw_clamp(x, y):
        def para(s, fill=INK2, weight="normal", size=11.5, w=70, lh=16):
            nonlocal y
            for ln in textwrap.wrap(s, w):
                b.append(text(x, y, ln, size, fill, weight=weight))
                y += lh
            y += 4

        # what the number means: two settings drawn as the 32 lanes cut into groups
        y -= 4
        para("The number holds two settings, the way 125 minutes is 2 hours and 5 minutes. "
             "Split it as big × 256 + small:")
        y -= 2
        for k, s_ in enumerate(["small: how far a lane may read inside its group. 31 means to the end.",
                                "big: sets the group size, 32 − big lanes (the hardware’s convention)."]):
            b.append(text(x + 8, y, "•", 11.5, INK2))
            b.append(text(x + 20, y, s_, 11.5, INK2))
            y += 16
        y += 8
        st = 12                                      # one square per lane
        for val, big, size, res in [("31", 0, 32, "one group, no walls"), ("7199", 28, 4, "a wall every 4 lanes")]:
            b.append(f'<text x="{x}" y="{y + 2}" font-size="12" fill="{INK}" font-family="{MONO}" xml:space="preserve">'
                     f'<tspan fill="{BLUE}" font-weight="700">{val:<4}</tspan> = {big:>2} × 256 + 31</text>')
            b.append(text(x + 172, y + 2, f"big {big}: groups of 32 − {big} = {size},", 11, INK2))
            b.append(text(x + 172, y + 16, res, 11, INK, weight="600"))
            y += 24
            for i in range(32):
                gx = x + i * st + (i // size) * 6
                b.append(rect(gx, y, st - 2, 14, BLUE_L if i < 4 else GREY_L, GRID, 1, 2))
                if i % size == 0 and i:              # a wall between two groups
                    b.append(line(gx - 4, y - 3, gx - 4, y + 17, ORANGE, 2.5))
            y += 32
        para("One square per lane, lane 0 on the left; blue is quad 0. In the name, “mask” is big and "
             "“clamp” is small.", MUTED, size=10.5, w=82, lh=14)

        # why it does not matter here
        para("Why it changes nothing in quad_reduce", INK, "600")
        y -= 4
        para("bfly’s partners, lane ^ 2 and lane ^ 1 (card 4), are always in the same quad, so they never cross "
             "a wall, even with groups of 4. 31 is also CuTe’s default; the code just spells it out.")

        # when it does matter
        para("When it matters", INK, "600")
        y -= 4
        para("down mode (card 2): partner = lane + 2. Quad 0 (lanes 0–3) works on one row of S and quad 1 "
             "(lanes 4–7) on the next, so lanes 2 and 3 would read lanes 4 and 5: another row’s values.")
        vals = [3, 7, 2, 5, 6, 1, 4, 8]              # quad 0's v, then quad 1's
        cx0, cs = x + 128, 46
        qx = cx0 + 4 * cs + 10                       # lanes 4, 5 sit past a gap
        y += 22
        b.append(text(x, y + 14, "each lane’s v", 11, INK2))
        b.append(text(cx0 + 2 * cs - 2, y - 18, "quad 0", 10, BLUE, "middle", "700"))
        b.append(text(qx + cs - 2, y - 18, "quad 1", 10, ORANGE_D, "middle", "700"))
        for i in range(6):
            cx = cx0 + i * cs if i < 4 else qx + (i - 4) * cs
            fill = BLUE_L if i < 4 else ORANGE_L
            b.append(rect(cx, y, cs - 4, 20, fill, GRID, 1, 3))
            b.append(text(cx + (cs - 4) / 2, y + 14, str(vals[i]), 11.5, INK, "middle"))
            b.append(text(cx + (cs - 4) / 2, y - 4, f"lane {i}", 9.5, MUTED, "middle"))
        y += 40
        b.append(text(x, y, "What lanes 0–3 receive from line 1, under three settings:", 11, INK2))
        y += 8
        cases = [("bfly", 32, "bfly, 31", "✓ as in quad_reduce"),
                 ("down", 32, "down, 31", "✗ lanes 2, 3 get\nquad 1’s 6 and 1"),
                 ("down", 4, "down, 7199", "✓ walls stop them:\nthey keep their v")]
        for mode, size, lab, verdict in cases:
            b.append(text(x, y + 19, lab, 11.5, INK, weight="600", family=MONO))
            for l in range(4):
                j = l ^ 2 if mode == "bfly" else l + 2
                if j // size != l // size:
                    j = l                            # partner past a wall: own v back
                own, cross = j == l, j // 4 != 0
                fill, edge = (ORANGE_L, ORANGE) if cross else (SURF, GRID)
                xx = cx0 + l * cs
                b.append(f'<rect x="{xx}" y="{y}" width="{cs - 4}" height="30" rx="3" fill="{fill}" stroke="{edge}" '
                         f'stroke-width="1"' + (' stroke-dasharray="3,2"' if own else '') + '/>')
                b.append(text(xx + (cs - 4) / 2, y + 13, str(vals[j]), 11.5, INK, "middle", "600"))
                b.append(text(xx + (cs - 4) / 2, y + 25, "own" if own else f"from {j}", 9, MUTED, "middle"))
            vcol = ORANGE_D if verdict.startswith("✗") else AQUA_D
            for k, ln in enumerate(verdict.split("\n")):
                b.append(text(cx0 + 4 * cs + 4, y + 13 + k * 13, ln, 10.5, vcol, weight="700"))
            y += 36
        y += 4
        para("Each box: the value the lane receives (big) and the lane it came from (small). "
             "Dashed: the partner was past a wall, so the lane got its own v back.", MUTED, size=10.5, w=82, lh=14)

    card(0, 0, 1, "op(my v, partner's v)", [
        "Combines my v with my partner's v, the value the whole shuffle call returns, and stores the result "
        "back in v. op itself is passed in where the kernel calls quad_reduce:"], draw_op)
    card(1, 0, 2, "shuffle_sync_bfly", [
        "The PTX instruction shfl.sync in bfly (butterfly) mode: partner\u00a0=\u00a0lane\u00a0^\u00a0offset (XOR, see 4), so lanes swap in pairs.",
        "Why bfly: every lane needs the row max to rescale its own 16 scores. down mode (partner = lane + offset) "
        "would leave it in lane 0 alone:"], draw_bfly)
    card(0, 1, 3, "v", [
        "My v again, the same register as op's 1st argument: the value this lane sends, here the maximum of its "
        "own 16 scores of the row. Every lane sends its v and gets its partner's back, in one instruction."], draw_v)
    card(1, 1, 4, "offset=2", [
        "^ is bitwise XOR, not a power. Write lane and offset in binary; each 1 bit of offset flips that bit of "
        "lane. offset=2 (0b10) flips bit 1, offset=1 (0b01) flips bit 0.",
        "Not 2 instead of 1: line 1 uses 2, line 2 uses 1. A quad's lanes 0–3 take 2 bits, and each line combines "
        "across one bit, so after both every lane has all 4. Either order works."], draw_offset)
    card(0, 2, 5, "mask=-1", [
        "Which lanes take part: one bit per lane, bit i for lane i. The other cards follow quad 0 (lanes 0–3), but "
        "one shuffle runs the whole warp: all 8 quads (lanes 0–31) swap at once, each on its own row. So the mask "
        "covers 32 lanes. -1 in 32 bits is all ones, 0xFFFFFFFF (add 1 and it wraps round to 0): every lane.",
        "Why: a shuffle reads my partner's register, so my partner must run the shuffle too. The mask lists the "
        "lanes that will, so the hardware knows whom to wait for. In quad_reduce every lane runs it: all 32."],
         draw_mask)
    card(1, 2, 6, "mask_and_clamp=31", [
        "Walls between lanes. It cuts the 32 lanes into equal groups, and a lane may read only from a partner "
        "in its own group. If the partner is past a wall, the lane gets its own v back instead. "
        "Not the same as mask (5): mask says which lanes run the shuffle."], draw_clamp)

    # trace both lines of quad_reduce on quad 0
    ty = Y0 + sum(RH) + 3 * 14 + 16
    b.append(text(20, ty, "Both lines of quad_reduce on quad 0, for the maximum of row 0", 13.5, weight="600"))
    b.append(text(20, ty + 18, "Each lane starts with the maximum of its own 16 scores. Quads 1–7 do the same "
                  "at the same time, on rows 1–7.", 11.5, INK2))
    cols = ["", "lane 0", "lane 1", "lane 2", "lane 3"]
    rows = [("", "v (own maximum)", [3, 7, 2, 5], False),
            ("offset=2", "partner = lane ^ 2", [2, 3, 0, 1], False),
            ("", "partner's v (received)", [2, 5, 3, 7], False),
            ("", "v = fmax(v, received)", [3, 7, 3, 7], True),
            ("offset=1", "partner = lane ^ 1", [1, 0, 3, 2], False),
            ("", "partner's v (received)", [7, 3, 7, 3], False),
            ("", "v = fmax(v, received)", [7, 7, 7, 7], True)]
    gx, lw, colw, rh = 20, 250, 80, 24
    y = ty + 44
    for i, c in enumerate(cols[1:]):
        b.append(text(gx + lw + i * colw + colw / 2, y, c, 11, MUTED, "middle"))
    y += 8
    for k, (step, lab, vals, res) in enumerate(rows):
        if step:
            b.append(line(gx, y, gx + lw + 4 * colw, y, GRID))
            b.append(text(gx, y + 16, step, 11.5, BLUE, weight="600", family=MONO))
        b.append(text(gx + 80, y + 16, lab, 11.5, INK if res else INK2, weight="600" if res else "normal"))
        for i, v in enumerate(vals):
            cx = gx + lw + i * colw + 6
            b.append(rect(cx, y + 2, colw - 12, rh - 4, BLUE_L if res else GREY_L, BLUE if res else GRID, 1, 3))
            b.append(text(cx + (colw - 12) / 2, y + 16, str(v), 12, INK, "middle"))
        y += rh
    nx = gx + lw + 4 * colw + 30
    b.append(text(nx, ty + 70, "After line 1, lanes 0 and 2 hold max(lanes 0, 2),", 11.5, INK2))
    b.append(text(nx, ty + 86, "and lanes 1 and 3 hold max(lanes 1, 3).", 11.5, INK2))
    b.append(text(nx, ty + 142, "After line 2 every lane holds 7, the maximum", 11.5, INK2))
    b.append(text(nx, ty + 158, "of the row: 2 shuffles for 4 lanes.", 11.5, INK2))
    b.append(text(nx, ty + 190, "For the row sum, op is + and the same two", 11.5, INK2))
    b.append(text(nx, ty + 206, "swaps leave every lane with 3 + 7 + 2 + 5.", 11.5, INK2))
    return svg(980, y + 20, b)


def warp_split():
    # Where row 0 of S lives under (4, 1, 1) and (2, 2, 1), and what its maximum costs. Ownership checked on
    # the GPU: under (2, 2, 1) warps 0 and 2 take turns every 8 keys of rows 0-15 and 32-47.
    b = [f'<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>']
    b.append(text(20, 30, "Where row 0 of S lives, and what its maximum costs", 16, weight="600"))
    fills = [BLUE_L, ORANGE_L, AQUA_L, GOLD_L]
    strong = [BLUE, ORANGE, AQUA, GOLD]
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
    """Every line of the copy of K into shared memory, followed for thread 19, one section per line."""
    # Checked on the GPU with CuTe DSL (bf16, D = 64, 128 threads, 256 keys):
    #   g2s tiler (16, 64)   gK (64,64,4):(64,1,4096)
    #   tKgK ((8,1),4,1,4):((1,0),1024,0,4096)   tKsK ((8,1),4,1):((1,0),1024,0)
    #   thread 19, step m: row 2 + 16 m of the tile, values 24-31
    import textwrap
    W, TX, TW = 1040, 640, 58
    FAINT, HL, DIM = PANEL, GOLD_L, DIM_LINE
    TR, TC = 2, 3                                    # thread 19's row and column in the 16 x 8 grid
    G2S = ["g2s = cute.make_tiled_copy_tv(",
           "    cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128),  # the atom",
           "    cute.make_layout((self.threads // tpr, tpr), stride=(tpr, 1)),  # threads: (16, 8):(8, 1)",
           "    cute.make_layout((1, vec)))                                     # values:  (1, 8)"]
    b = [f'<defs><marker id="ahc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>',
         text(20, 30, "Copying a tile of K into shared memory, line by line", 17, weight="600"),
         text(20, 52, "A tile is a 64 × 64 block of K: 64 keys × 64 values. 128 threads, numbered 0 to 127, copy it "
                      "together. They all run the same", 12.5, INK2),
         text(20, 70, "lines, but each copies a different part. The figure follows one of them, thread 19, and draws "
                      "its part in orange.", 12.5, INK2),
         text(20, 88, "19 is just an example; every thread works the same way. Thread 0 would hide the arithmetic, "
                      "since its numbers are all zero.", 12.5, INK2)]

    def para(y, s, size=12.5, fill=INK2, lh=18, n=TW):
        for ln in textwrap.wrap(s, n):
            b.append(text(TX, y, ln, size, fill))
            y += lh
        return y + 8

    def header(y, n, title, code, hl=None):
        # hl: indices of the lines this section explains; the other lines are greyed out
        b.append(line(20, y - 26, W - 20, y - 26, GRID))
        b.append(f'<circle cx="31" cy="{y - 5}" r="11" fill="{INK}"/>')
        b.append(text(31, y, str(n), 12.5, SURF, "middle", "700"))
        b.append(text(50, y, title, 15, weight="600"))
        y += 14
        h = 21 * len(code) + 12
        b.append(rect(20, y, W - 40, h, FAINT, GRID, rx=4))
        for i, c in enumerate(code):
            yy = y + 21 + 21 * i
            on = hl is None or i in hl
            if hl is not None and on:
                b.append(rect(21, yy - 15, W - 42, 21, HL))
            k = c.find("#")
            code_part = c if k < 0 else c[:k].rstrip()
            ind = len(code_part) - len(code_part.lstrip(" "))
            b.append(text(34 + 7.53 * ind, yy, code_part.lstrip(" "), 12.5, INK if on else DIM, family=MONO))
            if k >= 0:
                b.append(text(34 + 7.53 * k, yy, c[k:], 12.5, MUTED if on else DIM, family=MONO))
        return y + h + 34

    def arrow(x1, y1, x2, y2, label=None, dy=-7):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.4" '
                 f'marker-end="url(#ahc)"/>')
        if label:
            b.append(text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, 11.5, INK, "middle", "600"))

    def piece(x, y, cs, ch, fill=ORANGE_L, stroke=ORANGE, labels=None):
        # one piece: 8 values side by side
        for v in range(8):
            b.append(rect(x + v * cs, y, cs, ch, fill, stroke, 0.8))
            if labels:
                b.append(text(x + v * cs + cs / 2, y + ch / 2 + 4, labels[v], 10.5, INK, "middle"))

    def tile(x0, y0, px, owner_rows=(2, 18, 34, 50), band=None, lines=True):
        # a 64 x 64 tile; thread 19's pieces in orange; band = rows covered by one step, in light blue
        b.append(rect(x0, y0, 64 * px, 64 * px, FAINT, GRID))
        if band is not None:
            b.append(rect(x0, y0 + band * 16 * px, 64 * px, 16 * px, BLUE_L))
        if lines:
            for c in range(8, 64, 8):
                b.append(line(x0 + c * px, y0, x0 + c * px, y0 + 64 * px, GRID, 0.6))
            for r in range(16, 64, 16):
                b.append(line(x0, y0 + r * px, x0 + 64 * px, y0 + r * px, INK2, 1))
        for r in owner_rows:
            b.append(rect(x0 + 8 * TC * px, y0 + r * px, 8 * px, max(px, 3), ORANGE))

    def chips(x, y, steps, cs, ch, pitch):
        # thread 19's pieces as a tensor: one piece per step, stacked
        for m in range(steps):
            piece(x, y + m * pitch, cs, ch)

    # ---- 1: vec and tpr ------------------------------------------------------------------------
    y = header(134, 1, "How big is one piece?",
               ["vec = 128 // dtype.width                   # 8 bf16 values = 16 bytes per thread per step",
                "tpr = D // vec                             # 64 // 8 = 8 threads per row of the tile"])
    x0, cw, gap, ry = 56, 7, 6, y + 30
    b.append(text(x0, ry - 12, "One row of K: D = 64 values", 12, INK, weight="600"))
    for v in range(64):
        p = v // 8
        b.append(rect(x0 + v * cw + p * gap, ry, cw, 18, ORANGE_L if p == 0 else FAINT, GRID, 0.6))
    for p in range(8):
        px0 = x0 + p * (8 * cw + gap)
        b.append(text(px0 + 4 * cw, ry + 34, f"piece {p}", 10.5, INK, "middle"))
        b.append(text(px0 + 4 * cw, ry + 48, f"{8 * p}–{8 * p + 7}", 10, MUTED, "middle"))
    zy = ry + 92
    b.append(line(x0, ry + 54, x0, zy, ORANGE, 1, "3 3"))
    b.append(line(x0 + 8 * cw, ry + 54, x0 + 8 * 44, zy, ORANGE, 1, "3 3"))
    piece(x0, zy, 44, 24, labels=["2 bytes"] * 8)
    b.append(text(x0, zy + 46, "piece 0, zoomed in: 8 values × 2 bytes = 16 bytes = 128 bits", 12, INK, weight="600"))
    b.append(text(x0, zy + 66, "8 pieces per row, so 8 threads share each row: tpr = 8", 12, INK, weight="600"))
    pic_bottom = zy + 66
    ty = para(y, "dtype.width is the size of one value in bits. K is stored in bf16, a number format that "
                 "uses 16 bits (2 bytes) per value. So vec = 128 // 16 = 8 values.")
    ty = para(ty, "Why 128? It is the most one thread can move with a single copy instruction: 128 bits, "
                  "16 bytes. Below, the 8 values one thread copies at a time are called a piece.")
    ty = para(ty, "D is the head dimension: the length of one row of K, 64 values. A row holds 64 // 8 = 8 "
                  "pieces, so tpr = 8 threads share each row.")
    y = max(pic_bottom, ty) + 52

    # ---- 2: the atom ---------------------------------------------------------------------------
    y = header(y, 2, "The atom: the one instruction that moves a piece", G2S, hl={1})
    by, bw, bh = y + 20, 140, 104
    boxes = [(56, "global memory", "the GPU's main memory:", "big, slow", GOLD_L),
             (256, "registers", "thread 19's own storage:", "tiny, fastest", BLUE_L),
             (456, "shared memory", "on-chip memory that all", "128 threads of a block see", AQUA_L)]
    for bx, name, s1, s2, fill in boxes:
        b.append(rect(bx, by, bw, bh, fill, GRID, 1, 8))
        b.append(text(bx + bw / 2, by + 22, name, 12.5, INK, "middle", "600"))
        b.append(text(bx + bw / 2, by + 40, s1, 10.5, INK2, "middle"))
        b.append(text(bx + bw / 2, by + 54, s2, 10.5, INK2, "middle"))
        piece(bx + 14, by + 70, 14, 16)
    arrow(56 + bw + 4, by + 78, 256 - 6, by + 78, "1 load")
    arrow(256 + bw + 4, by + 78, 456 - 6, by + 78, "1 store")
    b.append(text(56, by + bh + 26, "CopyUniversalOp: a plain load into registers, then a plain store.", 12, INK,
                  weight="600"))
    b.append(text(56, by + bh + 44, "Each moves 128 bits = 16 bytes = one piece.", 12, INK2))
    pic_bottom = by + bh + 44
    ty = para(y, "An atom is the one instruction a thread runs to move one piece. make_copy_atom builds it from "
                 "three choices:")
    ty = para(ty, "CopyUniversalOp: the plain kind of copy. The thread loads its piece from global memory into "
                  "its registers, then stores it into shared memory.")
    ty = para(ty, "dtype: the values are bf16. num_bits_per_copy=128: one instruction moves 128 bits, a whole "
                  "piece.")
    ty = para(ty, "Change only this line and the instruction changes, and with it the speed. The next figure "
                  "compares atoms.")
    y = max(pic_bottom, ty) + 52

    # ---- 3: the thread layout ------------------------------------------------------------------
    y = header(y, 3, "The thread layout: where each thread sits", G2S, hl={2})
    x0, gy, cw, ch = 104, y + 50, 40, 17
    b.append(text(x0 + 4 * cw, gy - 36, "the 64 values of a row of K", 11, INK2, "middle"))
    for c in range(8):
        b.append(text(x0 + c * cw + cw / 2, gy - 20, f"col {c}", 10.5, INK if c == TC else MUTED, "middle"))
        b.append(text(x0 + c * cw + cw / 2, gy - 6, f"{8 * c}–{8 * c + 7}", 9.5, MUTED, "middle"))
    for r in range(16):
        b.append(text(x0 - 6, gy + r * ch + 12, f"row {r}", 10.5, INK if r == TR else MUTED, "end"))
        for c in range(8):
            hi = (r, c) == (TR, TC)
            base = FAINT if (r // 4) % 2 == 0 else BLUE_XL
            b.append(rect(x0 + c * cw, gy + r * ch, cw, ch, ORANGE_L if hi else base, ORANGE if hi else GRID,
                          2 if hi else 1))
            b.append(text(x0 + c * cw + cw / 2, gy + r * ch + 12, f"t{8 * r + c}", 9.5, INK, "middle",
                          "700" if hi else "normal"))
    gx = x0 + 8 * cw
    for w in range(4):
        wy = gy + 4 * w * ch
        b.append(line(gx + 8, wy + 2, gx + 8, wy + 4 * ch - 2, INK2, 1.5))
        b.append(text(gx + 14, wy + 2 * ch + 4, f"warp {w}", 11, INK2))
    b.append(text(x0, gy + 16 * ch + 22, "thread = 8 × row + column:  t19 = 8 × 2 + 3", 12, ORANGE, weight="600"))
    pic_bottom = gy + 16 * ch + 22
    ty = para(y, "self.threads is 128: the block's 4 warps (groups of 32 threads that run together) × 32. "
                 "128 // tpr = 16, so the layout puts the threads in a grid of 16 rows × 8 columns.")
    ty = para(ty, "stride=(tpr, 1) = (8, 1) numbers the spots. A stride is how much the number grows per step "
                  "in one direction: one row down adds 8, one column right adds 1. So thread = 8 × row + "
                  "column, and thread 19 sits at row 2, column 3.")
    ty = para(ty, "Lay the grid over the top 16 rows of the tile: grid row r copies row r of K, and column c "
                  "copies piece c, values 8c to 8c + 7.")
    y = max(pic_bottom, ty) + 52

    # ---- 4: the value layout, and the tiled copy as a whole ------------------------------------
    y = header(y, 4, "The value layout, and the whole plan", G2S, hl={0, 3})
    x0, gy, px = 136, y + 40, 5
    b.append(text(x0, gy - 12, "the 64 × 64 tile of K", 12, INK, weight="600"))
    tile(x0, gy, px, band=0)
    for m in range(4):
        b.append(text(x0 - 8, gy + (16 * m + 7) * px, f"step {m}", 11, INK, "end", "600"))
        b.append(text(x0 - 8, gy + (16 * m + 7) * px + 14, f"rows {16 * m}–{16 * m + 15}", 10, MUTED, "end"))
        b.append(text(x0 + 64 * px + 8, gy + (16 * m + TR) * px + 4, f"row {16 * m + TR}", 10.5, ORANGE, weight="600"))
    lx = x0
    ly = gy + 64 * px + 26
    b.append(rect(lx, ly - 10, 12, 12, BLUE_L, GRID))
    b.append(text(lx + 18, ly, "one step: 16 × 8 threads, 1 piece each = 16 rows × 64 values", 11.5, INK2))
    b.append(rect(lx, ly + 8, 12, 4, ORANGE))
    b.append(text(lx + 18, ly + 14, "thread 19's piece in each step: values 24–31", 11.5, INK2))
    pic_bottom = ly + 14
    ty = para(y, "make_layout((1, vec)) is the value layout: what each thread copies per step, 1 row × 8 "
                 "values. That is one piece, exactly what one atom moves.")
    ty = para(ty, "make_tiled_copy_tv (tv: thread and value) joins the atom and the two layouts into g2s "
                  "(global to shared): the plan for the whole copy.")
    ty = para(ty, "A step is one round in which every thread copies one piece. One step covers 16 rows × 64 values, a quarter of the tile. CuTe repeats the step 4 "
                  "times, 16 rows lower each time, so thread 19 copies values 24–31 of rows 2, 18, 34 and 50.")
    y = max(pic_bottom, ty) + 52

    # ---- 5: get_slice --------------------------------------------------------------------------
    y = header(y, 5, "Pick a thread: its share of the plan",
               ["thr_g2s = g2s.get_slice(tidx)              # this thread's share, e.g. thread 19"])
    tx0, tyy = 56, y + 20
    cols = [(0, "tidx"), (70, "grid spot"), (190, "rows of the tile"), (340, "values")]
    for dx, h in cols:
        b.append(text(tx0 + dx, tyy, h, 11.5, INK2, weight="600"))
    b.append(line(tx0, tyy + 8, tx0 + 440, tyy + 8, GRID))
    for i, (t, r, c) in enumerate([(0, 0, 0), (19, 2, 3), (127, 15, 7)]):
        ry_ = tyy + 30 + 26 * i
        if t == 19:
            b.append(rect(tx0 - 6, ry_ - 17, 452, 25, ORANGE_L, rx=3))
        vals = [str(t), f"row {r}, col {c}", ", ".join(str(r + 16 * m) for m in range(4)), f"{8 * c}–{8 * c + 7}"]
        for (dx, _), v in zip(cols, vals):
            b.append(text(tx0 + dx, ry_, v, 12, INK, weight="600" if t == 19 else "normal"))
    b.append(text(tx0, tyy + 30 + 26 * 3 + 4, "3 of the 128 threads shown. Each gets 4 pieces, one per step.",
                  11.5, INK2))
    pic_bottom = tyy + 30 + 26 * 3 + 4
    ty = para(y, "tidx is this thread's number in the block, 0 to 127. Every thread runs this same line with "
                 "its own tidx.")
    ty = para(ty, "get_slice keeps only this thread's part of the plan: its spot in the grid, and so its 4 "
                  "pieces of a tile.")
    ty = para(ty, "So far everything is positions: “row 2, values 24–31”. Steps 1–5 drew them on K, but the "
                  "plan never names a matrix and has not touched any data. The kernel uses the same plan to "
                  "copy Q and V too.")
    y = max(pic_bottom, ty) + 52

    # ---- 6: partition_S ------------------------------------------------------------------------
    y = header(y, 6, "Apply the plan to K: find thread 19's values in memory",
               ["tKgK = thr_g2s.partition_S(gK)             # its pieces of K in global memory (S: source)"])
    x0, gy, px, pitch = 56, y + 56, 2, 142
    b.append(text(x0, gy - 40, "gK: all of K, in key blocks of 64 keys, copied one at a time (first 4 drawn)", 12, INK, weight="600"))
    for j in range(4):
        tx = x0 + j * pitch
        b.append(text(tx, gy - 20, f"key block {j}: keys {64 * j}–{64 * j + 63}", 10, INK2))
        b.append(text(tx, gy - 6, f"copied in loop turn {j}", 10, MUTED))
        tile(tx, gy, px)
    ay = gy + 64 * px + 6
    for j in range(4):
        arrow(x0 + j * pitch + 64, ay, x0 + j * pitch + 64, ay + 24)
    cy0 = ay + 46
    b.append(text(x0, cy0 - 8, "tKgK", 12.5, INK, weight="600", family=MONO))
    b.append(text(x0 + 42, cy0 - 8, "thread 19's pieces in each key block", 11.5, INK2))
    for j in range(4):
        chips(x0 + j * pitch + 24, cy0 + 4, 4, 10, 8, 14)
    b.append(text(x0, cy0 + 4 + 4 * 14 + 16, "Same spots in every key block: column 3 (values 24–31) of rows 2, "
                  "18, 34, 50.", 11.5, ORANGE, weight="600"))
    b.append(text(x0, cy0 + 4 + 4 * 14 + 33, "Each chip points to 8 real values of K, e.g. key block 1, step 0: "
                  "key 66, values 24–31.", 11.5, INK2))
    sy = cy0 + 4 + 4 * 14 + 62
    b.append(text(x0, sy, "shape ((8, 1), 4, 1, n_blocks)", 12.5, INK, weight="600"))
    rows = [("(8, 1)", "one piece: 8 values in 1 row", ORANGE),
            ("4", "steps down the tile", INK),
            ("1", "piece across: one step already spans all 64 values", INK),
            ("n_blocks", "key blocks: N keys ÷ 64, so 64 for N = 4096", INK)]
    for i, (num, what, col) in enumerate(rows):
        b.append(text(x0 + 70, sy + 21 + 19 * i, num, 12.5, col, "end", "700"))
        b.append(text(x0 + 80, sy + 21 + 19 * i, what, 12, INK2))
    pic_bottom = sy + 21 + 19 * 3
    ty = para(y, "Why partition? To copy, thread 19 needs to know where its values of K actually sit in "
                 "memory, not just “row 2, values 24–31”. partition_S works that out by laying the plan over "
                 "real data.")
    ty = para(ty, "The data is gK: all of K for this sequence, in global memory. K has more keys than one "
                  "tile holds, so it is cut into key blocks of 64 keys. A key block is a piece of K, not the "
                  "block of 128 threads. gK's shape is (64, 64, n_blocks), with n_blocks = N ÷ 64 for N keys. N = 4096 gives 64 key "
                  "blocks; the figure draws the first 4.")
    ty = para(ty, "Thread 19 still copies only column 3 (values 24–31). The kernel copies the key blocks one "
                  "after another, one per loop turn (step 8), and in each one thread 19 copies the same spots.")
    ty = para(ty, "partition_S finds those spots in every key block at once, before the loop. Each loop turn "
                  "then picks key block j.")
    ty = para(ty, "The result, tKgK, is a list of places in memory. S stands for source: the side the copy "
                  "reads from. Nothing is copied yet.")
    ty = para(ty, "Read its shape from the left: one piece, steps, pieces across, key blocks.")
    y = max(pic_bottom, ty) + 52

    # ---- 7: partition_D ------------------------------------------------------------------------
    y = header(y, 7, "The same pieces at the destination: shared memory",
               ["tKsK = thr_g2s.partition_D(sK)             # the same pieces in shared memory (D: destination)"])
    x0, gy, px = 56, y + 40, 4
    b.append(text(x0, gy - 10, "sK: room for one key block in shared memory", 12, INK, weight="600"))
    tile(x0, gy, px)
    for m in range(4):
        b.append(text(x0 + 64 * px + 8, gy + (16 * m + TR) * px + 4, f"row {16 * m + TR}", 10.5, ORANGE,
                      weight="600"))
    cx = 420
    arrow(x0 + 64 * px + 56, gy + 40, cx - 10, gy + 40)
    b.append(text(cx, gy + 10, "tKsK", 12.5, INK, weight="600", family=MONO))
    chips(cx, gy + 24, 4, 12, 10, 16)
    b.append(text(cx, gy + 24 + 4 * 16 + 18, "shape ((8, 1), 4, 1)", 12.5, INK, weight="600"))
    b.append(text(cx, gy + 24 + 4 * 16 + 36, "no n_blocks: one block", 11.5, INK2))
    b.append(text(cx, gy + 24 + 4 * 16 + 52, "at a time lives here", 11.5, INK2))
    pic_bottom = gy + 64 * px
    ty = para(y, "sK is a 64 × 64 space in shared memory. It holds one key block at a time, stored row by "
                 "row like the tile in global memory.")
    ty = para(ty, "partition_D picks thread 19's spots in sK. D stands for destination: the side the copy "
                  "writes to. They are the same rows and values as in step 6: rows 2, 18, 34, 50, values 24–31.")
    ty = para(ty, "So tKsK has the shape of tKgK without its last part.")
    y = max(pic_bottom, ty) + 52

    # ---- 8: the copy ---------------------------------------------------------------------------
    y = header(y, 8, "The copy: key block j, 4 pieces per thread",
               ["cute.copy(g2s, tKgK[None, None, None, j], tKsK)   # copy key block j: 4 steps of 16 bytes"])
    x0, gy, pitch, mp = 56, y + 54, 52, 22
    b.append(text(x0, gy - 34, "tKgK", 12.5, INK, weight="600", family=MONO))
    b.append(text(x0 + 42, gy - 34, "global memory", 11.5, INK2))
    for j in range(4):
        for m in range(4):
            if j == 1:
                piece(x0 + j * pitch, gy + m * mp, 5, 8)
            else:
                piece(x0 + j * pitch, gy + m * mp, 5, 8, FAINT, GRID)
        b.append(text(x0 + j * pitch + 20, gy + 3 * mp + 24, f"j = {j}", 10.5,
                      ORANGE if j == 1 else MUTED, "middle", "600" if j == 1 else "normal"))
    b.append(rect(x0 + pitch - 4, gy - 4, 48, 3 * mp + 16, "none", ORANGE, 1.6, rx=2))
    b.append(text(x0, gy + 3 * mp + 44, "orange frame: tKgK[None, None, None, 1], key block 1", 11.5, ORANGE,
                  weight="600"))
    cx = 440
    b.append(text(cx, gy - 34, "tKsK", 12.5, INK, weight="600", family=MONO))
    b.append(text(cx + 42, gy - 34, "shared memory", 11.5, INK2))
    ax = x0 + 4 * pitch + 6
    for m in range(4):
        yy = gy + m * mp
        arrow(ax, yy + 4, cx - 8, yy + 4)
        b.append(text(ax + 4, yy, f"step {m}: key {64 + TR + 16 * m}", 9.5, INK2))
        piece(cx, yy, 12, 8)
        b.append(text(cx + 104, yy + 8, f"row {TR + 16 * m}", 10, INK2))
    b.append(text(ax, gy - 16, "each step: load 16 bytes, store 16 bytes", 11.5, INK, weight="600"))
    sy = gy + 3 * mp + 76
    b.append(text(x0, sy, "All 128 threads at once: 128 × 4 pieces × 16 bytes = 8,192 bytes", 12, INK, weight="600"))
    b.append(text(x0, sy + 18, "= 64 × 64 values × 2 bytes: the whole tile.", 12, INK2))
    pic_bottom = sy + 18
    ty = para(y, "tKgK[None, None, None, j] picks key block j. None keeps a whole part of the shape, like : "
                 "in NumPy, so only the last part is fixed, to j.")
    ty = para(ty, "cute.copy then runs the atom of step 2 once per piece: 4 times. Each time thread 19 loads "
                  "16 bytes from global memory into its registers and stores them at the same row and values "
                  "in sK.")
    ty = para(ty, "The kernel runs this line once per key block: j = 0, 1, 2, …, up to 63 when N = 4096.")
    y = max(pic_bottom, ty) + 20
    return svg(W, y, b)


def copy_atoms():
    """The atom half of a tiled copy: the same pieces, carried by different instructions."""
    b = []
    b.append(f'<defs><marker id="ah3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')
    b.append(text(20, 30, "The atom of a tiled copy: the instruction that carries each piece", 15, weight="600"))
    b.append(text(20, 50, "Who copies what stays the same; the instruction changes, and with it the speed.", 12.5, INK2))
    G, R, S = GOLD_L, BLUE_L, AQUA_L

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
    GREY_L = PANEL
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
    GREY_L = PANEL
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
    """Top-down: a block's 64 rows split over 4 warps, warp 0's 32 lanes in 8 quads, and which lanes hold
    each value of warp 0's 16 x 8 multiply result (quad q holds rows q and q + 8)."""
    b = []
    GREY_L = PANEL
    b.append(text(20, 30, "From a block down to its lanes: who holds which rows", 16, weight="600"))
    b.append(text(20, 50, "Blue follows one path: warp 0 of the block, then quad 0 of warp 0.", 12.5, INK2))

    # 1. one block = 4 warps, 16 rows each
    b.append(text(20, 88, "1. One block = 4 warps = 128 threads", 13.5, weight="600"))
    b.append(text(20, 106, "it computes 64 rows (tokens); each warp takes 16", 11.5, INK2))
    tx, ty, ts = 40, 122, 128
    for w in range(4):
        hit = w == 0
        y = ty + w * ts / 4
        b.append(rect(tx, y, ts, ts / 4, BLUE_L if hit else GREY_L, BLUE if hit else GRID, 1.5 if hit else 1))
        b.append(text(tx + ts + 14, y + 20, f"warp {w}: rows {16 * w}–{16 * w + 15}", 11.5,
                      BLUE if hit else INK2, weight="600" if hit else "normal"))
    b.append(text(tx, ty + ts + 18, "the block's 64 × 64 tile", 11, MUTED))

    # 2. warp 0 = 32 lanes in 8 quads
    y0 = 352
    b.append(text(20, y0 - 42, "2. Warp 0 = 32 lanes (threads)", 13.5, weight="600"))
    b.append(text(20, y0 - 24, "all 32 run the same instruction at the same time", 11.5, INK2))
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

    # 3. who holds each value of warp 0's 16 x 8 multiply result: lane 4r + c/2 for row r and row r + 8
    X = 420
    b.append(text(X, 88, "3. Who holds each value of warp 0's 16 × 8 multiply result", 13.5, weight="600"))
    b.append(text(X, 106, "one tensor-core multiply: warp 0's 16 rows × 8 columns", 11.5, INK2))
    gx, gw, rh, gap = X + 46, 52, 23, 10
    for c in range(8):
        b.append(text(gx + c * gw + gw / 2, 128, f"col {c}", 10, MUTED, "middle"))
    for r in range(16):
        y = 136 + r * rh + (gap if r >= 8 else 0)
        hit = r % 8 == 0
        b.append(text(gx - 8, y + 14, f"row {r}", 11, BLUE if hit else MUTED, "end", "600" if hit else "normal"))
        for c in range(8):
            lane = 4 * (r % 8) + c // 2
            b.append(rect(gx + c * gw, y, gw - 2, rh - 3, BLUE if hit else GREY_L, SURF if hit else GRID, 1, 3))
            b.append(text(gx + c * gw + gw / 2 - 1, y + 14, f"lane {lane}", 9.5, SURF if hit else INK2, "middle"))
    cy = 136 + 16 * rh + gap + 22
    for i, s in enumerate(["Why two rows per quad: 16 × 8 = 128 values ÷ 32 lanes = 4 per lane.",
                           "Each row (8 values) goes to one quad: 4 lanes × 2 neighbouring columns.",
                           "There are 16 rows but only 8 quads, so each quad takes 2 rows: row q",
                           "and row q + 8. A row's max or sum then needs only its quad's 4 lanes",
                           "to swap values (shuffles), not the whole warp."]):
        b.append(text(X, cy + 16 * i, s, 11.5, INK2))
    return svg(900, max(cy + 16 * 3, y0 + 8 * 28 + 34) + 20, b)


def ldmatrix_x4():
    """One ldmatrix.x4: a 16 x 16 piece of the K tile, its four 8 x 8 tiles, who receives each value."""
    b = []
    YEL_L = GOLD_L
    TILE = [(BLUE_L, BLUE), (ORANGE_L, ORANGE), (AQUA_L, AQUA), (YEL_L, GOLD)]
    GREY_L = PANEL
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
    GREY_L = PANEL
    b.append(f'<defs><marker id="ah7" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')
    b.append(text(20, 30, "The 64 × 64 tiles: what one block reads from Q, K and V and writes to O (one head, N = 4,096, D = 64)", 16, weight="600"))

    def arrow(x1, y1, x2, y2):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.4" marker-end="url(#ah7)"/>')

    # 1. global memory: one head's Q, O, K, V. Q and O side by side: tile 1 of Q in, tile 1 of O out
    top, h, w = 124, 288, 56
    b.append(text(20, 64, "1. Global memory: one head", 13.5, weight="600"))
    b.append(text(20, 82, "4,096 rows × 64 head dims each", 11.5, INK2))
    b.append(text(20, 97, "cut into 64 tiles of 64 × 64; tiles 0, 1, 2 and 63 drawn", 11.5, MUTED))
    # square tiles (64 rows x 64 dims, like panel 2), with a break standing for tiles 3 to 62
    ys = {0: top, 1: top + w, 2: top + 2 * w, 63: top + h - w}
    for name, x in (("Q", 24), ("O", 88), ("K", 168), ("V", 232)):
        b.append(text(x + w / 2, top - 10, name, 13, INK, "middle", "600"))
        for t, y in ys.items():
            if name in "QO" and t == 1:
                b.append(rect(x, y, w, w, ORANGE_L, ORANGE, 2))
                b.append(text(x + w / 2, y + w / 2 + 4, "1", 11, ORANGE, "middle", "600"))
            elif name in "KV" and t == 2:
                b.append(rect(x, y, w, w, AQUA_L, AQUA, 2))
                b.append(text(x + w / 2, y + w / 2 + 4, "step j", 10.5, AQUA, "middle", "600"))
            else:
                b.append(rect(x, y, w, w, GREY_L, INK2, 1))
                b.append(text(x + w / 2, y + w / 2 + 4, str(t), 11, MUTED, "middle"))
        b.append(text(x + w / 2, top + 3 * w + (h - 4 * w) / 2 + 7, "⋮", 20, MUTED, "middle"))
    arrow(304, top + 4, 304, top + h - 4)
    b.append(f'<text x="320" y="{top + h / 2}" font-size="11" fill="{INK2}" text-anchor="middle" '
             f'transform="rotate(90 320 {top + h / 2})">4,096 keys ÷ 64 per step = 64 steps</text>')
    b.append(text(20, top + h + 22, "Orange: this block's rows. Q tile 1 in, O tile 1 out.", 11.5, ORANGE))
    b.append(text(20, top + h + 38, "Green: the 64 keys of step j, one of the 64 steps.", 11.5, AQUA))
    b.append(text(20, top + h + 54, "The block steps down K and V, 64 rows at a time,", 11.5, INK2))
    b.append(text(20, top + h + 70, "adding each step's share into O tile 1.", 11.5, INK2))

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
             "same K and V tiles. Each step adds its 64 keys' share to O;",
             "after 64 steps, all 4,096 keys are counted. The division",
             "by each row's sum happens once, at the end."]
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
    G, GREY_L, PAUSE = GOLD_L, PANEL, GREY_M
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
            ["The K tile is 64 keys × 64 dims × 2 bytes (bf16) = 8,192 bytes. One cp.async copies at most 16 bytes (8 values),",
             "so the tile is 512 pieces. Split over the block's 128 threads, that is 4 pieces each: cp.async × 4. V is the same size: 4 more."])
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
    for x, h in zip(cx, ["What the thread does, in order", "K group", "V group", "What happens"]):
        b.append(text(x, hy, h, 11.5, MUTED, weight="600"))
    # one row per instruction of the strip at the top, in the same order; the K and V columns say where that
    # group's 4 copies are once the row's instruction has run
    rows = [
        ("cp.async × 4 (K)", "moving", "not started", False,
         ["Starts K's 4 copies. They begin to move;", "the thread does not wait for them."]),
        ("commit_group", "moving", "not started", False,
         ["Bundles the copies started so far (K's 4)", "into group 1. Later copies go in a new group."]),
        ("cp.async × 4 (V)", "moving", "moving", False,
         ["Starts V's 4 copies while K's are still moving.", "Still no waiting."]),
        ("commit_group", "moving", "moving", False,
         ["Bundles the copies started since (V's 4)", "into group 2."]),
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
        b.append(rect(32, ry, 856, rh - 4, PANEL if paused else SURF, GRID, 1, 6))
        if paused:
            b.append(rect(32, ry, 6, rh - 4, INK2, rx=2))
        b.append(text(cx[0], ry + 25, does, 12 if len(does) < 34 else 11, MUTED if does.startswith("(") else INK,
                      family=MONO if does.startswith(("cp.", "commit", "wait")) else None))
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
    b.append(text(50, yy, "K and V are the same size, so their copies take about as long; V lands a little after K because it was started later.",
                  12, INK2))
    yy += 22
    b.append(text(50, yy, "n is how many of the newest groups may still be moving; every older group must be done. "
                  "So wait_group(1) means \"K done,", 12, INK2))
    b.append(text(50, yy + 16, "V may still be moving\". wait_group(0) would wait for every group. "
                  "So the group the thread needs first must be the", 12, INK2))
    b.append(text(50, yy + 32, "oldest one: that is why K is started first. The three timelines below compare the choices.",
                  12, INK2))

    # why two groups, and why K first: three timelines stacked on one time scale, so the pauses line up
    y2 = yy + 66
    b.append(text(50, y2, "Why two groups, and why K first? Three ways to load K and V", 13, weight="600"))
    b.append(text(50, y2 + 18, "One time scale for all three rows. In every row the thread starts K's or V's copies first, and the "
                  "second set lands a little", 11.5, INK2))
    b.append(text(50, y2 + 34, "later because its copies queue behind the first. The dashed line is when the thread can start "
                  "S = Q Kᵀ.", 11.5, INK2))
    u, tx = 28, 300
    dur, lag, comp = 4, 1.5, 2.5     # copy time of one tile, how much later the second set lands, S = Q K^T
    top = y2 + 58
    rp = 92                          # row pitch

    def timeline(i, title, call, order, land, caption, one_group=False, best=False):
        ry = top + i * rp
        b.append(text(50, ry + 14, title, 12, BLUE if best else INK, weight="600"))
        b.append(text(50, ry + 30, call, 11, INK2, family=MONO))
        b.append(text(tx - 8, ry + 17, "copies", 11, MUTED, "end"))
        b.append(text(tx - 8, ry + 51, "thread", 11, MUTED, "end"))
        for row, g in enumerate(order):
            x0g = row * lag
            b.append(rect(tx + x0g * u, ry + 14 * row, dur * u - 2, 12, ORANGE, rx=3))
            end = x0g + dur
            lx = (land + end) / 2 if x0g < land < end else x0g + dur / 2   # keep the label off the dashed line
            b.append(text(tx + lx * u, ry + 14 * row + 10, g, 9.5, SURF, "middle", "600"))
        if one_group:
            b.append(f'<rect x="{tx - 3}" y="{ry - 3}" width="{(dur + lag) * u + 4}" height="32" fill="none" '
                     f'stroke="{INK2}" stroke-width="1" stroke-dasharray="3 3" rx="5"/>')
        b.append(rect(tx, ry + 34, land * u - 2, 26, PAUSE, rx=4))
        b.append(text(tx + land * u / 2, ry + 51.5, "paused", 11, INK2, "middle"))
        b.append(rect(tx + land * u, ry + 34, comp * u - 2, 26, BLUE_L, rx=4))
        b.append(text(tx + (land + comp / 2) * u, ry + 51.5, "S = Q Kᵀ", 10.5, INK, "middle"))
        b.append(line(tx + land * u - 1, ry - 6, tx + land * u - 1, ry + 64, INK, 1.5, "4 3"))
        if best:   # bracket under the thread bar: from this row's resume point to the other rows'
            sx0, sx1, by = tx + land * u - 1, tx + (land + lag) * u - 1, ry + 66
            b.append(line(sx0, by, sx1, by, BLUE, 1.5))
            b.append(line(sx0, by - 4, sx0, by + 4, BLUE, 1.5))
            b.append(line(sx1, by - 4, sx1, by + 4, BLUE, 1.5))
            b.append(text((sx0 + sx1) / 2, by + 15, "pause saved", 11, BLUE, "middle", "600"))
        for k, c in enumerate(caption):
            b.append(text(545, ry + 14 + 16 * k, c, 11.5, INK2))

    # faint guide where the thread resumes in the first two rows, carried down to the third
    gx = tx + (dur + lag) * u - 1
    b.append(line(gx, top - 6, gx, top + 2 * rp + 64, GRID, 1, "2 3"))
    timeline(0, "One group for K and V", "wait_group(0)", ("K", "V"), dur + lag,
             ["K lands at the same moment as in the K-first row,",
              "but the group also holds V, so the thread waits",
              "for V too, though S = Q Kᵀ does not use it."], one_group=True)
    timeline(1, "V first, then K", "wait_group(0)", ("V", "K"), dur + lag,
             ["K is now the newest group, so it lands last, and",
              "wait_group can only leave the newest groups",
              "unfinished. No better than one group."])
    timeline(2, "K first, then V (as above)", "wait_group(1)", ("K", "V"), dur,
             ["The thread waits only for K, the older group,",
              "and starts S = Q Kᵀ while V is still moving.",
              "The pause is shorter by the time V lands after K."], best=True)
    y_end = top + 2 * rp + 86
    b.append(text(50, y_end + 26, "So starting K first does not make K land sooner. It makes K the older group, so wait_group(1) "
                  "can stop waiting once K", 11.5, INK2))
    b.append(text(50, y_end + 42, "has landed. The time saved is small when memory is idle, larger when it is busy.", 11.5, INK2))

    # 3. waiting for everyone's copies
    y = y_end + 84
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


def key_block_v0():
    """One key block of v0, line by line: one panel per line of the abridged loop body, following thread 0."""
    # Facts used (layouts checked on the GPU with CuTe DSL, see mma_ownership and tiled_copy):
    #   g2s: thread t copies row t // 8 + 16 m (m = 0..3), values 8 (t % 8) to 8 (t % 8) + 7
    #   tSsQ ((2,2,2),1,4):((1,512,8),0,16)   tSsK ((2,2),8,4):((1,8),512,16)
    #   lane L = 4 g + t reads Q rows g, g + 8 and K key g + 8 n, at dims 16 k + {2t, 2t+1, 2t+8, 2t+9}
    #   SASS per key block per thread, unmasked loop (results/sass_counts.txt):
    #     LDG 8, STS 24 (8 K/V + 16 P), BAR 2, LDS 224 (96 x 32-bit: Q 16, K 64, P 16; 128 x 16-bit: V),
    #     HMMA 64 (32 for S, 32 for O), SHFL.BFLY 4, MUFU.EX2 34
    import textwrap
    W, TX, TW = 1040, 640, 58
    FAINT, HL, NEUT = PANEL, GOLD_L, FILL
    GLOB, SHR = GOLD_L, AQUA_L
    WARPC = [GOLD_L, AQUA_L, ORANGE_L, GREY_M]
    CODE = ["cute.arch.sync_threads()                                   # everyone is done with the old K, V",
            "cute.copy(g2s, tKgK[None, None, None, j], tKsK)            # LDG, then STS, for K and V",
            "cute.copy(g2s, tVgV[None, None, None, j], tVsV)",
            "cute.arch.sync_threads()                                   # K, V are in shared memory",
            "for k in cutlass.range_constexpr(cute.size(tSsQ, mode=[2])):",
            "    cute.autovec_copy(tSsQ[None, None, k], tSrQ[None, None, k])     # plain loads",
            "    cute.autovec_copy(tSsK[None, None, k], tSrK[None, None, k])",
            "    cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)",
            "# ... mask, online softmax with quad shuffles, P -> shared memory -> registers, O += P V"]
    # The figure is drawn as one tall column, then cut in two after panel 4: part 1 gets K and V into
    # shared memory (lines 1-4), part 2 computes with them (lines 5-9). Both parts open with the same
    # snippet and setup, with the other part's lines dimmed.
    DEFS = (f'<defs><marker id="ahk" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')
    b = []

    def code_line(x, yy, c, on=True):
        k = c.find("#")
        code_part = c if k < 0 else c[:k].rstrip()
        ind = len(code_part) - len(code_part.lstrip(" "))
        if code_part.strip():
            b.append(text(x + 7.53 * ind, yy, code_part.lstrip(" "), 12.5, INK if on else MUTED, family=MONO))
        if k >= 0:
            b.append(text(x + 7.53 * k, yy, c[k:], 12.5, MUTED, family=MONO))

    def head(title, on):
        start = len(b)
        b.append(text(20, 30, title, 17, weight="600"))
        b.append(text(20, 52, "The block's 128 threads (4 warps) run these lines once per key block j: 64 keys of K and "
                              "their 64 rows of V. Each panel takes one line.", 12.5, INK2))
        b.append(text(20, 70, "Blue: thread 0 (lane 0 of warp 0) and its values. Inside the k loop the figure shows "
                              "k = 1, head dims 16–31.", 12.5, INK2))
        y = snippet(on)
        hb = b[start:]
        del b[start:]
        return hb, y

    # ---- the whole snippet, each line tagged with its panel ------------------------------------
    def snippet(on):
        ly = 92
        b.append(rect(20, ly, W - 40, 21 * (len(CODE) + 1) + 12, FAINT, GRID, rx=4))
        code_line(60, ly + 21, "# one key block in v0 (abridged)")
        for i, c in enumerate(CODE):
            yy = ly + 42 + 21 * i
            b.append(f'<circle cx="38" cy="{yy - 4.5}" r="9" fill="{INK if i + 1 in on else MUTED}"/>')
            b.append(text(38, yy, str(i + 1), 11, SURF, "middle", "700"))
            code_line(60, yy, c, i + 1 in on)
        # the setup the snippet relies on: run once per thread, before the loop over key blocks
        ly = ly + 21 * (len(CODE) + 1) + 12 + 16
        SETUP = [
                "thr_g2s = g2s.get_slice(tidx)                                                   # idea 4: this thread's share of the copy",
                "tKgK, tKsK = thr_g2s.partition_S(gK), thr_g2s.partition_D(sK)                   # its pieces of K: global, shared",
                "tVgV, tVsV = thr_g2s.partition_S(gV), thr_g2s.partition_D(sV)                   # the same for V",
                "thr_mma = tiled_mma.get_slice(tidx)                                             # idea 3: this thread's share of the MMA",
                "tSsQ, tSsK = thr_mma.partition_A(sQ), thr_mma.partition_B(sK)                   # its values of Q, K in shared memory",
                "tSrQ, tSrK = thr_mma.make_fragment_A(tSsQ), thr_mma.make_fragment_B(tSsK)       # registers to hold them",
                "acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((64, 64)), Float32)     # its scores of S",
        ]
        b.append(rect(20, ly, W - 40, 21 * (len(SETUP) + 1) + 12, PANEL, GRID, rx=4))
        code_line(60, ly + 21, "# made once per thread, before the loop over key blocks (ideas 3 and 4 of the article)")
        for i, c in enumerate(SETUP):
            code_line(60, ly + 42 + 21 * i, c)
        return ly + 21 * (len(SETUP) + 1) + 12 + 60

    head1, y = head("One key block in v0, line by line (lines 1–4)", range(1, 5))
    head2, _ = head("One key block in v0, line by line (lines 5–9)", range(5, 10))
    y1 = y

    def para(y, s, size=12.5, fill=INK2, lh=18, n=TW):
        for ln in textwrap.wrap(s, n):
            b.append(text(TX, y, ln, size, fill))
            y += lh
        return y + 8

    def header(y, n, title, code):
        b.append(line(20, y - 26, W - 20, y - 26, GRID))
        b.append(f'<circle cx="31" cy="{y - 5}" r="11" fill="{INK}"/>')
        b.append(text(31, y, str(n), 12.5, SURF, "middle", "700"))
        b.append(text(50, y, title, 15, weight="600"))
        y += 14
        b.append(rect(20, y, W - 40, 33, HL, GRID, rx=4))
        code_line(34, y + 21, code.strip())
        return y + 33 + 34

    def arrow(x1, y1, x2, y2, label=None, dy=-7, size=11.5):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.4" '
                 f'marker-end="url(#ahk)"/>')
        if label:
            b.append(text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, size, INK, "middle", "600"))

    def tile(x0, y0, px, fill, rows=(), cols=(0, 8), mark=BLUE, frame=None):
        # a 64 x 64 tile; rows: rows whose columns cols[0]..cols[1]-1 are marked
        b.append(rect(x0, y0, 64 * px, 64 * px, fill, GRID))
        for r in rows:
            b.append(rect(x0 + cols[0] * px, y0 + r * px, (cols[1] - cols[0]) * px, max(px, 2.5), mark))
        if frame:
            r0, r1, c0, c1 = frame
            b.append(rect(x0 + c0 * px - 1.5, y0 + r0 * px - 1.5, (c1 - c0) * px + 3, (r1 - r0) * px + 3,
                          "none", ORANGE, 1.6))

    def cell(x, y, cs, fill, label=None, fg=INK, stroke=GRID):
        b.append(rect(x, y, cs, cs, fill, stroke, 0.6))
        if label is not None:
            b.append(text(x + cs / 2, y + cs / 2 + 3.5, label, 9, fg, "middle", "600"))

    def legend(x, yy, fill, s, stroke=GRID):
        b.append(rect(x, yy - 10, 12, 12, fill, stroke))
        b.append(text(x + 18, yy, s, 11.5, INK2))

    # ---- 1: the first barrier -------------------------------------------------------------------
    y = header(y, 1, "Wait until every warp is done with the old K and V", CODE[0])
    x0, BX, bh, pitch = 116, 470, 22, 30
    ends = [372, 404, 330, 452]
    ty0 = y + 30
    b.append(text(BX, ty0 - 12, "barrier", 11.5, INK, "middle", "600"))
    for w in range(4):
        wy = ty0 + w * pitch
        b.append(text(x0 - 10, wy + 15, f"warp {w}", 11.5, INK2, "end"))
        b.append(rect(x0, wy, ends[w] - x0, bh, NEUT, GRID, rx=3))
        b.append(text(x0 + 8, wy + 15, "block j − 1: S, softmax, O += P V", 10.5, INK))
        if BX - ends[w] > 8:
            b.append(rect(ends[w] + 2, wy + 3, BX - ends[w] - 6, bh - 6, "none", MUTED, 1, rx=2)
                     .replace("/>", ' stroke-dasharray="3 3"/>'))
        if BX - ends[w] > 44:
            b.append(text((ends[w] + BX) / 2, wy + 15, "waits", 10.5, MUTED, "middle"))
        b.append(rect(BX + 6, wy, 120, bh, SHR, GRID, rx=3))
        b.append(text(BX + 14, wy + 15, "copy block j", 10.5, INK))
    b.append(line(BX, ty0 - 6, BX, ty0 + 4 * pitch, INK, 2.5))
    hy = ty0 + 4 * pitch + 16
    b.append(line(ends[2], hy, ends[3], hy, ORANGE, 2))
    b.append(line(ends[2], hy - 4, ends[2], hy + 4, ORANGE, 2))
    b.append(line(ends[3], hy - 4, ends[3], hy + 4, ORANGE, 2))
    b.append(text(x0, hy + 22, "Without the barrier, warp 2 would start overwriting sK and sV here,", 11.5, ORANGE,
                  weight="600"))
    b.append(text(x0, hy + 39, "while warp 3 is still reading block j − 1's V for O += P V.", 11.5, ORANGE,
                  weight="600"))
    pic_bottom = hy + 39
    ty = para(y, "Shared memory has room for one key block of K (sK) and one of V (sV). Every key block "
                 "reuses the same spots, so block j can only be written once nobody needs block j − 1.")
    ty = para(ty, "The 4 warps reach this line at different times. sync_threads holds each of the 128 threads "
                  "here until all have arrived: one BAR.SYNC instruction.")
    y = max(pic_bottom, ty) + 52

    # ---- 2: copy K ------------------------------------------------------------------------------
    y = header(y, 2, "Copy key block j of K into shared memory, through registers", CODE[1])
    px, gx, rx, sx = 2, 56, 262, 452
    gy = y + 52
    T0 = (0, 16, 32, 48)
    b.append(text(gx, gy - 34, "K in global memory", 12, INK, weight="600"))
    b.append(rect(gx, gy - 20, 64 * px, 14, FAINT, GRID))
    b.append(text(gx + 32 * px, gy - 9, "block j − 1", 9.5, MUTED, "middle"))
    tile(gx, gy, px, GLOB, T0)
    b.append(rect(gx, gy + 64 * px, 64 * px, 14, FAINT, GRID))
    b.append(text(gx + 32 * px, gy + 64 * px + 11, "block j + 1", 9.5, MUTED, "middle"))
    b.append(text(gx + 32 * px, gy + 32 * px + 4, "key block j", 11, INK, "middle", "600"))
    # thread 0's registers
    b.append(text(rx, gy - 34, "thread 0's registers", 12, INK, weight="600"))
    b.append(rect(rx, gy - 8, 130, 144, BLUE_L, GRID, rx=6))
    for m in range(4):
        for v in range(8):
            b.append(rect(rx + 17 + v * 12, gy + 6 + m * 32, 12, 14, BLUE, SURF, 0.8))
        b.append(text(rx + 65, gy + 32 + m * 32, f"row {16 * m}, values 0–7", 9.5, INK2, "middle"))
    b.append(text(sx, gy - 34, "sK in shared memory", 12, INK, weight="600"))
    tile(sx, gy, px, SHR, T0)
    arrow(gx + 64 * px + 8, gy + 64, rx - 6, gy + 64, "LDG × 4")
    arrow(rx + 136, gy + 64, sx - 6, gy + 64, "STS × 4")
    cy = gy + 64 * px + 40
    b.append(text(gx, cy, "Blue: thread 0's 4 pieces, 16 bytes each: rows 0, 16, 32, 48, values 0–7.", 11.5, INK2))
    b.append(text(gx, cy + 18, "All 128 threads at once: 512 pieces × 16 bytes = 8 KB, the whole tile.", 11.5, INK,
                  weight="600"))
    pic_bottom = cy + 18
    ty = para(y, "tKgK[None, None, None, j] picks key block j; g2s gives each thread 4 pieces of it (the "
                 "copy figure in idea 4 takes this line apart).")
    ty = para(ty, "Its atom, CopyUniversalOp, is a plain copy, so each piece makes "
                  "two trips: a global load (LDG) into the thread's registers, then a shared store (STS).")
    ty = para(ty, "The store waits for its load, a trip to global memory of hundreds of cycles, and ④ then "
                  "waits for every store. Round 4's cp.async skips the registers and lets the copy run "
                  "behind other work.")
    y = max(pic_bottom, ty) + 52

    # ---- 3: copy V ------------------------------------------------------------------------------
    y = header(y, 3, "The same copy for V: the values of the same 64 keys", CODE[2])
    px, kx, vx = 1.75, 96, 270
    gy = y + 46
    for xx, name in ((kx, "K block j"), (vx, "V block j")):
        b.append(text(xx + 32 * px, gy - 10, name, 12, INK, "middle", "600"))
        tile(xx, gy, px, GLOB, T0)
    b.append(line(kx - 10, gy, kx - 10, gy + 64 * px, INK2, 1.2))
    b.append(text(kx - 16, gy + 32 * px - 4, "keys", 10.5, INK2, "end"))
    b.append(text(kx - 16, gy + 32 * px + 10, "64j to", 10.5, INK2, "end"))
    b.append(text(kx - 16, gy + 32 * px + 24, "64j + 63", 10.5, INK2, "end"))
    ry = gy + 22 * px
    b.append(line(kx, ry, vx + 64 * px, ry, ORANGE, 1.4, "4 3"))
    b.append(text((kx + 64 * px + vx) / 2, ry - 5, "same key", 10, ORANGE, "middle", "600"))
    sy = gy + 64 * px + 40
    for xx, name in ((kx, "sK"), (vx, "sV")):
        arrow(xx + 32 * px, gy + 64 * px + 6, xx + 32 * px, sy - 6)
        tile(xx, sy, px, SHR, T0)
        b.append(text(xx + 32 * px, sy + 64 * px + 18, name, 12, INK, "middle", "600", MONO))
    b.append(text((kx + 64 * px + vx) / 2, gy + 64 * px + 26, "same g2s,", 10.5, INK2, "middle"))
    b.append(text((kx + 64 * px + vx) / 2, gy + 64 * px + 40, "same spots", 10.5, INK2, "middle"))
    # shared memory map
    mx, my = 420, gy + 10
    b.append(text(mx, my - 6, "Shared memory of one block", 12, INK, weight="600"))
    smem = [("sQ", "Q tile, loaded once before the loop", FAINT),
            ("sK", "key block j of K (②)", SHR),
            ("sV", "key block j of V (③)", SHR),
            ("sP", "P, used after the loop (⑨)", FAINT)]
    for i, (name, what, fill) in enumerate(smem):
        yy = my + 6 + i * 44
        b.append(rect(mx, yy, 196, 36, fill, GRID, rx=3))
        b.append(text(mx + 10, yy + 16, name, 12, INK, weight="600", family=MONO))
        b.append(text(mx + 40, yy + 16, "8 KB", 11, INK2))
        b.append(text(mx + 10, yy + 30, what, 10.5, INK2))
    b.append(text(mx, my + 6 + 4 * 44 + 10, "32 KB in all", 11.5, INK, weight="600"))
    pic_bottom = sy + 64 * px + 18
    ty = para(y, "tVgV and tVsV come from the same g2s, so thread 0 copies the same spots of V: rows 0, 16, "
                 "32, 48, values 0–7. Another 4 loads and 4 stores.")
    ty = para(ty, "Row r of V belongs to the same key as row r of K. S = Q Kᵀ scores the 64 keys of block j; "
                  "O += P V then mixes their 64 rows of V. Both have to sit in shared memory together.")
    ty = para(ty, "The real kernel interleaves the two copies, one step of K then one of V. Nothing reads "
                  "either until ④, so the order does not matter.")
    y = max(pic_bottom, ty) + 52

    # ---- 4: the second barrier ------------------------------------------------------------------
    y = header(y, 4, "Wait until all of K and V has landed", CODE[3])
    px, x0 = 3, 100
    gy = y + 34
    b.append(text(x0, gy - 12, "sK: who wrote each row, and what thread 0 reads", 12, INK, weight="600"))
    for r in range(64):
        b.append(rect(x0, gy + r * px, 64 * px, px, WARPC[(r % 16) // 4]))
    b.append(rect(x0, gy, 64 * px, 64 * px, "none", GRID))
    for r in range(0, 64, 8):
        for k in range(4):
            for d in (0, 1, 8, 9):
                b.append(rect(x0 + (16 * k + d) * px, gy + r * px, px, px, BLUE))
        b.append(text(x0 - 6, gy + r * px + 4, f"key {r}", 9.5, INK2, "end"))
    for k in range(4):
        b.append(text(x0 + (16 * k + 8) * px, gy + 64 * px + 14, f"k = {k}", 9.5, MUTED, "middle"))
    lx = x0 + 64 * px + 30
    b.append(text(lx, gy + 4, "Rows written in ② by:", 11.5, INK, weight="600"))
    for w in range(4):
        legend(lx, gy + 26 + 20 * w, WARPC[w], f"warp {w} (threads {32 * w}–{32 * w + 31})")
    legend(lx, gy + 26 + 20 * 4 + 6, BLUE, "read by thread 0 in the k loop")
    b.append(text(lx, gy + 140, "keys 0, 16, 32, 48: written by warp 0", 11.5, INK2))
    b.append(text(lx, gy + 158, "keys 8, 24, 40, 56: written by warp 2", 11.5, ORANGE, weight="600"))
    pic_bottom = gy + 64 * px + 14
    ty = para(y, "In ② and ③ each thread wrote 8 small pieces. The k loop reads sK in a different pattern, "
                 "the tiled MMA's: thread 0 reads keys 0, 8, 16, …, 56, at head dims 16k + 0, 1, 8, 9.")
    ty = para(ty, "Half of those rows were written by warp 2 (threads 64–71), which may still be copying. The "
                  "same holds for V in O += P V. So every thread waits here until all 128 have finished ② "
                  "and ③: a second BAR.SYNC, two per key block.")
    y = max(pic_bottom, ty) + 52

    # ---- 5: the k loop (part 2 starts here) ------------------------------------------------------
    i5, y5 = len(b), y
    y = header(y, 5, "The k loop: what tSsQ's shape ((2, 2, 2), 1, 4) means, and why k = 0, 1, 2, 3", CODE[4])

    def sub(yy, s):
        b.append(text(56, yy, s, 13, INK, weight="600"))
        return yy + 30

    # 5a: partition_A cuts the Q tile into mma pieces
    y = sub(y, "5a. Where tSsQ and its shape come from")
    b.append(rect(56, y - 4, 560, 62, FAINT, GRID, rx=4))
    code_line(68, y + 14, "tSsQ = thr_mma.partition_A(sQ)    # setup, before the loop")
    code_line(68, y + 34, "print(tSsQ.layout)")
    b.append(text(68, y + 52, "# prints ((2,2,2),1,4):((1,512,8),0,16)", 12.5, ORANGE, weight="600", family=MONO))
    y += 84
    px, x0 = 3.5, 120
    gy = y + 34
    for r in range(64):
        b.append(rect(x0, gy + r * px, 64 * px, px, BLUE_L if r < 16 else FAINT))
    b.append(rect(x0, gy, 64 * px, 64 * px, "none", GRID))
    for c in (16, 32, 48):
        b.append(line(x0 + c * px, gy, x0 + c * px, gy + 64 * px, INK2, 1))
    for r in (16, 32, 48):
        b.append(line(x0, gy + r * px, x0 + 64 * px, gy + r * px, INK2, 1))
    for k in range(4):
        for r in (0, 8):
            for d in (0, 1, 8, 9):
                b.append(rect(x0 + (16 * k + d) * px, gy + r * px, px, px, BLUE))
        b.append(text(x0 + (16 * k + 8) * px, gy - 20, f"k = {k}", 10.5, ORANGE if k == 1 else INK, "middle",
                      "600"))
        b.append(text(x0 + (16 * k + 8) * px, gy - 6, f"dims {16 * k}–{16 * k + 15}", 9, MUTED, "middle"))
    for w in range(4):
        b.append(text(x0 - 6, gy + (16 * w + 8) * px + 4, f"warp {w}", 10.5, INK if w == 0 else MUTED, "end",
                      "600" if w == 0 else "normal"))
    b.append(rect(x0 + 16 * px - 1.5, gy - 1.5, 16 * px + 3, 16 * px + 3, "none", ORANGE, 1.8))
    b.append(text(x0, gy + 64 * px + 18, "sQ: 64 rows × 64 head dims,", 10.5, INK2))
    b.append(text(x0, gy + 64 * px + 33, "cut into 4 × 4 pieces of 16 × 16", 10.5, INK2))
    # zoom: the piece at warp 0, k = 1, thread 0's values numbered in mode-0 order
    zx, cs = 392, 13
    b.append(line(x0 + 32 * px + 2, gy, zx - 4, gy, ORANGE, 1, "3 3"))
    b.append(line(x0 + 32 * px + 2, gy + 16 * px, zx - 4, gy + 16 * cs, ORANGE, 1, "3 3"))
    num = {(0, 0): 0, (0, 1): 1, (8, 0): 2, (8, 1): 3, (0, 8): 4, (0, 9): 5, (8, 8): 6, (8, 9): 7}
    for r in range(16):
        for d in range(16):
            if (r, d) in num:
                cell(zx + d * cs, gy + r * cs, cs, BLUE, str(num[(r, d)]), SURF)
            else:
                cell(zx + d * cs, gy + r * cs, cs, BLUE_L, stroke=SURF)
    for r in (0, 8):
        b.append(text(zx - 5, gy + r * cs + 10, f"row {r}", 9.5, INK2, "end"))
    for d in (0, 8, 15):
        b.append(text(zx + d * cs + cs / 2, gy - 6, str(16 + d), 9, MUTED, "middle"))
    b.append(text(zx, gy - 20, "one piece: warp 0, k = 1", 10.5, ORANGE, weight="600"))
    zy = gy + 16 * cs + 18
    for i, s in enumerate(("0 → 1: next dim, offset + 1", "0 → 2: 8 rows down, offset + 512",
                           "0 → 4: 8 dims across, offset + 8")):
        b.append(text(zx, zy + 16 * i, s, 10.5, INK2))
    pic_bottom = max(gy + 64 * px + 33, zy + 32)
    ty = para(y - 84, "tSsQ is not made in the snippet: the setup line above makes it once, before any key "
                      "block. Printing its layout from CuTe, as print_partition_shapes.py does, gives the shape "
                      "and strides shown. Shape and stride are known when compiling, so print runs then.")
    ty = para(max(ty, y), "partition_A(sQ) lays the tiled MMA over the 64 × 64 Q tile. Each mma.m16n8k16 instruction "
                 "reads a 16 × 16 piece of Q, so the tile splits into 4 × 4 pieces: 4 warps down (the "
                 "(4, 1, 1) of make_tiled_mma), 4 steps of 16 head dims across.")
    ty = para(ty, "Thread 0 is in warp 0, so only the top row of pieces concerns it, and in each piece it holds "
                  "8 values (blue). The result, tSsQ, lists those values and nothing else.")
    ty = para(ty, "Each top-level entry of ((2, 2, 2), 1, 4) is a mode: one direction you can step through "
                  "thread 0's values in. 5b reads them one by one.")
    y = max(pic_bottom, ty) + 40

    # 5b: the three modes, as a table across the full width
    y = sub(y, "5b. The three modes of tSsQ = ((2, 2, 2), 1, 4) : ((1, 512, 8), 0, 16)")
    cols = [34, 96, 286, 530, 780]
    heads = ["mode", "size : stride", "what it counts", "why that size", "what [None, None, k] does"]
    rows = [("0", "(2, 2, 2) : (1, 512, 8)",
             ["thread 0's 8 values inside one", "piece, numbered 0–7 in 5a"],
             ["mma.m16n8k16 gives each of the", "32 lanes 8 of a piece's 256 values"],
             ["None: kept whole. All 8 values", "go into each mma of the step"], FAINT),
            ("1", "1 : 0",
             ["pieces down the rows,", "within thread 0's warp"],
             ["64 rows ÷ (16 rows per piece", "× 4 warps) = 1"],
             ["None: kept whole. There is", "only one"], FAINT),
            ("2", "4 : 16",
             ["pieces across the head dims"],
             ["64 dims ÷ 16 dims per piece = 4"],
             ["k: picks one piece per step,", "so the loop runs 4 times"], ORANGE_L)]
    hy = y + 4
    for x, h in zip(cols, heads):
        b.append(text(x, hy, h, 11.5, INK2, weight="600"))
    b.append(line(20, hy + 9, W - 20, hy + 9, GRID))
    ry = hy + 18
    for m, ss, c1, c2, c3, fill in rows:
        b.append(rect(20, ry, W - 40, 44, fill, rx=3))
        b.append(text(cols[0] + 12, ry + 18, m, 13, ORANGE if m == "2" else INK, "middle", "700"))
        b.append(text(cols[1], ry + 18, ss, 12, INK, weight="600", family=MONO))
        for x, ls in zip(cols[2:], (c1, c2, c3)):
            for i, s in enumerate(ls):
                b.append(text(x, ry + 18 + 16 * i, s, 11.5, INK))
        ry += 50
    b.append(text(34, ry + 12, "Strides are distances in sQ, in values. A row holds 64 values, so 8 rows down is "
                  "512; the next piece across starts 16 values later. A mode of size 1 never steps, so its "
                  "stride is 0.", 11.5, INK2))
    b.append(text(34, ry + 30, "K gets the same treatment: tSsK = partition_B(sK) prints ((2, 2), 8, 4). Its mode 1 is "
                  "8, one piece per mma (panel 7); its mode 2 is also 4, the same 4 steps.", 11.5, INK2))
    y = ry + 30 + 46

    # 5c: why the head dims come in 4 steps
    y = sub(y, "5c. Why k = 0, 1, 2, 3: one instruction covers only 16 head dims")
    x0, cw, gp = 120, 7, 16
    gy = y + 20

    def dims_row(yy, label):
        b.append(text(x0 - 8, yy + 12, label, 10.5, INK2, "end"))
        for k in range(4):
            sx0 = x0 + k * (16 * cw + gp)
            b.append(rect(sx0, yy, 16 * cw, 16, ORANGE_L if k == 1 else FAINT, ORANGE if k == 1 else GRID))
            b.append(text(sx0 + 8 * cw, yy + 12, f"dims {16 * k}–{16 * k + 15}", 9.5, INK, "middle"))

    dims_row(gy, "Q row 0")
    dims_row(gy + 38, "K key 0")
    py = gy + 82
    b.append(text(x0 - 8, py + 14, "S[0, 0] =", 11, INK, "end", "600"))
    for k in range(4):
        sx0 = x0 + k * (16 * cw + gp)
        b.append(rect(sx0, py, 16 * cw, 20, ORANGE if k == 1 else ORANGE_L, ORANGE, rx=3))
        b.append(text(sx0 + 8 * cw, py + 14, f"part k = {k}", 10, SURF if k == 1 else INK, "middle", "600"))
        if k:
            b.append(text(sx0 - gp / 2, py + 15, "+", 12, INK, "middle", "700"))
    pic_bottom = py + 20
    ty = para(y, "A score is a dot product over all 64 head dims. The tensor-core instruction is "
                 "m16n8k16: 16 rows × 8 keys, and only 16 head dims (the k16). The hardware fixes these sizes.")
    ty = para(ty, "So each score is built in 4 parts, one per 16 dims. Step k computes part k and adds it into "
                  "acc_S (panel 8). Both k's count head dims; neither is the key matrix K.")
    y = max(pic_bottom, ty) + 40

    # 5d: range_constexpr writes the 4 steps out
    y = sub(y, "5d. range_constexpr: the 4 steps are written out when compiling")
    x0 = 56
    uy = y
    for k in range(4):
        bx = x0 + k * 140
        b.append(rect(bx, uy + 10, 130, 62, ORANGE_L if k == 1 else FAINT, ORANGE if k == 1 else GRID, 1, rx=4))
        for i, s in enumerate((f"copy Q piece {k}", f"copy K pieces, k = {k}", f"gemm, k = {k}")):
            b.append(text(bx + 10, uy + 28 + 17 * i, s, 10.5, INK))
    b.append(text(x0, uy + 92, "What gets compiled: the body 4 times, k a fixed number in each", 11, INK2))
    pic_bottom = uy + 92
    ty = para(y, "cute.size(tSsQ, mode=[2]) is 4, the size of mode 2. range_constexpr runs the loop in "
                 "Python while compiling and writes the body out 4 times, with no loop counter at run time.")
    ty = para(ty, "This matters because tSrQ and tSrK are registers, which have no addresses: "
                  "tSrQ[None, None, k] must name exact registers when compiling. A k known only at run time "
                  "would push the arrays out to slow local memory.")
    y = max(pic_bottom, ty) + 52

    # ---- 6: Q into registers --------------------------------------------------------------------
    y = header(y, 6, "Q: 8 values into registers, 2 per load", CODE[5])
    px, x0 = 2, 56
    gy = y + 40
    b.append(text(x0, gy - 12, "sQ", 12, INK, weight="600", family=MONO))
    tile(x0, gy, px, SHR, frame=(0, 16, 16, 32))
    for r in (0, 8):
        for d in (16, 17, 24, 25):
            b.append(rect(x0 + d * px, gy + r * px, px, px, BLUE))
    b.append(text(x0, gy + 64 * px + 18, "orange: warp 0's", 10.5, ORANGE, weight="600"))
    b.append(text(x0, gy + 64 * px + 32, "Q piece at k = 1:", 10.5, ORANGE, weight="600"))
    b.append(text(x0, gy + 64 * px + 46, "rows 0–15, dims 16–31", 10.5, INK2))
    zx, cs = 250, 11
    b.append(line(x0 + 32 * px + 2, gy, zx - 4, gy, ORANGE, 1, "3 3"))
    b.append(line(x0 + 32 * px + 2, gy + 16 * px, zx - 4, gy + 16 * cs, ORANGE, 1, "3 3"))
    for r in range(16):
        for d in range(16):
            on = r in (0, 8) and d in (0, 1, 8, 9)
            cell(zx + d * cs, gy + r * cs, cs, BLUE if on else SHR, stroke=SURF if not on else GRID)
    for r in (0, 8):
        b.append(text(zx - 5, gy + r * cs + 9, f"row {r}", 9.5, INK2, "end"))
        for d in (0, 8):
            b.append(rect(zx + d * cs - 1.5, gy + r * cs - 1.5, 2 * cs + 3, cs + 3, "none", ORANGE, 1.6))
    for d in (0, 8, 15):
        b.append(text(zx + d * cs + cs / 2, gy - 4, str(16 + d), 9, MUTED, "middle"))
    b.append(text(zx + 8 * cs, gy + 16 * cs + 16, "orange frame: one load, 2 values", 10.5, ORANGE, "middle", "600"))
    rgx = 488
    arrow(zx + 16 * cs + 8, gy + 50, rgx - 8, gy + 50, "4 loads")
    b.append(text(rgx, gy - 4, "tSrQ[None, None, 1]", 11, INK, weight="600", family=MONO))
    for i, (r, d) in enumerate(((0, 16), (8, 16), (0, 24), (8, 24))):
        yy = gy + 12 + 26 * i
        for v in range(2):
            cell(rgx + v * cs, yy, cs, BLUE, stroke=SURF)
        b.append(text(rgx + 2 * cs + 6, yy + 9, f"row {r}, dims {d}–{d + 1}", 9.5, INK2))
    b.append(text(rgx, gy + 12 + 26 * 4 + 6, "4 registers", 10.5, INK2))
    pic_bottom = gy + 16 * cs + 16
    ty = para(y, "tSsQ[None, None, 1] is thread 0's share of Q piece 1: rows 0 and 8, head dims 16, 17, 24, "
                 "25. That is 8 bf16 values, 16 bytes.")
    ty = para(ty, "autovec_copy uses plain shared-memory loads (LDS), each as wide as the values allow. Only "
                  "neighbouring pairs sit next to each other in sQ, so one load moves 2 values, 4 bytes: 4 "
                  "loads per k, 16 per key block.")
    ty = para(ty, "Q never changes, yet the loop re-reads it for every key block. Round 2 replaces these "
                  "loads with ldmatrix.")
    y = max(pic_bottom, ty) + 52

    # ---- 7: K into registers --------------------------------------------------------------------
    y = header(y, 7, "K: 32 values into registers, and an 8-way bank conflict", CODE[6])
    x0, cs = 104, 11
    gy = y + 40
    b.append(text(x0 - 48, gy - 22, "sK at k = 1: the 8 keys thread 0 reads, dims 16–31", 12, INK, weight="600"))
    for i in range(8):
        yy = gy + i * 15
        for d in range(16):
            on = d in (0, 1, 8, 9)
            cell(x0 + d * cs, yy, cs, BLUE if on else SHR, stroke=SURF if not on else GRID)
        for d in (0, 8):
            b.append(rect(x0 + d * cs - 1.5, yy - 1.5, 2 * cs + 3, cs + 3, "none", ORANGE, 1.4))
        b.append(text(x0 - 6, yy + 9, f"key {8 * i}", 9.5, INK2, "end"))
    for d in (0, 8, 15):
        b.append(text(x0 + d * cs + cs / 2, gy - 4, str(16 + d), 9, MUTED, "middle"))
    rgx = 424
    arrow(x0 + 16 * cs + 8, gy + 56, rgx - 8, gy + 56, "16 loads")
    b.append(text(rgx, gy - 8, "tSrK[None, None, 1]", 11, INK, weight="600", family=MONO))
    for i in range(8):
        yy = gy + i * 15
        for v in range(4):
            cell(rgx + v * cs + (4 if v >= 2 else 0), yy, cs, BLUE, stroke=SURF)
        b.append(text(rgx + 4 * cs + 12, yy + 9, f"for mma {i}", 9.5, INK2))
    # one load across warp 0: lane 4g + t reads key g, dims 16 + 2t and 17 + 2t, in bank 8 + t
    by, bx, bc = gy + 8 * 15 + 52, 104, 14
    b.append(text(x0 - 48, by - 26, "One of those loads, by all 32 lanes of warp 0 at once (mma 0, dims 16 + 2t, 17 + 2t)",
                  12, INK, weight="600"))
    for bk in range(32):
        if bk % 4 == 0 or bk == 31:
            b.append(text(bx + bk * bc + bc / 2, by - 6, str(bk), 9, MUTED, "middle"))
    b.append(text(bx - 6, by - 6, "bank", 9.5, INK2, "end"))
    for g in range(8):
        yy = by + g * bc
        b.append(text(bx - 6, yy + 10, f"key {g}", 9.5, INK2, "end"))
        for bk in range(32):
            t = bk - 8
            if 0 <= t < 4:
                lane = 4 * g + t
                cell(bx + bk * bc, yy, bc, BLUE if lane == 0 else BLUE_L, str(lane), SURF if lane == 0 else INK)
            else:
                cell(bx + bk * bc, yy, bc, FAINT, stroke=SURF)
    b.append(rect(bx + 8 * bc - 1.5, by - 1.5, 4 * bc + 3, 8 * bc + 3, "none", ORANGE, 1.8))
    b.append(text(bx + 12 * bc + 10, by + 4 * bc + 4, "8 lanes per bank, 8 different words:", 11.5, ORANGE,
                  weight="600"))
    b.append(text(bx + 12 * bc + 10, by + 4 * bc + 21, "served one after another, 8 passes", 11.5, ORANGE,
                  weight="600"))
    pic_bottom = by + 8 * bc
    ty = para(y, "tSsK[None, None, 1]: for each of the 8 mmas, one key (0, 8, 16, …, 56) at head dims 16, 17, "
                 "24, 25. That is 32 values, again in pairs: 16 loads per k, 64 per key block.")
    ty = para(ty, "Shared memory serves a warp's load from 32 banks, 4 bytes wide. A row of K is 64 × 2 = 128 "
                  "bytes, exactly one sweep of the 32 banks, so a given head dim sits in the same bank in "
                  "every row.")
    ty = para(ty, "In one load, lanes 4g + t read key g, so 8 lanes need 8 different rows of the same bank: "
                  "an 8-way bank conflict. Q's loads in ⑥ have the same problem. Round 1's swizzle removes "
                  "it.")
    y = max(pic_bottom, ty) + 52

    # ---- 8: the tensor-core multiply --------------------------------------------------------------
    y = header(y, 8, "8 tensor-core instructions per warp, added into acc_S", CODE[7])
    cs, gp = 5, 3
    sx = 166
    ky = y + 46
    sy = ky + 16 * cs + 34

    def kx(c):
        return sx + c * cs + (c // 8) * gp

    b.append(text(sx, ky - 22, "K pieces 0–7 at k = 1 (Kᵀ: dims 16–31 × keys 0–63)", 11.5, INK, weight="600"))
    for c in range(64):
        for d in range(16):
            on = c % 8 == 0 and d in (0, 1, 8, 9)
            b.append(rect(kx(c), ky + d * cs, cs, cs, BLUE if on else BLUE_L))
    for n in range(8):
        b.append(text(kx(8 * n) + 4 * cs, ky - 5, f"n = {n}", 9, MUTED, "middle"))
    qx = sx - 16 * cs - 18
    b.append(text(qx + 8 * cs, sy - 8, "Q piece 1", 10.5, INK, "middle", "600"))
    for r in range(16):
        for d in range(16):
            on = r in (0, 8) and d in (0, 1, 8, 9)
            b.append(rect(qx + d * cs, sy + r * cs, cs, cs, BLUE if on else BLUE_L))
    b.append(text(qx + 8 * cs, sy + 16 * cs + 14, "rows 0–15", 9.5, INK2, "middle"))
    b.append(text(qx + 8 * cs, sy + 16 * cs + 27, "dims 16–31", 9.5, INK2, "middle"))
    for c in range(64):
        for r in range(16):
            on = r in (0, 8) and c % 8 in (0, 1)
            b.append(rect(kx(c), sy + r * cs, cs, cs, BLUE if on else BLUE_L))
    for n in range(8):
        b.append(text(kx(8 * n) + 4 * cs, sy + 16 * cs + 14, f"mma {n}", 9, INK, "middle", "600"))
    b.append(text(kx(63) + cs + 8, sy + 8 * cs - 4, "acc_S, warp 0:", 10.5, INK, weight="600"))
    b.append(text(kx(63) + cs + 8, sy + 8 * cs + 10, "rows 0–15 ×", 10, INK2))
    b.append(text(kx(63) + cs + 8, sy + 8 * cs + 23, "keys 0–63", 10, INK2))
    b.append(text(qx + 16 * cs + 9, sy + 8 * cs + 4, "×", 12, INK2, "middle"))
    ay = sy + 16 * cs + 56
    b.append(text(56, ay, "acc_S over the 4 steps: each step adds 16 more head dims to every score", 11.5, INK,
                  weight="600"))
    for k in range(4):
        fill = ORANGE if k == 1 else ORANGE_L if k < 1 else FAINT
        b.append(rect(56 + k * 100, ay + 10, 100, 16, fill, GRID, 0.6))
        b.append(text(56 + k * 100 + 50, ay + 22, f"k = {k}: dims {16 * k}–{16 * k + 15}", 9.5,
                      SURF if k == 1 else INK, "middle"))
    pic_bottom = ay + 26
    ty = para(y, "All three are in registers now. cute.gemm takes step k's pieces. For warp 0: Q piece 1 (rows 0–15, dims 16–31) times each "
                 "of the 8 K pieces, one mma.m16n8k16 (HMMA) per piece, each adding into its 16 × 8 piece "
                 "of acc_S.")
    ty = para(ty, "Each mma runs on the whole warp. Thread 0 brings only its blue values and holds 4 scores of "
                  "each S piece: rows 0, 8 × keys 8n, 8n + 1.")
    ty = para(ty, "acc_S is both input and output, so the 4 steps add up: 32 HMMA per warp per key block, and "
                  "after k = 3 every score covers all 64 head dims. Warps 1–3 do rows 16–63 at the same time.")
    y = max(pic_bottom, ty) + 52

    # ---- 9: the rest of the key block --------------------------------------------------------------
    y = header(y, 9, "The rest of the key block, in brief", CODE[8])
    bw, bh, gap, x0 = 122, 62, 22, 56
    gy = y + 20
    row1 = [("acc_S", "32 scores per thread", "", BLUE_L),
            ("mask", "diagonal blocks only:", "later keys → −1e6", BLUE_L),
            ("online softmax", "row max, row sum:", "4 SHFL, 34 exp2", BLUE_L),
            ("P", "exp2 of the scores,", "as bf16", BLUE_L)]
    row2 = [("sP", "STS × 16:", "registers → shared", SHR),
            ("P back", "LDS × 16:", "shared → registers", BLUE_L),
            ("O += P V", "V from sV: 128 LDS", "of 2 bytes; 32 HMMA", BLUE_L)]

    def flow(boxes, yy):
        for i, (name, l1, l2, fill) in enumerate(boxes):
            bx = x0 + i * (bw + gap)
            b.append(rect(bx, yy, bw, bh, fill, GRID, 1, 6))
            b.append(text(bx + 10, yy + 20, name, 12, INK, weight="600"))
            b.append(text(bx + 10, yy + 38, l1, 10, INK2))
            b.append(text(bx + 10, yy + 52, l2, 10, INK2))
            if i:
                arrow(bx - gap + 2, yy + bh / 2, bx - 3, yy + bh / 2)

    flow(row1, gy)
    r2y = gy + bh + 40
    flow(row2, r2y)
    ex = x0 + 3 * (bw + gap) + bw / 2
    b.append(f'<path d="M{ex},{gy + bh} V{gy + bh + 20} H{x0 + bw / 2} V{r2y - 3}" fill="none" stroke="{INK2}" '
             f'stroke-width="1.4" marker-end="url(#ahk)"/>')
    ly2 = r2y + bh + 26
    legend(x0, ly2, BLUE_L, "data in registers")
    legend(x0 + 160, ly2, SHR, "data in shared memory")
    pic_bottom = ly2
    ty = para(y, "The comment stands for the rest of the key block. Counts are per thread, per key block.")
    ty = para(ty, "The softmax combines each row across its quad with 4 warp shuffles (two per row max). P "
                  "then makes a round trip through shared memory to become the A operand of O += P V; Round 3 "
                  "keeps it in registers.")
    ty = para(ty, "V is read transposed, one 2-byte value per load: the 128 16-bit loads in v0's count. Round 2 "
                  "reads it with ldmatrix.trans.")
    y = max(pic_bottom, ty) + 20
    part1 = svg(W, y5 - 32, [DEFS] + head1 + b[:i5])
    part2 = svg(W, y - (y5 - y1), [DEFS] + head2 + [f'<g transform="translate(0 {y1 - y5})">'] + b[i5:] + ['</g>'])
    return part1, part2


def ldsm_copies():
    """Round 2's five lines, one section each: the ldmatrix atom, the three tiled copies built from the
    tiled MMA, and one copy of step k of Q. Counts per warp, per key block, from the v2 kernel:
    Q 4 x 1 + K 4 x 4 + P 4 x 1 + V 4 x 4 = 40 LDSM."""
    import textwrap
    W, TX, SX = 1040, 430, 50
    FAINT, YEL_L, YEL = PANEL, GOLD_L, GOLD
    WARP = [(BLUE_L, BLUE), (ORANGE_L, ORANGE), (AQUA_L, AQUA), (YEL_L, YEL)]
    b = [f'<defs><marker id="ahl" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>',
         text(20, 30, "Round 2's ldmatrix copies, line by line", 17, weight="600"),
         text(20, 52, "Shared memory → registers for the four operands of the two multiplies: Q and K for S = Q Kᵀ, "
                      "P and V for O = P·V.", 12.5, INK2),
         text(20, 70, "Q, K and V tiles are 64 × 64 bf16; 4 warps, each owning 16 rows of the block's 64. "
                      "A step k is 16 head dims of S's sum (or 16 keys of O's).", 12.5, INK2)]

    def arrow(x1, y1, x2, y2):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.4" '
                 f'marker-end="url(#ahl)"/>')

    def header(y, n, title, code):
        b.append(line(20, y - 26, W - 20, y - 26, GRID))
        b.append(f'<circle cx="31" cy="{y - 5}" r="11" fill="{INK}"/>')
        b.append(text(31, y, str(n), 12.5, SURF, "middle", "700"))
        b.append(text(50, y, title, 15, weight="600"))
        y += 14
        h = 21 * len(code) + 12
        b.append(rect(20, y, W - 40, h, FAINT, GRID, rx=4))
        for i, c in enumerate(code):
            yy = y + 21 + 21 * i
            k = c.find("#")
            code_part = c if k < 0 else c[:k].rstrip()
            ind = len(code_part) - len(code_part.lstrip(" "))
            b.append(text(34 + 7.53 * ind, yy, code_part.lstrip(" "), 12.5, INK, family=MONO))
            if k >= 0:
                b.append(text(34 + 7.53 * k, yy, c[k:], 12.5, MUTED, family=MONO))
        return y + h + 30

    def para(y, s, fill=INK2, n=84):
        for ln in textwrap.wrap(s, n):
            b.append(text(TX, y, ln, 12.5, fill))
            y += 18
        return y + 8

    def args(y, items, kw=170):
        # (code, meaning) pairs: the code in blue mono, its meaning in plain English beside it
        for code, s in items:
            b.append(text(TX, y, code, 12, BLUE, family=MONO))
            for ln in textwrap.wrap(s, 62):
                b.append(text(TX + kw, y, ln, 12.5, INK2))
                y += 18
            y += 8
        return y

    def piece(x, y, s, fill, edge, label=None):
        # a 16 x 16 piece = four 8 x 8 tiles = one ldmatrix.x4
        for i in range(2):
            for j in range(2):
                b.append(rect(x + j * s, y + i * s, s, s, fill, edge, 1))
        if label:
            b.append(text(x + 2 * s + 10, y + s + 4, label, 11.5, INK2))

    # ---- 1: the atom -----------------------------------------------------------------------------
    y = header(118, 1, "The atom: one ldmatrix instruction",
               ["def ldsm(transpose):",
                "    return cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=4), dtype)"])
    s = 30
    for t in range(4):
        fill, edge = WARP[t]
        tx, ty = SX + (t % 2) * (s + 4), y + 8 + (t // 2) * (s + 4)
        b.append(rect(tx, ty, s, s, fill, edge, 1))
        b.append(text(tx + s / 2, ty + s / 2 + 4, "8×8", 10.5, INK2, "middle"))
    b.append(text(SX + s + 2, y + 2 * s + 30, "4 tiles, bf16", 11.5, MUTED, "middle"))
    arrow(SX + 2 * s + 12, y + s + 10, SX + 2 * s + 64, y + s + 10)
    bx = SX + 2 * s + 70
    b.append(rect(bx, y + s - 8, 160, 36, BLUE_L, BLUE, 1, 6))
    b.append(text(bx + 80, y + s + 7, "32 lanes of one warp", 11.5, INK, "middle", "600"))
    b.append(text(bx + 80, y + s + 21, "8 values each", 11, INK2, "middle"))
    ty = args(y + 4, [
        ("LdMatrix8x8x16bOp", "the ldmatrix instruction: it reads 8 × 8 tiles of 16-bit values from shared memory."),
        ("num_matrices=4", "four tiles per instruction (ldmatrix.x4): 256 values, 8 for each of the 32 lanes."),
        ("transpose", "False: each tile as stored. True: rows and columns swapped (ldmatrix.trans)."),
        ("make_copy_atom", "packages the instruction, with the value type (dtype = bf16), as one copy step that CuTe calls an atom: one ldmatrix by one warp. The copies in steps 2–4 are built by repeating this step over the tile."),
    ])
    y = max(y + 2 * s + 40, ty) + 30

    # ---- 2: A operand --------------------------------------------------------------------------
    y = header(y, 2, "A copy shaped like the multiply's A operand: Q, and later P",
               ["s2r_A = cute.make_tiled_copy_A(ldsm(False), tiled_mma)     # Q and P"])
    s = 15
    b.append(text(SX, y + 4, "Q, one step k: 64 rows × 16 head dims", 11.5, INK, weight="600"))
    for w in range(4):
        fill, edge = WARP[w]
        py = y + 16 + w * (2 * s + 6)
        b.append(text(SX + 50, py + s + 4, f"rows {16 * w}–{16 * w + 15}", 11, MUTED, "end"))
        piece(SX + 60, py, s, fill, edge, f"warp {w}: 1 ldmatrix.x4")
    ty = para(y + 4, "This line builds the copy, a plan: no data moves yet (step 5 runs it).", INK)
    ty = para(ty, "make_tiled_copy_A copies the tiled MMA's A operand layout, so the copy hands each lane exactly "
                     "the values its multiply reads. No thread layout is written by hand.")
    ty = para(ty, "With (4, 1, 1), each warp multiplies only its own 16 rows, so on step k it loads only "
                  "those: 16 rows × 16 head dims, four 8 × 8 tiles, one ldmatrix.x4.")
    ty = para(ty, "P is the A operand of O = P·V, so the same copy loads it too.")
    y = max(y + 16 + 4 * (2 * s + 6), ty) + 30

    # ---- 3: B operand, K -------------------------------------------------------------------------
    y = header(y, 3, "The same for the B operand: K",
               ["s2r_K = cute.make_tiled_copy_B(ldsm(False), tiled_mma)"])
    b.append(text(SX, y + 4, "K, one step k: 64 keys × 16 head dims", 11.5, INK, weight="600"))
    for p in range(4):
        py = y + 16 + p * (2 * s + 6)
        b.append(text(SX + 50, py + s + 4, f"keys {16 * p}–{16 * p + 15}", 11, MUTED, "end"))
        piece(SX + 60, py, s, FAINT, INK2)
    top, bot = y + 16, y + 16 + 4 * (2 * s + 6) - 6
    bx = SX + 60 + 2 * s + 10
    b.append(f'<path d="M{bx},{top} h6 V{bot} h-6" fill="none" stroke="{INK2}" stroke-width="1.2"/>')
    b.append(text(bx + 14, (top + bot) / 2 - 4, "every warp loads all four:", 11.5, INK2))
    b.append(text(bx + 14, (top + bot) / 2 + 12, "4 ldmatrix.x4", 11.5, INK, weight="600"))
    ty = para(y + 4, "make_tiled_copy_B does the same for the B operand. Each warp computes all 64 keys of its "
                     "rows, so every warp needs the whole step: 64 keys × 16 head dims, four 16 × 16 pieces, "
                     "one ldmatrix.x4 each.")
    ty = para(ty, "Pieces are 16 keys wide because of permutation_mnk = (64, 16, 16). With the default 8 keys, "
                  "each instruction would carry only two tiles, and K would need twice as many instructions.")
    y = max(y + 16 + 4 * (2 * s + 6), ty) + 30

    # ---- 4: B operand, V, transposed --------------------------------------------------------------
    y = header(y, 4, "B again, for O = P·V: V, transposed on the way",
               ["s2r_V = cute.make_tiled_copy_B(ldsm(True), tiled_mma)      # V, transposed on the way"])
    c = 13

    def grid8(x, yy, cells, col, title):
        b.append(text(x, yy, title, 11.5, INK, weight="600"))
        yy += 22
        b.append(text(x + 4 * c, yy - 4, "head dims →", 10.5, MUTED, "middle"))
        b.append(f'<text x="{x - 8}" y="{yy + 4 * c}" font-size="10.5" fill="{MUTED}" text-anchor="middle" '
                 f'transform="rotate(-90 {x - 8} {yy + 4 * c})">← keys</text>')
        for r in range(8):
            for q in range(8):
                on = (r, q) in cells
                b.append(rect(x + q * c, yy + r * c, c, c, col if on else FAINT, GRID, 0.6))
        return yy + 8 * c

    g1 = grid8(SX + 12, y + 4, {(0, 0), (0, 1)}, ORANGE, "transpose=False")
    g2 = grid8(SX + 12 + 8 * c + 60, y + 4, {(0, 0), (1, 0)}, ORANGE, "transpose=True")
    b.append(text(SX + 12, g1 + 18, "One 8 × 8 tile of V as stored. Orange:", 11, INK2))
    b.append(text(SX + 12, g1 + 33, "the 2 values lane 0 receives.", 11, INK2))
    ty = para(y + 4, "In shared memory V is stored like K: one key per row, head dims along the row.")
    ty = para(ty, "The multiply wants each lane's 2 values to lie along the dimension it sums over. For S = Q Kᵀ "
                  "that is head dims, along a stored row: the plain load. For O = P·V it is keys, down a "
                  "stored column.")
    ty = para(ty, "transpose=True (ldmatrix.trans) swaps each tile's rows and columns as it loads, so V needs no "
                  "separate transpose. Otherwise the copy is the same as K's: 4 ldmatrix.x4 per step.")
    y = max(g1 + 40, ty) + 30

    # ---- 5: the copy -----------------------------------------------------------------------------
    y = header(y, 5, "Run it: steps 1–4 only built copies; cute.copy moves the data",
               ["thr_A = s2r_A.get_slice(tidx)                              # this thread's part of the copy",
                "ops = SimpleNamespace(sQ=thr_A.partition_S(sQ),            # from: Q in shared memory",
                "                      rQ=thr_A.retile(tSrQ), ...)          # to: the registers the multiply reads",
                "...",
                "for k in range(4):                                         # 4 steps of 16 head dims",
                "    cute.copy(ops.s2r_A, ops.sQ[None, None, k], ops.rQ[None, None, k])     # load step k of Q",
                "    cute.copy(ops.s2r_K, ops.sK[None, None, k], ops.rK[None, None, k])     # and of K",
                "    cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)"])
    q = 20
    b.append(text(SX, y + 4, "Warp 0, step k = 1 (head dims 16–31)", 11.5, INK, weight="600"))
    x0, y0 = SX + 30, y + 30
    lx = x0 + 4 * q + 14                                          # labels to the right of the flow
    for r in range(4):
        for cc in range(4):
            on = r == 0 and cc == 1
            b.append(rect(x0 + cc * q, y0 + r * q, q, q, BLUE if on else (BLUE_L if r == 0 else FAINT),
                          BLUE if r == 0 else GRID, 1))
        b.append(text(x0 - 6, y0 + r * q + q / 2 + 4, f"warp {r}", 10, MUTED, "end"))
    for cc in range(4):
        b.append(text(x0 + cc * q + q / 2, y0 - 5, f"k={cc}", 9.5, MUTED, "middle"))
    b.append(text(lx, y0 + 12, "sQ: the Q tile in shared memory,", 11, INK, weight="600"))
    b.append(text(lx, y0 + 27, "64 rows × 64 head dims. Each warp", 11, INK2))
    b.append(text(lx, y0 + 42, "owns 16 rows (blue: warp 0's).", 11, INK2))
    cx = x0 + 2 * q
    ay = y0 + 4 * q + 8
    arrow(cx, ay, cx, ay + 34)
    b.append(text(lx, ay + 16, "cute.copy, step k = 1:", 11, INK, weight="600", family=None))
    b.append(text(lx, ay + 31, "one ldmatrix.x4 per warp", 11, INK2))
    ry = ay + 40
    for cc in range(4):
        b.append(rect(x0 + cc * q, ry, q, q, BLUE if cc == 1 else BLUE_L, BLUE, 1))
    b.append(text(lx, ry + 9, "tSrQ (= ops.rQ): one lane's Q", 11, INK, weight="600"))
    b.append(text(lx, ry + 24, "registers, 8 values per step k", 11, INK2))
    gy = ry + q + 8
    arrow(cx, gy, cx, gy + 28)
    b.append(rect(x0 - 10, gy + 32, 4 * q + 20, 26, AQUA_L, AQUA, 1, 5))
    b.append(text(cx, gy + 49, "cute.gemm", 11, INK, "middle", "700", MONO))
    b.append(text(lx + 6, gy + 49, "multiply step k = 1 reads them", 11, INK2))
    pic_bottom = gy + 60
    ty = para(y + 4, "Steps 1–4 only build copies: plans of which lane reads which values with ldmatrix. No data "
                     "moves until cute.copy runs one, inside the loop that does the multiply.")
    ty = args(ty, [
        ("ops", "just a bundle (Python's SimpleNamespace) holding the copies and their views, so they can be handed "
                "to the loop function as one argument. Not a CuTe object."),
        ("ops.sQ", "where the copy reads from: the Q tile in shared memory (sQ), cut by partition_S into the pieces "
                   "this thread gives ldmatrix (S for source)."),
        ("ops.rQ", "where the copy writes to: tSrQ, the registers the multiply reads Q from. retile makes no new "
                   "registers; it orders tSrQ the way the copy fills it, so each value lands where cute.gemm "
                   "looks for it."),
        ("[None, None, k]", "None keeps a whole mode: all of this thread's values and row pieces. k picks one step: "
                            "head dims 16k to 16k + 15."),
    ], kw=130)
    ty = para(ty, "Why one step at a time: the multiply walks the 64 head dims in 4 steps of 16. Loading step k "
                  "right before multiply k lets the multiply start as soon as its values arrive, instead of "
                  "waiting for all of Q.")
    ty = para(ty, "K uses the same pattern (ops.sK, ops.rK, s2r_K) in the same loop; P and V use it in the loop "
                  "for O = P·V, with s2r_A and s2r_V.")
    y = max(pic_bottom, ty) + 14

    b.append(line(20, y, W - 20, y, GRID))
    y += 26
    b.append(text(20, y, "Per key block, each warp issues 40 ldmatrix: Q 4 × 1 + K 4 × 4 + P 4 × 1 + V 4 × 4. "
                         "Rounds 0–1 used 224 plain loads (LDS) for the same values.", 12.5, INK, weight="600"))
    return svg(W, y + 22, b)


# ---------------------------------------------------------------------------------------------
def p_roundtrip():
    """Round 3 before/after: v2 sends P through shared memory (16 STS + 4 LDSM per pass), v3 re-views the
    same registers with as_a_operand. Loop 0 of the non-causal SASS (results/cubin/v2_ldmatrix, v3_p_regs):
    STS 24 -> 8, LDSM 40 -> 36, NOP 2 -> 0, MOV 2 -> 3: 342 -> 321."""
    W = 1040
    FAINT = PANEL
    b = [f'<defs><marker id="ahp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>',
         text(20, 30, "Round 3: P stops making a round trip through shared memory", 17, weight="600"),
         text(20, 52, "One pass of the loop. P = softmax(S) is computed in registers; the next multiply, O += P·V, "
                      "needs P as its A operand, also in registers.", 12.5, INK2)]

    def header(y, n, title):
        b.append(line(20, y - 26, W - 20, y - 26, GRID))
        b.append(f'<circle cx="31" cy="{y - 5}" r="11" fill="{INK}"/>')
        b.append(text(31, y, str(n), 12.5, SURF, "middle", "700"))
        b.append(text(50, y, title, 15, weight="600"))
        return y + 24

    def para(x, y, lines, size=12.5, fill=INK2, lh=18):
        for s in lines:
            b.append(text(x, y, s, size, fill))
            y += lh
        return y

    def arrow(x1, y1, x2, y2):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.4" '
                 f'marker-end="url(#ahp)"/>')

    def box(x, y, w, title, sub, fill, edge, h=52):
        b.append(rect(x, y, w, h, fill, edge, 1.2, 6))
        b.append(text(x + w / 2, y + 21, title, 12.5, INK, "middle", "700"))
        b.append(text(x + w / 2, y + 39, sub, 11, INK2, "middle"))

    def flow(y, boxes):
        for i, (x, w, t, s, f, e) in enumerate(boxes):
            box(x, y, w, t, s, f, e)
            if i:
                px, pw = boxes[i - 1][0], boxes[i - 1][1]
                arrow(px + pw + 2, y + 26, x - 3, y + 26)

    # ---- 1: the two flows ------------------------------------------------------------------------
    y = header(100, 1, "What happens to P, before and after")
    common = [(50, 140, "S = Q·Kᵀ", "registers, fp32", FAINT, GRID),
              (220, 150, "softmax, to bf16", "rP: P in registers", FAINT, GRID)]
    last = (860, 150, "O += P·V", "cute.gemm reads P", FAINT, GRID)
    b.append(text(50, y + 8, "v2 (Round 2): P goes out to shared memory and comes back", 13, INK, weight="700"))
    flow(y + 22, common + [(400, 120, "16 STS", "write P out", ORANGE_L, ORANGE),
                           (550, 130, "sP", "shared memory, 8 KB", ORANGE_L, ORANGE),
                           (710, 120, "4 LDSM", "read P back", ORANGE_L, ORANGE), last])
    y2 = y + 22 + 52 + 44
    b.append(text(50, y2 + 8, "v3 (Round 3): P stays where it is", 13, INK, weight="700"))
    flow(y2 + 22, common + [(400, 430, "as_a_operand(rP)", "the same registers, read as the A operand: 0 instructions",
                             AQUA_L, AQUA), last])
    y = y2 + 22 + 52 + 32
    y = para(50, y, ["Why v2 needed the trip: the multiply S = Q·Kᵀ leaves each lane its values of S in the "
                     "accumulator's order, and O += P·V asks each lane",
                     "for its values of P in the A operand's order. Writing P to shared memory and reading it back "
                     "with the A-operand copy (s2r_A) reorders it."])
    y = para(50, y + 6, ["Why v3 can skip it: each lane already holds exactly the values of P that the A operand asks "
                         "it for; only the grouping into registers differs.",
                         "as_a_operand just relabels the registers, so no data moves. The figure after the code "
                         "checks this, register by register."], fill=INK)
    y += 40

    # ---- 2: the code change ----------------------------------------------------------------------
    y = header(y, 2, "The code change, v2 → v3 (inside the loop)")
    rows = [(" ", "rP.store(acc_S.load().to(self.dtype))", "P as bf16, in registers"),
            ("-", "cute.autovec_copy(rP, tPsP)", "16 STS: P to shared memory"),
            ("-", "cute.arch.sync_warp()", "wait until the warp's P is written"),
            ("+", "tOrP = as_a_operand(rP)", "same registers, new view: 0 instructions"),
            (" ", "for k in range(4):", ""),
            ("-", "    cute.copy(ops.s2r_A, ops.sP[None, None, k], ops.rP[None, None, k])", "4 LDSM: P back"),
            (" ", "    cute.copy(ops.s2r_V, ops.sVt[None, None, k], ops.rVt[None, None, k])", "V, as before"),
            (" ", "    cute.gemm(tiled_mma, acc_O, tOrP[None, None, k], tOrVt[None, None, k], acc_O)", ""),
            ("-", "cute.arch.sync_warp()", "wait before the next pass overwrites sP")]
    cy, lh = y + 4, 22
    b.append(rect(40, cy, W - 80, lh * len(rows) + 12, FAINT, GRID, 1, 5))
    for i, (mark, code, com) in enumerate(rows):
        yy = cy + 6 + i * lh
        if mark != " ":
            b.append(rect(41, yy, W - 82, lh, ORANGE_L if mark == "-" else AQUA_L))
        b.append(text(54, yy + 16, mark, 13, ORANGE if mark == "-" else AQUA, weight="700", family=MONO))
        ind = len(code) - len(code.lstrip(" "))
        b.append(text(74 + 7.53 * ind, yy + 16, code.lstrip(" "), 12.5, INK, family=MONO))
        if com:
            b.append(text(650, yy + 16, "# " + com, 12, MUTED, family=MONO))
    y = cy + lh * len(rows) + 12 + 22
    y = para(50, y, ["Outside the loop, v3 also drops the 8 KB buffer sP and the separate A-operand registers that v2 "
                     "loaded P into."])
    y += 40

    # ---- 3: the delta ----------------------------------------------------------------------------
    y = header(y, 3, "What it saves, per pass of the loop")
    chips = [("−16 STS", ORANGE_L), ("−4 LDSM", ORANGE_L), ("−2 NOP", FAINT), ("+1 MOV", FAINT)]
    x = 50
    for s, f in chips:
        w = 12 + 8.2 * len(s)
        b.append(rect(x, y, w, 28, f, GRID, 1, 14))
        b.append(text(x + w / 2, y + 19, s, 12.5, INK, "middle", "700", MONO))
        x += w + 10
    b.append(text(x + 8, y + 19, "= 342 → 321 instructions", 13, INK, weight="700"))
    y = para(50, y + 56, ["Shared memory per block: 32 KB → 24 KB (no sP).",
                          "Time at N = 4096: about 1% faster (2.19 → 2.17 ms). The round trip was already "
                          "conflict-free and only 20 of 342 instructions, so it was cheap."])
    return svg(W, y + 10, b)


# ---------------------------------------------------------------------------------------------
def a_operand_view():
    # as_a_operand step by step, following thread 0 (warp 0, lane 0) of v3.
    # Checked on the GPU with CuTe DSL (4 warps, permutation (64, 16, 16), 64 x 64 tile):
    #   acc   ((2,2),1,8):((1,2),0,4)
    #   pairs ((2,2),1,(2,4)):((1,2),0,(4,8))
    #   view  (((2,2),2),1,4):(((1,2),4),0,8)     partition_A of the same tile: ((2,2,2),1,4)
    #   register i holds the same (row, column) of P under partition_C and partition_A, for i = 0..31.
    W = 1040
    FAINT = PANEL
    TILE = [BLUE_L, ORANGE_L]                       # alternate 8-column tiles / 16-column steps
    STEP = [BLUE, ORANGE, AQUA, GOLD]          # steps k = 0..3
    STEP_L = [BLUE_L, ORANGE_L, AQUA_L, GOLD_L]
    b = [f'<defs><marker id="aha" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>',
         text(20, 30, "How as_a_operand works: the same 32 values, new labels", 17, weight="600")]

    def header(y, n, title):
        b.append(line(20, y - 26, W - 20, y - 26, GRID))
        b.append(f'<circle cx="31" cy="{y - 5}" r="11" fill="{INK}"/>')
        b.append(text(31, y, str(n), 12.5, SURF, "middle", "700"))
        b.append(text(50, y, title, 15, weight="600"))
        return y + 24

    def para(y, lines, x=50, size=12.5, fill=INK2, lh=18, weight="normal"):
        for s in lines:
            b.append(text(x, y, s, size, fill, weight=weight))
            y += lh
        return y

    def arrow(x1, y1, x2, y2, col=INK2):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="1.4" '
                 f'marker-end="url(#aha)"/>')

    def bracket(x1, x2, y, label, col=INK2, up=False, size=11):
        d = -5 if up else 5
        b.append(f'<path d="M{x1 + 1},{y} V{y + d} H{x2 - 1} V{y}" fill="none" stroke="{col}" stroke-width="1.3"/>')
        b.append(text((x1 + x2) / 2, y + (d - 5 if up else d + 13), label, size, col, "middle", "600"))

    # thread 0's cells: rows 0 and 8, columns 8n and 8n + 1. Accumulator number i = 4n + col_in_pair + 2 * (row == 8)
    def cells():
        for n in range(8):
            for dr in (0, 1):
                for dc in (0, 1):
                    yield 8 * dr, 8 * n + dc, 4 * n + dc + 2 * dr

    cw, ch, gx = 13, 11, 110

    def grid(y, groups, label_of, fill_of, glabel):
        """16 rows x 64 columns of warp 0's piece of P; columns split into groups of `groups`."""
        for r in range(16):
            for c in range(64):
                g = c // groups
                b.append(rect(gx + c * cw, y + r * ch, cw, ch, FAINT if (g % 2 == 0) else FILL, SURF, 0.5))
        for r, c, i in cells():
            b.append(rect(gx + c * cw, y + r * ch, cw, ch, fill_of(i), SURF, 0.5))
            b.append(text(gx + c * cw + cw / 2, y + r * ch + 8.5, label_of(i), 7.5, SURF, "middle", "700"))
        for g in range(64 // groups + 1):
            b.append(line(gx + g * groups * cw, y - 4, gx + g * groups * cw, y + 16 * ch, INK2, 1))
        for g in range(64 // groups):
            b.append(text(gx + (g + 0.5) * groups * cw, y - 8, glabel(g), 10.5, INK, "middle", "600"))
        b.append(text(gx - 8, y + 9, "row 0", 10, INK2, "end"))
        b.append(text(gx - 8, y + 8 * ch + 9, "row 8", 10, INK2, "end"))
        b.append(text(gx - 8, y + 16 * ch - 2, "row 15", 10, MUTED, "end"))
        b.append(text(gx, y + 16 * ch + 14, "key 0", 10, MUTED))
        b.append(text(gx + 64 * cw, y + 16 * ch + 14, "key 63", 10, MUTED, "end"))
        return y + 16 * ch + 14

    # ---- the message ---------------------------------------------------------------------------
    y = 58
    b.append(rect(20, y - 6, W - 40, 82, GOLD_L, GOLD_EDGE, 1, 6))
    para(y + 14, ["The idea in one sentence: after S = Q·Kᵀ, each thread already holds exactly the values of P that the "
                  "next multiply, O += P·V,",
                  "will ask it for, and even in the same order. Only the labels CuTe uses to find them differ. "
                  "as_a_operand changes the labels,",
                  "so P goes straight from one multiply into the next without moving. The steps below check this "
                  "for one thread, thread 0."], x=36, fill=INK)
    y += 76 + 46

    # ---- 1: what thread 0 holds ----------------------------------------------------------------
    y = header(y, 1, "What thread 0 holds: 32 values of P, numbered 0–31")
    y = para(y, ["Warp 0 computes rows 0–15 of P (all 64 keys), and its 32 threads share the 16 × 64 = 1,024 values: "
                 "32 each. Which 32 is not",
                 "our choice: the tensor-core instruction (mma.m16n8k16) fixes it. It produces S in tiles of 16 rows × "
                 "8 keys, and in every tile it",
                 "gives thread 0 the same 4 spots: rows 0 and 8, the tile's first 2 columns. Thread 0 numbers its "
                 "values tile by tile:"])
    y = grid(y + 30, 8, lambda i: str(i), lambda i: BLUE, lambda g: f"tile {g}")
    y = para(y + 22, ["Tile n holds values 4n to 4n + 3, in the order (row 0, 1st column), (row 0, 2nd column), "
                      "(row 8, 1st column), (row 8, 2nd column).",
                      "This is the accumulator view: CuTe labels value i by (spot in the tile, tile n)."])
    y += 46

    # ---- 2: what the next multiply asks for ----------------------------------------------------
    y = header(y, 2, "What O += P·V asks thread 0 for: 4 steps of 8 values")
    y = para(y, ["The next multiply reads P 16 keys at a time (4 steps, k = 0 to 3). The same instruction fixes which "
                 "values of each 16 × 16 piece",
                 "thread 0 must hand in, and in what order (slots 0–7): rows 0 and 8, columns 0, 1, 8 and 9 of the "
                 "piece."])
    slot = lambda i: str(i % 8)
    y = grid(y + 30, 16, slot, lambda i: STEP[i // 8], lambda g: f"step k = {g}")
    y = para(y + 22, ["Compare with step 1: the marked cells are the same 32 spots. Every value the multiply wants "
                      "from thread 0 is one it already holds.",
                      "This is the A-operand view: CuTe labels each value by (slot in the step, step k)."])
    y += 46

    # ---- 3: same order ------------------------------------------------------------------------
    y = header(y, 3, "Same values, same order: only the grouping differs")
    y = para(y, ["Line thread 0's 32 values up in its own order, 0 to 31. Above: the accumulator view groups them in "
                 "8 tiles of 4. Below: the",
                 "A-operand view groups them in 4 steps of 8. Step k is simply values 8k to 8k + 7, already in "
                 "slot order 0–7."])
    bw, rx = 27, 70
    ry = y + 40
    for n in range(8):
        bracket(rx + 4 * n * bw, rx + 4 * (n + 1) * bw, ry - 4, f"tile {n}", INK2, up=True)
    for i in range(32):
        b.append(rect(rx + i * bw, ry, bw, 26, STEP_L[i // 8], SURF, 1.5))
        b.append(text(rx + i * bw + bw / 2, ry + 17, str(i), 11, INK, "middle", "700", MONO))
        b.append(text(rx + i * bw + bw / 2, ry + 40, str(i % 8), 9.5, STEP[i // 8], "middle", "700"))
    b.append(text(rx - 8, ry + 17, "value", 10, MUTED, "end"))
    b.append(text(rx - 8, ry + 40, "slot", 10, MUTED, "end"))
    for k in range(4):
        bracket(rx + 8 * k * bw, rx + 8 * (k + 1) * bw, ry + 48, f"step k = {k}", STEP[k])
    y = ry + 48 + 52
    y = para(y, ["Check one: value 13 is tile 3 (keys 24–31), row 0, 2nd column, so key 25. In the A view it is "
                 "step 1, slot 5: row 0, column 9 of keys 16–31,",
                 "so key 25 again. The same holds for all 32 values (checked on the GPU)."])
    y = para(y + 6, ["Why it lines up: one 16 × 16 step is two neighbouring 16 × 8 tiles side by side, and the "
                     "instruction places thread 0's spots the same way",
                     "in both: the first 4 slots of step k are tile 2k, the last 4 are tile 2k + 1."], fill=INK)
    y += 46

    # ---- 4: the code ---------------------------------------------------------------------------
    y = header(y, 4, "The code only rewrites the labels")
    y = para(y, ["CuTe finds a thread's values through a layout: shape (how many along each direction) : stride (how "
                 "many values to jump for each step",
                 "along it). Each layout here has three parts: the values in one group, the groups down (always 1, "
                 "stride 0), the groups across."])
    code = ["def as_a_operand(acc):",
            "    pairs = cute.logical_divide(acc.layout, (None, None, 2))",
            "    return cute.make_tensor(acc.iterator, cute.make_layout(",
            "        ((pairs.shape[0], pairs.shape[2][0]), pairs.shape[1], pairs.shape[2][1]),",
            "        stride=((pairs.stride[0], pairs.stride[2][0]), pairs.stride[1], pairs.stride[2][1])))"]
    cy = y + 6
    b.append(rect(40, cy, W - 80, 21 * len(code) + 12, PANEL, GRID, 1, 5))
    for i, c in enumerate(code):
        ind = len(c) - len(c.lstrip(" "))
        b.append(text(54 + 7.53 * ind, cy + 22 + 21 * i, c.lstrip(" "), 12.5, INK, family=MONO))
    y = cy + 21 * len(code) + 12 + 34

    def chip(x, yy, s, fill, w=None):
        w = w or 14 + 7.6 * len(s)
        b.append(rect(x, yy, w, 28, fill, INK2, 1, 5))
        b.append(text(x + w / 2, yy + 19, s, 12.5, INK, "middle", "700", MONO))
        return x + w

    lx, cx0 = 50, 340
    rows = [
        ("acc (the input)", [("(2,2) : (1,2)", BLUE_L), ("1 : 0", FAINT), ("8 : 4", ORANGE_L)],
         "4 values per tile (next column +1, row 8 +2); 8 tiles across, 4 apart"),
        ("logical_divide(…, (None, None, 2))", [("(2,2) : (1,2)", BLUE_L), ("1 : 0", FAINT), ("2 : 4", GOLD_L),
                                               ("4 : 8", ORANGE_L)],
         "cut the 8 tiles into 4 pairs: 2 tiles per pair (4 apart), pairs 8 apart"),
        ("make_layout(…)", [("((2,2),2) : ((1,2),4)", BLUE_L), ("1 : 0", FAINT), ("4 : 8", ORANGE_L)],
         "move the tile-in-pair into the group: 8 values per step; 4 steps, 8 apart"),
    ]
    for j, (lab, chips, what) in enumerate(rows):
        yy = y + j * 64
        b.append(text(lx, yy + 19, lab, 12, INK, weight="700", family=MONO if j else None))
        x = cx0
        for s, f in chips:
            x = chip(x, yy, s, f) + 8
        b.append(text(cx0, yy + 46, what, 11.5, INK2))
    y = y + 3 * 64 + 18
    y = para(y, ["make_tensor(acc.iterator, layout) puts the new layout on acc's own values: acc.iterator is where "
                 "they start. No value is copied or moved,",
                 "and no instruction is generated. Read the new layout like step 3: value = slot + 8 × step "
                 "(the first part counts 0–7, the last jumps by 8)."])
    y += 46

    # ---- 5: using it ---------------------------------------------------------------------------
    y = header(y, 5, "Using it: step k of P is read straight from thread 0's own values")
    code = ["tOrP = as_a_operand(rP)                       # rP: P as bf16, in the accumulator layout",
            "for k in range(4):",
            "    cute.gemm(tiled_mma, acc_O, tOrP[None, None, k], tOrVt[None, None, k], acc_O)"]
    cy = y + 2
    b.append(rect(40, cy, W - 80, 21 * len(code) + 12, PANEL, GRID, 1, 5))
    for i, c in enumerate(code):
        k = c.find("#")
        part = c if k < 0 else c[:k].rstrip()
        ind = len(part) - len(part.lstrip(" "))
        b.append(text(54 + 7.53 * ind, cy + 22 + 21 * i, part.lstrip(" "), 12.5, INK, family=MONO))
        if k >= 0:
            b.append(text(54 + 7.53 * k, cy + 22 + 21 * i, c[k:], 12.5, MUTED, family=MONO))
    y = cy + 21 * len(code) + 12 + 28
    y = para(y, ["tOrP[None, None, k] means: all 8 slots of step k, which are values 8k to 8k + 7. The multiply "
                 "reads them where they already are.",
                 "Before Round 3, P was written to shared memory and read back just to arrive in this order, which "
                 "it already had."])
    return svg(W, y + 12, b)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "mma_ownership.svg").write_text(mma_ownership())
    (OUT / "bank_conflicts.svg").write_text(bank_conflicts())
    (OUT / "swizzle_banks.svg").write_text(swizzle_banks())
    (OUT / "swizzle_bms.svg").write_text(swizzle_bms())
    (OUT / "smem_layout.svg").write_text(smem_layout_fig())
    (OUT / "pipeline.svg").write_text(pipeline())
    (OUT / "gpu_hierarchy.svg").write_text(gpu_hierarchy())
    (OUT / "layouts.svg").write_text(layouts())
    (OUT / "mnk_shape.svg").write_text(mnk_shape())
    (OUT / "tiled_mma.svg").write_text(tiled_mma())
    (OUT / "ldmatrix_pairs.svg").write_text(ldmatrix_pairs())
    (OUT / "warp_shuffle.svg").write_text(warp_shuffle())
    (OUT / "shuffle_args.svg").write_text(shuffle_args())
    (OUT / "warp_split.svg").write_text(warp_split())
    (OUT / "tiled_copy.svg").write_text(tiled_copy())
    (OUT / "copy_atoms.svg").write_text(copy_atoms())
    (OUT / "memory_hierarchy.svg").write_text(memory_hierarchy())
    (OUT / "block_grid.svg").write_text(block_grid())
    (OUT / "warp_lanes.svg").write_text(warp_lanes())
    (OUT / "ldmatrix_x4.svg").write_text(ldmatrix_x4())
    (OUT / "qkv_tiles.svg").write_text(qkv_tiles())
    (OUT / "cp_async.svg").write_text(cp_async())
    (OUT / "ldsm_copies.svg").write_text(ldsm_copies())
    (OUT / "ldsm_count.svg").write_text(ldsm_count())
    kb1, kb2 = key_block_v0()
    (OUT / "key_block_v0_1.svg").write_text(kb1)
    (OUT / "key_block_v0_2.svg").write_text(kb2)
    (OUT / "p_roundtrip.svg").write_text(p_roundtrip())
    (OUT / "a_operand_view.svg").write_text(a_operand_view())
    print("wrote", *sorted(p.name for p in OUT.glob("*.svg")))
