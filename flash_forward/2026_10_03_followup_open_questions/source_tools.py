# source_tools.py
# Read Nsight Compute's Source page from the command line: per-SASS-instruction counters
# (instructions executed, warp-stall samples by reason), each attributed to the Python source
# line it came from. Used by items 5, 7 and 8.
#
#   ncu --import R --page source --print-source cuda,sass --csv --launch-skip K --launch-count 1
#
# prints, per source file, every source line that has SASS, followed by those SASS instructions.
import csv
import io
import subprocess

NCU = "/opt/nvidia/nsight-compute/2026.2.1/ncu"


def _num(x):
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return 0.0


def export(report, skip, csv_out=None, view="cuda,sass"):
    """Return a list of SASS instructions as dicts: file, line, source, address, sass, counters.

    `skip` selects the kernel inside the report (0 = first profiled launch). Kernels without
    line information (SDPA's) have no cuda,sass view; for those the plain sass view is used."""
    out = subprocess.run([NCU, "--import", str(report), "--page", "source", "--print-source", view, "--csv",
                          "--launch-skip", str(skip), "--launch-count", "1"], capture_output=True, text=True).stdout
    if view == "sass":
        rows = list(csv.reader(io.StringIO(out)))
        header = next(r for r in rows if r and r[0] == "Address")
        insts = []
        for row in rows[rows.index(header) + 1:]:
            if len(row) > 2 and row[0] not in ("", "...", "-"):
                d = {h: _num(v) for h, v in zip(header[2:], row[2:])}
                d.update(file=None, line=None, source="", address=row[0], sass=row[1].strip())
                insts.append(d)
        if csv_out:
            open(csv_out, "w").write(out)
        return insts
    insts, header, file, line, src = [], None, None, None, None
    for row in csv.reader(io.StringIO(out)):
        if not row:
            continue
        if row[0] == "File Path":
            file = row[1]
        elif row[0] in ("Function Name", "Kernel Name"):
            continue
        elif row[0] == "Line No":
            header = row
        elif header and row[0]:  # a source line (its own aggregate counters follow; skip them)
            line, src = int(row[0]), row[1].strip()
        elif header and len(row) > 3 and row[2] not in ("", "...", "-"):
            d = {h: _num(v) for h, v in zip(header[4:], row[4:])}
            d.update(file=file, line=line, source=src, address=row[2], sass=row[3].strip())
            insts.append(d)
    if not insts:
        return export(report, skip, csv_out, view="sass")
    if csv_out:
        open(csv_out, "w").write(out)
    return insts


def opcode(sass):
    """'@!P0 FMUL R1, R1, 0.5' -> 'FMUL'; 'HMMA.16816.F32.BF16 R4, ...' -> 'HMMA.16816.F32.BF16'."""
    parts = sass.split()
    if parts and parts[0].startswith("@"):
        parts = parts[1:]
    return parts[0] if parts else ""
