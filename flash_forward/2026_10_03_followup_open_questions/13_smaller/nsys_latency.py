# nsys_latency.py
# Item 13: what is the "Latency" in Nsight Systems' kernel tooltip? For every kernel in an
# exported timeline (nsys export --type sqlite), print the gap between its launch API call and
# the kernel's start, measured from the call's start and from its end, next to the duration.
# The Round 0 screenshot shows "Latency: <-14.1 ms" for one SDPA kernel; whichever column
# reproduces that number is what the tooltip measures.
#   python nsys_latency.py round0.sqlite
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1])
names = dict(con.execute("SELECT id, value FROM StringIds"))
rows = con.execute("""
    SELECT k.start, k.end, k.shortName, k.demangledName, r.start, r.end, r.nameId
    FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r ON k.correlationId = r.correlationId
    ORDER BY k.start""").fetchall()
t0 = min(r[4] for r in rows)
print(f"{'kernel':28s} {'API call':22s} {'call start':>11s} {'kernel start':>13s} "
      f"{'start->start':>13s} {'end->start':>11s} {'duration':>9s}")
for ks, ke, short, dem, rs, re_, nid in rows:
    kname = names.get(short, "?")
    if "pytorch_flash" in names.get(dem, ""):
        kname = "SDPA " + kname
    print(f"{kname[:28]:28s} {names.get(nid, '?')[:22]:22s} {(rs - t0) / 1e6:9.3f}ms {(ks - t0) / 1e6:11.3f}ms "
          f"{(ks - rs) / 1e6:11.3f}ms {(ks - re_) / 1e6:9.3f}ms {(ke - ks) / 1e6:7.3f}ms")
