# analyze_captures.py
# The numbers behind clock_dynamics.png, from the Nsight Systems GPC-clock samples (10 kHz):
#   wake_capture   per cold start: when the clock first passes 2.6 GHz, counted from the wake-up,
#                  from the first long kernel, and in GPU-busy time
#   ramp_capture   how long the clock holds 2.52 GHz from idle under the cold/warm capture's work
#   gap_capture    per burst of that work after an idle gap: the clock at its first kernel, how
#                  long until it falls below 2.56 GHz, and SDPA's time before and after that
#   all three      SDPA calls slower than 3 ms, and how long the SMs sat mostly idle (SMs Active
#                  under 50%) inside them, against normal calls
#   ramp, gap      which SDPA calls overlap graphics work (vertex/tess/geometry or pixel warps in
#                  flight: the desktop, which this GPU also drives) and how long they take
#   wake, gap      once boosted, how long the clock stays up after the last kernel, and when it
#                  falls back to 2.52 GHz and to the idle clock
#   python analyze_captures.py > captures.txt
import bisect
import sqlite3
import statistics


class Capture:
    def __init__(self, path):
        con = sqlite3.connect(path)
        names = dict(con.execute("SELECT id, value FROM StringIds"))
        mets = dict(con.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
        cid = [m for m, n in mets.items() if "GPC Clock" in n][0]
        sid = [m for m, n in mets.items() if n.startswith("SMs Active")][0]
        # The clock is stored in Hz in a signed 32-bit field: add 2^32 to negative values.
        self.clock = sorted((t, (v + 2 ** 32 if v < 0 else v) / 1e9) for t, v in con.execute(
            f"SELECT timestamp, value FROM GPU_METRICS WHERE metricId={cid}"))
        self.ct = [t for t, _ in self.clock]
        self.active = sorted(con.execute(f"SELECT timestamp, value FROM GPU_METRICS WHERE metricId={sid}"))
        self.at = [t for t, _ in self.active]
        self.kernels = sorted((s, e, "pytorch_flash" in names[d]) for s, e, d in con.execute(
            "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"))
        self.ranges = sorted((s, e, txt) for s, e, txt in con.execute(
            "SELECT start, end, text FROM NVTX_EVENTS WHERE text IS NOT NULL AND end IS NOT NULL")
            if txt not in ("ours", "sdpa"))

    def clock_in(self, a, b):
        return self.clock[bisect.bisect_left(self.ct, a):bisect.bisect_right(self.ct, b)]

    def idle_ms(self, a, b):
        return 0.1 * sum(v < 50 for _, v in self.active[bisect.bisect_left(self.at, a):bisect.bisect_right(self.at, b)])


wake = Capture("reports/wake_capture.sqlite")
print("wake_capture: when the clock first passes 2.6 GHz (ms)")
for s, e, txt in wake.ranges:
    seg = wake.clock_in(s - 5e6, e)
    woke = next(t for t, v in seg if v > 1.0)
    boost = next((t for t, v in seg if v > 2.6), None)
    long_k = [a for a, b, _ in wake.kernels if s <= a <= e and b - a > 1e6]
    busy = sum(min(b, boost) - a for a, b, _ in wake.kernels if s <= a < boost) / 1e6 if boost else float("nan")
    print(f"  {txt:24s} after wake-up {(boost - woke) / 1e6 if boost else float('nan'):7.1f}   after the first long kernel "
          f"{(boost - long_k[0]) / 1e6 if boost else float('nan'):6.1f}   GPU busy before it {busy:6.1f}   "
          f"highest clock {max(v for _, v in seg):.3f} GHz")

ramp = Capture("reports/ramp_capture.sqlite")
k0 = ramp.kernels[0][0]
seg = ramp.clock_in(k0, k0 + 1e9)
first_move = next(t for t, v in seg if v > 2.528) - k0
print(f"\nramp_capture: from idle the clock holds 2.52 GHz (no sample above 2.528) for {first_move / 1e6:.0f} ms of v0 + SDPA")

gap = Capture("reports/gap_capture.sqlite")
print("\ngap_capture: each 0.3 s burst of v0 + SDPA after an idle gap (cold = from idle)")
for s, e, txt in gap.ranges:
    if txt == "spin":
        continue
    ks = [k for k in gap.kernels if s <= k[0] <= e]
    first = ks[0][0]
    seg = gap.clock_in(first, e)
    drop = next(((seg[n][0] - first) / 1e6 for n in range(len(seg) - 20) if all(v < 2.56 for _, v in seg[n:n + 20])), None)
    sd = [((a - first) / 1e6, (b - a) / 1e6) for a, b, is_sdpa in ks if is_sdpa]
    before = [d for t, d in sd if t < drop - 2]
    after = statistics.median([d for t, d in sd if t > 150])
    print(f"  {txt:18s} clock at the first kernel {seg[0][1]:.3f} GHz; below 2.56 GHz from {drop:5.1f} ms; "
          f"SDPA before that {', '.join(f'{d:.2f}' for d in before) or '-'}; median SDPA after 150 ms {after:.3f}")

print("\nSDPA calls over 3 ms, and ms inside each call with the SMs mostly idle (SMs Active < 50%)")
for name, cap in (("ramp_capture", ramp), ("gap_capture", gap)):
    sd = [(a, b) for a, b, is_sdpa in cap.kernels if is_sdpa]
    normal = [cap.idle_ms(a, b) for a, b in sd if b - a < 2.2e6]
    slow = [((b - a) / 1e6, cap.idle_ms(a, b)) for a, b in sd if b - a > 3e6]
    print(f"  {name}: {len(slow)} of {len(sd)} calls over 3 ms: " + ", ".join(f"{d:.2f} ({i:.1f})" for d, i in slow))
    print(f"  {name}: calls under 2.2 ms have {statistics.median(normal):.1f} ms mostly idle (median), {max(normal):.1f} at most")


def series(con, name):
    mid = [m for m, n in con.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS") if n == name][0]
    rows = sorted(con.execute(f"SELECT timestamp, value FROM GPU_METRICS WHERE metricId={mid}"))
    return [t for t, _ in rows], [v for _, v in rows]


def window(ser, a, b):
    t, v = ser
    return v[bisect.bisect_left(t, a):bisect.bisect_right(t, b)]


print("\nGraphics work during SDPA calls: ms of 0.1-ms samples with vertex/tess/geometry or pixel warps in flight")
for name in ("ramp_capture", "gap_capture"):
    con = sqlite3.connect(f"reports/{name}.sqlite")
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    vert = series(con, "Vertex/Tess/Geometry Warps in Flight [Throughput %]")
    pix = series(con, "Pixel Warps in Flight [Throughput %]")
    sms = series(con, "SMs Active [Throughput %]")
    sd = sorted((s, e) for s, e, d in con.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL")
                if "pytorch_flash" in names[d])
    dur = [(e - s) / 1e6 for s, e in sd]
    gfx = [0.1 * sum(1 for x, y in zip(window(vert, s, e), window(pix, s, e)) if x > 0 or y > 0) for s, e in sd]
    hit = [d for d, g in zip(dur, gfx) if g > 0]
    clean = [d for d, g in zip(dur, gfx) if g == 0]
    stall = [v for (s, e), d in zip(sd, dur) if d > 3 for v in window(sms, s, e) if v < 50]
    print(f"  {name}: {len(hit)} of {len(sd)} calls overlap graphics work and take {min(hit):.2f} to {max(hit):.2f} ms; "
          f"the other {len(clean)} take at most {max(clean):.2f} ms; correlation of duration with graphics ms "
          f"{statistics.correlation(dur, gfx):.2f}; SMs Active in the stalls of the calls over 3 ms: "
          f"mean {statistics.mean(stall):.0f}%")

print("\nOnce boosted, how long the clock stays up after the last kernel of a range (s or ms after it)")
for name in ("wake_capture", "gap_capture"):
    cap = Capture(f"reports/{name}.sqlite")
    for s, e, txt in cap.ranges:
        last = max(b for a, b, _ in cap.kernels if s <= a <= e)
        nxt = min((a for a, b, _ in cap.kernels if a > last), default=last + 30e9)
        seg = cap.clock_in(last, nxt)
        up = next((t for t, v in seg if v >= 2.69), None)
        down = next((t for t, v in seg if up and t > up and v <= 2.53), None)
        leaves = next((t for t, v in seg if down and t > down and v < 2.4), None)
        below = next((t for t, v in seg if leaves and t > leaves and v < 0.5), None)
        idle = next((t for t, v in seg if below and t >= below and v <= 0.215), None)
        if up and down:
            f = lambda t: f"{(t - last) / 1e9:4.1f} s" if t else " n/a  "
            print(f"  {name:12s} {txt:24s} at least 2.69 GHz from {(up - last) / 1e6:3.0f} ms, back to 2.52 GHz at "
                  f"{(down - last) / 1e6:5.0f} ms, leaves 2.52 GHz at {f(leaves)}, below 0.5 GHz at {f(below)}, "
                  f"at the 210 MHz idle clock at {f(idle)}")
        elif nxt - last < 1e9:
            top = max((v for t, v in seg), default=float("nan"))
            print(f"  {name:12s} gap before the next range: {(nxt - last) / 1e6:4.0f} ms; at least 2.69 GHz from "
                  f"{(up - last) / 1e6 if up else float('nan'):3.0f} ms; highest clock in the gap {top:.3f} GHz")
