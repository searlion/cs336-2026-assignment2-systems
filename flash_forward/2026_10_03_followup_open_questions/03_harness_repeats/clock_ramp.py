# clock_ramp.py
# Item 6 and the harness's first row: how long the GPU stays at its starting clock after the
# 20-second pause before each harness run. Reads the 200 ms nvidia-smi samples taken during the
# 60-run batch (runs/nvidia_smi.csv) and the start time of each run (runs/timeline.log), and
# prints, per run, the SM clock from the first busy sample (clock above 1 GHz) onwards, and how
# long it stayed at 2,520 MHz before rising.
#   python clock_ramp.py > clock_ramp.txt
import csv
import datetime
import pathlib
import re
import statistics

HERE = pathlib.Path(__file__).resolve().parent
FMT = "%Y/%m/%d %H:%M:%S.%f"
smi = []
for r in csv.reader(open(HERE / "runs" / "nvidia_smi.csv")):
    if r[0].startswith("timestamp"):
        continue
    smi.append((datetime.datetime.strptime(r[0].strip(), FMT), int(r[1].split()[0])))
starts = [(datetime.datetime.strptime(m[1], FMT), m[2], m[3])
          for m in (re.match(r"(\S+ \S+) start (\S+) (\d+)", line) for line in open(HERE / "runs" / "timeline.log")) if m]

held = []
print("run                 SM clock (MHz) every 0.2 s from the first busy sample after the run starts")
for t0, version, run in starts:
    seq = [c for t, c in smi if 0 <= (t - t0).total_seconds() <= 6.0]
    busy = next((i for i, c in enumerate(seq) if c > 1000), None)
    if busy is None:
        continue
    seq = seq[busy:]
    n_start = next((i for i, c in enumerate(seq) if c != seq[0]), len(seq))
    if seq[0] == 2520:
        held.append(n_start)
    print(f"{version:15s} {run:>2s}  " + " ".join(str(c) for c in seq[:10]))
print(f"\nruns that start at 2,520 MHz: {len(held)} of {len(starts)}")
print(f"samples at 2,520 MHz before the clock rises: median {statistics.median(held)}, "
      f"min {min(held)}, max {max(held)} (x 0.2 s)")
