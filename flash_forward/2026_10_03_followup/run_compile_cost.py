# run_compile_cost.py
# Drive compile_cost.py, the unchanged harness and the unchanged test file in fresh
# processes with cold and warm Triton caches, and record wall-clock times.
#
#   python run_compile_cost.py <scratch dir for caches> results/compile_cost.json
import json
import os
import pathlib
import subprocess
import sys
import time

PY = sys.executable
SCRATCH = pathlib.Path(sys.argv[1])
OUT = sys.argv[2]
SHAPES = "4096:0,2048:0,1024:0,512:0,4096:1"
results = {"first_call": [], "harness": [], "tests": []}


def fresh_cache(name):
    d = SCRATCH / f"cache_{name}_{time.time_ns()}"
    d.mkdir(parents=True)
    return str(d)


def run(cmd, cache, extra_env=None):
    env = dict(os.environ, TRITON_CACHE_DIR=cache, **(extra_env or {}))
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-4000:])
        raise SystemExit(f"failed: {cmd}")
    return p.stdout, wall


# 1. First-call latency against the number of autotuning configs, cold then warm cache.
for k in (1, 2, 4, 8, 16):
    cache = fresh_cache(f"k{k}")
    for state in ("cold", "warm"):
        out, wall = run([PY, "compile_cost.py", "kernels/v1_tiles", "--configs", str(k), "--shapes", SHAPES], cache)
        rec = json.loads(out.strip().splitlines()[-1]) | {"cache": state, "process_wall_s": round(wall, 2)}
        results["first_call"].append(rec)
        print(k, state, [c["first_call_s"] for c in rec["calls"]], flush=True)

# 2. cache_results=True: the second process should not benchmark at all.
cache = fresh_cache("cache_results")
for state in ("cold", "warm"):
    out, wall = run([PY, "compile_cost.py", "kernels/v1_tiles", "--configs", "4", "--cache-results", "1",
                     "--shapes", SHAPES], cache)
    rec = json.loads(out.strip().splitlines()[-1]) | {"cache": state, "process_wall_s": round(wall, 2)}
    results["first_call"].append(rec)
    print("cache_results", state, [c["first_call_s"] for c in rec["calls"]], flush=True)

# 3. The unchanged harness and the unchanged test file, end to end.
for version in ("v1_tiles", "v2_autotune"):
    cache = fresh_cache(version)
    for state in ("cold", "warm"):
        _, wall = run([PY, "bench_flashattention.py"], cache, {"PYTHONPATH": f"kernels/{version}"})
        results["harness"].append(dict(version=version, cache=state, wall_s=round(wall, 2)))
        print("harness", version, state, round(wall, 2), flush=True)
    cache = fresh_cache(version + "_tests")
    for state in ("cold", "warm"):
        _, wall = run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_flashattention_triton.py"],
                      cache, {"PYTHONPATH": f"kernels/{version}"})
        results["tests"].append(dict(version=version, cache=state, wall_s=round(wall, 2)))
        print("tests", version, state, round(wall, 2), flush=True)

json.dump(results, open(OUT, "w"), indent=1)
