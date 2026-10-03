# make_variants.py
# Item 6: copies of kernels/v2_autotune that differ only in the @triton.autotune decorator,
# one per remedy the article suggests for autotuner noise.
#   python make_variants.py   -> writes kernels/<variant>/flashattention_autograd_function_triton.py
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SRC = (HERE.parent.parent / "2026_10_03_followup" / "kernels" / "v2_autotune" /
       "flashattention_autograd_function_triton.py").read_text()
DECORATOR = '@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N_QUERIES", "N_KEYS", "D", "is_causal"])\n'
CONFIG_LIST = "    for bq, bk, nw, ns in [(64, 32, 4, 3), (64, 32, 4, 4), (64, 32, 4, 2), (64, 16, 4, 5)]\n"
assert SRC.count(DECORATOR) == 1 and SRC.count(CONFIG_LIST) == 1

VARIANTS = {
    # 5x longer timing per candidate (the lambda matches how Triton 3.6's Autotuner._bench calls it).
    "v2_long_bench": [(DECORATOR,
        '@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N_QUERIES", "N_KEYS", "D", "is_causal"],\n'
        '                 do_bench=lambda fn, quantiles: triton.testing.do_bench(\n'
        '                     fn, warmup=100, rep=500, quantiles=quantiles))\n')],
    # Short and distinct: drop the two near-duplicates of (64, 32, 4, 3) that only change the stages.
    "v2_short_list": [(CONFIG_LIST,
        "    for bq, bk, nw, ns in [(64, 32, 4, 3), (64, 16, 4, 5)]\n")],
    # Timings written to the Triton cache; later processes reuse the first process's choice.
    "v2_cache_results": [(DECORATOR,
        '@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N_QUERIES", "N_KEYS", "D", "is_causal"],\n'
        '                 cache_results=True)\n')],
}

for name, edits in VARIANTS.items():
    src = SRC
    for old, new in edits:
        src = src.replace(old, new)
    d = HERE / "kernels" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "flashattention_autograd_function_triton.py").write_text(src)
    print("wrote", d)
