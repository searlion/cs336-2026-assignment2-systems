#!/usr/bin/env bash
# mma_types.sh
# Which input -> accumulator types does the tensor-core mma instruction accept on sm_89?
# Assembles one m16n8k16 mma.sync per type combination with the ptxas that ships with Triton
# and prints whether ptxas accepts it. bf16 inputs only come with an fp32 accumulator, so there is
# no bf16 counterpart to the fp16 -> fp16 path (and Nsight Compute has no counter for one).
#   bash mma_types.sh > mma_types.txt
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../../.venv/bin/python"
PTXAS="$("$PY" -c "import os, triton; print(os.path.join(os.path.dirname(triton.__file__), 'backends/nvidia/bin/ptxas'))")"
TMP="$(mktemp -d)"
"$PTXAS" --version | tail -1
# d.a.b.c types; an f32 accumulator takes 4 registers per thread, an f16/bf16 one 2 (packed pairs).
for types in f32.f16.f16.f32 f16.f16.f16.f16 f32.bf16.bf16.f32 bf16.bf16.bf16.bf16 f16.bf16.bf16.f16; do
  if [[ $types == f32.* ]]; then regs=".f32 d<4>, c<4>"; D="{d0,d1,d2,d3}"; C="{c0,c1,c2,c3}"
  else regs=".b32 d<2>, c<2>"; D="{d0,d1}"; C="{c0,c1}"; fi
  cat > "$TMP/mma.ptx" <<EOF
.version 8.0
.target sm_89
.address_size 64
.visible .entry k() {
  .reg .b32 a<4>, b<2>;
  .reg $regs;
  mma.sync.aligned.m16n8k16.row.col.$types $D, {a0,a1,a2,a3}, {b0,b1}, $C;
  ret;
}
EOF
  if (cd "$TMP" && "$PTXAS" -arch=sm_89 mma.ptx -o mma.cubin 2> err.txt); then
    printf '%-22s accepted\n' "$types"
  else
    printf '%-22s rejected: %s\n' "$types" "$(head -1 "$TMP/err.txt")"
  fi
done
rm -rf "$TMP"
