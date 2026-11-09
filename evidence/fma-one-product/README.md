# With fusion, an addition takes at most one operand unrounded (library issue L155)

Library evidence (not a kernel's), for library issue L155. Triton 3.8.0 (TVA's `.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-11-08. Script: [run.py](run.py); output:
[out.txt](out.txt) (about a minute).

**Question.** `AddRel fusion x y r` (`lean/TVA/Lang/Ops.lean`) is what a fusable float addition may
give. Before L155 it allowed, with `enable_fp_fusion`, the rounding of `p + q` for exact `p`, `q`
rounding to the operands, both replaced (now `AddRelWide`). Contraction makes `a * b + c` an
`fma(a, b, c)`, which takes one product unrounded and `c` as it is. L155 narrows the relation to one
replaced operand (`p = x.toRat` or `q = y.toRat`). That is sound if Triton 3.8.0's code never adds
two unrounded products in one addition. Does it?

**Source.** An addition reaches the device as a PTX `add`, `sub` or `fma`, or their packed
`f16x2`/`bf16x2` forms. Under `enable_fp_fusion` Triton lets LLVM fuse (`AllowFPOpFusion = Fast`,
`primitives/triton-3.8.0-llvm.cc:136-140`) and lets `ptxas` fuse (`--fmad=false` only without it,
`primitives/triton-3.8.0-nvidia-compiler.py:532`). The fused instruction NVPTX selects is
`fma.rn.<type> d, a, b, c`, the node `fma(a, b, c)`: one product `a × b` and an addend `c`, rounded
once (`primitives/llvm-NVPTXInstrInfo.td:1278-1315`: `FMA_F16`, `FMA_F16x2`, `FMA_BF16`,
`FMA_BF16x2`, `FMA_F32`, `FMA_F64`, and `FMA_F32x2`, which needs a later GPU; the packed forms take
one product per lane). Triton's combine pass rewrites `tl.dot(a, b) + c` into a `tl.dot` accumulating
into `c` (`dot_add`, which `tl.dot_add` states with the dot's own fact), not into an addition of two
products.

**Method** (`run.py`). For `float32`, `float64`, `float16` and `bfloat16`, five forms of an addition
of products, with Triton's default `enable_fp_fusion=True`: `a*b + c*d`, `a*b - c*d`, a `tl.sum` of
a row of two products (`tl.join`), `r = a*b` then `r += c*d` in a loop, and `(a*b + c*d) + e*f`.
`c = -a` and `d = b`, each moved by up to 3 units in the last place, so that the products nearly
cancel and each product's rounding shows in the result. 8 launches of 1024 elements per case. Each
device result is compared exactly (Python `Fraction`s, rounding to nearest even in the format) with
the values the narrow relation allows (the IEEE sum, and each addition with at most one operand
unrounded) and with the rounding of the exact products together, which only the wide relation
allows. The PTX and SASS instructions are listed.

**Result** (`out.txt`).
- No result outside the narrow relation: 0 of 163840.
- On 156264 inputs the rounding of both exact products differed from every value the narrow
  relation allows; the device gave it on none.
- The instructions: `fma.rn.f32` and `mul.f32` (SASS `FFMA`, `FMUL`); `fma.rn.f64`, `mul.f64`
  (`DFMA`, `DMUL`); `fma.rn.f16x2`, `mul.f16x2` (`HFMA2`, `HMUL2`); `fma.rn.bf16x2`, `mul.bf16x2`
  (`HFMA2.BF16_V2`, `HMUL2.BF16_V2`). Every addition of two products is one `fma` taking one product
  and the other product rounded by a `mul`.

**Conclusions.** On sm_90 an addition that Triton contracts takes one operand unrounded (a product)
and the other as it is. `AddRel` states that (library issue L155), and implies the earlier relation
(`AddRel.wide`), so every lemma about the wide relation still holds of it.
