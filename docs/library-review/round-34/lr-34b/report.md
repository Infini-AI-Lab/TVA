# Library review, round 34, by lr-34b

## Verdicts

- **L149** ACCEPT — the exclusion is now stated on the source and covers every round-33 form; `evidence/thread-locality` (Parts U, F, L, W) re-ran on the H200 with output identical to the shipped `output.txt` (empty diff, `work/rerun_tl/`). `SumDepth` is unchanged.
- **L155** ACCEPT — the narrowed `AddRel` holds on the device. `evidence/fma-one-product` re-ran identical (`work/rerun_fma/`). My own probe (`work/fma_narrow.py`, `fma_narrow.txt`) checks every result exactly against the narrow relation: `a*b + c`, `c + a*b`, `a*b - c`, `c - a*b`, `-(a*b) + c`, `tl.where(m, a*b, 0) + c`, `(a*b).to(wider) + c`, `(a*b)*a + c` in float32, float64, float16 and bfloat16; a `tl.dot` result scaled and then added; inputs near cancellation, with subnormal products and results, with a product that overflows when rounded while its `fma` is finite, and with infinite and NaN addends. 0 of 638976 results fall outside it. Every contracted spelling is `fma.rn.<type>`. The `where` and `.to` forms are not contracted (`work/fma_ptx.txt`, SASS in `work/fma_sass.py`). `AddRel` is a definition of the model, not a field of `Platform`, so `Platform.sm90ref` is untouched. The lemmas reach the relation only through `AddRel.ieee`, `AddRel.false_iff` and `AddRel.wide`, so they stay consistent. The reach list is complete. A related defect outside this issue, the contraction of `/`, is filed below.
- **L156** ACCEPT — conclusion 2 names `Rules.Num.ExpF32Overflow`, `Platform.ref.mathAt_far`, `sm90Special` and `Platform.sm90ref.expF32Overflow`, which all exist with these meanings; `1453635/16384` is `0x42b17218` = 88.72283935546875.
- **L157** ACCEPT — `evidence/relaxed-atomics` matches what I measured. My `work/atomics_unused.py` (`atomics_unused.txt`) ran 126 cases with the result unused: add on i32, i64, f32, f16, bf16 and f64; max and min on i32, i64 and f32 (with zeros of both signs); and, or and xor; xchg on i32, f32 and i64; cas on i32 and i64. Each case ran with `relaxed`, `acquire`, `release` and `acq_rel` at gpu, `relaxed` at sys, and `relaxed` at cta, under a random lane mask whose disabled lanes point at sentinel cells (-0.0 and NaN payloads), 20 contended launches each. No cell and no sentinel was wrong. Every case compiles to `atom` instructions with the source's order and scope and no `red`. The wording of "one `atom`" is filed below.
- **L158** ACCEPT — `evidence/bool-args` re-ran identical. In my own runs (`work/bool_more.txt`), float32 values `0.5, 1.5, 2.0, 256.0` store the bytes `0, 1, 2, 0` (`arith.fptosi`), and int64 `2^32 + 1` stores `1` (`arith.trunci`). A masked-off lane with `other=0.25` loads `false`. Each matches "converted to `int8`" as SEMANTICS.md states it.
- **L159** ACCEPT — `work/L159Check.lean` applies `sum_bcast_mul_depth_err` as a prover would: from `tl.sumWith`'s pick, the `tl.mul` products and `SumDepth.depth_le`, it obtains the bound at `D = Nat.clog 2 n`, and `bcastTreeBound_eq` holds. Its axioms are `propext`, `Classical.choice` and `Quot.sound`.

## New issues

- **NEW** soundness — the backend contracts `/` into a following addition. SEMANTICS.md ("Float `+` and `-`") counts only `*`, `tl.dot` and `tl.rand` as products, and models every other addition as IEEE. On float32, `div.full.f32` becomes `MUFU.RCP` followed by a multiply, and with `enable_fp_fusion` ptxas fuses that multiply with a following `add.f32` into one `FFMA`. The same holds for float16 and bfloat16 `/`, which run in float32 (`work/fn_add_sass.txt`). On the device, `x / y - q`, where `q` is the same quotient stored by an earlier launch, is nonzero on 65536 of 65536 elements, and no result is a multiple of `ulp(q)/2`, so no `P.div` value with an IEEE `-` gives it (`work/div_fused.txt`). With fusion off, all 65536 are 0. `tl.div_rn` and float64 `/` give 0 (`div_fused2.txt`). Claims are affected: in tutorial 05's layer-norm body, `var + eps` compiles to `FFMA(rcp(N), sum, eps)` (`work/ln_div_sass.txt`), while `claims/layer_norm_fwd_f16` models it as the IEEE `+`. Fix: count a float32 `/` (and float16/bfloat16 `/`) as a product for the fusable forms, with evidence that the fused exact product rounds to the quotient's own value, so that the narrow `AddRel` covers it. Then re-check the 13 claims that use `tl.div`.
- **NEW** wording — `evidence/relaxed-atomics` and SEMANTICS.md ("Atomics") say every atomic compiles to "one `atom`". A float `atomic_max` or `atomic_min` compiles to two predicated integer atoms (`max.s32` and `min.u32`), of which at most one runs per lane; with a NaN operand neither runs, which `fBound` allows. Say "one atomic read-modify-write per enabled lane".

## Re-ran

- `evidence/thread-locality/run.py`: identical.
- `evidence/fma-one-product/run.py`: identical.
- `evidence/bool-args/run.py`: identical.
- My own probes, all in `work/`: `fma_narrow.py`, `fma_ptx.py`, `fma_sass.py`, `fn_add_sass.py`, `div_fused.py`, `div_fused2.py`, `ln_div_sass.py`, `atomics_unused.py`, `atom_ptx_bar.py`, `bool_more.py` and `L159Check.lean`. These ran on the H200 NVL (sm_90) with Triton 3.8.0 on 2026-11-08.

## Not checked

- The kernel proof files that use `AddRel` (their own packages) against the narrowed relation.
- The release check.
- Descriptor reductions.
- `MemSem.unused`'s memory-model argument, beyond the device runs.
