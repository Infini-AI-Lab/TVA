# Review of softmax_f32, round 1, by r-softmax2

## Verdict

accept-partly
The model matches the source line by line on every covered instance by SEMANTICS.md. That covers the integer arguments equal to `1` (compile-time constants that each use makes `int32`), the masked load with the `float32` `−∞`, the `float32` max reduction, the IEEE `sub` and `sum` (no product feeds them), the platform's `exp` and `div`, and dropping `num_stages`, which is safe because no trip reads what any trip writes. The map, `kernel.json`'s classing of the source's functions and module-level lines, and the launch (grid, the one block shape, the defaults, part (A) points 4, 5, 6, 8 and 9 for the `warmup` `CompiledKernel`) are right. The conditions cover the host's test and every benchmark shape, and each narrowing is stated. All six `[rule]` citations are the library's facts for the operations the model uses. I do not accept the specification. Both of its tolerances are much looser than the cited facts and the library's tree bound allow. At `BLOCK_SIZE = 2^14` (35 of the 98 benchmark shapes), `relTol` accepts outputs rounded to bfloat16, and from `2048` on it accepts outputs rounded to float16. `absTol = 2^-22` leaves outputs below about `2.4·10^-7` unbounded, and at the test's shape it is 24 times the test's own `atol`. One wording finding on the README.

## Findings

- **SPEC_TOO_WEAK** `decl:Kernels.Softmax.relTol` — relTol is about 4 times the library's sum bound: at BLOCK_SIZE = 2^14 the exact softmax rounded to bfloat16 meets Post, and rounded to float16 from 2048 on
- **SPEC_TOO_WEAK** `decl:Kernels.Softmax.absTol` — absTol = 2^-22 is the same for every row: any output with exact value below about 2.4e-7 may be 0, and at the test's shape it is 24 times the test's own atol
- **DOC** `doc:kernels/softmax_f32/README.md` — README, Conditions: 'every other shape with n_cols ≤ 2^14 and n_rows below 2^31 - num_programs' leaves out in_off/out_off, so it claims shapes the conditions exclude

## Accepted

- `item:model`
- `item:map`
- `item:conditions`
- `item:assumption:faithful`
- `item:assumption:exp_total`
- `item:assumption:exp_acc`
- `item:assumption:exp_ninf`
- `item:assumption:div_total`
- `item:assumption:div_full`

## Checked

- Every Lean file of the kernel read in full (`Softmax.lean`, `Model.lean`, `Err.lean`, `Proof.lean`). Every name of the model, conditions and specification printed with full names (`work/names.lean`). The model also printed without notation and with instances: every operation is `TVA.*` (`instHAddPtrIntV`, `instHAddPtrTensorIntV`, `IntV.instMul`, `Tensor.ltS`, `tl.*`, `bc`, `FloatV.litF32` of `FloatV.ofPat 0xFFF0000000000000`). The only auxiliary declaration is `NeZero (2 + 1)`. The kernel declares no instance, notation or binder that changes a library name.
- `#print axioms` of `main`, `consistent` and `satisfiable`: `propext`, `Classical.choice`, `Quot.sound`. `Assumptions` is a structure with exactly `(K : Claim.Behaviour Inst) (P : Platform)`.
- Model against source, lines 84–109, against SEMANTICS.md:
  - argument order and dtypes (`model` passes `y`, `x`, the two strides, `n_rows`, `n_cols`);
  - the value-`1` constants at lines 90, 92, 98 and 107 ("Constants", "Loops");
  - `tl.range` as `forRange` with `RangeOk`, and dropping `num_stages`, safe because no conflict within a program;
  - the wrapping `int32` products and the pointer `+`;
  - the signed mask;
  - `other=-float('inf')` as `litF32` of the double `−∞`;
  - `tl.max(axis=0)` as `maxAll true` with no widening;
  - IEEE `sub` and `sum`: no product feeds them, and `tl.exp`'s result is not a product;
  - `tl.exp` as `P.math .exp`;
  - `/` as `tl.div`;
  - the store with no conversion.
- Model map: all 19 entries and each drop (the imports, the decorator). `kernel.json`: the launch runs no function outside lines 84–109 (`tl.max` and `tl.sum` are Triton's own). The module-level names it reads, `triton` and `tl`, are bound only by lines 26–27. No host line binds them or `softmax_kernel`, and `benchmark`'s decorator binds nothing.
- Supported subset: decorator, parameters (no defaults; annotations only `tl.constexpr`), statements, builtins (`float`, unary `-` on a compile-time value), the `tl.range` keyword, no use after the loop.
- Launch: `grid1 num_programs [[BLOCK_SIZE]]`, the only block shape (the reductions give scalars). The defaults are fusion `true`, `debug` false, `num_ctas` 1, `pdl` false.
- Part (A) against the host code:
  - `warmup` gives only `num_stages` and `num_warps` as options, plus `grid`;
  - the launch passes the warm-up's own arguments;
  - no descriptors, no host descriptors;
  - `do_bench` is not `do_bench_cudagraph`, and nothing captures.
- Conditions:
  - each field against the host (`next_power_of_2`, `min(NUM_SM·occupancy, n_rows)` positive when `occupancy ≥ 1`, `empty_like` strides, a fresh `y`);
  - the `int32` bounds against `RangeOk` and the products;
  - the test and all 98 benchmark shapes are covered (largest `BLOCK_SIZE` 2^14, `4095·12672 < 2^26`);
  - the input narrowings (no `+∞` or NaN, `|x| ≤ 2^126`, a finite element per row) are stated and harmless for `torch.randn`;
  - `satisfiable`'s instance is the source's test.
- Specification:
  - `softmaxRef` with `e^(−∞) = 0`, the finite outputs, and the frame on every byte outside the output elements;
  - the tolerances evaluated at the host's sizes against bfloat16 and float16 rounding, against the library's tree bound, and against the test's `torch.allclose` (`work/tolerance.lean`, checked; the randn figures in Python);
  - the error proof traced (`exp_lane`, `sum_rel`, `final_bound`) to find where the slack comes from.
- `[rule]` citations against `docs/rules/num.md`'s table and the definitions in `TVA.Rules.Num`: `MathTotal .exp .binary32`, `ExpF32`, `ExpF32NegInf`, `DivTotal .binary32`, `DivFull`. `Faithful P (model d) (launch d) (K d)` under `∀ d, d.ok`.
- The README against the Lean: the specification, the conditions table, the assumptions and the proof outline match, apart from the DOC finding and the README's wording that `relTol` is the library's depth bound (part of the first finding).

## Not checked

- That the tighter tolerances the findings suggest can be proved in Lean. The arguments are by hand from `ExpF32`, `DivFull`, `sum_err_covers` and `final_bound`'s steps; only the numeric comparisons are Lean-checked.
- The library itself (the meanings of the `tl.*` operations, the device facts, `Compiler.Faithful`), which is out of scope: I had no doubt to report.
- The proofs beyond what the gate and Lean check. I read them to judge names and to locate the tolerance's slack, not to re-verify them.
