# Review of softmax_f32, round 1, by r-softmax

## Verdict

accept-partly
The model is tutorial 02's `softmax_kernel` line by line (every operation printed with full names
is `TVA.*`, the masked load's `other` is the `float32` `−∞`, the reductions and the subtraction are
the IEEE forms since no product feeds them, `/` is `tl.div`), the map's drops are sound, the kernel
and host lines are classed right, the conditions cover the test and every benchmark shape with
their narrowings stated, and the six `[rule]` fields are the library's facts for the model's
operations. The specification is too weak at the covered sizes: its tolerance grows with
`BLOCK_SIZE` for every output separately and has a fixed absolute part, so at `BLOCK_SIZE = 2^14`
the same kernel with its output rounded to `float16` meets `Post` on every covered instance, and
from `2^11` on the correctly rounded `float16` softmax does. Only the row sum's error needs to grow
with `BLOCK_SIZE`, and it is one factor common to a row; I ask for that structure (or an argument).

## Findings

- **SPEC_TOO_WEAK** `item:spec` — Post's tolerance does not tell an fp32 softmax from a half-precision one: at BLOCK_SIZE = 2^14 this kernel with its output rounded to float16 meets Post on every covered instance
- **DOC** `doc:kernels/softmax_f32/README.md` — The benchmark's first launch per call runs on a fresh stream (lines 217-218) not ordered after the torch.randn writing x on the previous stream: say that scope point 1 is assumed there
- **DOC** `doc:kernels/softmax_f32/README.md` — README 'Covered in particular' says every shape with n_cols ≤ 2^14 and n_rows below 2^31 − num_programs; in_off, out_off also bound n_rows (≤ 131073 rows of stride 16384)

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

- Every Lean file of the kernel read in full (`Softmax.lean`, `Model.lean`, `Err.lean`, `Proof.lean`): one `open TVA TVA.Math.Float`, one `set_option linter.unusedVariables false in`, no instance, notation or look-alike; `#print` with `pp.fullNames` of `softmax_kernel`, `model`, `launch`, `Post` and `Assumptions` shows only `TVA.*` operations and the kernel's own definitions (work/Check.lean); `#print axioms` of `main`, `consistent`, `satisfiable`: `propext`, `Classical.choice`, `Quot.sound`.
- The model against lines 84–109 by SEMANTICS.md: `program_id`/`num_programs` axis 0; `tl.range` as `tl.forRange` with no carried name and `RangeOk` from `rows_grid`, `grid_lt`; the `int32` products under `in_off`/`out_off`; `tl.arange`, pointer `+`, the signed `ltS` mask; `other=-float('inf')` as `litF32` of `0xFFF0000000000000` (`−∞`, "Python's builtins", "Constants"); `tl.max(axis=0)` as `tl.maxAll true` with no widening for `float32`; `tl.sub`/`tl.sum` IEEE (no `*`, `tl.dot` or `tl.rand` feeds them); `tl.exp`, `tl.div` with the broadcast scalar; the masked store with no conversion; the integer arguments equal to `1` (each use makes the constant an `int32`, the same model).
- Each map entry and drop (imports 26–27, decorator 84, the `num_stages` keyword under "Loops"), and `kernel.json`'s classes: only `softmax_kernel` runs on the device or at compile time; `triton` and `tl` bound only by lines 26–27; no host line binds a name the kernel reads (`driver`, `DEVICE`, `properties`, …, `x`, `benchmark`).
- The launch: grid `(num_programs, 1, 1)`, the one block shape `[BLOCK_SIZE]`, `fusion = true`, `debug`, `num_ctas`, `pdl` at their defaults; part (A) of `Compiler.Faithful`'s scope point by point for the `CompiledKernel` of `warmup` (options `num_warps`, `num_stages` only; the launch passes the warm-up's own arguments; no descriptor; `do_bench`, not `do_bench_cudagraph`).
- The conditions against the host: the test `1823 × 781` and every benchmark `4096 × N`, `N = 256 … 12672` (`BLOCK_SIZE ≤ 2^14`, `4095·12672 < 2^31`, fresh contiguous `y`); `grid_pos` against `min(NUM_SM·occupancy, n_rows)`; the input narrowings (finite `≤ 2^126` or `−∞`, a finite element per row) against `torch.randn`; satisfiability (`satisfiable`'s test instance).
- `Post` at the largest covered sizes, with exact arithmetic (work/tolerance.py) and Lean-checked inequalities (work/Check.lean); the frame (every byte outside the output elements unchanged).
- Each `[rule]` field against rules/num.md's table (`ExpF32`, `ExpF32NegInf` for `tl.exp` on `float32`; `DivFull` for `float32` `/`; `MathTotal .exp .binary32`, `DivTotal .binary32`) and `faithful` in the form `∀ d, d.ok → Faithful P (model d) (launch d) (K d)`; `Assumptions` is a structure with exactly `(K : Claim.Behaviour Inst) (P : Platform)`.
- The README against the Lean (specification, conditions table, assumptions, proof outline, narrowings).

## Not checked

- The proofs' steps beyond their statements (Lean and the gate's replay check them).
- Part (B) of `Compiler.Faithful`'s scope (assumed), except the benchmark's stream use, which the source shows.
