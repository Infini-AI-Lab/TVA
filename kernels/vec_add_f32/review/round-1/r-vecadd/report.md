# Review of vec_add_f32, round 1, by r-vecadd

## Verdict

accept
The model is the source under the transcription rules, line by line, and the launch is the
host's (grid `cdiv(n, BLOCK_SIZE)` on axis 0, one block shape `[BLOCK_SIZE]`, fusion `true`,
`num_ctas = 1`, `debug = false`). The conditions cover the host's real use (float32, contiguous,
fresh output, `BLOCK_SIZE = 1024`, sizes 98432 and `2^12 … 2^27`), every value including NaN and
infinities, and in-place use; each narrowing is stated with its reason. `Post` is exact (the IEEE
`float32` sum up to NaN payload, no tolerance) with a full frame over all memory outside
`output`'s first `n` elements, so it rejects lower-precision, dropped-term and out-of-bounds
kernels. The one assumption is `Compiler.Faithful` cited exactly as the library states it. My only
finding is a non-blocking wording error in the README about why `n = 2^31` is excluded.

## Findings

- **DOC** `doc:kernels/vec_add_f32/README.md` — Narrowings: "for n ≥ 2^31 the int32 offsets wrap" is wrong at n = 2^31 (largest offset 2^31 − 1); that size is excluded because n_elements is then int64

## Accepted

All items.

## Checked

- `model`: each line of `add_kernel` (source 39–54) against TRANSCRIPTION.md §1–§4: `program_id` axis 0 (`int32`); `pid * BLOCK_SIZE` as an `int32` product with the constexpr as an `int32` constant; `block_start + tl.arange(0, BLOCK_SIZE)` with the library's scalar–tensor `+`; `offsets < n_elements` as `ltS` (signed `int32`, `n_elements : I32` since `n < 2^31`); both loads with the source's mask and no `other`; `x + y` as IEEE `tl.add` (operands are loads, no product); the store with the source's mask and no conversion (already `float32`).
- `model`: printed with `pp.fullNames` (work/Print.lean): every operation is `TVA.*`; `model d` passes `d.x.ptr 0`, `d.y.ptr 0`, `d.output.ptr 0`, `IntV.ofNat d.n`, `d.BLOCK_SIZE`.
- `model`: the launch against rules §9 and host lines 70 and 75: `Launch.grid1 ((n + B − 1) / B)` is `triton.cdiv`; grid y, z are 1; `blocks = [[BLOCK_SIZE]]` is the shape of every block value (`arange`, `offsets`, `mask`, pointer tensors, `x`, `y`, `output`); fusion `true` (default), `numCtas = 1`, `debug = false` by `Launch` defaults; no autotune.
- `map`: all 10 entries against the source; the only drop (line 29, `decl`) is a decorator; the remaining lines of 29–54 are comments.
- `conditions`: work/Cover.lean proves `Inst.ok` for the host's instance (fresh disjoint buffers, `BLOCK_SIZE = 1024`) at `n = 98432`, `2^27` and `2^31 − 1024`, and in place (`output = x`); `Pre` holds with every byte present; `Inst.ok` fails at `2^31 − 1023` (the stated L8 narrowing).
- `conditions`: each narrowing against the README (float32 only, `n + BLOCK_SIZE ≤ 2^31`, contiguous buffers, `y` length, aliasing); `x`/`y` may overlap freely; `n = 0` covered; `Pre` allows every `float32` value.
- `spec`: `Post` determines each output element exactly (`FloatV.add` up to `NaNEq`) and keeps every byte outside `output.elems {j | j < n}`; work/Spec.lean evaluates `FloatV.add` on `float32` (round to nearest even on ties, overflow to `+∞`, `∞ + −∞` NaN, `−0 + −0 = −0`, subnormals kept), and `1 + 2^-10` shows a bfloat16-precision kernel is rejected. No tolerance, so nothing to evaluate at the largest size.
- `assumption:faithful`: the statement is `∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`, the citation form of docs/rules/compiler-faithful.md; the rule page's four checks (model, launch, coverage, `consistent` via `exists_total_faithful`) hold. No approximate operation, so no `Num` rule is needed.
- Axioms: `#print axioms` on `main` and `consistent` gives only `propext`, `Classical.choice`, `Quot.sound`.
- The write-up: Specification, Conditions, Assumptions and Proof sections against the Lean (`meets_independent`, `launch_valid`, `safe_program`, the disjointness and `hpost` steps).

## Not checked

- The proofs themselves beyond the statement and axioms (Lean checks them).
- The library (`tl.*` meanings, `Compiler.Faithful`, Triton's argument specialisation of `n_elements`), which kernel review trusts.
