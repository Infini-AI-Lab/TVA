# Review of vec_add_f32, round 2, by r-vecadd2

## Verdict

accept
Judged in full at gate `990243026f1289fb`. The model is the source line by line by SEMANTICS.md
(`int32` index arithmetic with the constexpr built in `int32`, the signed `int32` mask, the same
mask on both loads and the store, the IEEE `tl.add` that no product feeds, no conversion before the
store), and the launch is the host's (`cdiv(n, BLOCK_SIZE)` programs, one block shape, every
default). `Inst.ok` covers every launch the script makes, and its narrowings (`float32`, `n < 2^31`,
contiguous buffers, `y` long enough, no overlap of `output` with an input) are stated with reasons.
`Post` is exact (the IEEE sum up to NaN payload) with a frame over every other byte, and rejects the
plausible wrong kernels. The one assumption is `Compiler.Faithful` in the library's form, and the
kernel and its launches stay within part (A) of its scope. I have no finding.

## Findings

None.

## Accepted

All items.

## Checked

- Every Lean file of the kernel (`lean/VecAdd.lean`) in full: its one `open TVA`, `open Block` in the
  proof section after the claim, no `notation`, `instance`, `attribute`, `export` or `set_option`, and
  the local binders (`ctx`, `pid`, `block_start`, `offsets`, `mask`, `x`, `y`, `output`), none
  shadowing a library name. `work/Names2.lean` prints the model, launch, conditions, `Post` and
  `Assumptions` with full names and explicit instances: every operation is `TVA.*` or core `Nat`
  arithmetic in the grid formula; `Assumptions` is a structure with exactly `(K)` and `(P)`;
  `#print axioms` of `main`, `consistent`, `satisfiable` is `propext`, `Classical.choice`, `Quot.sound`.
- The model map, entry by entry, against SEMANTICS.md: the drops of lines 23, 24 (imports binding only
  `triton` and `tl`, which no other statement rebinds) and 29 (bare `@triton.jit`, `debug` unset);
  the signature (pointers `*fp32`, `n_elements` an `int32` for `n` in `[2, 2^31)` and `0`, the constant
  `1` at `n = 1`, which the comparison makes an `int32` tensor first); lines 39 to 54.
- The functions and module-level code: only `add_kernel` runs at compile time or on the device; `add`,
  `benchmark` and the lambdas are host code; lines 21, 26, 84 to 93 and 135 bind `torch`, `DEVICE`,
  `size`, `x`, `y`, `output_torch`, `output_triton`, `benchmark` and nothing the launch reads (the
  kernel assigns its own `x` and `y` before reading them); the module docstring binds only `__doc__`.
- The supported subset: a top-level `def` under a bare `@triton.jit`, no defaults, parameters of
  admitted kinds, only `name = e` and one call statement, admitted `tl.*` calls and operators.
- The launch: `Launch.grid1 (cdiv n B) [[B]] (fusion := true)` with the defaults `num_ctas = 1`,
  `pdl = false`, `debug = false`; `[BLOCK_SIZE]` is the only block shape (the scalars have shape `[]`).
- Part (A) of `Compiler.Faithful`'s scope, point by point against the source (4, 5, 6, 8, 9); the
  source is the pinned `tutorials/01-vector-add.py` (same sha256), whose host the rule page states
  makes no capture. No visible breach of part (B) in the source.
- The conditions: `work/Cover3.lean` proves `Inst.ok` for every `n < 2^31` with `BLOCK_SIZE = 1024`
  and fresh disjoint buffers, and for `add(x, x)`; satisfiability by the prover's test instance.
- The specification, with the library as it is now: `work/Spec2.lean` evaluates `FloatV.add` on
  `float32` patterns (exactness, ties to even, rounding up, subnormals, overflow, NaN, signed zeros),
  checks what `NaNEq` accepts and rejects, and proves that a write past `output`'s first `n` elements
  fails `Post`.
- The `[rule]` assumption `faithful`: `∀ d, d.ok → Faithful P (model d) (launch d) (K d)`, the
  library's form, with its docstring.
- The write-up against the Lean: Specification, Conditions (with part (A) and part (B)),
  Assumptions, Proof, Narrowings and modelling decisions.

## Not checked

- Triton's `triton.testing.do_bench` itself, which TVA does not pin: for point 9 I rely on the rule
  page's statement that no in-scope tutorial's host captures.
- Part (B) of the scope, the library's facts and the device, which are outside a kernel reviewer's
  scope.
