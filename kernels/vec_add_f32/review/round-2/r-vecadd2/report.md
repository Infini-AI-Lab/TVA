# Review of vec_add_f32, round 2, by r-vecadd2

## Verdict

accept
I read the claim at gate `401e1fa92db421d7` in full, as if for the first time: the source, the
kernel's whole Lean, the model map, the write-up, and the library pages and definitions the claim
reaches. The model is the source line by line by SEMANTICS.md: `program_id` on axis 0, the
constexpr `BLOCK_SIZE` built in `int32` for `pid * BLOCK_SIZE`, the scalar-tensor `+` with
`tl.arange`, the signed `int32` comparison with `n_elements` (also at `n = 1`, where the constant
becomes an `int32` tensor first), masked loads without `other`, the IEEE `tl.add` (no product feeds
it), and the masked store with no conversion. Every name resolves to the library's operation or to
core `Nat` arithmetic. The launch is the host's `cdiv` grid with the one block shape `[BLOCK_SIZE]`
and the defaults. The conditions cover the script's use (float32, `n` up to `2^27`, a fresh
`output`, one stream, no options, knobs or hooks) and every `n < 2^31`, with each narrowing stated
and justified; they are satisfiable. `Post` is exact up to NaN payload and keeps every other byte,
so it rejects plausible wrong kernels at every size. The one assumption is `Compiler.Faithful` in
the library's form. I have no blocking finding.

## Findings

None.

## Accepted

All items.

## Checked

- The model against the source, lines 39–54, with SEMANTICS.md ("Index arithmetic", "Constants", "Operators and promotion", "Scalar arguments", `tl.load`, `tl.store`, "Float `+` and `-`"): `pid * BLOCK_SIZE` in wrapping `int32` with the constant `IntV.ofNat BLOCK_SIZE` (equal to `IntV.ofInt`, checked in work/Checks.lean), `block_start + tl.arange(0, BLOCK_SIZE)` as the scalar-tensor `+`, `offsets < n_elements` as the signed `int32` `ltS` (at `n = 1` the constant `1` becomes an `int32` tensor first, the same comparison), the masks and the absence of `other`, `x + y` as IEEE `tl.add` (both operands are loads), and the store of `float32` values to a `*fp32` pointer.
- Every name in `lean/VecAdd.lean`: one `namespace`, `open TVA`, and `open Block` only in the proof section; no notation, instance, `export`, `set_option` or binder named like a library namespace. work/Names.lean and work/Instances.lean print `add_kernel`, `model`, `launch`, `Inst`, `Inst.ok`, `Pre`, `Post`, `Spec` and `Assumptions` with full names and explicit instances: every operator resolves to a `TVA.*` instance or to core `Nat` arithmetic; `add_kernel._proof_1` is the `Fin 3` literal's proof Lean generates.
- The launch: `Launch.grid1 ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) [[BLOCK_SIZE]] (fusion := true)` is the host's `triton.cdiv(n_elements, meta['BLOCK_SIZE'])` 1-D grid (lines 70, 75); `[BLOCK_SIZE]` is the shape of every block value of the source (`tl.arange`, `offsets`, `mask`, the pointer tensors, `x`, `y`, `output`); `debug`, `num_ctas` and `launch_pdl` keep their defaults, and `enable_fp_fusion` is Triton's default and changes nothing here.
- The model map: all 12 entries, the drops of lines 23, 24 (imports of Triton's own modules) and 29 (a bare `@triton.jit`), and the comment-only lines in the ranges. The function classes: `add_kernel` is the only function the launch runs; `add`, `benchmark` and the four lambdas run only on the host. The module-level lines: the kernel reads only `triton` and `tl`, bound at lines 23–24 (kernel lines); the host lines 21, 26, 84–93 and 135 rebind neither, and the module's `x` and `y` (lines 86–87) are shadowed by the kernel's own assignments before its read at line 52.
- The supported subset: a top-level `def` under a bare `@triton.jit`; parameters three pointers, one unannotated integer, one `tl.constexpr` with an `int` argument, none with a default; statements `name = e` and one call; expressions from the list; only Triton's modules read at module level.
- The host against `Compiler.Faithful`'s scope: one stream, a fresh `output` per call, no launch option but the constexpr, no autotune, heuristics, hook, knob, `CompiledKernel` or descriptor; `do_bench` and `perf_report` run `add` sequentially on the same stream.
- The conditions: `block_pow2`/`block_le` are Triton's block limits; `n_lt` is the `int32` class of `n_elements` and gives `Block.Fits` for every program (grid and last offset evaluated at `n = 2^31 - 1` with `BLOCK_SIZE = 1` and `2^20`, and at the host's sizes, work/Checks.lean); the length conditions and the disjointness of `output`'s first `n` elements from those of `x` and `y` are needed (an overlap is a conflict within a program or a race) and cover the host, which allocates `output` fresh; `x` and `y` may overlap; `n = 0` and `n = 1` are covered; `Pre` admits every `float32` value. `satisfiable` uses the source's own test instance.
- The specification: `Post` gives each `output[j]`, `j < n`, as the IEEE round-to-nearest-even sum of `x[j]` and `y[j]` up to NaN payload, with no tolerance, so it rejects a lower-precision rounding, a dropped term or a wrong index at every covered size; its second conjunct keeps every byte outside `output`'s first `n` elements, the inputs and `output`'s tail included.
- The assumption `faithful`: `∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`, the form docs/rules/compiler-faithful.md gives; the kernel uses no approximate operation, so it needs no fact about `P`.
- The write-up against the Lean: the Specification, Conditions (table, Pre, scope, subset), Assumptions, Proof outline (`meets_independent` with `hdebug`, `hvalid`, `hsafe`, `hdisj`, `hRW`, `hpost`), `consistent`, `satisfiable` and Narrowings match the definitions; the Narrowings' account of `n ≥ 2^31` (an `int64` `n_elements`, and the `int32` wrap of `pid * BLOCK_SIZE` beyond `2^31`) is right.
- `#print axioms` of `main`, `consistent` and `satisfiable`: only `propext`, `Classical.choice` and `Quot.sound` (work/Names.lean).

## Not checked

- The proofs of the library lemmas the proof uses (`meets_independent`, `safe_load_block`, `safe_store_block`, `fits_of_grid_pow2`, …): they are the library's, and Lean checks the kernel's proof against their statements.
- Whether the script is run with Triton's environment variables and knobs at their defaults and on an H200: the script sets none and names none, but the environment it runs in is outside what its text shows; the claim covers only such runs, as its scope says.
- Whether the kernel compiles at every covered `BLOCK_SIZE` (out of every claim's scope).
