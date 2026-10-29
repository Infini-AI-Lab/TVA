# Review of vec_add_f32, round 2, by r-vecadd

## Verdict

accept
Judged in full at gate `a2e8d45aaf58a6d2`, as if for the first time. The model is the source line
by line by SEMANTICS.md: the `int32` program id, the constexpr `BLOCK_SIZE` built in `int32` for
`pid * BLOCK_SIZE`, the scalar–block `+` of the offsets, the signed `int32` comparison with
`n_elements` (also at `n = 1`, where the constant becomes an `int32` tensor first), the masked loads
without `other`, the IEEE `tl.add` (no product feeds it), and the masked store with no conversion.
Every name of the model is the library's or core Lean's, and the launch is the host's grid
`triton.cdiv(n, BLOCK_SIZE)` with every default. `Inst.ok` covers every launch the script makes, and
its narrowings (`float32`, `n < 2^31`, contiguous buffers, no overlap of the output with an input)
are stated with their reasons. `Post` is the exact IEEE `float32` sum up to NaN payload, with a frame
on every other byte, and it rejects plausible wrong kernels. The one assumption is
`Compiler.Faithful` in the library's form, and the launches stay within part (A) of its scope. The
three wording remarks of the last round are fixed. My two findings are library reports, which do
not block; I have no blocking finding.

## Findings

- **LIBRARY_REPORT** `decl:TVA.I8` — `TVA.I8`'s docstring describes `tl.int32` (it belongs to `I32`); this kernel uses `I32` and is not affected
- **LIBRARY_REPORT** `lib:docs/rules/compiler-faithful.md` — scope 11's reuse clause omits the runtime arguments' specialisation, and its example "a repeated launch of the same instance" reads narrower than intended

## Accepted

All items.

## Checked

- What changed since gate `990243026f1289fb`: the two gate snapshots hold the same kernel
  constants with the same hashes (model, launch, `Inst`, `Inst.ok`, `Pre`, `Post`, `Spec`, the
  field's statement). What changed is the docstring of `Assumptions.faithful`, the README, and the
  library files of L116 (SEMANTICS.md, exclusions.md, compiler-faithful.md, `TVA.Claim`,
  `TVA.Rules.Compiler`). I read all of them as they are now; nothing in L116 reaches this kernel
  beyond wording (no `@triton.autotune`; the source is byte for byte the pinned tutorial).
- The three remarks are fixed: the README's host wording (`-O`, `-OO`, `PYTHONOPTIMIZE`, the
  floating-point environment), the `faithful` docstring ("one of the outcomes of the model's
  launch"), and the README's account of the `int32` block added to a `*fp32` pointer. The two
  passages aligned with L116 match the rule page's "Which launch".
- Every map entry against the source (lines 23–24, 29, 30–36, 39, 44, 45, 47, 50, 51, 52, 54) by
  SEMANTICS.md: Values ("Index arithmetic", pointers), Constants (an operator's operand; "everywhere
  else" for the comparison at `n = 1`), Operators and promotion, Scalar arguments, `tl.load`,
  `tl.store`, Float `+` and `-`. The drops of lines 23, 24 and 29 (imports and a bare decorator: no
  step, `debug` unset). The uncovered lines of the ranges are comments.
- The ranges and module-level names: `add_kernel` is the only function the launch runs; `add`,
  `benchmark` and the lambdas run on the host only; no function is unused. The launch reads only
  `tl` and `triton` (lines 23–24, mapped), each bound only by its import; the host lines bind
  nothing the kernel reads (its `x`, `y`, `output` are assigned before they are read). The kernel
  stays in the supported subset.
- The whole kernel Lean file, read in full: `open TVA`, `open Block` inside the proof section, no
  instance, notation, attribute, `export` or `set_option`. `work/Names4.lean` prints the claim with
  full names and explicit instances (every operation `TVA.*`, `tl.load` with `other = none`, the
  grid formula on core `Nat` instances), `Assumptions` with exactly `(K)` and `(P)`, and the axioms of
  `main`, `consistent` and `satisfiable` (`propext`, `Classical.choice`, `Quot.sound`).
- The launch: a 1-D grid `(n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `program_id` axis 0, the one block
  shape `[BLOCK_SIZE]`, `debug = false`, `enable_fp_fusion = true`, `num_ctas = 1`,
  `launch_pdl = False`.
- Part (A) of `Compiler.Faithful`'s scope against the script, points 4, 5, 6, 8 and 9: no option,
  `warmup`, `triton.compile`, autotune or heuristics; no descriptor; no `CompiledKernel`; no host
  `TensorDescriptor`; no CUDA graph capture (`triton.testing.do_bench`, not `do_bench_cudagraph`).
  The source visibly breaks no point of part (B).
- The conditions and the specification, by re-running `work/Cover3.lean`, `work/Checks3.lean` and
  `work/AddCheck3.lean` under the library as it is now: `Inst.ok` holds at the script's sizes, at
  `n = 0`, `1` and `2^31 - 1`, and for `add(x, x)`, and fails at `n = 2^31` and in place; the grid
  is `triton.cdiv` there; `FloatV.add` equals two `float32` references on all 600 pairs (ties to
  even, overflow, infinities, signed zeros, subnormals, NaN). `Post` has no tolerance, and its frame
  covers every byte outside the output's first `n` elements.
- The assumption `faithful`: `∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`,
  the library's fact for this model and launch, with a docstring that matches the rule page.
- The write-up against the Lean: Specification, Conditions (each field, the narrowings, `Pre`, part
  (A) point by point, the supported subset), Assumptions, Proof outline (the four goals of
  `meets_independent`), and Narrowings and modelling decisions. Two wording remarks are in the
  filing's notes.

## Not checked

- Part (B) of `Compiler.Faithful`'s scope (points 1, 2, 3, the environment's part of 4, 7, 10 and
  11), which the claim assumes and the scope does not ask a kernel reviewer to check.
- What `triton.testing.do_bench` and `triton.testing.perf_report` do inside Triton: I relied on the
  README and on the library naming `do_bench_cudagraph` as the variant that captures, and read no
  Triton source.
- The library's own facts (what each `tl.*` operation means, `Compiler.Faithful`), which I trust.
