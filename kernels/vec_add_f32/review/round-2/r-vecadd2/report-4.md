# Review of vec_add_f32, round 2, by r-vecadd2

## Verdict

accept
Judged in full at gate 5f65eb1f7a961315. Since my last acceptance the library changed (round 23:
L117, L118) and the prover revised the README on `n_elements` and point 11; no kernel constant and
no map entry changed. The model is the source line by line by SEMANTICS.md (an `int32` program id
times the constexpr built in `int32`, the scalar–tensor `+`, the signed `int32` comparison with
`n_elements`, which at `n = 1` is the constant `1` made an `int32` tensor, two masked loads without
`other`, the IEEE `tl.add` that no product feeds, the masked store without conversion); the launch
is the host's; every name of the kernel's Lean is the library's. The conditions cover the host's
use, and the runtime argument `n` is constrained by them (`n < 2^31`, `n ≤ x.len`, `y.len`,
`output.len`), as point 11 now asks. `Post` is the exact IEEE `float32` sum up to NaN payload, with
every other byte unchanged. `faithful` is the library's `Compiler.Faithful` in its form. My library
report F4 is settled as the library now states point 11, so I release it, and I accept every item.

## Findings

- **release F4** — point 11 now asks for the code Triton compiles for the launch's own argument values, compile-time and runtime, with no reuse clause; code compiled for another runtime value counts only when it is that code

## Accepted

All items.

## Checked

- Read in full: `claim.md`, `source/01-vector-add.py`, `lean/VecAdd.lean`, `model-map.json`,
  `kernel.json`, `README.md`, `REVIEW.md`; `docs/rules/compiler-faithful.md`, SEMANTICS.md (the
  introduction, the supported subset, how the front end evaluates the source up to "Annotated
  names", the `tl.*` operations up to the integer conversions, Launches with Scalar arguments),
  `docs/exclusions.md`; the docstring of `Rules.Compiler.Faithful`, `TVA.Claim`'s point-11 and
  point-7 text, and the library definitions the claim reaches (`TVA.Lang.Ops`, `Tensor`, `Mem`,
  `Scalar`, `TVA.Sem.Launch`, `TVA.Lemmas.Buf`, `Access`, `TVA.Claim`).
- What changed: the snapshots of gates a2e8d45aaf58a6d2 and 5f65eb1f7a961315 differ only in
  `TVA.I8`, `TVA.I32`, `TVA.Rules.Compiler.Faithful` and library files; the model, launch, `Inst`,
  `Inst.ok`, `Pre`, `Post`, `Spec` and the field's statement print as before (`work/Names5.lean`).
- F4 (L117): point 11 on the rule page, in the Lean docstring, in SEMANTICS.md ("Launches") and in
  the exclusions row now states the code for the launch's own argument values, compile-time and
  runtime, and excludes code compiled for another runtime value unless it is that code; the README's
  Conditions and Modelling decisions follow it.
- L118: the docstrings of `TVA.I8` (`tl.int8`) and `TVA.I32` (`tl.int32`, …) as the library now has
  them (`work/Checks5.lean`); the model uses `I32 = IntV true 32`.
- Names (`work/Names5.lean`, full names and explicit instances): every operation of the model is
  `TVA.*`, the grid uses core `Nat` instances, the only other kernel constant is
  `add_kernel._proof_1`; `Assumptions` is a structure with exactly `(K) (P)` and one field in the
  library's `Faithful` form; `main`, `consistent`, `satisfiable` use only `propext`,
  `Classical.choice`, `Quot.sound`. The file's `open TVA` and `open Block` (in the proof section)
  change no model name; it declares no instance, notation or option.
- Every map entry and every drop (lines 23, 24, 29) against SEMANTICS.md; the code lines of 29–54
  all mapped; the parameters' kinds (three pointers, an unannotated `int`, a `tl.constexpr` `int`).
- The ranges and module-level names: the launch runs only `add_kernel` and reads only `triton` and
  `tl` (lines 23–24, mapped); the host lines bind nothing the launch reads (the kernel's `x`, `y` are
  its own, assigned before line 52 reads them; `benchmark`'s decorator and `add`'s annotations bind
  no module-level name).
- Part (A) of `Compiler.Faithful`'s scope against the script's launches (line 75, called at 89 and
  through `triton.testing.do_bench` at 127): no launch option, `warmup`, `triton.compile`, autotune
  or heuristics, a bare `@triton.jit`, no descriptor, no `CompiledKernel`, no host
  `TensorDescriptor`, no capture. The source visibly breaks no point of part (B).
- Conditions (`work/Checks5.lean`): `Inst.ok` for every `n < 2^31` with `BLOCK_SIZE = 1024` and
  fresh disjoint buffers of at least `n` elements, and for `add(x, x)`; not for `n = 2^31`, an
  in-place output or a short `y`, each a narrowing the README states with its reason.
- Launch and indexing at the extremes: the grid is Python's `(n + B - 1) // B`; the last lane's
  `int32` offset at `n = 2^31 - 1` with `B = 2^20` and `B = 1` is the true offset (no wrap);
  `launch` keeps `debug = false`, `fusion = true`, one CTA, no PDL, `blocks = [[BLOCK_SIZE]]`.
- Specification: no tolerance; `FloatV.add` evaluated on `float32` patterns (exact `1 + 2^-23` and
  `1 + 2^-10`, ties to even, rounding up, subnormals, overflow, signed zeros, NaN); `NaNEq` tells
  `-0` from `+0`; the frame covers every byte outside `output`'s first `n` elements.
- The write-up against the Lean: Specification, Conditions (with the new `n_elements` paragraph),
  Assumptions, Proof, Narrowings and Modelling decisions say what the Lean says; two wording remarks
  are in the filing's notes.

## Not checked

- Part (B) of `Compiler.Faithful`'s scope, which the protocol assumes and no one checks.
- The library's facts (the meaning of each `tl.*` operation, `Compiler.Faithful` itself, the
  semantics of launches), which I trust; whether the L117 and L118 fixes are right is for library
  review.
- SEMANTICS.md sections this kernel does not reach (dot, descriptors, block pointers, scans, sorts,
  loops, runtime `if`), which I skimmed only by heading.
