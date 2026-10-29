# Review of vec_add_f32, round 2, by r-vecadd2

## Verdict

accept
Judged in full at gate a2e8d45aaf58a6d2. The model is the source line by line by SEMANTICS.md (an
`int32` program id times the constexpr built in `int32`, the scalar–tensor `+`, the signed `int32`
comparison with `n_elements`, which at `n = 1` is the constant `1` made an `int32` tensor, the two
masked loads without `other`, the IEEE `tl.add` that no product feeds, the masked store without
conversion), the launch is the host's (`triton.cdiv` grid, the one block shape `[BLOCK_SIZE]`,
every option at its default), and every name of the kernel's Lean is the library's. The
conditions cover the host's use (fresh `float32` buffers, `BLOCK_SIZE = 1024`, the script's sizes up
to `2^27`, `add(x, x)`), and each narrowing (`n < 2^31`, `float32`, contiguous buffers, `y` long
enough, no overlap of `output` with an input) is stated with its reason. `Post` is the exact IEEE
`float32` sum up to NaN payload, with every byte outside `output`'s first `n` elements unchanged.
`faithful` is the library's `Compiler.Faithful` in its form, and the launches are within part (A)
of its scope. The three wording remarks of my last filing are fixed. I raise one non-blocking
`LIBRARY_REPORT` on the wording of the rule's scope 11, and accept every item.

## Findings

- **LIBRARY_REPORT** `lib:docs/rules/compiler-faithful.md` — scope 11 admits code compiled "earlier for the same source, compile-time values and settings" without asking for the same specialisation of the runtime arguments (integer class, pointer alignment)

## Accepted

All items.

## Checked

- Read in full: `claim.md`, `source/01-vector-add.py`, `lean/VecAdd.lean`, `model-map.json`,
  `kernel.json`, `README.md`, `REVIEW.md`; SEMANTICS.md (the supported subset, how the front end
  evaluates the source, the `tl.*` operations, Constants, Operators and promotion, Launches, Scalar
  arguments), `docs/rules/compiler-faithful.md`, `docs/exclusions.md`, and the library definitions
  the claim reaches (`TVA.Claim`, `TVA.Sem.Launch`, `TVA.Rules.Compiler`, `TVA.Lang.Ops`, `Tensor`,
  `Mem`, `Scalar`, `TVA.Lemmas.Buf`, `Access`).
- What changed since gate 990243026f1289fb: the two snapshots differ only in library files and in
  the digest of `Rules.Compiler.Faithful`, whose statement `work/Names4.lean` prints in the same form;
  no kernel constant changed. I re-read the whole README and the whole Lean file all the same.
- The three wording remarks are fixed (README's environment bullet: without `-O` or `-OO` and with
  `PYTHONOPTIMIZE` unset; the `faithful` docstring: "one of the outcomes of the model's launch";
  README's supported-subset paragraph: an `int32` block added to a `*fp32` pointer), and the L116
  passages (every launch a direct launch of the `@triton.jit` function; point 11) match the rule page.
- Names (`work/Names4.lean`, full names and explicit instances): every operation of the model is
  `TVA.*`, `Disjoint` is Mathlib's on `Set Int`, the grid uses core `Nat` instances, the only other
  kernel constant is `add_kernel._proof_1`; `Assumptions` is a structure with exactly `(K) (P)`;
  `main`, `consistent`, `satisfiable` use only `propext`, `Classical.choice`, `Quot.sound`. The
  file's `open TVA` resolves every model name to the library's (as printed), its `open Block` comes
  after the model, in the proof section, and it declares no instance, notation or option.
- Every map entry and every drop (lines 23, 24, 29) against SEMANTICS.md; the code lines of 29–54
  all mapped; the parameters' kinds (three pointers, an unannotated `int`, a `tl.constexpr` `int`).
- The ranges: the launch runs only `add_kernel`; `add`, `benchmark` and the lambdas run on the host;
  the launch reads only `triton` and `tl` (lines 23–24, mapped); the host lines bind `torch`,
  `DEVICE`, `add`, `size`, `x`, `y`, `output_torch`, `output_triton`, `benchmark`, none of which the
  kernel reads (its own `x`, `y` are assigned before line 52 reads them); `add`'s annotations and
  `benchmark`'s decorator bind nothing.
- Part (A) of `Compiler.Faithful`'s scope against the script's launches (line 75, called at 89 and
  through `triton.testing.do_bench` at 127): 4, no option, no `warmup` or `triton.compile`, no
  autotune or heuristics, a bare `@triton.jit`; 5, no descriptor; 6, no `CompiledKernel`; 8, no host
  `TensorDescriptor`; 9, no capture. The source visibly breaks no point of part (B): it sets no knob,
  installs no hook, rebinds no Triton name, sets no `kernel.debug`, calls no `preload`; line 26 only
  reads `triton.runtime.driver.active`.
- Conditions (`work/Checks4.lean`): `Inst.ok` holds for every `n < 2^31` with `BLOCK_SIZE = 1024` and
  any three buffers of at least `n` elements whose output bytes meet neither input, and for
  `add(x, x)`; it fails at `n = 2^31` and for an in-place output, as the README says; the script's
  sizes are below `2^31`.
- Launch: the grid is `triton.cdiv` at the host's sizes and at the extremes of `Inst.ok` (checked
  against Python), and `launch` keeps `debug = false`, `fusion = true`, one CTA, no PDL,
  `blocks = [[BLOCK_SIZE]]`.
- Specification: no tolerance; `FloatV.add` evaluated on `float32` patterns (exact `1 + 2^-23`, ties
  to even, rounding up, subnormals kept, overflow to `+inf`, `-0 + -0 = -0`, NaN for `inf - inf` and
  a NaN operand); `NaNEq` tells `-0` from `+0`; the frame covers every byte outside `output`'s first
  `n` elements. So `Post` rejects a lower-precision or flushing add, a missed tail and a write past `n`.
- The write-up against the Lean: Specification, Conditions, Assumptions, Proof and Narrowings say
  what `Post`, `Inst.ok`, `Pre`, `faithful`, `main`, `consistent` and `satisfiable` say.

## Not checked

- Part (B) of `Compiler.Faithful`'s scope, which the protocol assumes and no one checks.
- The library's facts (the meaning of each `tl.*` operation, `Compiler.Faithful` itself, the
  semantics of launches), which I trust; my one doubt is the `LIBRARY_REPORT`.
- Whether the kernel compiles, which no claim covers.
- A diff of the README and the Lean file against the versions I accepted at gate 990243026f1289fb,
  which no file keeps; I re-read both whole instead.
