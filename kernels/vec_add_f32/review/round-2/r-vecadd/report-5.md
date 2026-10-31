# Review of vec_add_f32, round 2, by r-vecadd

## Verdict

accept
Judged in full at gate `5f65eb1f7a961315`. The kernel's own constants are those I accepted at gate
`a2e8d45aaf58a6d2` (same hashes in both snapshots); what changed is library wording (L117, L118) and
the README. Both of my library reports are settled as the library now states them: `I32` carries
the `int32` docstring and `I8` its own (F2), and point 11 of `Compiler.Faithful`'s scope now asks for
the code Triton compiles for the launch's own argument values, compile-time and runtime, with no
reuse clause, in every copy (F3). The model is the source line by line, `Inst.ok` covers the host's
use (fresh disjoint `float32` buffers, `BLOCK_SIZE = 1024`, `n` from `98432` and `2^12 … 2^27`, all
below `2^31`) and constrains the runtime value `n` as the user's decision on L117 asks, `Post` is
the exact IEEE `float32` sum up to NaN payload with a full frame, and the one `[rule]` field is
`Compiler.Faithful` as the library states it. I have no blocking finding and accept every item.

## Findings

- **release F2** — L118 fixed as asked: `lean/TVA/Lang/Scalar.lean:103` gives `I8` "`tl.int8` values.", `:106` gives `I32` the `int32` docstring; the rule page and `TVA.Claim` now state point 7's host in full.
- **release F3** — L117 fixed in every copy of point 11 (Lean docstring, rule page, SEMANTICS.md, exclusions.md, `TVA.Claim`): the launch's own argument values, compile-time and runtime, no reuse clause, a repeated launch only "for example"; `Inst.ok` constrains `n` as the decision asks.

## Accepted

All items.

## Checked

- What changed since gate `a2e8d45aaf58a6d2`: the two snapshots' cones differ only in library entries (`TVA.I32`, `TVA.I8`, `TVA.Rules.Compiler.Faithful`, and the files `docs/SEMANTICS.md`, `docs/exclusions.md`, `docs/rules/compiler-faithful.md`, `lean/TVA/Claim.lean`, `lean/TVA/Lang/Scalar.lean`, `lean/TVA/Rules/Compiler.lean`); every kernel constant has the same hash. The gate's recorded sha256s of the kernel files equal the tree's.
- L117 and L118 as the library now states them: `Scalar.lean:103-107`; `Rules/Compiler.lean` (the docstring of `Faithful`, points 7 and 11, and its unchanged statement); `docs/rules/compiler-faithful.md` (What it says, Scope 11, What it does not cover, How a bundle cites it); `docs/SEMANTICS.md` (Parameters and arguments, Compile-time values Triton keys apart, Launches, Scalar arguments); `docs/exclusions.md` (row on code other than the launch's own); `lean/TVA/Claim.lean` (module docstring). No copy of the old reuse clause is left in the library's pages or Lean. I read the entries L117 and L118 of `docs/library-review/ISSUES.md` only to learn what was fixed.
- The whole of `lean/VecAdd.lean`: every `def`, `structure`, `abbrev`, `open` (`TVA`, and `Block` inside the proof section, after every claim definition), binder and field access; no `instance`, `notation`, `attribute`, `export` or `set_option`. `work/Names5.lean` prints the model, launch, conditions, `Post`, `Spec` and `Assumptions` with full names and explicit instances: every operation is `TVA.*` or core `Nat` arithmetic in the grid formula; the axioms of `main`, `consistent` and `satisfiable` are `propext`, `Classical.choice`, `Quot.sound`.
- The model against the source by SEMANTICS.md, every map entry: line 39 (`program_id` axis 0 of the 1-D grid, `int32`); 44 (the constexpr built in `int32`, wrapping product); 45 (scalar plus `int32` block); 47 (signed `int32` comparison, also at `n = 1`, where the constant `1` becomes an `int32` tensor first); 50, 51 (pointer plus offsets in elements, the source's mask, no `other`); 52 (IEEE `tl.add`: no product feeds it); 54 (no conversion, the source's mask); the drops of 23, 24 (imports binding `triton`, `tl`) and 29 (bare `@triton.jit`, so `debug` false); 30-36 (three pointers, an unannotated `int`, a `tl.constexpr`, no defaults).
- The ranges and module-level names: the launch runs only `add_kernel`; it reads only `triton` and `tl`, bound only by lines 23-24; the host lines bind `torch`, `DEVICE`, `add`, `size`, `x`, `y`, `output_torch`, `output_triton`, `benchmark`, none of which the kernel reads (its `x`, `y`, `output` are assigned before they are read); `add`'s annotations and `benchmark`'s decorator bind nothing. The source is in the supported subset.
- The launch: grid `cdiv(n, BLOCK_SIZE)` as `triton.cdiv`, `fusion = true`, `debug = false`, `numCtas = 1`, `pdl = false`, blocks `[[BLOCK_SIZE]]`. Part (A) of the scope against the script's launches (line 75, called at 89 and through `triton.testing.do_bench` at 127): no option, no `warmup` or `triton.compile`, no autotune or heuristics, no descriptor, no host `TensorDescriptor`, no CUDA graph capture. The source visibly breaks no point of part (B).
- The conditions: `Inst.ok` holds for the host's sizes and edges and for `add(x, x)`, and fails at `n = 2^31` and in place (`work/Cover3.lean`, re-run now); every narrowing is in the README with its reason; `satisfiable` holds.
- The specification: exact IEEE `float32` sum up to NaN payload for every `j < n`, every other byte unchanged; `work/AddCheck3.lean`, re-run now, finds `FloatV.add` equal to two float32 references on 600 pairs, so `Post` rejects a lower-precision, flushing or truncating add.
- The assumption: `faithful` is `Compiler.Faithful P (model d) (launch d) (K d)` under `∀ d, d.ok →`, with a docstring that matches the rule; there is no `[kernel]` assumption.
- The README against the Lean: Specification, Conditions (the new paragraph on `n` and point 11), Assumptions, Proof, Narrowings and Modelling decisions. One wording remark in the filing's notes.

## Not checked

- Part (B) of `Compiler.Faithful`'s scope (the environment, Triton's caches, the target and host), which the claim assumes and the protocol does not ask me to check.
- Whether the kernel compiles, and anything on a GPU; the library's own review material beyond the two issue entries above.
- r-vecadd2's new filing, which I have not read.
