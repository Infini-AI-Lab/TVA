# Review of vec_add_f32, round 2, by r-vecadd2

## Verdict

accept
A confirmation at gate `5f65eb1f7a961315`, the gate of my last acceptance. Since then the prover
changed only `README.md`, to fix the two wording remarks of my last filing. Both are fixed, and the
README still matches the Lean. No item's digest changed, I have no open finding, and I accept every
item again.

## Findings

None.

## Accepted

All items.

## Checked

- What changed: REVIEW.md gives the gate as PASS at digest `5f65eb1f7a961315`, with every item's
  digest the one I accepted in `filing-4.json`. `lean/VecAdd.lean`, `model-map.json` and the source
  are unchanged since that filing. Only `README.md` is newer.
- Remark 1 (Conditions): the README now says `n` is "the argument `n_elements`, a runtime argument
  at every `n` but `n = 1`, where Triton passes it as the compile-time constant `1`". This matches
  SEMANTICS.md, Launches, "Scalar arguments": an `int` equal to `1` passed to a parameter without
  annotation is a compile-time constant, and any other `int` below `2^31` is a runtime `int32`. It
  also agrees with the README's next paragraph, with `model`'s docstring and with Modelling
  decisions.
- Remark 2 (Modelling decisions): the README now says "Under point 11, only a launch of a
  `CompiledKernel` runs code that is not the code compiled for its own arguments (point 6), and
  there is none; this is the assumption, not a statement about Triton's caches". This is how point
  11 reads in SEMANTICS.md, Launches, "the kernel is new": cached code counts when it is that code,
  and so does a `CompiledKernel`'s code within its specialisation.
- I re-read `README.md` whole against `lean/VecAdd.lean`, and these all match:
  - the fields of `Inst.ok`;
  - `Pre`, `Post`, `Spec` and the `faithful` field;
  - the lemmas the proof names (`meets_independent`, `launch_valid`, `grid_lt`, `fits`,
    `safe_program`, `body_eq`, `launch_pid`, `tile_disjoint_in`, `div_lt_grid`,
    `exists_total_faithful`);
  - `testInst`'s sizes and addresses.

## Not checked

- I did not re-check anything unchanged since my last filing (`filing-4.json`, `report-4.md`): the
  model line by line, the map and its drops, the names of the kernel's Lean, the conditions'
  coverage, the specification and the `faithful` citation. All of them have the digests I accepted
  there, and that report records how I checked them.
- Part (B) of `Compiler.Faithful`'s scope, which the protocol assumes and no one checks.
- The library's facts, which I trust.
