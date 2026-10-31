# Review of vec_add_f32, round 2, by r-vecadd

## Verdict

accept
A confirmation at gate `5f65eb1f7a961315`, the gate I accepted every item at in `filing-5.json`.
The prover's only change since is to `README.md`, which no item's digest holds: the gate's digest,
every item's digest and the recorded hashes of the Lean, the source, the model map and
`kernel.json` are unchanged. Both wording remarks of this round are fixed, mine on Modelling
decisions and the one on Conditions, and the README still matches the Lean. I have no finding and
accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- That the claim is unchanged: `gate.json`'s digest is `5f65eb1f7a961315` and its item digests are
  those of my `filing-5.json` acceptance; the recorded sha256s of `lean/VecAdd.lean`,
  `source/01-vector-add.py`, `model-map.json` and `kernel.json` equal the files in the tree.
  `work/Names5.lean`, re-run now, prints `main`, `consistent` and `satisfiable` as before, on
  `propext`, `Classical.choice` and `Quot.sound` only.
- My remark (README, Modelling decisions): it now reads "Under point 11, only a launch of a
  `CompiledKernel` runs code that is not the code compiled for its own arguments (point 6), and
  there is none; this is the assumption, not a statement about Triton's caches". That is point 11
  as the rule page (Scope 11) and `docs/SEMANTICS.md` (Launches) state it. The parenthetical, "the
  script's later launches run code the JIT compiled at its first launch", agrees with
  `docs/SEMANTICS.md` ("Compile-time values Triton keys apart", "Scalar arguments"): the first
  launch (line 89, `n = 98432`) and the benchmark's (`2^12 … 2^27`) all have an `n` that is a
  multiple of 16 other than `1`, and `BLOCK_SIZE = 1024`. Nothing in the README rests on it.
- The other remark (README, Conditions): "`n` is the argument `n_elements`, a runtime argument at
  every `n` but `n = 1`, where Triton passes it as the compile-time constant `1`". That is what
  "Scalar arguments" says, and it agrees with the docstrings of `add_kernel` and `model`.
- The whole README against the Lean as it is now: the instance and the eight fields of `Inst.ok`
  with their reasons, `Pre`, the two parts of `Post`, `Spec`, the launch, the one `[rule]` field
  `faithful`, the absence of a `[kernel]` assumption, the proof's steps, `consistent`,
  `satisfiable`, the narrowings and the modelling decisions. It still matches.

## Not checked

- Anything that has not changed since `filing-5.json`: the model against the source, the model
  map and its drops, the ranges and module-level names, the conditions' coverage, the
  specification's strength and the assumption. I judged them in full at this same gate then and
  rely on that review; I did not re-run `work/Cover3.lean`, `work/Checks3.lean` or
  `work/AddCheck3.lean`.
- Part (B) of `Compiler.Faithful`'s scope, which the claim assumes; whether the kernel compiles;
  anything on a GPU.
- r-vecadd2's new filing, which I have not read.
