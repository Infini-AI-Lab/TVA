---
name: tva-proof
description: Prove the lemmas of a TVA kernel's design and main, consistent, satisfiable (meets_independent or shared, Safe rules per line, footprints, int32 arithmetic), and fix Lean errors. Use at stage 8 (the statements come from stage 6, skill tva-design), and when writing the program walk, or whenever a Lean proof does not build.
---

# The proofs (PROVER_GUIDE §4)

The statements come first (stage 6, skill **tva-design**): every lemma stated, `sorry` only as a whole
proof, `main` and the program walk proved from the statements. Then replace one `sorry` at a time, the
hardest first (each loop's step lemma), building after each (the hook builds after every edit).
`tva-kit stage` lists what is left. Keep the example's lemma structure; it is the shortest route. Change a
statement when its proof does not close (and the statements that use it), never weaken one silently.

Keep each try fast. Lean re-checks the whole file at every build, so do not grow one file around a big
proof: a hard step (one loop trip, one numeric bound) is its own lemma, with the hypotheses it uses, in
a small file of `lean/` imported by the file that uses it; move lemmas that are done out of the file
you are editing. The whole output of each build is in `~/scratch/build/<Module>.log`.

## The route for kernels whose programs touch disjoint tiles: `TVA.meets_independent`

```lean
theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent hA.faithful (fun _ _ => rfl) (fun _ hd => launch_valid hd)
    (fun d _ i => R d i) (fun d _ i => W d i) (fun d init i m => Q d init i m) ?_ ?_ ?_ ?_
  · -- hsafe: each program is safe        (rw [body_eq, launch_pid]; exact safe_program …)
  · -- hdisj: W i disjoint from R j ∪ W j for i ≠ j
  · -- hRW:   R i disjoint from W i
  · -- hpost: the programs' results give Post
```

The lemmas that feed it are in the closest example, `{{example_file}}` (`tva-kit lookup <name>` prints one
with its signature). The usual set for a 1-D grid: `grid`, `body_eq` (the body for program `c.pid 0`, by
`rfl`), `launch_size`, `launch_pid`, `block_pos`, `grid_lt`, `launch_valid`, `fits`
(`Block.fits_of_grid_pow2`), `div_lt_grid`, the footprints `R`, `W` and what a program leaves `Q`,
`safe_program` (one program: one Safe rule per model line), and a disjointness lemma for tiles.

## One program: one rule per model line, in order

`refine safe_load_block … fun vx hx => ?_` then the next line, … , and the last line with
`safe_last (… fun m' hm' _ _ => safe_ret ?_)`. The table (from PROVER_GUIDE §4):

| Model line | Rule |
|---|---|
| `tl.load (b.ptr 0 + offs p B) (some (mask p B n))` | `safe_load_block` |
| `tl.store (b.ptr 0 + offs p B) v (some (mask …))` | `safe_store_block` |
| other loads/stores | `safe_load`, `safe_store`, `safe_store_any` |
| `tl.add`, `sub`, `mul`, `fma`, `div_rn`, `sqrt_rn`, `neg`, `abs` | `safe_add`, … ; `safe_exact` |
| `tl.mul_add`, `tl.addc`, `tl.subc` | `safe_mul_add`, `safe_addc`, `safe_subc` |
| `tl.cvt`, `tl.cvtFrom8`, `tl.itof`, `tl.ftoi`, `tl.cvt8` | `safe_cvt`, `safe_cvtFrom8`, `safe_itof`, `safe_ftoi`, `safe_cvt8` |
| `tl.maximum`, `tl.minimum` | `safe_maximum`, `safe_minimum` |
| `//`, `%`, `<<`, `>>` | `safe_idiv`, `safe_irem`, `safe_shl`, `safe_ashr`, `safe_lshr` |
| `tl.exp` …, `/`, `tl.dot` | `safe_math`, `safe_div`, `safe_dot` (accuracy: the named Rules.Num fact) |
| `tl.dot(a, b) + c` | `safe_dot_add` |
| `tl.sum`, `sum1`, `max1`, … | `safe_sum`, `safe_max1`, … ; accuracy `sum_err_covers` |
| hints | `safe_multiple_of`, `safe_max_contiguous`, … (prove the hint) |
| `for` loop | `safe_forRange` / `safe_forN_bind` with an invariant and per-trip write footprints |
| `while` loop | `safe_while` with an invariant and a decreasing measure |
| the last line / `return` | `safe_last` / `safe_ret` |

`Safe.bind` and `Safe.mono` compose and weaken; `TVA.Lemmas.Int` turns `int32` arithmetic into integer
arithmetic once you show it does not wrap. For anything else, read PROVER_GUIDE §4 (`tva-kit doc
"Proving Safe"`) and grep the examples: `tva-kit examples 'safe_forRange' --in lean`.

**Atomics / programs sharing memory** (split-K, Stream-K, histograms): `TVA.shared` with `SafeA` instead
of `meets_independent`; see PROVER_GUIDE §4 "Programs that share memory through atomics" and the example
`kernels/streamk_matmul_f16/`.

## consistent and satisfiable

```lean
theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.any model launch
  exact ⟨K, Platform.any, htot, ⟨fun d _ => hf d⟩⟩      -- one entry per Assumptions field
```

With device facts, use `Platform.sm90ref` and its fields (`Platform.sm90ref.expF32`, …; `tva-kit lookup
sm90ref`). `satisfiable` gives one covered instance and memory: the source's own test instance, buffers at
aligned, non-overlapping addresses (`testInst`, `testMem := fun _ => some 0`), and proves each `Inst.ok`
field and `Pre` (`norm_num [testInst]`, `simp [testMem]`).

## Lean errors: what to do

- Fix the first error; rebuild. `unknown identifier`/`unknown constant` → `tva-kit lookup NAME` (or the
  name is in another namespace: `TVA.` or `Block.`; `open Block` as the example does).
- `unsolved goals`: the goal is shown; finish it (`simp [defs]`, `omega`, `norm_num`, `exact …`).
- `type mismatch`: compare expected and given; often a missing coercion (`(i : Nat)`, `IntV.ofNat`).
- `omega` fails on `/` and `*` of variables: give it the facts (`Nat.div_add_mod`, `Nat.div_lt_iff_lt_mul`)
  as `have`s first, as `grid_lt` and `div_lt_grid` in the example do.
- `simp` loops or times out: `simp only [the lemmas you need]`.
- `maximum recursion depth` / `deterministic timeout`: split the proof into lemmas; avoid `decide` on big
  numbers (use `norm_num`).
- To see a goal mid-proof, put `trace_state` (or leave `sorry`) at that point and build; the goal is in
  the build's log (`~/scratch/build/<Module>.log`), in full.
