---
name: tva-design
description: Design the proof before proving it - read the closest example's design (`tva-kit outline`), state every lemma in layers (facts, numbers, the program walk, composition) with `sorry` only as whole proofs, prove main from the statements, then have the claim-audit subagent check the claim and the design. Use at stages 6 and 7, before any long proof.
---

# Designing the proof (stages 6 and 7)

The statements of the lemmas are the design of a proof. When they are right, each proof is short work;
when one is wrong (too strong, missing a bound, asking for what the program walk cannot give), hours of
proof are lost on it. So state every lemma first, let Lean check that the statements fit together, have
them audited, and only then prove them.

## 1. Read the example's design

`tva-kit outline {{example}}` prints the closest example's Lean files bottom-up, each with its docstring
(what it proves, for which source lines) and its declarations. Read all of it. Then read the statements
of the files that do what your kernel's hard part does (a loop with a running state, a reduction, an
error bound): `tva-kit outline {{example}} FILE`. Decide which of your files corresponds to each of its
files: kernels of one family have the same files.

Every certified kernel in `kernels/` is an example, not only the closest ones. Their designs, each file
bottom-up with its title (`tva-kit outline KERNEL` for the whole outline):

{{designs}}

## 2. The layers

Every certified kernel is built in the same layers, and each file uses only the layers below it:

| Layer | What it holds | Memory or `Safe` in it? |
|---|---|---|
| 1. facts | the launch and grid (`launch_size`, `launch_pid`, `launch_valid`), `int32` arithmetic without wrap-around, addresses and their bounds, the constants | no |
| 2. numbers | one operation's rounding (`f32_mul`, `f16_cvt`, …) and real-number lemmas; for each loop the **invariant** on its values, the **step lemma** (one trip keeps it) and the **end lemma** (the invariant after the last trip gives what `Post` asks); without a loop, one lemma for an output's error | no: values (`F32`, `F16`, `ℝ`) and the instance only |
| 3. the program walk | the loop's walk (one trip per step of `safe_forRange`) and `safe_program`: one `Safe` rule per model line | yes |
| 4. composition | the footprints `R`, `W`, `Q`, their disjointness, `main`, `consistent`, `satisfiable` | |

The walk is thin. Each rule hands it the relation of one line's result (`FloatV.NaNEq (FloatV.sub a b) x`,
`P.math .exp …`, a reduction tree's `Eval`, a dot's error bound), and at the end of a trip it passes them
all to the step lemma. All the hard reasoning (error bounds, the running maximum, rescaled sums) is in
layer 2, on plain values, in small files that build in seconds. Never do layer-2 reasoning inside the
walk: a `sorry` left there is a step lemma nobody stated, and the walk around it takes minutes per build.

## 3. The step lemma: one hypothesis per line

State the step lemma with one hypothesis for each model line of the loop body, each exactly what that
line's rule gives, and the invariant before the trip; it concludes the invariant after the trip:

```lean
theorem row_step … {m mx m' l l' : F32} {s x p : Fin B → F32} …
    (hs : ∀ j, (s j).isFinite = true ∧ |(s j).toReal - R.sv (B * t + j)| ≤ …)  -- the scores (line …)
    (hmx : ∃ tr : Tree, tr.Covers B ∧ tr.Eval (FloatV.MaxRel true) (laneSeq s) mx)  -- tl.max (line …)
    (hm' : FloatV.MaxRel true m mx m')                                           -- tl.maximum (line …)
    (hx : ∀ j, FloatV.NaNEq (FloatV.sub (s j) m') (x j))                         -- s - m_new (line …)
    (hp : ∀ j, P.math .exp .binary32 (x j) (p j))                                -- tl.exp (line …)
    …
    (hold : St R t m l acc) : St R (t + 1) m' l' acc'
```

Read real ones before you write yours (each is stated this way, and the list names walks that call them):

{{design_lemmas}}

To find the hypotheses, write the trip's walk with the rules (the table in skill **tva-proof**) and read
what each rule gives. The invariant (`St`) says what the loop-carried values are after `t` trips against
the exact computation over the first `t` trips, with an error that may grow per trip. Give it room for
what one trip adds (an absolute term per trip, a factor per trip), or the step lemma is false. At `t = 0`
it holds of the loop's initial values.

## 4. The skeleton (stage 6)

- One file per part (as the example has), each with a docstring that lists its lemmas and the source
  lines they cover, like the example's.
- Every lemma stated; a lemma of layer 1 or 2 may have `sorry` as its whole proof (`:= by sorry`).
- The upper layers proved from the statements below them: the walk from the step lemma and the facts,
  `main` from `safe_program`, the footprints and the disjointness. A `sorry` inside one of these is a
  lemma not yet stated: state it.
- The whole build green.

`tva-kit stage` marks stage 6 done when `main`, `consistent` and `satisfiable` are stated, `main` has no
`sorry`, and no `sorry` sits inside a proof longer than 150 lines; it lists every `sorry` with its
lemma. When writing the walk shows that a hypothesis cannot be supplied, change the step lemma's
statement now: that is what the skeleton is for.

## 5. The audit (stage 7)

Start the **claim-audit** subagent ("audit the claim and the design of the kernel"). Before you prove,
it checks every integer operation for wrap-around under `Inst.ok`, the tolerances at the extremes in
floating point, each invariant's room for one trip's error, and that every `sorry` is a stated lemma;
it writes `~/scratch/audit.md`. Fix what it finds in the claim and the statements (state each narrowing
in README.md). In an earlier run the prover learned only after three hours of proofs that its claim was
false (an `int32` product could wrap around) and that an invariant had no room for one trip's error.

## 6. Then prove (stage 8, skill tva-proof)

Hardest first: the step lemma of each loop, since that is where a statement turns out wrong. Give each
lemma about an hour; if it does not close, change its statement (split it, add the bound it needs to the
invariant, add a condition to `Inst.ok` and say so in README.md) rather than pushing on. Delegate one
lemma per subagent, in its own file, with its statement and the example's lemma to follow. For a lemma
that resists, run several attempts at it at once with different approaches: `tva-kit try start LEMMA`
(skill **tva-proof**, "Several attempts at one lemma").
