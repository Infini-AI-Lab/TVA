# The choice sites of `Prog`, and what `exec` needs at each

Working notes for phase 1 (2026-11-10). They check the three kinds of `PLAN.md` ("The oracle")
against the code of `lean/TVA/Lang`, which has 63 `.pick` sites. `exec` and `exec_sound` are in
`Exec.lean`.

## The obstacle, and the change made (2026-11-10, pending zhuominc's review)

`Prog.pick` stored only `P : β → Prop` and `k`. In compiled code `P` and `β` are erased, so an
executor cannot tell the sites apart: it can neither build a witness nor decide `P b`. The
alternatives considered were a meta-level executor over `Expr`, an annotated copy of `Prog` in
`validation/`, both together, and changing the library. The user chose to change the library,
which reverses the 2026-11-09 decision "a lemma, not a new construct":

- `Prog.pick P solve k`. `solve : Solve P` is code that proposes a witness, asking the oracle
  (`Hints`, a closed family `Query` of typed questions: a tree, a permutation, a rational, …) for
  the parts it cannot compute. It returns the witness with a proof (`Ans`). `Step.pick` and
  `Step.pickUB` ignore it, so no model's meaning changes, and a wrong solver can only make `exec`
  give up.
- `Prog.plat q k` (new). `q : PlatQ` is an operation the platform defines, on concrete operands
  (`math`, `div`, `cvt` into fp8, `dot`, `dotScaled`); `Step.plat` takes a result `q.Rel E.P`
  allows, and `Step.platUB` applies when there is none. `PlatQ.Rel` is the predicate the old
  `env`-then-`pick` had, so the meaning is the same. The run takes one silent step fewer.
  A kind-3 site cannot be a `pick` with a solver: `E.P` is erased too, so neither a solver nor an
  oracle could certify an answer about it.

Proof impact: `Safe` and `SafeA` gain a `plat` rule; the rules `safe_math`, `safe_div`,
`safe_cvt8`, `safe_dot`, `safe_dot_scaled` keep their statements and change their proofs. Every
case analysis on `Step` gained `plat`/`platUB`. No kernel proof mentions `pick`.

## Status of each site

| sites | solver |
|---|---|
| `Unit` guards (`assume`, `cvt`, `cvt8`, `cvtFrom8`, `itof`, `ftoi`, block pointers, descriptors, `dot_scaled`) | `Solve.guard`: decides, predicts UB when false |
| hints `max_contiguous*`, `multiple_of*`, `max_contiguous_multiple_of*` | `Solve.hint x`: decides, predicts UB when violated |
| hints `max_constancy`, `max_constancyN` | none: the element type has no `DecidableEq` in the signature |
| `idiv`, `irem`, `shl`, `ashr`, `lshr` | `Solve.lanes`: computes, predicts UB |
| `ftoi`, `histogram`, `make_tensor_descriptor` | the only witness, predicts UB |
| `upToNaN`, `maximum`, `minimum`, `mul_add`, `addc`, `cvtFrom8`, `itof` | the exact (IEEE, unfused) value |
| `math`, `div`, `cvt8`, `dot`, `dot_scaled` | `plat` |
| reduction trees (`sum*`, `max*`), `associative_scan`, `argmax*`, `sort`, `reshape_any`, `pad` | none yet (next). Since library issue L149 the float sums also require `t.depth ≤ sumBound e.P n`; `Platform.sumDepth` is data, not a relation, so it survives compilation and a solver can check the bound |
| atomics (junk, lane order), `rmw` | none yet: `upd` is still an erased relation. `Lemmas/AtomicResult.lean` writes out `atomicWith`'s `pick` terms, `Solve.none` included, so a solver there must be mirrored in those proofs |
| mixed: `dot_add`, `sum_bcast_mul`, `dot_scaled_add`, `convert` (`ConvRel`) | none yet. Splitting a disjunction into "choose a branch, then `plat`" would change when undefined behaviour is reachable, so these need care |

The decision procedures the solvers need (`Format.Std`, `RoundOk`, `ContigAlong`, `FirstsDiv`,
`TensorDesc.Access`, `BlockPtr.Valid`, `ScaledCfg.ShapeOk`, …) are instances next to the
definitions. `Idx.SameGroup` needed a bounded form, using `Idx.coord_of_ge`.

## The sites, by kind

The plan's three kinds hold, but several sites mix kinds, and some freedoms are not picks.

**Kind 1: guards and partial functions (one witness, or undefined behaviour).**
- `Unit` guards: `BlockPtr` 93, 100; `Desc` 182, 191, 199, 210, 219; `Scaled` 156, 169;
  `Ops` 443, 448, 457, 478, 488 (format conditions of `cvt`, `itof`, `ftoi`); `assume` (826).
- Hints, `y = x ∧ …`: `Ops` 834–890.
- Integer `div?`, `rem?`, `shl?`, `ashr?`, `lshr?`: `Ops` 694–717.
- Also of this kind: `ftoi` (489: a range condition plus a function), `histogram`
  (`Scan` 137: a range condition plus counts), `make_tensor_descriptor` (`Desc` 175: a sign
  condition plus `D = ⟨…⟩`).

**Kind 2: freedom independent of the platform.**
- NaN payload, `upToNaN` (342), used by `add`, `sub`, `neg`, `abs`, `cvtWith`, …. Its canonical
  witness is the exact value `v` itself, so `exec` can choose it without the oracle.
- `maximum`/`minimum` (417, 419: `MaxRel`): NaN payload, and either zero for `(+0, -0)`.
- `pad` (202: padding of out-of-bounds lanes, as `Padding.Holds` allows).
- fp8 → wider conversion of non-finite values (458).
- `itof` (479: `ItofRel` allows three roundings, direct or through `f64`/`f32`). This one is
  not in the plan's list.
- Reduction trees: `sumWith`, `sum1With`, `sum0With` (502–512), `max1`, `maxAll`, `max0`
  (539–552), `associative_scan` (`Scan` 78), `cumsum` (`Scan` 97, via `cumsumWith`).
- `argmax*` (620–634: ties when `tieLeft = false`, and NaN).
- `sort` (`Scan` 121). A row containing a NaN is unconstrained, because the condition is an
  implication. Inputs must avoid NaN or the check means nothing there.
- `reshape_any` (898: a bijection `Idx s' ≃ Idx s`).
- Atomics: the disabled lanes' junk (240) and the lane order (241).

**Kind 3: platform relations.**
- `addc`, and the fused additions in reductions: `AddRel fusion` (368). With `fusion = true` its
  second disjunct is `∃ p q : Rat, RoundsTo p x ∧ …`. That is not decidable as stated, so the
  oracle must supply `p`, `q` and `z`. With `fusion = false` it is `NaNEq` of the IEEE sum, which
  is kind 2.
- `mul_add` (360): IEEE or `fma`, depending on `e.fusion`. This is the fusion choice of kind 2,
  but it is read from `Env`.
- `div` (385), `math` (395), `cvt` into fp8 (449), `dot` (643), `dotScaled` (`Scaled` 157, 170).
- Mixed: `dot_add` (654: either `P.dot` with `c`, or `P.dot` then `AccAdd`), `sum_bcast_mul`
  (677: a tree of products, or `P.dot` with `ieee`), `convert` (`Promote` 425: `ConvRel`, which
  is kind 1 for integers, kind 2 for most floats, and kind 3 when it reaches `P.cvt` into fp8).

**Freedom outside `pick`: the `Prog` effects themselves.** The plan does not list these.
- `load`: the values of disabled lanes (`Step.load` constrains only the enabled ones). `exec`
  asks `Oracle.fill`. These are not observable in a correct model (masked-out lanes go through
  `where`), but nothing enforces that.
- `store`: the order of the lanes (it matters for duplicate addresses). `exec` uses index order.
- `rmw`: the new value is a relation `upd`. For integer atomics it is a function (`n = o + v`).
  For float `atomic_add` it is `P.atomicAdd` (kind 3). The same erasure problem applies.
- `Step.defer`: the order of a program's atomics. `exec` never defers. That is sound, but it is
  one order only.

## What `exec_sound` does and does not say

It says: if `exec` reports `done a m'`, then `Runs E m p m' (ret a)`; if it reports `ub`, then
`Runs` reaches a step whose outcome is `ub`. It is about one program, and it does not track
events, so it says nothing about `IntraRace`. Phase 4 runs programs one at a time, as
`independent` allows, so this is enough there, but a race-free check of the model is separate.
Completeness (every `Step` outcome is some oracle's `exec`) is not claimed, and it fails for
`defer` orders and for store orders other than index order.
