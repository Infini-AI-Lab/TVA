# The choice sites of v3, and what `exec` does at each

Working notes for phase 3 of [PLAN.md](PLAN.md) (2026-11-13). The executor and its soundness
theorems are in [Exec.lean](Exec.lean); the change to `Prog` they need is described in PLAN.md
("The change to `Prog` that phase 3 needs") and is pending zhuominc's review.

> **Status (2026-11-14, branch `validation-v3-sites`, commit `6beae0a`).** Stage 1 is done: the
> solvers have left the library. The semantics is written for any signature `Sig`
> (`TVA.Program`); Triton's is the catalog `TVA.Site`, so far only `Site.other P` and
> `Upd.other upd`, with no solver. `Exec.lean` does not build until stage 2. The sections after
> the plan describe the previous design (commit `2d69b79`), whose solver code stage 2 reuses.

## Stage 2 progress (2026-11-14)

Done: (1) `validation/Solve.lean`: `Ans` with four outcomes (`assumed` needs a proof of
`Solvers.Platform`), `Gap`, `Hints`/`Oracle` with occurrence numbers, typed `Record`s and
`Record.Assumptions`. (2) The catalog (`TVA.Site`): `guard`, `hint`, `lanes`, `upToNaN`, `addc`,
`mulAdd`, `maxMin`, `itof`, `ftoi`, `fromF8`, `convert`, `plat`, `dotAdd`, `dotScaledAdd`,
`sumBcastMul`, `fsum`/`fsum1`/`fsum0`, `fmax`/`fmax1`/`fmax0`, `arg`/`arg1`/`arg0`, `histogram`,
`pad`, `anyVals`, `order`; updates `add`, `xchg`, `cas`, `max`, `bit`, `red`. Each `Allows` is the
old predicate by definition (`Site.allows_*`, `validation/Preserve.lean`). The relations the
catalog needs moved, unchanged, to `TVA/Triton/<File>/Rel.lean`, imported before `TVA.Site`.
`Safe.pick`/`SafeA.pick` take any `Site`; new kit rules `Safe.guard`, `Safe.hint`. Still
`Site.other`: `reshape_any`, `max_constancy`/`max_constancyN`, the scans, and
`make_tensor_descriptor` (in no claim). `Upd.other` is unused. Oracle edits: three lines
(`upd :=` → `u :=` in `route_histogram_i32` and `streamk_matmul_f16`). All 41 pass after every
family.

Next: (3) solvers per constructor, (4) `Exec.lean` on `Ans`/`Record`, (5) verdicts.

## Stage 2 plan: the catalog, the solvers, replay

**Catalog: one constructor per semantic family, not per `tl.*` function.** Start with the
reusable families: guards (`guard c` with `Decidable c`), compiler hints, partial lanewise
computations (`lanes g`: `idiv`, `irem`, shifts), values up to NaN (`upToNaN v`), platform requests
(`plat q P`, including the mixed `dot_add`/`dot_scaled_add`), float and index reductions (tree
sites), padding, atomic lane order and disabled-lane values, and the atomic updates (`Upd`). Add a
constructor only when the allowed-result relation or the runtime data a solver needs differs.
Keep `other P` for arbitrary relations (scans' caller-given combiners, `sort`, `reshape_any`,
`max_constancy`); the executor reports them unsupported.

Rules for a constructor:
- It carries **executable** data enough to produce or check a witness (a `Decidable` instance, a
  lanewise partial function, the operands of a platform request), never an erased predicate
  wrapped in a constructor.
- It carries **only** what its relation needs. Run-time bookkeeping (source location, occurrence
  number, solver settings) lives in the executor's trace, outside the site.

**Migration preserves meaning, and each step proves it.** For each migrated operation, a theorem
that the new `Site.Allows` is equivalent to the old predicate (usually `Iff.rfl`), so the allowed
results and the cases with no result (undefined behaviour) are both unchanged. Mixed operations stay
one site with their original disjunctive relation (`dot_add`: platform dot, or the fused form):
splitting into a free branch choice followed by an operation adds undefined behaviour through an
empty branch. Equal `Allows` is not enough for atomics: `UsesSharedAtomicResult` and `Step.defer`
compare continuations with `=`, so a site whose data depends on an atomic's old value while its
relation does not makes the semantics more conservative; check each migrated family against the
oracle proofs that use atomics (`streamk_matmul_f16`, `lora_shrink_bf16`, `route_histogram_i32`,
`count_expert_tokens_i32`).

**Solver answers: four outcomes.** A solver for site `s` (in `validation/`, keyed on the
constructor) returns one of:
1. a value with a proof that `s` allows it;
2. a proof that `s` allows no value (predicted undefined behaviour);
3. unsupported, or search failed: inconclusive, never undefined behaviour;
4. an unproved answer, only at an explicitly platform-dependent site (`plat`, float
   `atomic_add`, conversions into fp8).

The same interface for `Upd`, given the old value. `exec_sound` stays conditional on outcome 4.

**Typed replay records.** Replace the predicate-only assumption log with records of the operation,
its operands, its configuration and the answer returned; occurrence numbers are added outside the
site, so repeated requests may receive different answers. A supplied answer (a device value, a
tree, a lane order) is checked against the actual request: an invalid tree, a non-permutation or
mismatched operands is a diagnostic, and a default is used only when no answer was supplied (today
`Solve.askOr` silently falls back, which hides operand swaps).

**Verdicts are separate from the executor's correctness.** `exec_sound` is the executor's contract.
The differential tests report, per case: agreement with a certified execution; agreement
conditional on platform answers; a candidate discrepancy (to investigate, and bring to zhuominc);
inconclusive search; unsupported operation or exhausted fuel.

**Order of work.** (1) the solver-answer and replay types in `validation/`; (2) families one by
one, each with its `Allows` equivalence theorem, the proof kit's rule, a rebuild and all 41
oracles; (3) the solvers per family; (4) `Exec.lean` and `exec_sound` on the new types; (5) the
verdict layer in the harness (`ops/`, `run_test.py`).

## What changed in the library

- `Prog.pick P solve k`: `solve : Solve P` proposes a value with its proof (`Ans`), asking an
  oracle (`Hints`, the closed family `Query`: a tree, an order of lanes, an element, a number)
  for what it cannot compute. `Step.pick`/`Step.pickUB` ignore it.
- `Prog.plat q k` (a definition, not a constructor): `q : PlatQ` is an operation the platform
  defines on concrete operands (`math`, `div`, `cvt` into fp8, `dot`, `dotScaled`), and
  `Prog.plat q k` is `env fun e => pick (q.Rel e.P) (Solve.assume (.plat q)) k`, the old
  `env`-then-`pick` pattern with the predicate `PlatQ.Rel`.
- `Ans.assume b` (2026-11-13): a value with no proof, from the oracle, at a site whose
  predicate mentions the platform. `Solve.assume q` asks a `Query` that carries the operands
  (`plat`, `convert`, `atomicAdd`). `exec` reports each assumed value (`Assumed`), and its
  soundness theorems are conditional on them (PLAN.md, "Assumed answers").
- `Prog.rmw d a upd solve sem sc k`: `solve old : Solve (upd old)` proposes the new value.
  `tl.atomicWith` and `tl.rmwSeq` take one solver per lane.
- `Safe.plat` and `SafeA.plat` are theorems (from `env` and `pick`); `safe_div`, `safe_math`,
  `safe_cvt8`, `safe_dot`, `safe_dot_scaled` keep their statements and proofs. The atomic rules take
  the solver as an implicit argument. `UsesSharedAtomicResult` quantifies over the solver too.
- `Prog.SameUpToSolvers` (2026-11-14, from PR review): programs compared up to their solvers.
  With `=`, a continuation that uses an atomic's old value only in a solver (`Solve.ok old _`)
  counted as using the result in `UsesSharedAtomicResult`, and made `Step.defer`'s premise
  vacuous (a deferred `pick` could choose per old value). Both now use `SameUpToSolvers`, and so
  do the "result unused" premises of `Safe.rmw`, `SafeA.rmw`, `SafeA.shared` (proofs that had `=`
  pass `.of_eq`; `SameUpToSolvers.bind_left` replaces the `rw` in the sequencing rules).
- Decidability instances for the guards' conditions (hints, `RoundOk`, block pointers,
  descriptors, `dot_scaled`'s checks), next to the definitions.

**Claims and proofs.** No claim file changes (their sha256 are unchanged). All 41 oracle proofs
check against the changed library (each claim's `Claim` and `Proof/**` compiled in import order,
`main : claim.Holds`, axioms `propext`, `Classical.choice`, `Quot.sound`), with no proof file
changed: since `plat` is a definition, no `induction` over `Safe` has a case to add.

## Status of each site

| sites | solver |
|---|---|
| `Unit` guards: `assume`, `static_assert`, `cvt`, `cvt8`, `cvtFrom8`, `itof`, `ftoi`, block pointers, descriptors, `dot_scaled`, `dot_scaled_add` | `Solve.guard`: decides; a false condition is a predicted undefined behaviour |
| hints `max_contiguous`, `multiple_of`, `max_contiguous_multiple_of` (rank 1 and N), `multiple_of1` | `Solve.hint x`: decides; a false promise is a predicted undefined behaviour |
| hints `max_constancy`, `max_constancyN` | none: the element type has no `DecidableEq` in the signature |
| `idiv`, `irem`, `shl`, `ashr`, `lshr` | `Solve.lanes`: computes, predicts undefined behaviour |
| `ftoi`, `histogram`, `make_tensor_descriptor` | the only candidate, decided |
| `upToNaN` (`add`, `sub`, `mul`, `neg`, `abs`, `fma`, `div_rn`, `sqrt_rn`, `cvt`), `maximum`, `minimum`, `cvtFrom8` | the exact value |
| `addc`, `subc`, `mul_add` | the unfused IEEE value: one of the allowed results; a fused device result is not it (run with `enable_fp_fusion=False`, or choose inputs where both agree) |
| `itof` | the direct rounding (`ItofRel` also allows two double roundings) |
| `convert` (`ConvRel`) | computed for every pair of dtypes, "no result" where `ConvRel` has none; into fp8 from a float, the platform's: assumed (`Query.convert`) |
| `div`, `math`, `cvt8`, `dot`, `dot_scaled` (`Prog.plat`) | assumed: the oracle's answer (`Query.plat`), the device's value for those operands |
| float reductions `sum*`, `max1`, `maxAll`, `max0`; `argmax*` | the oracle's tree (`Query.tree`), else the sequential tree, checked (`Covers`, and the depth bound `sumBound e.P n`, which is data); evaluated with the IEEE sum, `FloatV.max`/`min`, `argCombine` |
| `pad` | `.zero`: zero; `.nan`: the all-ones pattern; `.undef`: the oracle's element, else zero |
| atomics: disabled lanes' values, lane order | zero bits; the oracle's order (`Query.order`) if it is a permutation of the enabled lanes, else index order |
| atomic updates: integer `add`, `xchg`, `cas`, `max`/`min`, bitwise; float `max`/`min`; `desc_atomic` | computed (float `max`/`min`: the candidate `fBound` accepts) |
| float `atomic_add` | assumed (`Query.atomicAdd`): `addUpd` mentions `e.P.atomicAdd`, a platform relation |
| mixed platform and fusion: `dot_add`, `dot_scaled_add` | assumed: the oracle's dot (scaled dot) with `c` as the accumulator (`Query.plat`), the first branch |
| `sum_bcast_mul` | none yet: its first branch (products, then a tree of additions) can be certified with the oracle's tree, as the reductions are; the second (`tl.dot`) assumed |
| `associative_scan` (`cumsum`, `cumprod`, …), `sort`, `reshape_any` | none yet. A scan's combine is a relation given by the caller; `sort` and `reshape_any` need a permutation from the oracle and a check |

`exec` gives up (`none`) at a site without an answer. Since `Ans.assume`, every site the 41
claims use has one (by `grep tl\.` in their `Claim.lean`) except `cumsum` (`chunk_cumsum_f32`).
Per operation: [COVERAGE.md](COVERAGE.md).

## Freedom outside `pick`: the effects

- `load`: the values of disabled lanes, from `Oracle.fill`.
- `store`: the order of the lanes (matters only for duplicate addresses). `exec` uses index order.
- `Step.defer`: the order of a program's atomics. `exec` never defers: sound, one order only.
- the launch: `execLaunch` runs program `0` to the end, then program `1`, …: one interleaving.

## What the soundness theorems say

`exec_sound`: if `exec` reports `done a m'`, then `Runs E m p m'.toMem (ret a)`; if it reports
`ub`, `Runs` reaches a step whose outcome is `ub`. `execLaunch_sound`: a finished launch is a
`Reach` trace from `Config.start` to a configuration where every program is `ret ()`, with the
reported memory; a reported `ub` is a trace to a `StuckUB` configuration. Neither says anything
about races (`WellBehaved`); phase 4 runs claims whose launches are proved well behaved.
Completeness is not claimed.

## Running it

`python3 validation/run_test.py vec_add_f32` (after `lake build` in `lean/`) compiles the
claim and `Exec.lean` into `/tmp/tva-validation` and elaborates `validation/tests/vec_add_f32.lean`:
the claim's model on `n = 10`, `BLOCK_SIZE = 4` (three programs, the last masked), checked against
`x + y` by `#guard`. With 1000 elements and 8 programs it takes about a second in the interpreter.

Memory in `exec` is a hash table (`CMem`, read through `CMem.toMem`, with `toMem_writeAll`): a
`Mem` is a function, and a memory after `k` writes is `k` nested closures (and a memory returned
by `writeAll` recomputes the whole store at every read), which made a 1000-element run take
minutes. Tensors are still functions, so a value computed from many others is recomputed at each
use; kernels with long loops may need tensors memoised, which phase 4 will show.
