# The choice sites of v3, and what `exec` does at each

Working notes for phase 3 of [PLAN.md](PLAN.md) (2026-11-13). The executor and its soundness
theorems are in [Exec.lean](Exec.lean); the change to `Prog` they need is described in PLAN.md
("The change to `Prog` that phase 3 needs") and is pending zhuominc's review.

## What changed in the library

- `Prog.pick P solve k`: `solve : Solve P` proposes a value with its proof (`Ans`), asking an
  oracle (`Hints`, the closed family `Query`: a tree, an order of lanes, an element, a number)
  for what it cannot compute. `Step.pick`/`Step.pickUB` ignore it.
- `Prog.plat q k` (new): `q : PlatQ` is an operation the platform defines on concrete operands
  (`math`, `div`, `cvt` into fp8, `dot`, `dotScaled`); `Step.plat` takes any result
  `q.Rel E.P` allows, `Step.platUB` applies when there is none. `PlatQ.Rel` is the predicate the
  old `env`-then-`pick` had, so the meaning is unchanged.
- `Prog.rmw d a upd solve sem sc k`: `solve old : Solve (upd old)` proposes the new value.
  `tl.atomicWith` and `tl.rmwSeq` take one solver per lane.
- `Safe` and `SafeA` gain a `plat` rule. `safe_div`, `safe_math`, `safe_cvt8`, `safe_dot`,
  `safe_dot_scaled` keep their statements; their proofs use `Safe.plat`. The atomic rules take
  the solver as an implicit argument. `UsesSharedAtomicResult` quantifies over the solver too.
- Decidability instances for the guards' conditions (hints, `RoundOk`, block pointers,
  descriptors, `dot_scaled`'s checks), next to the definitions.

**Claims and proofs.** No claim file changes (their sha256 are unchanged). All 41 oracle proofs
check against the changed library (each claim's `Claim` and `Proof/**` compiled in import order,
`main : claim.Holds`, axioms `propext`, `Classical.choice`, `Quot.sound`). Two needed one line
each, a `plat` case in an `induction` over `Safe`: `recompute_wu_bf16/oracle/Proof/Stable.lean`
and `streamk_matmul_f16/oracle/Proof/Iter.lean`.

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
| `convert` (`ConvRel`) | computed for every pair of dtypes, "no result" where `ConvRel` has none; gives up into fp8 from a float, which is the platform's (`convAns`) |
| `div`, `math`, `cvt8`, `dot`, `dot_scaled` | `plat`: the oracle's answer for its platform |
| float reductions `sum*`, `max1`, `maxAll`, `max0`; `argmax*` | the oracle's tree (`Query.tree`), else the sequential tree, checked (`Covers`, and the depth bound `sumBound e.P n`, which is data); evaluated with the IEEE sum, `FloatV.max`/`min`, `argCombine` |
| `pad` | `.zero`: zero; `.nan`: the all-ones pattern; `.undef`: the oracle's element, else zero |
| atomics: disabled lanes' values, lane order | zero bits; the oracle's order (`Query.order`) if it is a permutation of the enabled lanes, else index order |
| atomic updates: integer `add`, `xchg`, `cas`, `max`/`min`, bitwise; float `max`/`min`; `desc_atomic` | computed (float `max`/`min`: the candidate `fBound` accepts) |
| float `atomic_add` | none: `addUpd` mentions `e.P.atomicAdd`, a platform relation, so no solver can certify a value. Open: an `rmw` whose update is a `PlatQ` |
| mixed platform and fusion: `dot_add`, `sum_bcast_mul`, `dot_scaled_add` | none: a disjunction of a platform result and a platform result followed by a fused addition. Open: split into the choice of a branch, then `plat`, which must keep when undefined behaviour is reachable |
| `associative_scan` (`cumsum`, `cumprod`, …), `sort`, `reshape_any` | none yet. A scan's combine is a relation given by the caller; `sort` and `reshape_any` need a permutation from the oracle and a check |

`exec` gives up (`none`) at a site without an answer. Of the operations the models of the seven
kernels phase 4 starts with use (`grep tl\.` in their `Claim.lean`), only `convert` into fp8
(`kv_cache_write_bf16`'s fp8 cache) has no answer; the matmul and attention kernels need
`dot_add`.

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
