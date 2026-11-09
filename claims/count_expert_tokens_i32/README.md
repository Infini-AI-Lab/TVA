# `count_expert_tokens_i32`: prepared claim

vLLM's `_count_expert_num_tokens` (`source/count_expert_tokens.py`,
kernel lines 45–74, host `count_expert_num_tokens` lines 78–107). `Claim.lean`, namespace
`Claims.CountExpertTokens`; the claim to prove is `Statement : ∀ K P, Assumptions K P → Spec K`.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.CountExpertTokens.main : Statement` (`Statement`:
`Claim.lean` line 254). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 178), and every initial memory `init` that `Pre` accepts (line 205), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 236).

**What `Post` asks** (line 236), two parts:
1. for every local expert `e < numLocal`: element `e` of `out` reads in `fin` as an `int32` `c`, and
   `c.toInt = countOf d e` (line 231), the number of positions `j < numel` with `routed d j = e`.
   `routed` (line 226) is computed from `Inst`'s contents: the entry `entryInt dtype (idsVal j)`
   (line 51) without a map; with a map, `(mapVal a.toNat).toInt` for an entry `a ≥ 0` and `-1` for a
   negative entry. Exact: no tolerance;
2. the frame: every byte outside the first `numLocal` elements of `out` is unchanged.

**What you are given** (`Assumptions`, line 245): one field, `faithful` (`Compiler.Faithful`): the
device's outcomes `K d` are outcomes of the model's launch, when the launch is valid, has
`debug = false`, and the model's launch is well behaved from `init` (no out-of-bounds access, no race).
No numeric fact: the kernel is integer-only.

**The kernel, as the model runs it** (`count_kernel` line 123, `loopBody` line 99, `model` line 212,
`launch` line 220): a 1-D grid of `numLocal` programs, block shape `[B]`. Program `e` computes the
trip count `cdiv(numel, B)` with two `tl.idiv` lines (the second is the bound), then runs
`tl.forRange 0 trips 1` over the state `(acc, topk_ids_ptrs)`. Each trip loads the masked block of
`topk_ids` (lanes `< numel − x·B`, `other` the `-1` of `m1`, line 44); with a map it gathers
`expert_map[id]` where `geZero` holds (`other = -1`); it adds `1` to `acc` where the id equals `e`,
and advances the pointers by `B`. After the loop, under the runtime `if e < numLocal`, it stores
`Tensor.foldAll (+) 0 acc` (the `int32` sum) to element `e` of `out` (`tl.store1`). Every program
reads all of `topk_ids` (and the map); each writes one element, and no atomics are used: the route is
`TVA.meets_independent`, and the loop is `safe_forRange` with an invariant (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1` with `numLocal_lt`; the block
  `[B]` from `b_pow2`, `b_le`);
- each program is `Safe`: the two `tl.idiv` lines have a nonzero divisor (`safe_idiv`) and
  `numel + (B − 1)` does not wrap (`numel_b`); every trip's masked load is in bounds (`ids_len`, and
  the `int32` values `x·B` and `numel − x·B` do not wrap for `x < trips`, from `numel_b`); with a
  map, every gathered index is in `[0, numGlobal)` (`Pre`'s bound on entries, `emap_len`); the store
  is in bounds (`out_len`); the loop body writes nothing (`fun _ => ∅`, `iterW_empty`);
- the loop invariant: after `t` trips, lane `l` of `acc` is the number of positions `j = x·B + l`,
  `x < t`, `j < numel`, with `routed d j = e`, read through `Pre`'s `Buf.Holds`; the masked-off lanes
  add `0` (the loaded `-1` of `m1` matches no expert `e ≥ 0`, and with a map the `-1` from the
  disabled gather does not either). No lane or sum wraps: each is at most `numel < 2^31`
  (`TVA.Lemmas.Int`, `I32.toInt_ofNat`);
- the programs' write elements are disjoint, and no program writes what any program reads
  (`out_ids`, `out_emap`);
- the stored sums give part 1 (`countOf` as a sum over lanes of the per-lane counts), and the bytes no
  program writes give part 2.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `mk_ok` and the
`setup_*_ok` theorems; `bytes_disj` (line 279) can be reused.

## Specification

`Post d init fin`:
- for every local expert `e < num_local_experts`, element `e` of `expert_num_tokens` is an `int32`
  equal to `countOf d e`, the number of positions `j < topk_ids.numel()` with `routed d j = e`;
  `routed d j` is, from the host's inputs, the entry `topk_ids.view(-1)[j]` itself without an expert
  map, and with one `expert_map[entry]` for an entry `≥ 0` and `-1` for a negative entry. Exact: no
  tolerance;
- frame: every byte outside the output's first `num_local_experts` elements is unchanged.

The reference is computed from `Inst`'s contents (`idsVal`, `mapVal`, which `Pre` ties to memory with
`Buf.Holds`), by position, independently of how the kernel scans or indexes.

## Conditions

`Inst`: the ids' dtype, `hasMap`, `numLocal`, `numGlobal`, `numel`, `BLOCK_SIZE`, the buffers
`ids`, `emap`, `out` and the contents `idsVal`, `mapVal`.

`Inst.ok`: ids `int32` or `int64`; `BLOCK_SIZE` a power of two `≤ 2^20` (the wrapper's is
`next_power_of_2(min(numel, 1024))`); `numel + BLOCK_SIZE ≤ 2^31`; buffer lengths (`numel` ids,
`numGlobal` map entries with a map, `numLocal` outputs); `numLocal < 2^31`; the output shares no byte
with `topk_ids` or (with a map) `expert_map`.

`Pre`: the buffers valid and holding the contents; with a map, every entry `< numGlobal` (a global
expert id or negative). Without a map every value is covered; every `int32` map entry is covered.

The launch: `Launch.grid1 numLocal [[BLOCK_SIZE]]`, fusion on (default), `debug = false`. Part (A) of
`Compiler.Faithful`'s scope: the wrapper launches with `num_warps=4`, `num_stages=3` only (no other
option, no heuristics, no descriptors, no precompiled kernel). The claim relies on
part (B), the environment.

## Assumptions

One `[rule]` field, `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`.
The kernel is integer-only: no device fact. No `[kernel]` field.

## Coverage

68 instances `setup_*` with proofs `setup_*_ok`: every launch of the host's test (lines 146–180):
the 21 vLLM shapes (1–7317 tokens × top-2/6/8, 64 experts, `ep_size` 1/2/4, `int64`), the six `numel`
cases (1 … 8191, `int64`), the six model configurations (`int32`: DeepSeek-V3 linear and
round-robin, Qwen3, an uneven 128/3 split, Mixtral, one token), each at the ranks `0` and
`ep_size − 1`, with the map and without it (`local_ids`), `num_local_experts` as `make_expert_map`
gives it, `BLOCK_SIZE` the wrapper's. Both id dtypes and both `HAS_EXPERT_MAP` paths are covered.

## Tolerances

None: every clause is exact.

## Exclusions

- **`numel + BLOCK_SIZE > 2^31`** — a limit of the kernel. `tl.cdiv(topk_numel, BLOCK_SIZE)` is the
  `int32` `(topk_numel + BLOCK_SIZE − 1) // BLOCK_SIZE`, which wraps negative for
  `numel ∈ [2^31 − BLOCK_SIZE + 1, 2^31)`: the loop does not run and every count is `0`. First launch
  that hits it: `numel = 2^31 − 1023` with `BLOCK_SIZE = 1024`. At `numel ≥ 2^31` Triton passes
  `topk_numel` as `int64` (another specialisation), and the `int32` accumulator could wrap for an
  expert with `≥ 2^31` entries. Real use does not reach it (tens of thousands of tokens × top-8, an
  8 GiB `int32` `topk_ids` at the limit).
- **Dtypes other than `int32`/`int64`** — outside `SETUP.md`'s use (unsigned dtypes are rejected by
  the wrapper's assert; `int8`/`int16` ids are not used).
- **With a map, entries `≥ num_global_experts`** — outside real use (the routers emit global ids or
  `-1`); the gather would read past the map.
- **`numel = 0`** is covered by `Inst.ok` but the wrapper computes `BLOCK_SIZE = 0`, which
  `tl.arange` rejects (no launch).

## Modelling decisions

- `range(tl.cdiv(…))` with one argument evaluates its argument twice: two `tl.idiv` lines, the
  second the bound.
- At the value `1`, `topk_numel` and `num_experts` are compile-time constants; every use makes them
  `int32` tensors first (line 61's `cdiv` is then Python's, also `1`), so the values are the
  model's.
- `int64` ids: `expert_ids == curr_expert` sign-extends the program id; `expert_ids >= 0` promotes
  the `int32` `0`; `other=-1` is the sign-extended `-1` (`m1`).
- `tl.sum(acc)` on `int32` is `Tensor.foldAll (+)` (order-independent wrapping sum).

## Library requests

None.

## Checks

- Builds: `lake env lean ../Claim.lean`, no errors or warnings, no
  `sorry`/`axiom`; `consistent`, `satisfiable`, `setup_*_ok` use only `propext`,
  `Classical.choice`, `Quot.sound`.
- `check_gpu.py` (H200): 126 launches (the host's 112 and 14 edges: arbitrary `int32`/`int64`
  values without a map, `int64` values whose low 32 bits are a local id, arbitrary map entries,
  `numel = 1`, one expert, `2^26` entries, 4096 programs): every count exact, inputs unchanged, and
  in a guarded relaunch every byte outside the counts unchanged.
