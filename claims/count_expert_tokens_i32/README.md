# `count_expert_tokens_i32`: prepared claim

vLLM's `_count_expert_num_tokens` (`source/count_expert_tokens.py`,
kernel lines 45–74, host `count_expert_num_tokens` lines 78–107). `Claim.lean`, namespace
`Claims.CountExpertTokens`; the claim to prove is `Statement : ∀ K P, Assumptions K P → Spec K`.

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
