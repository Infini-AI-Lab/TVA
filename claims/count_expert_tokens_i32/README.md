# `count_expert_tokens_i32`

**The kernel.** vLLM's `_count_expert_num_tokens` (`source/count_expert_tokens.py`, kernel lines
45–74; host wrapper `count_expert_num_tokens` lines 78–107). A 1-D grid of `num_local_experts`
programs; program `e` scans all of `topk_ids` (flat, `int32` or `int64`) in blocks of `BLOCK_SIZE`,
maps each entry through `expert_map` when `HAS_EXPERT_MAP` (a negative entry gives `-1`), counts the
entries equal to `e` in an `int32` accumulator, and stores `tl.sum(acc)` at `expert_num_tokens[e]`.
The claim is `Claims.CountExpertTokens.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and in the
final memory:
- for every `e < num_local_experts`, element `e` of `expert_num_tokens` is an `int32` equal to
  `countOf d e`, the number of positions `j < topk_ids.numel()` with `routed d j = e`; `routed d j`
  is, from the host's inputs, the entry itself without an expert map, and with one
  `expert_map[entry]` for an entry `≥ 0` and `-1` for a negative entry. Exact: no tolerance;
- every byte outside the output's first `num_local_experts` elements is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `topk_ids` is `int32` or `int64`;
- `BLOCK_SIZE` a power of two `≤ 2^20` (the wrapper's is `next_power_of_2(min(numel, 1024))`);
- `numel + BLOCK_SIZE ≤ 2^31`; `num_local_experts < 2^31`;
- the buffers hold `numel` ids, `numGlobal` map entries (with a map) and `num_local_experts`
  outputs, are valid and hold the instance's contents in the initial memory;
- the output shares no byte with `topk_ids` or (with a map) `expert_map`;
- with a map, every entry is below `numGlobal` (a global expert id, or negative).

Without a map every entry value is covered; every `int32` map entry is covered.

**The launches covered** (`SETUP.md`, `setup_*_ok`): the 68 launches of the host's test (lines
146–180): 21 vLLM shapes (`int64`), six `numel` cases (`int64`) and six model configurations
(`int32`), each with the map and without it, `num_local_experts` as `make_expert_map` gives it, and
the wrapper's `BLOCK_SIZE`.

**Device facts.** None: the kernel is integer-only.

## Exclusions

- **`numel + BLOCK_SIZE > 2^31`**: a limit of the kernel. `tl.cdiv(topk_numel, BLOCK_SIZE)` is the
  `int32` `(topk_numel + BLOCK_SIZE − 1) // BLOCK_SIZE`, which wraps negative for
  `numel ∈ [2^31 − BLOCK_SIZE + 1, 2^31)`: the loop does not run and every count is `0` (first
  launch: `numel = 2^31 − 1023` with `BLOCK_SIZE = 1024`). At `numel ≥ 2^31` Triton passes
  `topk_numel` as `int64`. Real use does not reach it.
- **Dtypes other than `int32`/`int64`**: unsigned dtypes are rejected by the wrapper's assert;
  `int8`/`int16` ids are not used.
- **With a map, entries `≥ num_global_experts`**: outside real use (the routers emit global ids or
  `-1`); the gather would read past the map.
- **`numel = 0`** is covered by `Inst.ok`, but the wrapper computes `BLOCK_SIZE = 0`, which
  `tl.arange` rejects (no launch).

## Modelling decisions

- `range(tl.cdiv(…))` with one argument evaluates its argument twice: two `tl.idiv` lines, the
  second the bound.
- At the value `1`, `topk_numel` and `num_experts` are compile-time constants; every use makes them
  `int32` tensors first (line 61's `cdiv` is then Python's, also `1`), so the values are the
  model's.
- `int64` ids: `expert_ids == curr_expert` sign-extends the program id; `expert_ids >= 0` promotes
  the `int32` `0`; `other=-1` is the sign-extended `-1` (`m1`).
- `tl.sum(acc)` on `int32` is `Tensor.foldAll (+)` (an order-independent wrapping sum).

`check_gpu.py` (one H200) checks `Post` exactly on the host's launches and on edges of `Inst.ok`.
