# `route_histogram_i32`: prepared claim

SGLang's `_route_histogram_kernel` (`source/route_histogram.py`, kernel
lines 109–144 with the helpers `route_bucket_ids` 62–89 and `_add_counts` 92–106; host
`route_histogram` 154–185), on both of its paths: `BINS == 0` (one `tl.atomic_add` of `1` per
pair) and the `tl.histogram` path (`BINS > 0`). `Claim.lean`, namespace
`Claims.RouteHistogram`; the claim to prove is `Statement`.

## Specification

`Post d init fin`:
- for every bucket `b < NUM_BUCKETS`, `counts[b] = init[b] + pairsIn d b` (the wrapping `int32`
  sum; with the host's zeroed `counts`, exactly the count), where `pairsIn d b` is the number of
  pairs `p < num_pairs` with `bucketOf d p = b`;
- `bucketOf d p`, from the host's inputs (`Inst`'s contents) as the source's docstring and `reference` define it: slot
  `token_slots[p / WIDTH]`, group `group_ids.view(-1)[p]`; live when `0 ≤ slot < MAX_LORAS` and,
  with groups, `group ≥ 0` and (unless `GROUPS_PER_SLOT = 1`) `group < GROUPS_PER_SLOT`; the
  bucket is `slot·GROUPS_PER_SLOT + group` with groups and more than one group per slot, `slot`
  otherwise, and `NUM_BUCKETS − 1` when not live;
- frame: every byte outside `counts`' first `NUM_BUCKETS` elements is unchanged.

The atomics are modelled as `tl.atomic_add` (`acq_rel`, `gpu`), integer adds whose order does not
matter to `Post`: one per nonzero bin below `NUM_BUCKETS` on the `tl.histogram` path, one of the
`int32` `1` per pair (enabled by `pair_mask`) at its bucket with `BINS == 0`.

## Conditions

`Inst`: `num_pairs`, `WIDTH`, `GROUPS_PER_SLOT`, `MAX_LORAS`, `NUM_BUCKETS`, `BLOCK`, `BINS`,
`HAS_GROUPS`, the buffers and their contents (`tsVal`, `giVal`, `cntVal`).

`Inst.ok`: `BLOCK` a power of two `≤ 2^20` (any, not only the host's `512`); `BINS = 0`, or `BINS`
a power of two `≤ 2^20` with `NUM_BUCKETS < BINS` (any, not only the host's
`1 << NUM_BUCKETS.bit_length()`, and either path at any `num_pairs` and bucket count, not only
where the host's `_histogram_bins` takes it); `NUM_BUCKETS = GROUPS_PER_SLOT·MAX_LORAS + 1 < 2^31`
(the host's check); `1 ≤ GROUPS_PER_SLOT < 2^31`; `1 ≤ WIDTH < 2^31`; `num_pairs < 2^31`; `token_slots` holds
`cdiv(num_pairs, WIDTH)` tokens, `group_ids` (with groups) `num_pairs` pairs, `counts`
`NUM_BUCKETS` cells; `counts` shares no byte with the inputs. (`num_pairs < 2^31` is enough for
every program's `int32` pair index: `BLOCK` divides `2^31`.)

`Pre`: buffers valid and holding the contents. Every `int32` value is covered in `token_slots`,
`group_ids` and the initial `counts`.

The launch: `Launch.grid1 (cdiv(max(num_pairs, 1), BLOCK))` with the block shapes `[BLOCK]` and,
on the `tl.histogram` path, `[BINS]`; `debug = false`,
`USE_PDL = False`. Part (A) of `Compiler.Faithful`'s scope: `num_warps=8`, `num_stages=3` only. The
claim relies on part (B).

## Assumptions

One `[rule]` field, `faithful` (`Compiler.Faithful` as the library states it). Integer-only kernel:
no device fact; no `[kernel]` field.

## Coverage

`setup_*` / `setup_*_ok`, with `BLOCK = 512` and the host's `BINS`:
- the `tl.histogram` path: the host's nine test launches on it (dense 16384, 16385, 100000, 32768
  tokens; MoE per expert 2048×8, 8193×2, 16384×8; shared outer 3000×6; per-head 1024×32) and a
  dense `2^24`-token route with 64 adapters;
- `BINS == 0`: the host's two test launches on it (1000 dense tokens; 4096×8 pairs with 513
  buckets), MoE LoRA per expert with 128 local experts × 4 adapters at 16384 tokens × top-8
  (513 buckets) and 256 × 8 at 65536 × top-8 (2049 buckets), `384 × 32 ≥ 8192` route buckets at
  96 tokens × top-8 (12289 buckets), and a shared-outer route of 2000 × top-6 pairs (fewer than
  16384).

## Tolerances

None: exact.

## Exclusions

- **`num_pairs ≥ 2^31`** — a limit of the kernel. `pair_ids = tl.program_id(0) * BLOCK +
  tl.arange(0, BLOCK)` is `int32` and wraps for the programs past `2^31 / BLOCK`; `num_pairs`
  itself becomes an `int64` argument. First launch that hits it: `num_pairs = 2^31` (an 8 GiB
  `int32` `group_ids`). Real use does not reach it.
- **`NUM_BUCKETS ≥ 2^31`** — outside real use: the host raises before the launch.
- **`GROUPS_PER_SLOT = 0`, `GROUPS_PER_SLOT ≥ 2^31`** — outside real use (the host's groups per
  slot is `1`, a number of experts or heads). `0` would also be a kernel limit without groups:
  `NUM_BUCKETS = 1` while a live slot `≥ 1` is a histogram value past `BINS`, or with `BINS == 0` an
  atomic past `counts`.
- **`WIDTH = 0`, `WIDTH ≥ 2^31`** — outside real use (`group_ids.shape[1]` is a top-k).
- `USE_PDL = True` (programmatic dependent launch) — outside TVA's launches (`SETUP.md`).

## Modelling decisions

- The helpers are inlined Lean functions; the compile-time `if`s on `HAS_GROUPS`,
  `GROUPS_PER_SLOT == 1` and `BINS == 0` are Lean `if`s on the constexprs; `if USE_PDL:` is not
  modelled (`config`: `False` on every covered launch).
- The `1` of line 102 is an `int32` constant broadcast to `[BLOCK]` (`tl.full`), the pointer's
  dtype already.
- `slots.to(tl.int32)` and `groups.to(tl.int32)` are identities (`castInt`).
- At `num_pairs = 1` the argument is a compile-time constant; its one use makes it an `int32`
  tensor, the same values.
- `Post` holds for any initial `counts` (wrapping add), wider than the host's zeroed buffer.

## Library requests

None.

## Checks

- Builds with no errors or warnings, no `sorry`/`axiom`; `consistent`, `satisfiable`, `setup_*_ok`
  use only the three standard axioms.
- `check_gpu.py` (H200): 30 launches. The host's 11 test launches and 4 real routes with
  `BINS == 0` (MoE LoRA 128 × 4 and 256 × 8 experts × adapters, 384 × 32 at 768 pairs, 12000
  shared-outer pairs), and 15 edges: arbitrary `int32` slots and groups on both paths, 511 buckets /
  `BINS = 512`, one group per slot with arbitrary groups, `2^26` pairs, `BINS > 0` below 16384
  pairs, `BINS` above the host's, `BINS == 0` at `2^25` pairs and 9 buckets, `BLOCK` 128 and 2048,
  one pair and no pair on both paths. The host's counts are exact; a guarded relaunch with random
  initial counts (one near `2^31 − 1`, so the sum wraps) gives exactly `init + count` and leaves the
  guards and inputs unchanged; on every launch four wrong outputs (nothing added, one pair in the
  wrong bucket, the last program's pairs missing, the invalid pairs not counted) fail `Post`.
