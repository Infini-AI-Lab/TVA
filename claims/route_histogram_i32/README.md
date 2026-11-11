# `route_histogram_i32`

**The kernel.** SGLang's `_route_histogram_kernel` (`source/route_histogram.py`, kernel lines
109–144 with the helpers `route_bucket_ids` 62–89 and `_add_counts` 92–106, inlined; host
`route_histogram` 154–185). A 1-D grid of `cdiv(max(num_pairs, 1), BLOCK)` programs; program `i`
takes the pairs `i·BLOCK + [0, BLOCK)`, masked by `< num_pairs`, computes each pair's bucket from
its token's slot (`token_slots[p / WIDTH]`) and, with groups, its group (`group_ids.view(-1)[p]`),
and adds it into `counts`: with `BINS == 0`, one `tl.atomic_add` of `1` per pair; with `BINS > 0`, a
`tl.histogram` over `BINS` bins and one `tl.atomic_add` per nonzero bin below `NUM_BUCKETS`. The
claim is `Claims.RouteHistogram.claim` in `Claim.lean`.

**What is claimed.** On both paths, every covered launch ends normally (no fault, no hang, no
race), and in the final memory:
- each element `b < NUM_BUCKETS` of `counts` is its initial value plus `pairsIn d b`, the number of
  pairs `p < num_pairs` whose bucket `bucketOf d p` is `b` (the wrapping `int32` sum; with the
  host's zeroed `counts`, exactly the count). `bucketOf` is stated from the host's inputs: live when
  `0 ≤ slot < MAX_LORAS` and, with groups, `group ≥ 0` and (unless `GROUPS_PER_SLOT = 1`)
  `group < GROUPS_PER_SLOT`; the bucket is `slot·GROUPS_PER_SLOT + group` with groups and more than
  one group per slot, `slot` otherwise, and `NUM_BUCKETS − 1` when not live;
- every byte outside the first `NUM_BUCKETS` elements of `counts` is unchanged.

It is exact: integers only.

**The conditions** (`Inst.ok`, `Pre`):
- `BLOCK` a power of two `≤ 2^20` (any, not only the host's `512`);
- `BINS = 0`, or `BINS` a power of two `≤ 2^20` with `NUM_BUCKETS < BINS` (either path at any
  `num_pairs` and bucket count, not only where the host's `_histogram_bins` takes it);
- `NUM_BUCKETS = GROUPS_PER_SLOT·MAX_LORAS + 1 < 2^31` (the host's check);
  `1 ≤ GROUPS_PER_SLOT < 2^31`; `1 ≤ WIDTH < 2^31`; `num_pairs < 2^31`;
- `token_slots` holds `cdiv(num_pairs, WIDTH)` tokens, `group_ids` (with groups) `num_pairs` pairs,
  `counts` `NUM_BUCKETS` cells; all are valid and hold the instance's contents in the initial
  memory; `counts` shares no byte with the inputs.

Every `int32` value is covered in `token_slots`, `group_ids` and the initial `counts`.
`USE_PDL = False` on every covered launch (`if USE_PDL:` is not modelled).

**The launches covered** (`SETUP.md`, `setup_*_ok`), all with `BLOCK = 512` and the host's `BINS`:
on the `tl.histogram` path, the host's nine test launches on it and a dense `2^24`-token route with
64 adapters; with `BINS == 0`, the host's two test launches on it and four real MoE LoRA /
shared-outer routes.

**Device facts.** None: the kernel is integer-only.

## Exclusions

- **`num_pairs ≥ 2^31`**: a limit of the kernel. `pair_ids = tl.program_id(0) * BLOCK +
  tl.arange(0, BLOCK)` is `int32` and wraps; `num_pairs` itself becomes an `int64` argument.
- **`NUM_BUCKETS ≥ 2^31`**: the host raises before the launch.
- **`GROUPS_PER_SLOT = 0` or `≥ 2^31`**: outside real use (the host's groups per slot is `1`, a
  number of experts or heads). `0` would also be a kernel limit without groups: `NUM_BUCKETS = 1`
  while a live slot `≥ 1` is a histogram value past `BINS`, or with `BINS == 0` an atomic past
  `counts`.
- **`WIDTH = 0` or `≥ 2^31`**: outside real use (`group_ids.shape[1]` is a top-k).
- **`USE_PDL = True`**: outside the covered launches.

`check_gpu.py` (one H200) checks `Post` on the host's launches, real routes and edges of `Inst.ok`.
