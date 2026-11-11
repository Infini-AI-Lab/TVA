# `kv_cache_write_bf16`

**The kernel.** vLLM's `reshape_and_cache_kernel_flash` (`source/reshape_and_cache_flash.py`,
kernel lines 73-195; host `reshape_and_cache_flash` lines 198-298), which writes each token's keys
and values into the paged KV cache at the token's slot. A grid of `num_tokens × cdiv(n, TILE_SIZE)`
programs (`n = num_heads·head_size`); program `(t, i)` reads `slot_mapping[t]`, returns on a
negative slot (padding), and otherwise copies tile `i` of token `t`'s keys and values to the cache
elements `(s // block_size, s % block_size, h, e)`. Three configurations (`Mode`): `bfloat16`
inputs and caches; `bfloat16` inputs and `float8e4nv` caches (divided by the `float32` scales
`k_scale`, `v_scale` and rounded to fp8); `float8e4nv` inputs and caches. The caches are two views
of one allocation (vLLM's flash and Triton-attention layouts). The configuration is `SETUP.md`'s:
`USE_HEAD_MAJOR_LAYOUT = False`, `INTERLEAVED_V_PACK_FACTOR = 0`, `x = 1`,
`dim_stride_k = dim_stride_v = 0`. The claim is `Claims.KvCacheWrite.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and:
- for every token `t` with a slot `s ≥ 0`, every head `h` and component `e`, the key cache's
  element `(s // block_size, s % block_size, h, e)` holds what `Stored` says of `key[t, h, e]`: the
  value itself, bit for bit, when the cache has the inputs' dtype; with `bfloat16` inputs and an
  fp8 cache, when `x` and the scale `s` are finite, `s ≠ 0` and `|x/s| ≤ 2^126`, the saturating
  fp8 rounding to nearest even of a finite `float32` within `2^-22·|x/s| + 2^-148` of `x/s`; the
  value cache's element likewise of `value[t, h, e]` with `v_scale`;
- every byte outside the written cache elements is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `TILE_SIZE` is a power of two, at most `2^20`;
- at most `2^22` tokens; the grid within CUDA's limits (`cdiv(n, TILE_SIZE) < 2^16`, fewer than
  `2^31` programs);
- `n < 2^31` and `head_size < 2^31`; the strides, `block_size` and the buffers' lengths below
  `2^63`;
- the buffers hold every element the kernel touches; the cache view is non-overlapping;
- the caches' storage shares no byte with `key`, `value`, `slot_mapping` (and, with an fp8 cache
  and `bfloat16` inputs, the scales); no element of the key cache shares a byte with one of the
  value cache (their storage ranges may overlap);
- the buffers are valid; every slot is negative or a slot of the cache, and the non-negative slots
  of two tokens differ. Any input values and scales are covered.

**The launches covered** (`SETUP.md`, `setup_*_ok`): vLLM's test cases (21 tokens at 8 × 64, 80,
256; one token; padding), serving shapes (8192 tokens at 8 × 128, 4096 tokens at 32 × 128 and
32 × 256) in both layouts, and the fp8 cache and fp8 input configurations at the test's and the
serving shapes. `mk_ok` covers every instance of vLLM's buffers with at most `2^22` tokens, in each
configuration and both layouts.

**Device facts** used (all in `Device P`), only with an fp8 cache and `bfloat16` inputs:
`Device.DivFull` (`/` on `float32`: relative `2^-22`, absolute `2^-148`, for quotients up to
`2^126`), `Device.CvtSatF32E4m3` (the conversion `float32 → float8e4nv` is the saturating rounding
to nearest even), and the totality facts for `/` and that conversion.

## Tolerances

With the inputs' dtype in the cache: none, `Post` is bit-exact. With an fp8 cache and `bfloat16`
inputs, `Stored` composes the two facts on the element's path: the load and the widening to
`float32` are exact, `DivFull` gives a finite `r` within `2^-22·|x/s| + 2^-148` of `x/s`, and the
stored byte is `cvtSat .rn r`, one of the at most two fp8 values next to `x/s` (`±448` beyond the
range). On vLLM's fp8 test this allows at most half an fp8 ulp in fp8's normal range; a cache that
skips the scale, multiplies by it, rounds toward zero, or uses `k_scale` for the values fails
`Post`.

## Exclusions

- **`n ≥ 2^31`**: a kernel limit (the tile positions and the mask are `int32`); real models have
  `n ≤ 8192`.
- **`head_size ≥ 2^31`**: outside real use; Triton rejects the `int32` `//` and `%` of lines
  150-151.
- **Grids beyond CUDA's limits or with `2^31` programs or more**: a limit of the launch; real
  batches are far below.
- **More than `2^22` tokens**: real shapes only.
- **Strides, `block_size` or tensors of `2^63` or more**: outside real use (the `int64` offsets
  would wrap).
- **A self-aliasing cache view, a key and a value cache sharing an element, a cache overlapping an
  input, two tokens with the same slot, a slot past the cache**: outside real use; the kernel would
  race or write outside the cache.
- **Non-finite inputs or scales, a zero scale, `|x/s| > 2^126`** (fp8 cache): the element is only
  known to hold an fp8 value.
- **Other dtypes, a `float8e5` cache, fp8 inputs into a 16-bit cache, per-head scales, the
  head-major layout**: not part of `SETUP.md`.

`check_gpu.py` (one H200) checks `Post` on the setup launches and edge cases.
