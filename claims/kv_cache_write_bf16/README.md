# Prepared claim: `kv_cache_write_bf16`

Kernel: `source/reshape_and_cache_flash.py`, vLLM's
`reshape_and_cache_kernel_flash` (lines 73-195), launched by `reshape_and_cache_flash` (lines
198-298). Claim: `Claim.lean`, namespace `Claims.KvCacheWrite`; the statement to prove is
`Statement : ∀ K P, Assumptions K P → Spec K`. The model map is `model-map.json`.

One program per (token, tile of `TILE_SIZE` of its `n = num_heads * head_size` values): a token
with a slot `s ≥ 0` has its key and value rows stored at block `s // block_size`, row
`s % block_size` of the key and value caches; a padding token (negative slot) writes nothing. The
claim covers SETUP.md's path (`USE_HEAD_MAJOR_LAYOUT = False`, `INTERLEAVED_V_PACK_FACTOR = 0`,
`x = 1`, `dim_stride_k = dim_stride_v = 0`) in **both** of vLLM's cache layouts, which are two
views of one allocation whose byte ranges overlap, in three configurations (`Mode`):

| `Mode` | inputs | caches | `kv_cache_dtype`, `FP8_KV_CACHE` | lines 164-183 |
|---|---|---|---|---|
| `bf16` | `bfloat16` | `bfloat16` | `"auto"`, `False` | `key_tile = key_load` (169, 183) |
| `fp8` | `bfloat16` | `float8e4nv` | `"fp8"`, `True` | `is_fp8()` false: `key_load / tl.load(k_scale)` (167, 181), stored with the conversion to fp8 |
| `fp8in` | `float8e4nv` | `float8e4nv` | `"fp8"`, `True` | `is_fp8()` true: `key_load` (167, 177) |

## Specification

`Spec K := Claim.Meets Inst.ok Pre Post K`. `Post d init fin`, with `s_t` the initial
`slot_mapping[t]` and `kvIdx(b, r, h, e) = b * block_stride + r * page_stride + h * head_stride +
e` (the key cache's strides, which the value cache shares):

- for every token `t < num_tokens` with `s_t ≥ 0`, every head `h < num_heads` and component
  `e < head_size`, element `kvIdx(s_t // block_size, s_t % block_size, h, e)` of the key cache
  holds a value `v` that `Stored` relates to the initial element `x = key[t, h, e]` (element
  `t * key.stride(0) + h * head_size + e` of `key`) with the initial `k_scale`, and the same
  element of the value cache a value related to `value[t, h, e]` with `v_scale`;
- every byte outside those elements of the two caches is unchanged (the rest of the shared
  allocation included).

`Stored`:
- `Mode.bf16`, `Mode.fp8in`: `v = x`, bit for bit (a copy: no tolerance);
- `Mode.fp8`: when `x` and the scale `s` are finite, `s ≠ 0` and `|x / s| ≤ 2^126`, there is a
  finite `float32` `r` with `|r − x/s| ≤ 2^-22·|x/s| + 2^-148` and `v = cvtSat .rn r`, the
  saturating rounding to nearest even into `float8e4nv` (`±448` beyond the range). So `v` is one of
  the at most two fp8 values next to `x / s` (one, unless `x / s` lies within `2^-22` of a midpoint
  between two fp8 values). With a NaN or an infinite input or scale, a zero scale or `|x/s| > 2^126`,
  the element holds some fp8 value (what the device gives is not stated by the library there).

The reference is the host's `reference` (`cache[s // bs, s % bs] = key[t]`, every other element
unchanged), from the host's inputs and the caches' own strides, and with an fp8 cache the exact
quotient `key[t] / k_scale`, `value[t] / v_scale`. A kernel that leaves the scale out, multiplies
by it, rounds toward zero or divides the value by `k_scale` fails `Post` (Checks).

## Conditions

`Inst`: the sizes (`num_tokens`, `num_heads`, `head_size`, `block_size`, `num_blocks`,
`TILE_SIZE`), the strides (`key.stride(0)`, `value.stride(0)`, the cache's `stride(0..2)`), and the
storage of `key`, `value`, both caches, `slot_mapping`, `k_scale`, `v_scale`.

`Inst.ok`:
- `TILE_SIZE` a power of two `≤ 2^20` (the host's `min(2048, next_power_of_2(n))` is one);
- the grid `(num_tokens, cdiv(n, TILE_SIZE))` within CUDA's limits, fewer than `2^31` programs;
- `n < 2^31` (the mask's constant is an `int32`, the tile positions do not wrap; a limit of the
  kernel, Exclusions); `head_size < 2^31` (the divisor of lines 150-151 is an `int32` constant,
  also when `num_heads = 0`); `block_size < 2^63`; the strides `< 2^63`;
- `slot_mapping` has `num_tokens` elements; `key` and `value` hold every element
  `t * stride + p` (`t < num_tokens`, `p < n`); both caches hold every element of the
  `[num_blocks, block_size, num_heads, head_size]` view;
- `kv_inj`: the view is non-overlapping (distinct indices, distinct elements);
- the caches share no byte with `key`, `value`, `slot_mapping`;
- `scales` (`Mode.fp8`, where the kernel reads them): `k_scale` and `v_scale` hold an element and
  share no byte with the caches (vLLM's `layer._k_scale`, `_v_scale` are tensors of their own);
- `disj_caches`: no element of the key cache's view shares a byte with an element of the value
  cache's view (the caches' whole buffers may overlap: in both vLLM layouts the two views share
  one allocation).

`Pre d init`: `key`, `value`, both caches and `slot_mapping` valid, and in `Mode.fp8` `k_scale`
and `v_scale`; every slot negative (padding) or below `num_blocks * block_size`; the non-negative
slots of two tokens differ (vLLM gives each token its own slot). Any input values and scales.

`block_size = 0` and `num_blocks = 0` are covered: the cache has no slot, every slot is padding,
and `Post` holds. `n = num_heads * head_size = 0` is not: the host's `TILE_SIZE =
next_power_of_2(0)` is `0`, `tl.arange(0, 0)` does not compile, and the launch is rejected on the
host (outside the claim: `Inst.ok.tile_pow2`).

`launch d`: grid `(num_tokens, cdiv(n, TILE_SIZE), 1)`, block `[TILE_SIZE]`, Triton's default
`enable_fp_fusion`; `num_warps = 16`, `num_stages = 10` change no result; `num_ctas = 1`, no
`launch_pdl`, `debug = false`.

`Compiler.Faithful`'s scope, part (A), checked against the source: (4) the launch passes only the
kernel's parameters, `num_warps` and `num_stages` (the grid is a `lambda` of `meta`, evaluated by
Triton with the launch's constexprs); no `@triton.heuristics` or `@triton.autotune`; (5) no
descriptor; (6) no `CompiledKernel`; (8) no host `TensorDescriptor`. Part (B), the environment,
is assumed and not checked.

**The supported subset.** Lines 164-183 use, besides the forms of the other lines, a conditional
expression and a compile-time `if` on `key_load.dtype.is_fp8()`, `value_load.dtype.is_fp8()`
(SEMANTICS.md, "A dtype's predicates": compile-time `bool`s of the loads' dtype; only the side
they pick is visited), the scalar `tl.load(k_scale)`, the float `/` of a `bfloat16` tile by a
`float32` scalar (computed in `float32`, "`/` on floats"), and the `tl.store` of a `float32` tile
to a `float8e4nv` pointer, which converts by `tl.cvt8 x .rn` (SEMANTICS.md, "`tl.store`").

## Assumptions

Five `[rule]` fields, each cited as the library states it (`docs/PROVER_GUIDE.md` section 3,
[num.md](../../docs/rules/num.md)): `faithful` (`Compiler.Faithful`); for `Mode.fp8`,
`div_total` (`Rules.Num.DivTotal P .binary32`: `/` always has a result, the masked lanes'
unspecified values included), `div_full` (`Rules.Num.DivFull`), `cvt_total`
(`Rules.Num.CvtTotal P .binary32 .e4m3 .rn`) and `cvt_sat` (`Rules.Num.CvtSatF32E4m3`). No
`[kernel]` field. `consistent` instantiates them at `Platform.sm90ref`.

## Coverage

`Claim.lean` builds vLLM's buffers (`mk m L T nh hs bs nb TILE`: configuration `m`; `key`,
`value` as `qkv[:, 1]`, `qkv[:, 2]` of a `[T, 3, nh, hs]` tensor; the caches as `make_caches` lays
them out, layout `L`) and proves `mk_ok`: every such instance is covered in each configuration and
both layouts, the non-overlap (`kv_inj`) and
the element-wise disjointness of the two caches (`disj_caches`) proved for any sizes (mixed-radix
uniqueness, `mr`). The host's test cases and SETUP.md's serving shapes:

| Launch (tokens, heads × size, block, blocks, layout) | `TILE_SIZE` | Instance |
|---|---|---|
| 21, 8 × 64, 16, 512, flash | 512 | `setup_t21_h64_flash` |
| 21, 8 × 80, 8, 512, flash | 1024 | `setup_t21_h80_flash` |
| 21, 8 × 256, 32, 5000, flash | 2048 | `setup_t21_h256_flash` |
| 1, 8 × 128, 16, 64, flash | 1024 | `setup_t1_flash` |
| 1 (padding), 8 × 128, 16, 64, triton_attn | 1024 | `setup_t1_triton` |
| 256 (64 padding), 8 × 128, 16, 64, triton_attn | 1024 | `setup_t256_triton` |
| 300 (17 padding), 4 × 96, 64, 20, triton_attn | 512 | `setup_t300_h96_triton` |
| 4096 (100 padding), 32 × 128 (two tiles), 16, 300, flash | 2048 | `setup_t4096_h32_flash` |
| serving: 8192, 8 × 128, 16, 65536, flash and triton_attn | 1024 | `setup_serving_flash`, `setup_serving_triton` |
| serving: 4096, 32 × 256 (four tiles), 32, 4096, flash | 2048 | `setup_serving_h32x256` |
| fp8 cache: vLLM's fp8 test (21, 8 × 64 / 80 / 256, 16 / 8 / 32, 512 / 5000, flash) | 512 / 1024 / 2048 | `setup_fp8_t21_h64_flash`, `setup_fp8_t21_h80_flash`, `setup_fp8_t21_h256_flash` |
| fp8 cache, serving: 8192, 8 × 128, 16, 65536, flash and triton_attn | 1024 | `setup_fp8_serving_flash`, `setup_fp8_serving_triton` |
| fp8 inputs and cache: 8192 / 256, 8 × 128, 16, 65536 / 64, flash / triton_attn | 1024 | `setup_fp8in_flash`, `setup_fp8in_triton` |

Any token stride, any number of tokens and blocks, heads 1 to 32 and beyond, head sizes not a
power of two, any block size and any padding are covered.

## Tolerances

`Mode.bf16`, `Mode.fp8in`: none, `Post` is bit-exact.

`Mode.fp8`: `Stored` composes the two facts on the element's path and nothing else. The load
and the widening `bfloat16 → float32` are exact, so the division's operands are `x` and `s`;
`DivFull` (`2^-22`, `2^-148`, `|x/s| ≤ 2^126`) gives a finite `r` within `2^-22·|x/s| + 2^-148`
of `x/s`; `CvtSatF32E4m3` makes the stored byte `cvtSat .rn r` (up to NaN payload, and `cvtSat`
of a finite `r` is finite, so equal). As a bound against the exact quotient, `FloatV.cvtSat_rn_err`
(`float8e4nv`: `u = 2^-4`, `η = 2^-10`) and the clamp's 1-Lipschitz property give
`|v − clamp(x/s)| ≤ 2^-4·|clamp(x/s)| + 2^-10 + (1 + 2^-4)·(2^-22·|x/s| + 2^-148)`, `clamp` to
`[−448, 448]`; `Post` is stronger: `v` is one of the at most two roundings. **Meaningful**: on
vLLM's fp8 test (`randn` inputs, `k_scale = key.amax() / 64`), the largest deviation `Post` allows
from `clamp(x/s)`, over `|v|`, is `0.021` at the median and at most `0.0625` (half an fp8 ulp,
`2^-4`) in fp8's normal range (`|x/s| ≥ 2^-6`); below it, in the subnormal range, the median is
`0.06` and the worst `0.5` (an absolute `2^-10`). This is the fp8 format's own rounding: a cache
that skips the scale, multiplies by it, rounds toward zero, or uses `k_scale` for the values
fails `Post` (Checks).

## Exclusions

- **`n = num_heads * head_size ≥ 2^31`: a limit of the kernel.** The tile positions and the mask
  are `int32`; at `n ≥ 2^31` the constant `num_heads * head_size` is an `int64` and the positions
  past `2^31` wrap. One token row would be 4 GiB; real models have `n ≤ 8192`. Never reached.
- **`head_size ≥ 2^31`** (with `num_heads = 0`, where `n < 2^31` still holds): outside real use
  (real head sizes are a few hundred at most); the constexpr divisor `head_size` of lines 150-151
  is then outside `int32`, and Triton rejects the `int32` `//` and `%` at compile time, even with
  an empty tile range.
- **Grids beyond CUDA's limits or with `2^31` programs or more** (`num_tokens ≥ 2^31`,
  `cdiv(n, TILE_SIZE) ≥ 2^16`, `num_tokens * cdiv(n, TILE_SIZE) ≥ 2^31`): CUDA rejects the first
  two; Triton 3.8.0's launcher runs no program for the third without an error (a limit of the
  launch). At `n = 8192` (four tiles) the third needs `2^29` tokens; real batches have thousands.
  Never reached.
- **Strides or `block_size` of `2^63` or more**: outside real use.
- **A cache view that aliases itself, a key and a value cache sharing an element, a cache
  overlapping `key`, `value` or `slot_mapping`, two tokens with the same slot, a slot past the
  cache**: outside real use (vLLM's allocations and slot mapping); the kernel would race or write
  outside the cache there.
- **Inputs or scales that are NaN or infinite, a zero scale, `|x / s| > 2^126`** (`Mode.fp8`): the
  element is only known to hold an fp8 value; the library states `/` only for finite operands
  and a non-zero finite divisor with the quotient's magnitude at most `2^126` (`DivFull`; an
  infinite divisor, `DivF32InfDivisor`, gives `±0`, not used here). Real use has finite
  activations and calibrated positive scales.
- **`float16` or `float32` inputs, a `float8e5` cache, fp8 inputs into a 16-bit cache, mixed input
  dtypes, per-head scales**: not part of the setup (`SETUP.md`).
- The head-major layout is not part of the setup (`SETUP.md`).

## Modelling decisions

- Line 104's `.to(tl.int64)` is written (`tl.convert`, the identity on the `int64` slot); line
  105's comparison is the `int64` `<` with `0` built in `int64`.
- Lines 152-157 in the source's order: the `int64` scalar sum first, then the blocks.
- `dim_stride_k = dim_stride_v = 0` (the host's values) are constants of the model.
- Lines 161-195 are given once per configuration (`match m`): `FP8_KV_CACHE`, the loads' dtype
  (and so `is_fp8()`) and the stores' conversion are compile-time values fixed by `m`.

## Library requests

None.

## Checks

- **Builds**: `lake env lean ../claims/kv_cache_write_bf16/Claim.lean`: no error or warning (the
  `dim_stride_k`, `dim_stride_v` of the source's signature, and `k_scale`, `v_scale` outside
  `Mode.fp8`, are unused, with the unused-variable linter off for the model), no `sorry`, no
  `axiom`; the theorems use only `propext`, `Classical.choice`, `Quot.sound`.
- **Not vacuous**: `consistent` (`Platform.sm90ref`); `satisfiable` (the one-token flash case on
  the all-zero memory, slot `0`).
- **Coverage**: the 18 instances above (`setup_*_ok`), from `mk_ok`.
- **GPU** (`check_gpu.py`, H200, Triton 3.8.0): 33 launches through the host, each with the
  caches carved from one random-bit allocation between guard regions. `bfloat16` caches (19
  launches): the 8 setup shapes in their layouts, and edges (two and four tiles in both layouts,
  `n = 400`, one head of 1, block size 1, every slot used, only padding, one block, a token stride
  `2n`, 4096 tokens in the triton layout, negative slots other than `-1`), every written element
  bit-exact. `float8e4nv` caches (14 launches, the `uint8` allocation the host views as fp8):
  vLLM's fp8 test shapes and larger ones in both layouts with `k_scale = key.amax() / 64`; inputs
  with NaN, infinities, values beyond `448·scale` and fp8-subnormal values; scales `1`, `0.0137`
  and `−3.5`, `10^-30` and `2^-126` (saturation, and quotients beyond `2^126`), `0` and `∞`
  (unconstrained); every constrained element one of the values `Stored` allows; and `float8e4nv`
  inputs, bit for bit. Every other element of each allocation, every guard, every input and the
  scales unchanged. Negative tests (on two fp8 launches): the scale left out, the scale
  multiplied, a rounding toward zero, the value cache divided by `k_scale`: `Post` rejects each.
