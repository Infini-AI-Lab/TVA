# `seeded_dropout_f32`: prepared claim

Triton tutorial 04's `_seeded_dropout` (`source/04-low-memory-dropout.py`, kernel lines 110–131 with
the imports of lines 38–39; host `seeded_dropout` 134–140). `Claim.lean`, namespace
`Claims.SeededDropout`; the claim to prove is `Statement`. The launches to cover are in `SETUP.md`.

## Specification

The reference is computed from the host's inputs (`seed`, `p`, `x[j]`) and the element's index `j`
by the definition of Triton's generator:
- `rbits seed j`: the first word of Philox 4x32 with 10 rounds (`Philox.impl`, the library's
  definition of `philox_impl`) on the counter `(j mod 2^32, 0, 0, 0)` and the key `(key0, key1)`,
  `key0 = seed mod 2^32`, `key1` the high 32 bits of `seed mod 2^64`;
- `rint seed j`: `r = rbits seed j` if `r < 2^31`, else `2^32 − 1 − r` (the bits read as an `int32`
  and complemented when negative, as `uint_to_uniform_float` does);
- `uRef seed j`: `rint` rounded to `float32`, times the `float32` constant `4.6566127342e-10`
  (`Philox.scale32`, bit pattern `0x2FFFFFFF`), one IEEE product; in `[0, 1)`;
- `p32 = FloatV.hostF32 p`, the `float32` the kernel receives; `keep j` is `p32 < uRef seed j`
  (ordered);
- `dRef = FloatV.sub one32 p32`, the IEEE `float32` `1 − p32` (exact for `p32 ≥ 1/2`).

`Post d init fin`: for every `j < n`, element `j` of `x` can be read in `init` (value `a`) and
element `j` of `output` in `fin` (value `c`), and
- if `keep j` is false, `c` is `+0` exactly (bit pattern `0`);
- if `keep j` is true and `a` is finite with `|a/dRef| ≤ 2^126`, `c` is finite and
  `|c − a/dRef| ≤ 2^-22·|a/dRef| + 2^-148` (in `ℚ`);
- frame: every byte outside the first `n` elements of `output` is unchanged (`x`, the rest of
  `output`, every other byte).

The keep/drop decision is exact: which elements are zeroed is fixed bit for bit by `seed`, `p` and
`j`. The kept values carry the bound of `/` on `float32` (the kernel's `x / (1 - p)` is
`div.full.f32`, not IEEE division). A kept element whose `x` is NaN or infinite, or whose quotient
exceeds `2^126` in magnitude, is only required to be readable (Exclusions).

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch that
runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

`Inst`: `n` (`n_elements`), `BLOCK_SIZE`, `p` (the Python double), `seed` (the Python integer), and
the contiguous `float32` buffers `x`, `output`.

`Inst.ok`:

| Field | Condition | Why |
|---|---|---|
| `block_pow2`, `block_le` | `BLOCK_SIZE` a power of two, at most `2^20` | the block sizes Triton accepts; `Launch.Valid`. Every one is covered, not only the host's `1024` |
| `n_lt` | `n < 2^31` | `n_elements` is an `int32` argument (a compile-time `1` at `n = 1`, the same model); the `int32` offsets of the grid `cdiv(n, BLOCK_SIZE)` do not wrap (`Block.fits_of_grid_pow2`) |
| `p_fin`, `p_ge`, `p_le` | `p32` finite, `0 ≤ p32 ≤ 1` (`-0.0` included) | a dropout probability (Exclusions). At `p32 = ±0`, `dRef = 1`; at `p32 = 1`, `dRef = 0`, but `uRef ≤ 1 − 2^-24`, so no element is kept and no stored lane takes a quotient |
| `seed_ge`, `seed_lt` | `-2^63 ≤ seed < 2^64` | the integers Triton passes as arguments; others are rejected on the host |
| `x_len`, `output_len` | each buffer holds at least `n` elements | the kernel accesses elements `0 … n−1` of each |
| `output_x` | the first `n` elements of `output` share no byte with the first `n` of `x` | an in-place launch would be a conflict within a program; the host's `output` is fresh |

`n = 0` is covered (empty grid).

`Pre`: the two buffers are valid (`Buf.Valid`). Every `float32` value of `x` is covered, NaN and
infinities included.

The model's seed type follows the argument (SEMANTICS.md, Scalar arguments): `model` passes an
`int32` for `[-2^31, 2^31)`, an `int64` for the rest of `[-2^63, 2^63)`, a `uint64` for
`[2^63, 2^64)`, each as `IntV.ofInt seed`; at `seed = 1` Triton passes a constant, which `tl.rand`
turns into an `int32` tensor before use, so the `int32` model applies. In every case
`seed.to(tl.uint64)` (sign extension from a signed type) is `seed mod 2^64`, which is how the
reference states the key.

The launch: `Launch.grid1 (cdiv(n, BLOCK_SIZE)) [[BLOCK_SIZE]]`, `enable_fp_fusion = true`,
`debug = false`, `num_ctas = 1`, no `launch_pdl`. Part (A) of `Compiler.Faithful`'s scope
(`docs/rules/compiler-faithful.md`): (4) the launch gives no option, only the constexpr; no
`warmup`, `triton.compile`, `@triton.autotune` or `@triton.heuristics`, and a bare `@triton.jit`;
(5) no tensor descriptor; (6) no `CompiledKernel`; (8) no host `TensorDescriptor`. The claim relies on part (B), the environment, which is assumed and not checked.

The kernel is inside the supported subset (`docs/SEMANTICS.md`): a top-level `def` under a bare
`@triton.jit`, two pointer parameters, three unannotated scalars (`n_elements`, `p`, `seed`), one
`tl.constexpr`; it reads only the module-level names `triton` and `tl`, each bound only by its own
`import`; it calls `tl.rand` (SEMANTICS.md, Random numbers), no helper of its own file.

## The model

Line by line in `model-map.json`. The points that decide the meaning:
- `tl.rand(seed, offsets)` is `tl.rand` of the library (`TVA.Lang.Random`): `randint4x` with the
  32-bit offsets as the first counter word (the second is `0`, offsets are not wider than 32 bits),
  then `tl.itof` and the IEEE product with `Philox.scale32`.
- Fusion: the product inside `tl.rand` reaches only the comparison `random > p`, never an addition,
  so no operation of the kernel is the fusable form; `1 - p` is an IEEE `tl.sub` (no product feeds
  it). The launch keeps `enable_fp_fusion = true`, which then changes nothing.
- `random > p` is the ordered comparison, written `Tensor.flt (bc p) random`.
- `x / (1 - p)` is `tl.div`, the platform's `float32` division, computed on every lane (also the
  dropped and the disabled ones, whose quotients no stored lane uses).
- `tl.where(x_keep, …, 0.0)`: the `0.0` is built in `float32` as `+0`.

## Assumptions

Three `[rule]` fields, each the library's fact as `docs/rules/num.md` states it: `faithful`
(`Compiler.Faithful`), `div_total` (`Num.DivTotal P .binary32`, line 130), `div_full`
(`Num.DivFull`: relative `2^-22`, absolute `2^-148` for quotients up to `2^126`, line 130). The
generator is pure integer arithmetic; the conversion and the product of `tl.rand` and the
subtraction are fully determined (one IEEE rounding each), and the comparison is exact: none needs
a device fact. No `[kernel]`
field.

## Coverage

`setup_seed123` and `setup_seed512`, with `setup_*_ok`: the file's launches of lines 145–147 (`n = 10`,
`p = 0.5`, `BLOCK_SIZE = 1024`, fresh buffers; the launch of line 146 is the one of line 145).
`mk_ok` covers the host's launch with `p = 0.5` at every `n < 2^31`, every block size `2^e ≤ 2^20`
and every seed in `[-2^63, 2^64)`; `half_facts` shows that the `float32` of `0.5` is in `[0, 1]`.

## Tolerances

Only the kept values have one, and it is `Num.DivFull`'s card at the kernel's own operands, so it is
provable directly: on a kept lane with a finite `a` and `|a/dRef| ≤ 2^126`,
- the divisor is `dRef` exactly: `tl.sub` is `FloatV.sub one32 p32` up to NaN payload, and it is not
  NaN (`p32` finite); it is finite and nonzero: a kept lane has `p32 < uRef ≤ 1 − 2^-24`
  (`2^31·Philox.scale32 = 1 − 2^-24` bounds every random number), and `0 ≤ p32` then gives
  `2^-24 ≤ dRef ≤ 1`;
- `div_full` then gives a finite result within `2^-22·|a/dRef| + 2^-148` of `a/dRef`.

The bound is meaningful: about 2–4 ulp of the quotient. The device's worst error is 0.500 of it
(Checks). The keep decision needs no tolerance: `tl.randint4x` is exact (`Philox.randint4x`); the
conversion of `rint < 2^31` to `float32` is `FloatV.ofRat rint` by the library's lemma
`tl.itofRel_binary32` (`TVA.Lemmas.Random`, for `|z| < 2^53`), so `tl.rand`'s value is `uRef` (the
product is not NaN); and the comparison and `tl.where` are exact.

## Exclusions

- **`p` outside `[0, 1]`** (as its `float32`: negative, above `1`, infinite or NaN) — outside real
  use: `p` is a dropout probability, and the file uses `0.5`.
- **Kept elements with a NaN or infinite `x`, or with `|x/(1−p)| > 2^126`** — a library gap:
  `Num.DivFull` promises nothing for such operands, so `Post` constrains those outputs only to be
  readable (dropped elements are `+0` whatever `x` is). The first launch it touches: any launch whose
  `x` holds a NaN at a kept element; `randn` inputs never reach it.
- **`n = 2^31`** — another specialisation, outside this claim: Triton passes `n_elements` as an
  `int64` argument there, which this model does not take. It is not a kernel limit (the `int32`
  offsets reach only `2^31 − 1`), and it is outside the file's launches (`n = 10`).
- **`n > 2^31`** — a kernel limit: the last program's `int32` product `pid * BLOCK_SIZE` wraps
  (first launch: `n = 2^31 + 1` with `BLOCK_SIZE = 1024`), and its lanes would access memory before
  the buffers. Real use does not reach it.
- **Seeds outside `[-2^63, 2^64)`** — rejected by Triton on the host, with no outcome.
- **Dtypes other than `float32`** — outside the file's launches (the kernel is dtype-generic in
  `x`, the file uses `float32`).
- **`output` overlapping the first `n` elements of `x`** (in place) — outside the file's launches
  (`output` is always fresh); it would also be a conflict within one program, which the library
  does not order.
- **Strided tensors** — the host asserts `x` contiguous; the kernel ignores strides.
- **The baseline `_dropout`** (lines 44–63) — another kernel of the file, not part of this claim.

## Library requests

None.

## Checks

- `Claim.lean` builds with no errors or warnings, no `sorry`/`axiom`; `consistent`, `satisfiable`
  and every `setup_*_ok` use only `propext`, `Classical.choice`, `Quot.sound`.
- `check_gpu.py` (H200, Triton 3.8.0): its numpy Philox 4x32-10 matches the three Random123
  known-answer vectors, the claim's `rbits`/`rint`/`uRef`/`keep` evaluated in Lean (5 seeds × 4
  indices), and `tl.randint4x` (all four words) and `tl.rand` on the GPU bit for bit (16 seeds of
  every class, `2^20` offsets each). The file's three launches and 37 more (sizes `0` to `2^26`,
  `p` from `0` and `-0.0` to `1`, every seed class, `randn`, random bit patterns and special
  values), with direct launches at `BLOCK_SIZE` `1, 16, 128, 1024, 4096, 65536` (`2^20`, the top of
  `Inst.ok`'s range, did not finish compiling within 25 minutes and is not run): every dropped element is
  `+0` bit for bit; every kept element with a finite `x` and `|x/dRef| ≤ 2^126` is within the bound,
  worst error / tolerance 0.500 (`p = 0.6`), and 94803836 of the 101027006 kept-element checks (all
  launches together) equal the IEEE quotient (all at `p = 0.5`, where `dRef = 1/2`); `x` is unchanged; no element
  outside `output`'s first `n` is written. Nine negative tests (a kept element zeroed, a dropped
  one given a value or `-0`, a kept value off by `2^-19` relative or not rescaled, references with
  the seed `+1`, the offsets `+1` or 9 rounds, a write outside `output`) are rejected: 76 of 76
  cases pass.
