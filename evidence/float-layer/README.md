# The float layer against the device

Library evidence (not a kernel's), for library issue L65. Triton 3.8.0 (TVA's Python environment,
torch 2.11.0+cu130), NVIDIA H200 NVL (sm_90, driver 610.57.04), GPU 0; Lean 4.35.0; 2026-10-28.
Scripts: [build.sh](build.sh), [Driver.lean](Driver.lean), [run.py](run.py), [e2m1.py](e2m1.py),
[bytes.py](bytes.py), [Bytes.lean](Bytes.lean), [nan_patterns.py](nan_patterns.py) (part F, round
19). Output: [results.out](results.out), [e2m1.out](e2m1.out), [bytes.out](bytes.out),
[ptx.out](ptx.out), [nan_patterns.out](nan_patterns.out).

**Question.** Do the library's float definitions (`lean/TVA/Math/Defs.lean`: what a pattern means,
the rounding of an exact value, the IEEE operations, the conversions) and its memory encoding
(`TVA.Math.Bytes`, `DType.toBits`/`ofBits`) give what Triton 3.8.0 computes on the device, for
the operations the library takes as exact: `+ - *` (when not fused), `tl.fma`, `tl.div_rn`,
`tl.sqrt_rn`, `-x`, `tl.abs`, `tl.maximum`/`tl.minimum` (both `propagate_nan`), the comparisons,
`x.to(dtype)` between float formats and between integers and floats, and `tl.load`/`tl.store`?

**Method.** The library side is the definitions themselves, not a re-implementation:
`build.sh` compiles `lean/TVA/Math/Defs.lean` (core Lean only, so it compiles alone) and
`Driver.lean` to one native executable with `lean -c` and `leanc`. `Driver.lean` maps a test
(`add:f16`, `cvt:f32:e4m3:rn:1`, …) to `TVA.Math.Float.Format.add`, `Format.cvt`, … and evaluates
it on a range of inputs. `run.py` runs the same operation in a Triton kernel on the device, on the
same inputs, and both sides reduce each chunk of `2^20` consecutive inputs to one 64-bit number
`h = Σ out(i)·(2iK + 1) mod 2^64` (`K = 0x9E3779B97F4A7C15`). The multiplier is odd, so a single
differing output always changes `h`. A NaN output is replaced by the format's canonical NaN first
on both sides (the library's operations are up to NaN payload). For a chunk whose numbers differ,
both sides print every output of the chunk and the differing inputs are listed (four chunks at
most).

Inputs:
- `all`: every pattern of the source format (`2^8`, `2^16` or `2^32`), or every pair of `16`-bit
  patterns (`x = i >> 16`, `y = i mod 2^16`: `2^32` pairs);
- `file`: records written from a fixed seed (`gen_records`): a quarter uniform bit patterns, a
  quarter significands with only their top bits set (short products, ties), a quarter structured
  (random exponents, for the second and third operands near the first's exponent: cancellation,
  carries, ties), a quarter special values (zeros, the smallest and largest subnormals and normals,
  `1`, `1 ± ulp`, `maxVal`, infinities, NaNs, both signs). `.sample` is such a sample of `2^26`
  inputs of a test whose full domain is `2^32`.

Some outputs are compared only where the library says something: `cvtfin` (out of fp8) only on
finite inputs (`tl.cvtFrom8` says nothing else), `ftoi` only where the truncation is in the integer
type's range (undefined behaviour elsewhere; both sides put the same marker there).

`e2m1.py`: Triton has no fp4 dtype, so the 16 `e2m1` values come out of `tl.dot_scaled` with
scales 1 against the `bfloat16` identity (a zero comes out `+0`, the sum adding `+0` products).
`bytes.py` with `Bytes.lean` (part E): for every dtype, 512 random values stored by a kernel and
read back byte by byte through a `uint8` pointer, and 512 random byte strings loaded by a kernel as
the dtype; the library side is `leBytes d.bytes (d.toBits …)` and `d.toBits (d.ofBits (leValue …))`
(interpreted, `lake env lean --run`). `run.py --ptx` lists the floating-point instructions of each
test's kernel ([ptx.out](ptx.out)).

**Result.** 150 tests, 1.26·10^11 inputs ([results.out](results.out), one line per test with its
spec, domain and size). 143 agree on every input. The 7 that differ are variants of conversions
into fp8 that the device does not perform: six non-saturating conversions to nearest (`"rtne"`,
`sat := false`) and one saturating conversion toward zero (`"rtz"` with `sat := true`, `float32`
into `float8e5`), below. Exhaustive (`all`; "every" below) and sampled (`file`):

| Part | Every input or pair | Sampled | Result |
|---|---|---|---|
| A: decoding, conversions | `float32` into `float64` (every `float32` pattern decoded), `float16`, `bfloat16` (`"rtne"` and `"rtz"`), `float8e4nv`, `float8e5` (both with `sat`), `float8e5` `"rtz"` (without `sat`); `float16`, `bfloat16` into `float32`, `float64`, each other and both fp8 formats (`sat`); both fp8 formats into `float16`, `bfloat16`, `float32` (finite inputs); the 16 `e2m1` values ([e2m1.out](e2m1.out)) | `float64` into `float32`, `float16`, `bfloat16` (`2^22` each) | agree |
| A: fp8, the variants the device does not run | `float16`, `bfloat16` into both fp8 formats, `"rtne"` without `sat` | `float32` into both, `"rtne"` without `sat` (`2^22`); `float32` into `float8e5`, `"rtz"` with `sat` (`2^26`) | differ, see below |
| B: arithmetic | `float16` pairs: `+ - *`, `<`, `<=`, `==`, `!=`, `tl.maximum`/`tl.minimum` (both `propagate_nan`); `bfloat16` pairs: `+ - *`; `tl.sqrt_rn` on every `float32` | `float32` `+ - *`, `tl.div_rn`, `tl.fma`, comparisons, max/min (`2^24` each); `float64` `+ - * /`, `tl.fma`, comparisons, max/min (`2^22`); `tl.fma` on `float16`, `bfloat16` (`2^22`) | agree |
| C: sign | `-x`, `tl.abs` on every `float16`, `bfloat16`, `float32` | `float64` (`2^22`) | agree |
| D: integers | every `int8`, `uint8`, `int16`, `uint16` into `float16`, `bfloat16`, `float32`; every `int32`, `uint32` into `float32`; every `float16`, `bfloat16` into the 8 integer types and every `float32` into `int32`, `uint32` (truncation, in range) | `int32`, `uint32` into `float16`, `bfloat16`, `float64` (`2^24`); `int64`, `uint64` into the four formats (`2^22`); `float32`, `float64` into `int64`, `uint64` (`2^24`, `2^22`) | agree |
| E: memory | 15 dtypes, 512 values written and 512 byte strings read each ([bytes.out](bytes.out)) | | agree |

The differences. The six `"rtne"` tests, every difference counted in
[results.out](results.out): with `sat := false` the library's conversion into fp8 gives `±∞`
(`float8e5`) or NaN (`float8e4nv`) where the rounding overflows, and `±∞` or NaN for an infinite
input; the device gives `±maxVal` there (`0x7b`/`0xfb`, `0x7e`/`0xfe`). That is all of them:
14720, 258, 30512 and 28706 inputs from `float16` and `bfloat16`, and 1711569 and 1624771 of the
`2^22` sampled `float32` inputs, each one "the library gives an infinity or NaN, the device the
largest finite magnitude". The `sat := true` variants agree on every input. Into `float8e5` with
`"rtz"` it is the other way round: the device keeps `±∞` (it converts through `float16` with
`cvt.rz` and keeps the high byte), so `sat := false` agrees on every `float32` input
(`A.cvt.f32.e5m2.rz.sat0`, exhaustive), and the saturating `A.cvt.f32.e5m2.rz.sat1.sample`
differs exactly on its `±∞` inputs. Exactly, by this argument rather than by the listing:
toward zero, saturation changes only the result for `±∞` (`Format.cvt_rz_sat`, proved: on every
other pattern `cvt … .rz true` and `cvt … .rz false` are equal, since an overflow toward zero
gives `±maxVal` either way), and there `sat := true` gives `±maxVal` (`0x7b`/`0xfb`) where the
device, as `sat := false`, gives `±∞` (`0x7c`/`0xfc`). The sample holds `±∞` among its special
values, so 64 of its chunks differ; [results.out](results.out) lists 24 differing inputs, the
first six of each of the four chunks `run.py` dumps (`bad[:4]`, `diffs[:6]`), every one
`7f800000` or `ff800000`.

The 64-bit `ftoi` tests first ran with a fault in `run.py`'s range test (`2^63 − 1` is not a
`float64`, so `2^63` passed as in range); with the range bounded by `2^63` exclusive they agree,
and [results.out](results.out) keeps that run. `evidence/literals` (the constants
`FloatV.litF64` … `litBF16`, which round with this layer), re-run on it, printed exactly its
recorded `out.txt` (0 differences in each dtype, 9364 literals).

The native code that ran is what the final `lean/TVA/Math/Defs.lean` compiles to (`lean -c`
output identical, checked with `cmp`).

**What Triton compiles them to** ([ptx.out](ptx.out)): `add.f16`, `sub.f16`, `mul.f16`,
`fma.rn.f16` and the `bf16` forms (no widening to `float32`), `add/sub/mul.f32`, `fma.rn.f32`,
`div.rn.f32` (`tl.div_rn`), `sqrt.rn.f32` (`tl.sqrt_rn`), `div.rn.f64` (`/` on `float64`),
`max.f16`/`max.f32`/`max.f64` (`tl.maximum`), `max.NaN.f16`/`f32` and, for `float64`,
`max.f64` with a NaN test (`propagate_nan=ALL`), `setp.lt/le/eq.f*` and `setp.neu.f*` for `!=`;
`-x` and `tl.abs` as bit operations; `cvt.rn`/`cvt.rz` between standard formats,
`cvt.rn.satfinite.e4m3x2`/`e5m2x2` into fp8 (from `float32` and `float16`; `bfloat16` widened to
`float32` first), `float32` into `float8e5` with `"rtz"` as `cvt.rz.f16.f32` and a byte select;
out of fp8 `cvt.rn.f16x2.e4m3x2`/`e5m2x2`, and `float8e5` into `bfloat16` by a bit trick with
`mul.rn.bf16x2`; integers by `cvt.rn.f*.s*/u*` (`int8` into `bfloat16` by a bit trick with
`sub.bf16x2`), floats into integers by `cvt.rzi`.

**Part F: which NaN the device gives** (round 19, library issue L99; [nan_patterns.py](nan_patterns.py),
[nan_patterns.out](nan_patterns.out); Triton 3.8.0, H200 NVL, GPU 0, 2026-10-28). The comparisons
above replace every NaN by the canonical one, as the library's operations are up to NaN payload;
this part asks which NaN the device actually returns, for `FloatV.nan`'s docstring. Method: per
standard format, a kernel computes from runtime operands (a NaN with a payload, of either sign,
and the invalid operations `inf − inf`, `0·inf`, `0/0`, `fma(inf, 0, 1)`, `sqrt(−1)`) `+ - *`,
`tl.fma`, `tl.div_rn` and `/`, `tl.sqrt_rn` (`tl.div_rn` and `tl.sqrt_rn` take `float32` only;
the other formats run `/` and `tl.sqrt`, the 16-bit ones through `float32`), `tl.maximum` and
`tl.minimum` with `propagate_nan=ALL`, `-x`, `tl.abs`, and every conversion between the standard
formats and into fp8, and prints the patterns. Result:
- `float32`, `float16`, `bfloat16`: every NaN result of these operations is the library's
  canonical NaN (`0x7fffffff`, `0x7fff`), except `-x` and `tl.abs`, bit operations that keep the
  payload (`-(0xffc00001)` is `0x7fc00001`);
- `float64`: an invalid operation gives `0xfff8000000000000` (negative), a NaN operand keeps its
  quieted payload (`0x7ff0000000012345 + 1` is `0x7ff8000000012345`), and
  `propagate_nan=ALL` gives `0x7ff8000000000000`;
- conversions: between `float16`, `bfloat16` and `float32` the canonical NaN, except
  `bfloat16 → float32`, which keeps the payload (`0x7fa3` becomes `0x7fa30000`); into fp8 the
  canonical `0x7f`; into `float64` the payload is kept, and out of `float64` the sign
  (`0xfff8000000000001` becomes `0xffc00000`, `0xfe00`, `0xffc0`).

So `FloatV.nan` is "the library's canonical NaN", the device's NaN for `float32`, `float16` and
`bfloat16` arithmetic and for the conversions into fp8, and no more than some NaN elsewhere; every
float operation of the library is up to NaN payload (`FloatV.NaNEq`), which covers all of these.

**Conclusions.**
1. The library's float definitions are what Triton 3.8.0 computes on the H200 NVL, bit for bit
   up to NaN payload, on every input of 29 operations over all `2^32` inputs or pairs, on every
   input of the smaller formats, and on every sampled input elsewhere: the meaning of each
   pattern (`toRat`, `isNaN`, `isInf`), the rounding (`roundRat`, both modes, with ties,
   subnormals, overflow and the sign of zero), `add`, `sub`, `mul`, `fma`, `div`, `sqrt`, `neg`,
   `absPat`, `maxNum`, `minNum`, `maxNaN`, `minNaN`, `cmp` and `cvt`.
2. `tl.maximum`/`tl.minimum` give exactly `maxNum`/`minNum` (`maxNaN`/`minNaN` with
   `propagate_nan=ALL`), signed zeros included, on every `float16` pair; `FloatV.MaxRel` allows
   more.
3. Every integer-to-float conversion tested is one rounding to nearest (`FloatV.ofRat`, the
   first case of `tl.ItofRel`); float-to-integer is truncation where it is in range.
4. Conversions into fp8 saturate: `x.to(float8e4nv)` and `x.to(float8e5)` are
   `Format.cvt … .rn true` (`FloatV.cvtSat .rn`) on every `float16`, `bfloat16` and `float32`
   input (NaN giving a NaN: outputs are compared up to payload); `float32` into `float8e5` with
   `"rtz"` is `Format.cvt … .rz false` (`FloatV.cvtWith .rz`). The library states the first (the
   `CvtSat…` facts, rounds 14 and 20: library issues L64, L108), not the second.
5. Out of fp8, finite values are exact; the 16 `e2m1` values are `0, ±0.5, ±1, ±1.5, ±2, ±3, ±4,
   ±6`.
6. Memory is little-endian for every dtype; `int1` is stored as the byte `0`/`1` and a nonzero
   byte loads as `true`.

**Re-run.** From this folder, with GPU 0 and a build folder `B` outside the tree:
`sh build.sh B`, then `CUDA_VISIBLE_DEVICES=0 python -I run.py B <tests or prefixes>`
(`python -I run.py B --list` lists them; `--jobs N` Lean processes, 40 by default; results are
appended to `results.out` in the current folder, so run it from another folder, by its path, to
keep this one's), `python -I e2m1.py B`, `python -I bytes.py B/bytes`, and
`python -I run.py B --ptx A B C D`. `build.sh` compiles with the Lean toolchain that
`lean/lean-toolchain` names, which it finds through `lake` run in `lean/` (elan's `lake` runs
that toolchain, and `lake env` puts its `bin` folder first on `PATH`); it needs no `lean` or
`leanc` on `PATH`, only `lake` (on `PATH`, or elan's, or the one `LAKE` names). Round 12's run
called `lean` and `leanc` from `PATH`, with that toolchain's `bin` there (library issue L99).
Round 19 (2026-10-28, GPU 0): `build.sh` run with `PATH=/usr/bin:/bin` built the driver, its
`Defs.c` identical (`cmp`) to the one compiled from `lean/TVA/Math/Defs.lean` before the round's
docstring changes, and `A.cvt.f16.e4m3.sat1` and `A.cvt.f32.e5m2.rz.sat1.sample` re-ran with
their recorded results. The native code: `TVA.Math.Defs` defines `^` on `Nat` and `Int`
for compiled code as `powFast`/`intPowFast` (a `@[csimp]` replacement with a proof that it is the
same function), since `2^k` through big numbers dominated the run time. At about 1–7 µs per
operation an exhaustive run over `2^32` inputs took 2 to 14 minutes on 40 cores of a shared
machine.
