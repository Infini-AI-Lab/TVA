import TVA.Base.Float.Value

/-!
# Python numbers as float constants

Where a Python `float` `c` meets a tensor (`x + 0.5`, `tl.where(m, x, 0.0)`,
`tl.full(shape, c, dtype)`), Triton builds a constant of the operation's dtype with
`scalar_constant` (`semantic.py:601-613`, called at `semantic.py:205-206` and by `full` at
`semantic.py:622-623`), which calls one builder function per dtype (`ir.cc:952-972`):
- a zero (`-0.0` included) is `+0` (`get_null_value`);
- `float64`: the Python `float` itself (`get_fp64(double)`): `litF64`;
- `float32`: the nearest `float32`, by C's conversion of the double (`get_fp32(float)`):
  `litF32`, `hostF32`;
- `float16`: the nearest `float16` to that `float32` (`getF16FloatAttr(float)`, two roundings):
  `litF16`;
- `bfloat16`: the `bfloat16` nearest to the *decimal text* that `std::to_string` prints for that
  `float32`, six digits after the point (`APFloat(BFloat, std::to_string(v))`): `litBF16`. So a
  constant below `5·10⁻⁷` in magnitude is `±0`, and `1.23456e-5` is the `bfloat16` nearest to
  `1.2e-5`;
- fp8: the `float32` constant truncated by `arith.truncf`, which MLIR folds only when the
  conversion is exact; otherwise the kernel is rejected on the host: `litF8`.

A NaN keeps its sign and is quiet. In `float32` and `float16` it keeps the top bits of its
payload (`nanNarrow`: C's and LLVM's conversions drop the low bits and set the quiet bit); in
`bfloat16` it has none (its text is `nan` or `-nan`); in `float64` it is the double itself. So
`float('nan')` (the double `0x7ff8000000000000`) is `0x7fc00000`, `0x7e00` and `0x7fc0`.

Each function takes `c` as the Python `float` it is (`float(c)`, an `F64`). A Python `int` in an
integer dtype is `IntV.ofInt c`. Where the number instead first becomes a tensor of its own type
(a comparison, the `tl.*` functions, `other=`), `TVA.Triton.Convert` says what happens.
-/
namespace TVA.FloatV

variable {f : Format}

/-- The quiet NaN with sign `s` whose fraction is the quiet bit (its top bit) and the low
`m − 1` bits of `p` (for an `.ieee` format with `m ≥ 1`). -/
def qNaN (s : Bool) (p : Nat) : FloatV f :=
  ofPat (f.setSign s (f.infMag + 2 ^ (f.m - 1) + p % 2 ^ (f.m - 1)))

/-- A NaN of `f` converted on the host into the narrower `.ieee` format `g`, as C's `(float)` of
a `double` (x86-64) and LLVM's `APFloat::convert` do: the NaN's sign, the quiet bit set, and the
top `g.m − 1` bits of the rest of its fraction. So `float('nan')`, the double
`0x7ff8000000000000`, is `0x7fc00000` in `float32`, and that is `0x7e00` in `float16`. -/
def nanNarrow {g : Format} (x : FloatV f) : FloatV g :=
  qNaN x.sign (f.fracField x.pat / 2 ^ (f.m - g.m))

/-- `q` rounded to a multiple of `10⁻⁶`, ties to even: the number C's `%f` (six digits after
the point) prints for the exact binary value `q`. -/
def round6 (q : Rat) : Rat :=
  let a : Int := q.num * 1000000
  let d : Int := q.den
  let n := a / d                      -- `⌊q · 10⁶⌋` (`Int` division rounds down for `d > 0`)
  let r2 := 2 * (a - n * d)           -- twice the remainder, against `d`
  let m : Int := if r2 < d then n else if d < r2 then n + 1 else if n % 2 = 0 then n else n + 1
  mkRat m 1000000

/-- C's conversion `(float)c` of a `double` on the host, in the default floating-point
environment (round to nearest even, no flush to zero). It is what the launcher passes for a
`float` kernel argument (`extractFP32`, `nvidia-driver.c:1152-1157`) and what the constant
builders receive for a Python `float`. A number rounds to nearest even (an overflow is `±∞`,
`-0.0` stays); a NaN keeps its sign and the top bits of its payload, made quiet (`nanNarrow`). -/
def hostF32 (c : F64) : F32 := if c.isNaN then nanNarrow c else cvt c

/-- The `float64` constant for the Python number `c`: the double itself, NaN payload and sign
included (`+0` for a zero). -/
def litF64 (c : F64) : F64 := if c.isZero then ofPat 0 else c

/-- The `float32` constant for the Python number `c`: `hostF32 c` (`+0` for a zero). A NaN keeps
its sign and the top bits of its payload: `float('nan')` is `0x7fc00000`, `-float('nan')` is
`0xffc00000`. -/
def litF32 (c : F64) : F32 := if c.isZero then ofPat 0 else hostF32 c

/-- The `float16` constant for the Python number `c`: the nearest `float16` to its `float32`
constant (`+0` for a zero); a NaN keeps its sign and the top bits of the `float32`'s payload:
`float('nan')` is `0x7e00`. -/
def litF16 (c : F64) : F16 :=
  if c.isZero then ofPat 0
  else
    let x := litF32 c
    if x.isNaN then nanNarrow x else cvt x

/-- The `bfloat16` constant for the Python number `c`: the `bfloat16` nearest to the six-decimal
text of its `float32` constant (`round6`); a zero result keeps the sign of the text
(`-0.000000` for a small negative value); `±∞` stay; a NaN, whose text is `nan` or `-nan`, is
the quiet NaN of that sign with no payload, `0x7fc0` or `0xffc0` (`+0` for a zero `c`). -/
def litBF16 (c : F64) : BF16 :=
  if c.isZero then ofPat 0
  else
    let x := litF32 c
    if x.isNaN then qNaN x.sign 0
    else if x.isInf then ofPat (Format.bfloat16.setSign x.sign Format.bfloat16.infMag)
    else ofPat (Format.bfloat16.roundRat .rn false x.sign (round6 x.toRat))

/-- The fp8 constant for the Python number `c` in `g` (`e5m2` or `e4m3`), or `none` when Triton
rejects the kernel on the host. `scalar_constant` builds `+0` for a zero, and otherwise the
`float32` constant `litF32 c` followed by `arith.truncf` into `g`, which MLIR folds only when
`APFloat::convert` reports no loss (`llvm-ArithOps.cpp:1588-1611, 1788-1825`); an unfolded one
does not compile. So:
* a number: its `float32` constant, when `g` holds that value exactly, else `none` (`0.1`, and
  `1000.0` in `float8e4nv`, are rejected; `-1e-50`, whose `float32` constant is `-0`, is `-0`);
* `±∞` (also a `c` whose `float32` overflows, `1e39`): `±∞` in `float8e5`, `none` in
  `float8e4nv`;
* a NaN: in `float8e5`, the `float32` NaN's sign, quiet bit and next fraction bit (`nanNarrow`)
  when no lower fraction bit is set (`float('nan')` is `0x7e`), else `none`; in `float8e4nv`
  `none`.
A number converted into fp8 at run time (`.to`, a stored value) uses the device's saturating
conversion instead (`TVA.Triton.Convert`). -/
def litF8 {g : Format} (c : F64) : Option (FloatV g) :=
  if c.isZero then some (ofPat 0)
  else
    let x := litF32 c
    if x.isNaN then
      if g.specials = .ieee ∧ Format.binary32.fracField x.pat % 2 ^ (Format.binary32.m - g.m) = 0 then
        some (nanNarrow x)
      else none
    else if x.isInf then
      if g.specials = .ieee then some (ofPat (g.setSign x.sign g.infMag)) else none
    else
      let y : FloatV g := cvt x
      if y.isFinite = true ∧ y.toRat = x.toRat then some y else none

end TVA.FloatV
