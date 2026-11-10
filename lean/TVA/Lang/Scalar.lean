import TVA.Math
import Mathlib.Basic.Real.Basic

/-!
# Scalar types and values

The element types of Triton tensors (`tl.dtype`, Triton 3.8.0) and the Lean values that stand
for them. A prover's model of a kernel computes with these values.

* `IntV signed w`: a `w`-bit integer (`tl.int8` … `tl.uint64`), held as its bit pattern. `+ - *`
  wrap modulo `2^w` (Triton lowers them to LLVM `add`/`sub`/`mul` without `nsw`). Division,
  remainder and shifts can be undefined, so they return `Option` (`none` = undefined behaviour).
* `Bool` is `tl.int1`.
* `FloatV f`: a value of the binary format `f`, held as its bit pattern (NaN payloads, signed
  zeros and infinities are kept). The IEEE operations are those of `TVA.Math.Arith` with
  round-to-nearest-even; `evidence/float-layer` compares them with the device.
* `DType` names the Triton dtypes the library covers; `DType.Val d` is the Lean type of its
  values; `DType.bytes`, `DType.toBits` and `DType.ofBits` are the memory encoding (little-endian
  bit pattern, `int1` as one byte `0`/`1`, read back as `≠ 0`).

Not covered (`docs/exclusions.md`): `float8e4b15` (Triton's software format), the AMD formats
`float8e4b8` and `float8e5b16`, and pointer-to-pointer element types.
-/
namespace TVA

open TVA.Math.Float TVA.Math.Bytes

/-! ## Integers -/

/-- A `w`-bit integer, signed (`tl.intW`) or unsigned (`tl.uintW`), as its bit pattern. -/
structure IntV (signed : Bool) (w : Nat) where
  bits : BitVec w
  deriving DecidableEq, Repr

namespace IntV

variable {s : Bool} {w : Nat}

/-- The integer with value `v mod 2^w` (two's complement). -/
def ofInt (v : Int) : IntV s w := ⟨BitVec.ofInt w v⟩
/-- The integer with value `v mod 2^w`. -/
def ofNat (v : Nat) : IntV s w := ⟨BitVec.ofNat w v⟩

/-- The mathematical value: two's complement when signed, the natural number otherwise. -/
def toInt (a : IntV s w) : Int := if s then a.bits.toInt else (a.bits.toNat : Int)

instance : Add (IntV s w) := ⟨fun a b => ⟨a.bits + b.bits⟩⟩
instance : Sub (IntV s w) := ⟨fun a b => ⟨a.bits - b.bits⟩⟩
instance : Mul (IntV s w) := ⟨fun a b => ⟨a.bits * b.bits⟩⟩
instance : Neg (IntV s w) := ⟨fun a => ⟨-a.bits⟩⟩
instance {n : Nat} : OfNat (IntV s w) n := ⟨ofNat n⟩
instance : Inhabited (IntV s w) := ⟨⟨0⟩⟩

/-- Bitwise and, or, xor, not (`&`, `|`, `^`, `~`). -/
def and (a b : IntV s w) : IntV s w := ⟨a.bits &&& b.bits⟩
def or (a b : IntV s w) : IntV s w := ⟨a.bits ||| b.bits⟩
def xor (a b : IntV s w) : IntV s w := ⟨a.bits ^^^ b.bits⟩
def not (a : IntV s w) : IntV s w := ⟨~~~a.bits⟩

/-- `<`, `<=` by the type's signedness; `==` and `!=` compare bits. -/
def lt (a b : IntV s w) : Bool := if s then a.bits.slt b.bits else a.bits.ult b.bits
def le (a b : IntV s w) : Bool := if s then a.bits.sle b.bits else a.bits.ule b.bits
def eq (a b : IntV s w) : Bool := a.bits == b.bits

/-- `max`/`min` by the type's signedness (`tl.maximum`, `tl.minimum`). -/
def max (a b : IntV s w) : IntV s w := if lt a b then b else a
def min (a b : IntV s w) : IntV s w := if lt a b then a else b

/-- Division truncating toward zero (`//` on tensors: LLVM `sdiv`/`udiv`, not Python's floor
division). Undefined (`none`) for a zero divisor and for signed `min / -1`. -/
def div? (a b : IntV s w) : Option (IntV s w) :=
  if b.bits = 0 then none
  else if s then
    (if a.bits = BitVec.intMin w ∧ b.bits = -1 then none else some ⟨a.bits.sdiv b.bits⟩)
  else some ⟨a.bits.udiv b.bits⟩

/-- Remainder with the sign of the dividend (`%`: LLVM `srem`/`urem`). Undefined as `div?`. -/
def rem? (a b : IntV s w) : Option (IntV s w) :=
  if b.bits = 0 then none
  else if s then
    (if a.bits = BitVec.intMin w ∧ b.bits = -1 then none else some ⟨a.bits.srem b.bits⟩)
  else some ⟨a.bits.umod b.bits⟩

/-- `a << n`; undefined when `n ≥ w` (LLVM poison). -/
def shl? (a : IntV s w) (n : Nat) : Option (IntV s w) :=
  if n < w then some ⟨a.bits <<< n⟩ else none

/-- `a >> n`, arithmetic (`ashr`); undefined when `n ≥ w`. -/
def ashr? (a : IntV s w) (n : Nat) : Option (IntV s w) :=
  if n < w then some ⟨a.bits.sshiftRight n⟩ else none

/-- `a >> n`, logical (`lshr`); undefined when `n ≥ w`. -/
def lshr? (a : IntV s w) (n : Nat) : Option (IntV s w) :=
  if n < w then some ⟨a.bits >>> n⟩ else none

/-- Integer-to-integer cast: sign-extend when the *source* is signed, zero-extend otherwise,
truncate when narrower; the value modulo `2^w'` (`IntV.cast_eq_ofInt`). It is the one result of
`x.to(tl.intW)` (`tl.convert`, `ConvRel`) between integer dtypes other than `int1`
(`ConvRel.iff_cast_i32_i64` and the other pairs, in `TVA.Lemmas.IntW`). -/
def cast {s' : Bool} {w' : Nat} (a : IntV s w) : IntV s' w' :=
  ⟨if s then a.bits.signExtend w' else a.bits.zeroExtend w'⟩

end IntV

/-- `tl.int8` values. -/
abbrev I8 := IntV true 8
abbrev I16 := IntV true 16
/-- `tl.int32` values, the type of `tl.program_id`, `tl.arange` and most index arithmetic. -/
abbrev I32 := IntV true 32
abbrev I64 := IntV true 64
abbrev U8 := IntV false 8
abbrev U16 := IntV false 16
abbrev U32 := IntV false 32
abbrev U64 := IntV false 64

/-! ## Floating point -/

/-- A value of the binary format `f`, as its bit pattern. -/
structure FloatV (f : Format) where
  bits : BitVec f.width
  deriving DecidableEq, Repr

namespace FloatV

variable {f : Format}

instance : Inhabited (FloatV f) := ⟨⟨0⟩⟩

/-- The pattern as a natural number (the representation `Format` works on). -/
def pat (x : FloatV f) : Nat := x.bits.toNat
/-- The value with pattern `n mod 2^width`. -/
def ofPat (n : Nat) : FloatV f := ⟨BitVec.ofNat f.width n⟩

def isNaN (x : FloatV f) : Bool := f.isNaN x.pat
def isInf (x : FloatV f) : Bool := f.isInf x.pat
def isFinite (x : FloatV f) : Bool := f.isFinite x.pat
def isZero (x : FloatV f) : Bool := f.isZero x.pat
def sign (x : FloatV f) : Bool := f.sign x.pat

/-- The exact value of a finite pattern (`0` for both zeros). Of `±∞` it is `±2^(emax+1)`
(`Format.toRat_of_isInf`, `emax` the bias), beyond `±maxVal`, so the comparisons, `max`/`min` and
the order key order the infinities below and above every number; only a NaN's value is
meaningless (the formula's, which every operation tests for first). Every error bound of the
library is about finite values: it concludes that its result is finite (or, as `RoundsTo.err`,
assumes it), so the value it bounds is a number. -/
def toRat (x : FloatV f) : Rat := f.toRat x.pat
/-- The exact value as a real number. -/
noncomputable def toReal (x : FloatV f) : ℝ := (x.toRat : ℝ)

/-- The library's canonical NaN: positive, all-ones exponent and fraction (`0x7fffffff` for f32,
`0x7fff` for f16 and bf16, `0x7f` for both fp8 formats). The device gives it for every NaN result
of `float32`, `float16` and `bfloat16` arithmetic (`+ - *`, `tl.fma`, `/`, `tl.maximum`/`tl.minimum`
with `propagate_nan=ALL`, and on `float32`, the only one of the three formats Triton accepts them on,
`tl.div_rn` and `tl.sqrt_rn`), of the conversions between those three formats other than
`bfloat16 → float32`, and of the conversions into fp8 at run time. Not otherwise
(`evidence/float-layer`, part F): `-x` and `tl.abs` keep the payload (bit operations); `float64`
arithmetic gives `0xfff8000000000000` for an invalid operation and keeps a NaN operand's quieted
payload, and `tl.maximum`/`tl.minimum` with `propagate_nan=ALL` on `float64` give
`0x7ff8000000000000`; `bfloat16 → float32` and the conversions into `float64` keep the payload,
those out of `float64` the sign; and the constants of Python NaNs keep the sign (`litF32` …,
`litF8`). The `tl.*` float operations are up to NaN payload (`FloatV.NaNEq`), so for them it stands
for any NaN. -/
def nan : FloatV f := ofPat f.nanMag

/-- The quiet NaN with sign `s` whose fraction is the quiet bit (its top bit) and the low
`m − 1` bits of `p` (an `.ieee` format with `m ≥ 1`). -/
def qNaN (s : Bool) (p : Nat) : FloatV f :=
  ofPat (f.setSign s (f.infMag + 2 ^ (f.m - 1) + p % 2 ^ (f.m - 1)))

/-- A NaN of `f` converted on the host into the narrower `.ieee` format `g`, as C's `(float)` of a
`double` (x86-64) and LLVM's `APFloat::convert` do: the NaN's sign, the quiet bit set, and the top
`g.m − 1` bits of the rest of its fraction (`x`'s fraction shifted right by `f.m − g.m`, its low
bits dropped). So `float('nan')`, the double `0x7ff8000000000000`, is `0x7fc00000` in `float32`,
and that is `0x7e00` in `float16`; the signalling `0x7ff4000000000000` is `0x7fe00000`. -/
def nanNarrow {g : Format} (x : FloatV f) : FloatV g :=
  qNaN x.sign (f.fracField x.pat / 2 ^ (f.m - g.m))

/-! ### IEEE operations, round to nearest even

Each is one IEEE 754 operation of the format with one rounding (`TVA.Math.Arith`: `Format.add`
etc.). These are the meanings of Triton's `+`, `-`, `*` on one element when no fusion takes place,
of `tl.fma`, `tl.div_rn` and `tl.sqrt_rn` (`TVA.Lang.Ops` decides which source operation means
which). -/

def add (x y : FloatV f) : FloatV f := ofPat (f.add .rn x.pat y.pat)
def sub (x y : FloatV f) : FloatV f := ofPat (f.sub .rn x.pat y.pat)
def mul (x y : FloatV f) : FloatV f := ofPat (f.mul .rn x.pat y.pat)
def fma (x y z : FloatV f) : FloatV f := ofPat (f.fma .rn x.pat y.pat z.pat)
def divRn (x y : FloatV f) : FloatV f := ofPat (f.div .rn x.pat y.pat)
def sqrtRn (x : FloatV f) : FloatV f := ofPat (f.sqrt .rn x.pat)

/-- `-x`: flips the sign bit (LLVM `fneg`). The device may return any NaN for a NaN input; the
`tl.*` operations allow that (`FloatV.NaNEq`). -/
def neg (x : FloatV f) : FloatV f := ofPat (f.neg x.pat)
/-- `tl.abs`: clears the sign bit (LLVM `fabs`). -/
def abs (x : FloatV f) : FloatV f := ofPat (f.absPat x.pat)

/-- The reference `max`/`min` (default `propagate_nan=NONE`): a NaN operand gives the other
operand; `max(-0, +0) = +0`, `min(-0, +0) = -0`. The device may give either zero
(`FloatV.MaxRel` in `TVA.Lang.Ops`). -/
def max (x y : FloatV f) : FloatV f := ofPat (f.maxNum x.pat y.pat)
def min (x y : FloatV f) : FloatV f := ofPat (f.minNum x.pat y.pat)
/-- `tl.maximum(…, propagate_nan=ALL)`: a NaN operand gives the library's canonical NaN. -/
def maxNaN (x y : FloatV f) : FloatV f := ofPat (f.maxNaN x.pat y.pat)
def minNaN (x y : FloatV f) : FloatV f := ofPat (f.minNaN x.pat y.pat)

/-- Comparisons as Triton lowers them: `<`, `<=`, `>`, `>=`, `==` are ordered (false when a NaN
is involved); `!=` is unordered (true when a NaN is involved). `-0 == +0`. -/
def lt (x y : FloatV f) : Bool := f.cmp x.pat y.pat == some .lt
def le (x y : FloatV f) : Bool := f.cmp x.pat y.pat == some .lt || f.cmp x.pat y.pat == some .eq
def eq (x y : FloatV f) : Bool := f.cmp x.pat y.pat == some .eq
def ne (x y : FloatV f) : Bool := !(eq x y)

/-- The value nearest to the rational `q` (ties to even), as the conversion of an exact value:
`+0` for `0`; an overflow gives what `f`'s `Specials` gives, `±∞` (`.ieee`: the standard formats
and `float8e5`), the NaN (`.nanOnly`: `float8e4nv`) or `±maxVal` (`.finite`: `e2m1`). -/
def ofRat (q : Rat) : FloatV f := ofPat (f.roundRat .rn false false q)

/-- Conversion between formats with one rounding to nearest even; NaN gives the library's
canonical NaN.
This is `cvt.rn` between standard formats (`x.to(tl.float16)` from `tl.float32`, and every
widening between standard formats, which is exact). Conversions into fp8 are platform
operations (`tl.cvt8`); conversions out of fp8 are `tl.cvtFrom8`. -/
def cvt {g : Format} (x : FloatV f) : FloatV g := ofPat (Format.cvt f g .rn false x.pat)

/-- Conversion with rounding mode `rnd`: `.rn` (`cvt`) or `.rz` (truncation, `cvt.rz`). -/
def cvtWith {g : Format} (rnd : RoundMode) (x : FloatV f) : FloatV g :=
  ofPat (Format.cvt f g rnd false x.pat)

theorem cvtWith_rn {g : Format} (x : FloatV f) : (cvtWith .rn x : FloatV g) = cvt x := rfl

/-- The saturating conversion: as `cvtWith rnd`, except that a finite value beyond `g`'s range
and `±∞` give `±maxVal` of `g` (PTX `.satfinite`); NaN gives the library's canonical NaN. Within range it
is `cvtWith rnd` (`FloatV.cvtSat_of_le`). `evidence/float-layer` (part A) compares Triton's
conversions into fp8 with it on every input of `float16`, `bfloat16` and `float32`. -/
def cvtSat {g : Format} (rnd : RoundMode) (x : FloatV f) : FloatV g :=
  ofPat (Format.cvt f g rnd true x.pat)

end FloatV

/-- The standard float formats: `float16`, `bfloat16`, `float32`, `float64`. -/
def Math.Float.Format.Std (f : Format) : Prop :=
  f = .binary16 ∨ f = .bfloat16 ∨ f = .binary32 ∨ f = .binary64

/-- The fp8 formats the library covers: `float8e5` and `float8e4nv`. -/
def Math.Float.Format.F8 (f : Format) : Prop := f = .e5m2 ∨ f = .e4m3

abbrev F16 := FloatV Format.binary16
abbrev BF16 := FloatV Format.bfloat16
abbrev F32 := FloatV Format.binary32
abbrev F64 := FloatV Format.binary64
abbrev F8E5 := FloatV Format.e5m2
abbrev F8E4 := FloatV Format.e4m3

/-! ### Python number constants (library issue L23)

Where a Python `int` or `float` `c` is an operand of an arithmetic or bitwise operator on a tensor
(`+ - * / // % & | ^ << >>`) or a value of `tl.where`, or fills `tl.full(shape, c, dtype)`,
Triton 3.8.0 builds the constant directly in the dtype the operation computes in
(`TVA.Lang.Promote`; `scalar_constant`, `triton-3.8.0-semantic.py:601-613`, called at `:205-206`
and by `full` at `:622-623`), with a builder function per dtype (`triton-3.8.0-ir.cc:952-972`):
- a zero (`-0.0` included) is `+0` (`get_null_value`);
- `float64`: the Python `float` itself (`get_fp64(double)`);
- `float32`: the nearest `float32` (`get_fp32(float)`: C's conversion of the double, to nearest
  even, `hostF32`);
- `float16`: the nearest `float16` to that `float32` (`getF16FloatAttr(float)`, two roundings);
- `bfloat16`: the `bfloat16` nearest to the *decimal text* `std::to_string` prints for that
  `float32`, that is C's `%f`, six digits after the point (`APFloat(BFloat, std::to_string(v))`).
  So a constant below `5·10⁻⁷` in magnitude is `±0`, and `1.23456e-5` is the `bfloat16` nearest to
  `1.2e-5`.
A NaN keeps its sign in every dtype, and is quiet. In `float32` and `float16` it keeps the top bits
of its payload (`nanNarrow`: C's conversion and LLVM's `APFloat::convert` drop the low bits and set
the quiet bit); in `bfloat16` it has none (the text is `nan` or `-nan`); in `float64` it is the
double itself. So `float('nan')` (the double `0x7ff8000000000000`) is `0x7fc00000`, `0x7e00` and
`0x7fc0`, and `-float('nan')` the same with the sign bit set (`evidence/literals`, "NaN
constants", bit for bit). These patterns are exact; the arithmetic on them is up to NaN payload, as
every float operation is. A NaN that reaches the kernel as a compile-time value (a `tl.constexpr`
argument, or a helper's compile-time argument) is told apart from another NaN by neither Triton's
caches nor its helper names, which write every NaN as `nan`; `docs/SEMANTICS.md` ("Compile-time
values Triton keys apart") says when a helper call uses the copy made for its own NaN, and
`Rules.Compiler.Faithful`'s scope 11 assumes the kernel is new, compiled for the launch's own
values (library issues L98, L103).
The functions below take `c` as the Python `float` it is (`float(c)`, as a `float64` pattern).
Integer dtypes take `c` modulo `2^w` (`IntV.ofInt c`; `tl.full` does not check the range below 64
bits). An fp8 dtype takes the `float32` constant and truncates it with `arith.truncf`, which MLIR
folds at compile time only when the conversion loses nothing (`litF8`; library issue L64): the
constant is that exact conversion, never the device's saturating one, and when the conversion would
lose something the kernel is rejected on the host. Everywhere else (a comparison, `tl.maximum` and
the other `tl.*` functions, the condition of `tl.where`, `other=`, a stored value) the number first
becomes a tensor of its own type, `int32` or `float32` most often (`Operand.ownType?`), which is
then converted, into an fp8 dtype by the device at run time (`tl.cvt8`): `x < 0.1` with a `float64`
`x` compares against `float32(0.1)`, widened (`evidence/promotion` K1; `docs/SEMANTICS.md`,
"Constants", its two cases). -/

namespace FloatV

/-- `q` rounded to a multiple of `10⁻⁶`, ties to even: the number C's `%f` prints, with six digits
after the point, for an exact binary value `q`. -/
def round6 (q : Rat) : Rat :=
  let a : Int := q.num * 1000000
  let d : Int := q.den
  let n := a / d                      -- `⌊q · 10⁶⌋` (`Int` division rounds down for `d > 0`)
  let r2 := 2 * (a - n * d)           -- twice the remainder, against `d`
  let m : Int := if r2 < d then n else if d < r2 then n + 1 else if n % 2 = 0 then n else n + 1
  mkRat m 1000000

/-- C's conversion `(float)c` of a `double` on the host (x86-64): what the launcher's
`extractFP32` passes for a `float` argument (`float f32 = (float)temp_double`,
`triton-3.8.0-nvidia-driver.c:1152-1157`), and what the constant builders `get_fp32`, `get_fp16`
and `get_bf16` receive for a Python `float` (C++ `float` parameters, `triton-3.8.0-ir.cc:952-967`).
A number rounds to nearest even (`cvt`: an overflow is `±∞`, `-0.0` stays); a NaN keeps its sign
and the top bits of its payload, made quiet (`nanNarrow`; the device's `cvt` would give the
library's canonical NaN). `evidence/literals` ("NaN constants", N1, N2, N4) compares it bit for bit
on NaNs with every single payload bit, quiet and signalling, both signs. This is the conversion in
the host process's default floating-point environment, which `Rules.Compiler.Faithful`'s scope 7
assumes (part (B), not checked; round to nearest even; flush-to-zero and denormals-are-zero off;
library issue L110): C's
conversion follows the process's rounding mode and MXCSR bits, so after `fesetround(FE_TOWARDZERO)`
the argument `0.1` and the constant `tl.full([1], 0.1, tl.float32)` were `0x3dcccccc`, not
`0x3dcccccd`, and with flush-to-zero and denormals-are-zero set `tl.full([1], 1e-40, tl.float32)` was
`0` (`evidence/host-round21` H2, H3). -/
def hostF32 (c : F64) : F32 := if c.isNaN then nanNarrow c else cvt c

/-- The `float64` constant for the Python number `c`: the double itself, NaN payload and sign
included (`+0` for a zero). -/
def litF64 (c : F64) : F64 := if c.isZero then ofPat 0 else c

/-- The `float32` constant for the Python number `c`: the nearest `float32`, by C's conversion
(`hostF32`; `+0` for a zero). A NaN keeps its sign and the top bits of its payload, made quiet:
`float('nan')` is `0x7fc00000`, `-float('nan')` is `0xffc00000`. -/
def litF32 (c : F64) : F32 := if c.isZero then ofPat 0 else hostF32 c

/-- The `float16` constant for the Python number `c`: the nearest `float16` to its `float32`
constant (`+0` for a zero), by LLVM's `APFloat::convert`; a NaN keeps its sign and the top bits of
the `float32`'s payload (`nanNarrow`): `float('nan')` is `0x7e00`, `-float('nan')` is `0xfe00`. -/
def litF16 (c : F64) : F16 :=
  if c.isZero then ofPat 0
  else
    let x := litF32 c
    if x.isNaN then nanNarrow x else cvt x

/-- The `bfloat16` constant for the Python number `c`: the `bfloat16` nearest to the six-decimal
text of its `float32` constant (`round6`); a zero result keeps the sign the text has
(`-0.000000` for a small negative value); `±∞` stay so; a NaN, whose text is `nan` or `-nan`, is
the quiet NaN of that sign with no payload, `0x7fc0` or `0xffc0` (`+0` for a zero `c`). -/
def litBF16 (c : F64) : BF16 :=
  if c.isZero then ofPat 0
  else
    let x := litF32 c
    if x.isNaN then qNaN x.sign 0
    else if x.isInf then ofPat (Format.bfloat16.setSign x.sign Format.bfloat16.infMag)
    else ofPat (Format.bfloat16.roundRat .rn false x.sign (round6 x.toRat))

/-- The fp8 constant for the Python number `c` in `g` (`float8e5` is `e5m2`, `float8e4nv` is
`e4m3`), or `none` when Triton rejects the kernel on the host (library issue L64). `scalar_constant`
builds `+0` for a zero (`get_null_value`), and otherwise the `float32` constant `litF32 c`, then
`arith.truncf` into `g` (`triton-3.8.0-semantic.py:601-613`). MLIR folds that truncation at compile
time only when `APFloat::convert` reports no loss (`convertFloatValue`, `llvm-ArithOps.cpp:1588-1611,
1788-1825`); an unfolded one does not compile, and the kernel is rejected on the host. So the
constant is the exact conversion of the `float32` constant, with no saturation and no conversion on
the device:
* a number: the value of its `float32` constant `litF32 c` in `g`, when `g` holds that value exactly
  (`cvt` then gives it), else `none`. So `1 + 2^-30`, whose `float32` constant is `1.0`, is `1.0`,
  `-1e-50`, whose `float32` constant is `-0`, is `-0`, and `0.1`, `460.0` and a value beyond `maxVal`
  (`1000.0` in `float8e4nv`) are rejected;
* `±∞` (also a `c` whose `float32` overflows, `1e39`): `±∞` in a format with infinities
  (`float8e5`: `0x7c`, `0xfc`), else (`float8e4nv`) `none`;
* a NaN: in a format with NaN payloads (`float8e5`), the `float32` NaN's sign, its quiet bit and the
  next fraction bit (`nanNarrow`), when no lower fraction bit is set (`float('nan')` is `0x7e`,
  `-float('nan')` is `0xfe`), else `none`; in `float8e4nv` `none`.
`evidence/fp8-constants` compares it bit for bit with what `tl.full` and `tl.where` store, and the
host rejections, on every value of both formats, their neighbours and the special values. A number
converted into fp8 at run time (`.to`, a stored value, `other=`) is `tl.cvt8`, the device's
saturating conversion, instead. -/
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

end FloatV

/-! ## Triton dtypes -/

/-- The Triton scalar types the library covers. -/
inductive DType where
  /-- `tl.int1` (boolean). -/
  | i1
  | i8 | i16 | i32 | i64
  | u8 | u16 | u32 | u64
  /-- `tl.float16`, `tl.bfloat16`, `tl.float32`, `tl.float64`. -/
  | f16 | bf16 | f32 | f64
  /-- `tl.float8e5` (OCP E5M2) and `tl.float8e4nv` (OCP E4M3FN). -/
  | f8e5 | f8e4nv
  deriving DecidableEq, Repr

namespace DType

/-- The Lean type of a dtype's values. -/
@[reducible] def Val : DType → Type
  | .i1 => Bool
  | .i8 => I8 | .i16 => I16 | .i32 => I32 | .i64 => I64
  | .u8 => U8 | .u16 => U16 | .u32 => U32 | .u64 => U64
  | .f16 => F16 | .bf16 => BF16 | .f32 => F32 | .f64 => F64
  | .f8e5 => F8E5 | .f8e4nv => F8E4

/-- Bytes an element occupies in memory (`int1` takes a byte). -/
def bytes : DType → Nat
  | .i1 | .i8 | .u8 | .f8e5 | .f8e4nv => 1
  | .i16 | .u16 | .f16 | .bf16 => 2
  | .i32 | .u32 | .f32 => 4
  | .i64 | .u64 | .f64 => 8

theorem bytes_pos (d : DType) : 0 < d.bytes := by cases d <;> decide

/-- The bit pattern of a value, stored as `bytes` little-endian bytes. -/
def toBits : (d : DType) → d.Val → Nat
  | .i1, b => if b then 1 else 0
  | .i8, x => x.bits.toNat | .i16, x => x.bits.toNat | .i32, x => x.bits.toNat
  | .i64, x => x.bits.toNat
  | .u8, x => x.bits.toNat | .u16, x => x.bits.toNat | .u32, x => x.bits.toNat
  | .u64, x => x.bits.toNat
  | .f16, x => x.pat | .bf16, x => x.pat | .f32, x => x.pat | .f64, x => x.pat
  | .f8e5, x => x.pat | .f8e4nv, x => x.pat

/-- The value with the given bit pattern; an `int1` byte reads as `true` when non-zero. -/
def ofBits : (d : DType) → Nat → d.Val
  | .i1, n => n != 0
  | .i8, n => IntV.ofNat n | .i16, n => IntV.ofNat n | .i32, n => IntV.ofNat n
  | .i64, n => IntV.ofNat n
  | .u8, n => IntV.ofNat n | .u16, n => IntV.ofNat n | .u32, n => IntV.ofNat n
  | .u64, n => IntV.ofNat n
  | .f16, n => FloatV.ofPat n | .bf16, n => FloatV.ofPat n | .f32, n => FloatV.ofPat n
  | .f64, n => FloatV.ofPat n
  | .f8e5, n => FloatV.ofPat n | .f8e4nv, n => FloatV.ofPat n

instance (d : DType) : Inhabited d.Val := ⟨d.ofBits 0⟩

instance (d : DType) : DecidableEq d.Val := by
  cases d <;> exact inferInstance

/-- The value is a NaN (never, for integer and boolean dtypes). -/
def IsNaN : (d : DType) → d.Val → Prop
  | .f16, x => x.isNaN = true | .bf16, x => x.isNaN = true | .f32, x => x.isNaN = true
  | .f64, x => x.isNaN = true | .f8e5, x => x.isNaN = true | .f8e4nv, x => x.isNaN = true
  | _, _ => False

/-- Its float format, for the floating dtypes. -/
def format? : DType → Option Format
  | .f16 => some .binary16 | .bf16 => some .bfloat16 | .f32 => some .binary32
  | .f64 => some .binary64 | .f8e5 => some .e5m2 | .f8e4nv => some .e4m3
  | _ => none

/-- Every float dtype has a NaN. -/
theorem exists_nan : (d : DType) → d.format?.isSome = true → ∃ z : d.Val, d.IsNaN z
  | .f16, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .bf16, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .f32, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .f64, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .f8e5, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .f8e4nv, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .i1, h | .i8, h | .i16, h | .i32, h | .i64, h | .u8, h | .u16, h | .u32, h | .u64, h => by
      simp [format?] at h

end DType

end TVA
