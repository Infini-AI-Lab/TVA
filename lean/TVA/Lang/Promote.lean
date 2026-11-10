import TVA.Lang.Ops

/-!
# Dtype promotion and conversion (library issue L45)

What an operator of the source computes in when its operands differ in dtype, or one of them is a
Python number, as Triton 3.8.0's front end decides it, and the conversion it then applies to each
operand. These are the library's one statement of Triton's promotion: every implicit conversion
of a mixed-dtype operator in a model is `tl.convert` to the dtype these definitions give, and a
prover evaluates them on the kernel's operands at proof time (`decide`, `rfl`, `#eval`).
`evidence/promotion` compares them with the front end on 40240 operand pairs (no difference) and
checks the conversions and the arithmetic on the device.

The sources: `semantic.py:53-116` (`integer_promote_impl`, `computation_type_impl`), `:118-158`
(`to_tensor`, `to_tensor_type`), `:175-210` (`binary_op_type_checking_impl`), `:230-577` (the
operators), `:811-916` (`cast`), `:1638-1652` (`where`); `core.py:900-1105` (the tensor's
operators: a comparison first makes a Python number a tensor of its own type), `:2912-2954`
(`tl.minimum`, `tl.maximum`: also `bfloat16` to `float32`), `:3882-3929` (Python's `min` and
`max` in a kernel); `code_generator.py:795-808` (which operand's method runs), `:1097-1114` (unary
operators), `:1481-1531` (`and`, `or`).

* `Operand`: an operand as the source has it: a tensor of a dtype (a block, or a scalar of the
  program), or a Python number (a literal or a `tl.constexpr` holding one). A name that a statement
  assigns a Python number holds a tensor of the number's own dtype, not a Python number (library
  issue L59; see `Operand`).
* `Operand.ownType?`: the dtype a Python number becomes as a tensor of its own (`to_tensor_type`).
* `computationType?`: the dtype a binary operator computes in.
* `BinOp.typing?`, `UnOp.typing?`, `whereTyping?`, `boolOp?`: per operator, the dtype both operands
  are converted to, the result's dtype, and whether the result is a block; `none` when the front
  end rejects it (an error on the host before any program runs: no outcome, `TVA.Claim.Behaviour`).
* `ConvRel`, `tl.convert`: `semantic.cast` with the default rounding, element by element.
* `tl.bitcast` (`DType.bitwidth`): `bitcast=True`, each element's bits read in the target dtype
  (library issue L143).
* `Operand.constIn?`: the value Triton builds for a Python number in a dtype (`docs/SEMANTICS.md`,
  "Constants").

When both operands are Python numbers, Python computes the result at compile time; these
definitions do not apply. Pointer operands (`p + i`) are `Ptr` arithmetic (`docs/SEMANTICS.md`,
"Values"), not covered here.
-/

namespace TVA

open TVA.Math.Float

namespace DType

/-- Triton's name of the dtype, as `tl.dtype` prints it. -/
def tname : DType → String
  | .i1 => "int1" | .i8 => "int8" | .i16 => "int16" | .i32 => "int32" | .i64 => "int64"
  | .u8 => "uint8" | .u16 => "uint16" | .u32 => "uint32" | .u64 => "uint64"
  | .f16 => "fp16" | .bf16 => "bf16" | .f32 => "fp32" | .f64 => "fp64"
  | .f8e5 => "fp8e5" | .f8e4nv => "fp8e4nv"

/-- Every dtype the library covers. -/
def all : List DType :=
  [.i1, .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .f8e5, .f8e4nv, .f16, .bf16, .f32, .f64]

/-- An integer dtype, `int1` included: Triton's `dtype.is_int` (`core.py:518`). -/
def isTritonInt : DType → Bool
  | .i1 | .i8 | .i16 | .i32 | .i64 | .u8 | .u16 | .u32 | .u64 => true
  | _ => false

/-- A signed integer dtype; `int1` is unsigned (`dtype.SINT_TYPES`, `UINT_TYPES`,
`core.py:391-392`). -/
def isSigned : DType → Bool
  | .i8 | .i16 | .i32 | .i64 => true
  | _ => false

/-- An fp8 dtype. -/
def isFp8 : DType → Bool
  | .f8e5 | .f8e4nv => true
  | _ => false

/-- The bit width of an integer dtype (`1` for `int1`), `0` for a float. -/
def intWidth : DType → Nat
  | .i1 => 1 | .i8 | .u8 => 8 | .i16 | .u16 => 16 | .i32 | .u32 => 32 | .i64 | .u64 => 64
  | _ => 0

/-- `dtype.kind()` (`core.py:401-404, 524-532`): `0` for `int1`, `1` for the other integers, `2`
for floats. -/
def kind : DType → Nat
  | .i1 => 0
  | d => if d.isTritonInt then 1 else 2

/-- The smallest and largest values of an integer dtype. -/
def intMin (d : DType) : Int := if d.isSigned then -(2 ^ (d.intWidth - 1) : Int) else 0
def intMax (d : DType) : Int :=
  if d.isSigned then (2 ^ (d.intWidth - 1) : Int) - 1 else (2 ^ d.intWidth : Int) - 1

end DType

/-- An operand of an operator, as the source has it: a tensor of dtype `d` (`block` when it is a
block; otherwise a scalar of the program, such as a loaded scalar or a scalar argument), or a
Python number: an `int`, a `float` (its `float64` pattern) or a `bool`, written as a literal in
place, held by a `tl.constexpr` (a parameter, a name bound by `name: tl.constexpr = e`, a
compile-time expression of these), or passed directly to a helper or given by a helper's default,
where its parameter holds it.

A name that a statement assigns a Python number (`=`, `+=` and the other augmented assignments, a
tuple assignment, element by element) holds instead a scalar tensor of the number's own dtype
(`ownType?`), with the constant `constIn?` gives in that dtype, unless the statement is
`name: tl.constexpr = e` on a name not defined before (`visit_Assign`, `visit_AnnAssign`,
`code_generator.py:694-760`; library issues L59, L84, `evidence/python-numbers` A1-A12,
`evidence/front-end-subset` C1-C9; a later plain assignment to that name makes a tensor again).
Such a name is `.tensor (ownType? c) false` here: after
`e = 1e-7`, `x * e` with a `bfloat16` `x` computes in `float32`, and after `a = -7; b = 2`, `a // b`
is a tensor `//` (`-3`), not Python's (`-4`). -/
inductive Operand where
  | tensor (d : DType) (block : Bool)
  | int (v : Int)
  | float (x : F64)
  | bool (b : Bool)

namespace Operand

/-- A Python number. -/
def isPy : Operand → Bool
  | .tensor .. => false
  | _ => true

/-- A block (a Python number is not). -/
def block : Operand → Bool
  | .tensor _ b => b
  | _ => false

/-- `float32`'s range for a Python `float` (`to_tensor_type`): zero, infinite, NaN, or of
magnitude in `[2^-126, (2 - 2^-23)·2^127]`. -/
def f32Range (x : F64) : Bool :=
  x.isZero || x.isInf || x.isNaN ||
    (let a := if x.toRat < 0 then -x.toRat else x.toRat
     decide ((1 : Rat) / 2 ^ 126 ≤ a) && decide (a ≤ (2 - (1 : Rat) / 2 ^ 23) * 2 ^ 127))

/-- `to_tensor_type` (`semantic.py:129-158`): a tensor's own dtype, or the dtype a Python number
becomes as a tensor of its own: a `bool` is `int1`; an `int` is `int32`, `uint32`, `int64` or
`uint64`, the first whose range holds it (`none` beyond: rejected); a `float` is `float32` in
`f32Range`, and `float64` otherwise. -/
def ownType? : Operand → Option DType
  | .tensor d _ => some d
  | .bool _ => some .i1
  | .int v =>
    if -2 ^ 31 ≤ v ∧ v < 2 ^ 31 then some .i32
    else if 2 ^ 31 ≤ v ∧ v < 2 ^ 32 then some .u32
    else if -2 ^ 63 ≤ v ∧ v < 2 ^ 63 then some .i64
    else if 2 ^ 63 ≤ v ∧ v < 2 ^ 64 then some .u64
    else none
  | .float x => some (if f32Range x then .f32 else .f64)

/-- `to_tensor`: a Python number as a scalar tensor of its own dtype; a tensor unchanged. -/
def toTensor? (x : Operand) : Option Operand := x.ownType?.map (fun d => .tensor d x.block)

/-- A Python integer's value (a `bool` is `0` or `1`); `none` for a tensor or a `float`. -/
def intVal? : Operand → Option Int
  | .int v => some v
  | .bool b => some (if b then 1 else 0)
  | _ => none

/-- A Python number's value as a double (`float(c)`); `none` for a tensor. -/
def toF64? : Operand → Option F64
  | .float x => some x
  | .int v => some (FloatV.ofRat (v : Rat))
  | .bool b => some (FloatV.ofRat (if b then 1 else 0))
  | .tensor .. => none

/-- The value Triton builds for the Python number `c` in dtype `d` (`scalar_constant`,
`semantic.py:601-613`; `docs/SEMANTICS.md`, "Constants"): the `FloatV.litF64`, `litF32`, `litF16`
and `litBF16` functions of its double for the standard float dtypes; for an fp8 dtype, `FloatV.litF8`,
the exact conversion of its `float32` constant, which MLIR folds at compile time (library issue
L64), and `none` when that conversion is not exact, where Triton rejects the kernel on the host;
`c` modulo `2^w` for an integer dtype, `c ≠ 0` for `int1`; `none` for a tensor and for a `float` in
an integer dtype. -/
def constIn? (c : Operand) : (d : DType) → Option d.Val
  | .f64 => c.toF64?.map FloatV.litF64
  | .f32 => c.toF64?.map FloatV.litF32
  | .f16 => c.toF64?.map FloatV.litF16
  | .bf16 => c.toF64?.map FloatV.litBF16
  | .f8e5 => c.toF64?.bind FloatV.litF8
  | .f8e4nv => c.toF64?.bind FloatV.litF8
  | .i1 => c.intVal?.map (· ≠ 0)
  | .i8 => c.intVal?.map IntV.ofInt | .i16 => c.intVal?.map IntV.ofInt
  | .i32 => c.intVal?.map IntV.ofInt | .i64 => c.intVal?.map IntV.ofInt
  | .u8 => c.intVal?.map IntV.ofInt | .u16 => c.intVal?.map IntV.ofInt
  | .u32 => c.intVal?.map IntV.ofInt | .u64 => c.intVal?.map IntV.ofInt

end Operand

/-- `integer_promote_impl` (`semantic.py:53-66`), C's usual arithmetic conversions: the wider of
two integers of the same signedness; with mixed signedness, the unsigned one when it is at least
as wide, the signed one otherwise (`int1` counts as unsigned, of width 1). -/
def intPromote (a b : DType) : DType :=
  if a.isSigned = b.isSigned then (if a.intWidth > b.intWidth then a else b)
  else if !a.isSigned then (if a.intWidth ≥ b.intWidth then a else b)
  else (if b.intWidth ≥ a.intWidth then b else a)

/-- `computation_type_impl` (`semantic.py:68-116`): the dtype a binary operator computes in, from
its operands' dtypes (a Python number's own dtype, `Operand.ownType?`) and which of them are
Python numbers; `divMod` for `/`, `//` and `%`. In order:
0. one operand a Python number of a kind no higher than the tensor's (`DType.kind`): the tensor's
   dtype (`float32` for `float16`/`bfloat16` under `divMod`);
1. either `float64`: `float64`; 2. either `float32`: `float32`;
3. either `float16`: `float16` (`float32` under `divMod`), so `float16` with `bfloat16` is
   `float16`;
4. both `bfloat16`: `bfloat16` (`float32` under `divMod`); one `bfloat16`: `float32`;
5. two fp8: the same one, or `float16` for two different ones;
6. otherwise both must be integers (an fp8 with an integer is rejected), not of mixed signedness
   under `divMod`, and the result is `intPromote`. -/
def computationType? (a : DType) (aPy : Bool) (b : DType) (bPy : Bool) (divMod : Bool) :
    Option DType :=
  let step0 : Option DType :=
    if aPy != bPy then
      let (sc, tn) := if aPy then (a, b) else (b, a)
      if sc.kind ≤ tn.kind then
        some (if divMod && (tn == .f16 || tn == .bf16) then .f32 else tn)
      else none
    else none
  match step0 with
  | some d => some d
  | none =>
    if a == .f64 || b == .f64 then some .f64
    else if a == .f32 || b == .f32 then some .f32
    else if a == .f16 || b == .f16 then some (if divMod then .f32 else .f16)
    else if a == .bf16 && b == .bf16 then some (if divMod then .f32 else .bf16)
    else if a == .bf16 || b == .bf16 then some .f32
    else if a.isFp8 && b.isFp8 then some (if a == b then a else .f16)
    else if !a.isTritonInt || !b.isTritonInt then none
    else if divMod && a.isSigned != b.isSigned then none
    else some (intPromote a b)

/-- How the front end types an operator: the dtype both operands are converted to (`opnd`), the
result's dtype (`res`), and whether the result is a block. -/
structure Typing where
  opnd : DType
  res : DType
  block : Bool
  deriving DecidableEq, Repr

/-- `binary_op_type_checking_impl` (`semantic.py:175-210`) on two operands, at least one a tensor:
the computation dtype, after rejecting a negative Python number with an unsigned result and a
Python integer outside an integer result's range. -/
def common? (x y : Operand) (divMod : Bool) : Option DType := do
  let a ← x.ownType?
  let b ← y.ownType?
  let r ← computationType? a x.isPy b y.isPy divMod
  let okPy (o : Operand) : Bool :=
    match o with
    | .tensor .. => true
    | .float v => !(r.isTritonInt && !r.isSigned && decide (v.toRat < 0))
    | o => match o.intVal? with
      | some v => !(r.isTritonInt && !r.isSigned && decide (v < 0)) &&
          (!r.isTritonInt || (decide (r.intMin ≤ v) && decide (v ≤ r.intMax)))
      | none => true
  if okPy x && okPy y then some r else none

/-- The binary operators of the source on tensors, and `tl.minimum`/`tl.maximum` (also Python's
`min` and `max` of two operands in a kernel when one is a tensor, `core.py:3882-3929`). -/
inductive BinOp where
  | add | sub | mul | div | floordiv | mod | band | bor | bxor | shl | shr
  | lt | le | gt | ge | eq | ne | minimum | maximum
  deriving DecidableEq, Repr

/-- `float32` for `bfloat16` (`_promote_bfloat16_to_float32`, `core.py:3083-3089`), applied to each
operand of `tl.minimum` and `tl.maximum`. -/
def bf16ToF32 : Operand → Operand
  | .tensor .bf16 b => .tensor .f32 b
  | o => o

/-- How the front end types `x op y` when at least one operand is a tensor (`none`: rejected on
the host).
* `+ - *`: `common?`; the result is that dtype.
* `/` (`truediv`, `semantic.py:292-315`): `common?` with `divMod`; two integers are then both
  converted to `float32` (`int64` included), so `opnd` is `float32`.
* `//` (`:317-329`): `common?` with `divMod`; integers only. `%` (`:343-360`): `common?` with
  `divMod`; a float `%` is `frem`.
* `& | ^ << >>` (`:415-438, 459-469`): `common?`; integers only. Which shift `>>` is (arithmetic or
  logical) depends on the source dtype of the operand whose method runs (`tl.ashr`, `tl.lshr`).
* comparisons (`:501-577`): each Python number is first a tensor of its own dtype
  (`core.py:900-1105`), so a constant does not take the tensor's dtype: `x < 0.1` with a `float64`
  `x` compares against the `float32` constant, widened; then `common?`; the result is `int1`.
* `tl.minimum`, `tl.maximum`: the same own-dtype step, then `bfloat16` becomes `float32`, then
  `common?`. -/
def BinOp.typing? (op : BinOp) (x y : Operand) : Option Typing :=
  if x.isPy && y.isPy then none else
  let blk := x.block || y.block
  match op with
  | .add | .sub | .mul => (common? x y false).map fun r => ⟨r, r, blk⟩
  | .div => (common? x y true).map fun r => if r.isTritonInt then ⟨.f32, .f32, blk⟩ else ⟨r, r, blk⟩
  | .floordiv => (common? x y true).bind fun r => if r.isTritonInt then some ⟨r, r, blk⟩ else none
  | .mod => (common? x y true).map fun r => ⟨r, r, blk⟩
  | .band | .bor | .bxor | .shl | .shr =>
    (common? x y false).bind fun r => if r.isTritonInt then some ⟨r, r, blk⟩ else none
  | .lt | .le | .gt | .ge | .eq | .ne => do
    let x' ← x.toTensor?
    let y' ← y.toTensor?
    let r ← common? x' y' false
    some ⟨r, .i1, blk⟩
  | .minimum | .maximum => do
    let x' ← x.toTensor?
    let y' ← y.toTensor?
    let r ← common? (bf16ToF32 x') (bf16ToF32 y') false
    some ⟨r, r, blk⟩

/-- Unary operators on a tensor: `-x`, `~x`, `not x`, `+x`. -/
inductive UnOp where
  | neg | invert | not | pos
  deriving DecidableEq, Repr

/-- How the front end types a unary operator on a tensor (`code_generator.py:1097-1114`): `-x`
keeps the dtype (`fneg` for a float, exact up to NaN payload, `tl.neg`; `0 - x`, wrapping, for an
integer; `semantic.py:478-485`); `~x` needs an integer (`x ^ all-ones`, `:487-492`); `not x` needs
`int1` (`logical_not`, `:440-457`); `+x` is rejected (a tensor has no `__pos__`). -/
def UnOp.typing? (op : UnOp) : Operand → Option Typing
  | .tensor d b =>
    match op with
    | .neg => some ⟨d, d, b⟩
    | .invert => if d.isTritonInt then some ⟨d, d, b⟩ else none
    | .not => if d == .i1 then some ⟨.i1, .i1, b⟩ else none
    | .pos => none
  | _ => none

/-- `tl.where(c, x, y)` (`core.py:2837-2856`, `semantic.py:1638-1652`): the condition becomes a
tensor (a Python number of its own dtype) and is converted to `int1` (`c != 0`; an fp8 condition
is rejected); `x` and `y` are typed as by `common?`, Python numbers included, so a constant takes
the other operand's dtype; the result is a block if any of the three is. -/
def whereTyping? (c x y : Operand) : Option Typing := do
  let cd ← c.ownType?
  if cd.isFp8 then none else
  let r ← common? x y false
  some ⟨r, r, c.block || x.block || y.block⟩

/-- What `a and b` / `a or b` gives (`code_generator.py:1481-1531`). -/
inductive BoolOpRes where
  /-- a Python value, at compile time -/
  | python
  /-- a tensor -/
  | tensor (d : DType) (block : Bool)
  deriving DecidableEq, Repr

/-- `a and b` (`isAnd`) and `a or b`: a Python value that decides the result (false for `and`,
true for `or`) is the result, and the operands after it are not evaluated; another Python value
is dropped, so `x and True` is `x`; two tensors must both be `int1`, and the result is their
elementwise `&` (`|`). -/
def boolOp? (isAnd : Bool) (x y : Operand) : Option BoolOpRes :=
  let decides (o : Operand) : Bool :=
    match o with
    | .tensor .. => false
    | .bool b => b != isAnd
    | .int v => (v != 0) != isAnd
    | .float f => (!f.isZero) != isAnd
  if x.isPy && decides x then some .python
  else match x, y with
  | .tensor a ba, .tensor b bb =>
    if a == .i1 && b == .i1 then some (.tensor .i1 (ba || bb)) else none
  | .tensor a ba, _ => if decides y then some .python else some (.tensor a ba)
  | _, .tensor b bb => some (.tensor b bb)
  | _, _ => some .python

/-! ## Conversions (`semantic.cast`) -/

/-- An integer value `z` (a source integer's value, or `0`/`1` for `int1`) converted to dtype
`d'` (`semantic.py:870-898`): `z ≠ 0` for `int1`; `z` modulo `2^w` for an integer (sign- or
zero-extension by the source's signedness, then truncation, which is the value modulo `2^w`);
`tl.ItofRel` for a standard float (`sitofp`/`uitofp`); no conversion into fp8 (rejected). -/
def ofIntRel (z : Int) : (d' : DType) → d'.Val → Prop
  | .i1 => fun r => r = decide (z ≠ 0)
  | .i8 => fun r => r = IntV.ofInt z | .i16 => fun r => r = IntV.ofInt z
  | .i32 => fun r => r = IntV.ofInt z | .i64 => fun r => r = IntV.ofInt z
  | .u8 => fun r => r = IntV.ofInt z | .u16 => fun r => r = IntV.ofInt z
  | .u32 => fun r => r = IntV.ofInt z | .u64 => fun r => r = IntV.ofInt z
  | .f16 => fun r => tl.ItofRel .binary16 z r | .bf16 => fun r => tl.ItofRel .bfloat16 z r
  | .f32 => fun r => tl.ItofRel .binary32 z r | .f64 => fun r => tl.ItofRel .binary64 z r
  | .f8e5 | .f8e4nv => fun _ => False

/-- A float `x` of format `f` converted to an integer dtype (`fptosi`/`fptoui`, `semantic.py:883-891`):
`tl.ftoi`'s truncation, from a standard format only; no result (undefined) when `x` is NaN,
infinite or out of range. -/
def FtoiRel {f : Format} (sg : Bool) (w : Nat) (x : FloatV f) (r : IntV sg w) : Prop :=
  f.Std ∧ x.isFinite = true ∧
    (Format.intRange sg w).1 ≤ x.toRat.num.tdiv x.toRat.den ∧
    x.toRat.num.tdiv x.toRat.den ≤ (Format.intRange sg w).2 ∧
    r = IntV.ofInt (x.toRat.num.tdiv x.toRat.den)

/-- A float `x` of format `f` converted to format `g` with the default rounding
(`semantic.py:838-868`): the same value when `f = g` (no conversion); between standard formats one
rounding to nearest even, up to NaN payload (`tl.cvt`; a `float16`/`bfloat16` source goes through
`float32`, which is exact); from `float16`, `bfloat16` or `float32` into fp8, the platform's
conversion (`tl.cvt8`); from fp8 into `float16`, `bfloat16` or `float32`, exact on finite values
(`tl.cvtFrom8`); any other pair (fp8 to fp8, fp8 to or from `float64`) has no result. -/
def FloatRel (P : Platform) (f g : Format) (x : FloatV f) (r : FloatV g) : Prop :=
  if f = g then r.pat = x.pat
  else (f.Std ∧ g.Std ∧ FloatV.NaNEq (FloatV.cvt x) r) ∨
    ((f = .binary16 ∨ f = .bfloat16 ∨ f = .binary32) ∧ g.F8 ∧ P.cvt f g .rn x r) ∨
    (f.F8 ∧ (g = .binary16 ∨ g = .bfloat16 ∨ g = .binary32) ∧ (x.isFinite = true → r = FloatV.cvt x))

/-- A float `x` of format `f` converted to dtype `d'`: `x != 0` for `int1` (unordered: a NaN gives
`true`, `-0.0` gives `false`; `semantic.py:884-887`), `FtoiRel` for an integer, `FloatRel` for a
float. -/
def ofFloatRel (P : Platform) {f : Format} (x : FloatV f) : (d' : DType) → d'.Val → Prop
  | .i1 => fun r => r = FloatV.ne x (FloatV.ofPat 0)
  | .i8 => fun r => FtoiRel true 8 x r | .i16 => fun r => FtoiRel true 16 x r
  | .i32 => fun r => FtoiRel true 32 x r | .i64 => fun r => FtoiRel true 64 x r
  | .u8 => fun r => FtoiRel false 8 x r | .u16 => fun r => FtoiRel false 16 x r
  | .u32 => fun r => FtoiRel false 32 x r | .u64 => fun r => FtoiRel false 64 x r
  | .f16 => fun r => FloatRel P f .binary16 x r | .bf16 => fun r => FloatRel P f .bfloat16 x r
  | .f32 => fun r => FloatRel P f .binary32 x r | .f64 => fun r => FloatRel P f .binary64 x r
  | .f8e5 => fun r => FloatRel P f .e5m2 x r | .f8e4nv => fun r => FloatRel P f .e4m3 x r

/-- `semantic.cast(x, d')` (`semantic.py:811-916`) with the default rounding, for one element of
dtype `d`: what the result `r` may be. No `r` exists for a conversion that does not compile or is
undefined (`tl.convert` is then undefined behaviour). -/
def ConvRel (P : Platform) : (d d' : DType) → d.Val → d'.Val → Prop
  | .i1, d' => fun x r => ofIntRel (if x then 1 else 0) d' r
  | .i8, d' => fun x r => ofIntRel x.toInt d' r | .i16, d' => fun x r => ofIntRel x.toInt d' r
  | .i32, d' => fun x r => ofIntRel x.toInt d' r | .i64, d' => fun x r => ofIntRel x.toInt d' r
  | .u8, d' => fun x r => ofIntRel x.toInt d' r | .u16, d' => fun x r => ofIntRel x.toInt d' r
  | .u32, d' => fun x r => ofIntRel x.toInt d' r | .u64, d' => fun x r => ofIntRel x.toInt d' r
  | .f16, d' => fun x r => ofFloatRel P x d' r | .bf16, d' => fun x r => ofFloatRel P x d' r
  | .f32, d' => fun x r => ofFloatRel P x d' r | .f64, d' => fun x r => ofFloatRel P x d' r
  | .f8e5, d' => fun x r => ofFloatRel P x d' r | .f8e4nv, d' => fun x r => ofFloatRel P x d' r

namespace tl

/-- `x.to(d')` with the default rounding, and every implicit conversion of an operand: the
promotion of a mixed-dtype operator (`BinOp.typing?` and the others give the target), the
condition of `tl.where` (to `int1`). Each element as `ConvRel` allows; undefined behaviour if some
element has no result. -/
def convert {s : Shape} (d d' : DType) (x : Tensor s d.Val) : Prog (Tensor s d'.Val) :=
  .env fun e => .pick (fun r : Tensor s d'.Val => ∀ i, ConvRel e.P d d' (x.get i) (r.get i)) Solve.none .ret

end tl

/-! ## Bit casts (library issue L143) -/

/-- A dtype's width in bits, Triton's `primitive_bitwidth`: `1` for `int1`, and eight bits per
byte of the others. -/
def DType.bitwidth (d : DType) : Nat := if d = .i1 then 1 else 8 * d.bytes

/-- Reading a value's bit pattern back in its own dtype gives the value. -/
theorem DType.ofBits_toBits : (d : DType) → (v : d.Val) → d.ofBits (d.toBits v) = v
  | .i1, b => by cases b <;> rfl
  | .i8, ⟨b⟩ | .i16, ⟨b⟩ | .i32, ⟨b⟩ | .i64, ⟨b⟩ | .u8, ⟨b⟩ | .u16, ⟨b⟩ | .u32, ⟨b⟩ | .u64, ⟨b⟩ => by
      simp [DType.ofBits, DType.toBits, IntV.ofNat]
  | .f16, ⟨b⟩ | .bf16, ⟨b⟩ | .f32, ⟨b⟩ | .f64, ⟨b⟩ | .f8e5, ⟨b⟩ | .f8e4nv, ⟨b⟩ => by
      simp [DType.ofBits, DType.toBits, FloatV.ofPat, FloatV.pat]

namespace tl

/-- `x.to(d', bitcast=True)`, `tl.cast(x, d', bitcast=True)` and `x.cast(d', bitcast=True)` of a
tensor of dtype `d` (`triton-3.8.0-core.py:2322-2345`, `triton-3.8.0-semantic.py:793-809`): each
element's bit pattern read as a value of `d'` (`DType.toBits`, `DType.ofBits`), exact, NaN payloads
included; `fp_downcast_rounding` is ignored. Triton requires the same width (`DType.bitwidth`, the
proof `h`; for literal dtypes the default proof is found by itself) and rejects every other pair on the
host. For `d' = d` it is `x` itself (`bitcast_self`), as Triton returns its input. A pointer operand or
target is not this operation: Triton converts it as without `bitcast` (`semantic.py:801-802`). -/
def bitcast {s : Shape} (d d' : DType) (x : Tensor s d.Val)
    (_h : d.bitwidth = d'.bitwidth := by decide) : Tensor s d'.Val :=
  x.map fun v => d'.ofBits (d.toBits v)

theorem bitcast_self {s : Shape} (d : DType) (x : Tensor s d.Val) (h : d.bitwidth = d.bitwidth) :
    bitcast d d x h = x := by
  ext i
  exact DType.ofBits_toBits d (x.get i)

end tl

end TVA
