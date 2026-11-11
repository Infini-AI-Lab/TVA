import TVA.Triton.Convert

/-!
# Dtype promotion and Python numbers

What an operator of the source computes in when its operands differ in dtype or one of them is a
Python number, as Triton's front end decides it, and the constant Triton builds for a Python
number. These are TVA's one statement of Triton's promotion: a model writes each implicit
conversion as `tl.convert d d' x` to the dtype these definitions give, and a prover evaluates them
on the kernel's dtypes (`decide`, `rfl`).

Sources: `semantic.py:53-116` (`integer_promote_impl`, `computation_type_impl`), `:118-158`
(`to_tensor`), `:175-210` (`binary_op_type_checking_impl`), `:230-577` (the operators),
`:601-613` (`scalar_constant`), `:1638-1652` (`where`); `core.py:900-1105` (a comparison first makes
a Python number a tensor), `:2912-2954` (`tl.minimum`, `tl.maximum`), `:3882-3929` (Python's `min`
and `max`); `code_generator.py:1097-1114` (unary operators), `:1481-1531` (`and`, `or`).

* `Operand`: an operand as the source has it, a tensor of a dtype or a Python number.
* `Operand.ownType?`: the dtype a Python number becomes as a tensor of its own.
* `Operand.constIn?`: the constant Triton builds for a Python number in a dtype.
* `computationType?`, `BinOp.typing?`, `UnOp.typing?`, `whereTyping?`, `boolOp?`: per operator,
  the dtype both operands are converted to, the result's dtype, and whether the result is a block;
  `none` when the front end rejects it on the host (no program runs).

**Where a Python number takes which dtype.** A Python number `c` (a literal, a `tl.constexpr`
holding one, a builtin such as `float('inf')`) used as an operand of `+ - * / // % & | ^ << >>`
on a tensor, as a value of `tl.where`, or in `tl.full`, is built directly in the dtype the operation
computes in (`constIn?` of that dtype): `x * 0.1` with a `bfloat16` `x` computes in `bfloat16`.
Everywhere else (a comparison, the condition of `tl.where`, an argument of another `tl.*`
function, a stored value, `other=` of `tl.load`, `tl.cast`) it first becomes a tensor of its own
dtype (`ownType?`), then the operation's own promotion or conversion applies: `x < 0.1` with a
`float64` `x` compares against `float32(0.1)`, widened; a stored `1000.0` into `float8e4nv` goes
through `tl.cvt8`. A *name* a statement assigned a Python number holds a scalar tensor of the
number's own dtype, not a Python number (`.tensor (ownType? c) false`), unless the statement is
`name: tl.constexpr = e`.

When both operands are Python numbers, Python computes the result at compile time and these
definitions do not apply. Pointer arithmetic (`p + i`) is `Ptr` arithmetic (`TVA.Memory`).
-/
namespace TVA

namespace DType

/-- Triton's name of the dtype, as `tl.dtype` prints it. -/
def tname : DType → String
  | .i1 => "int1" | .i8 => "int8" | .i16 => "int16" | .i32 => "int32" | .i64 => "int64"
  | .u8 => "uint8" | .u16 => "uint16" | .u32 => "uint32" | .u64 => "uint64"
  | .f16 => "fp16" | .bf16 => "bf16" | .f32 => "fp32" | .f64 => "fp64"
  | .f8e5 => "fp8e5" | .f8e4nv => "fp8e4nv"

/-- Every dtype TVA covers. -/
def all : List DType :=
  [.i1, .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .f8e5, .f8e4nv, .f16, .bf16, .f32, .f64]

/-- An integer dtype, `int1` included: Triton's `dtype.is_int` (`core.py:518`). -/
def isTritonInt : DType → Bool
  | .i1 | .i8 | .i16 | .i32 | .i64 | .u8 | .u16 | .u32 | .u64 => true
  | _ => false

/-- A signed integer dtype; `int1` is unsigned (`core.py:391-392`). -/
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

/-- The smallest value of an integer dtype. -/
def intMin (d : DType) : Int := if d.isSigned then -(2 ^ (d.intWidth - 1) : Int) else 0

/-- The largest value of an integer dtype. -/
def intMax (d : DType) : Int :=
  if d.isSigned then (2 ^ (d.intWidth - 1) : Int) - 1 else (2 ^ d.intWidth : Int) - 1

end DType

/-- An operand of an operator as the source has it: a tensor of dtype `d` (`block` when it is a
block, otherwise a scalar of the program such as a loaded scalar or a scalar argument), or a Python
number (an `int`, a `float` given by its `float64` pattern, a `bool`), written in place, held by a
`tl.constexpr`, or passed directly to a helper. -/
inductive Operand where
  /-- A tensor of dtype `d`; `block` unless it is a scalar. -/
  | tensor (d : DType) (block : Bool)
  /-- A Python `int`. -/
  | int (v : Int)
  /-- A Python `float`. -/
  | float (x : F64)
  /-- A Python `bool`. -/
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

/-- The constant Triton builds for the Python number `c` in dtype `d` (`scalar_constant`,
`semantic.py:601-613`), a function of `c`'s double (`TVA.Base.Float.Literal`):
* `float64`: `FloatV.litF64`, the double itself; `float32`: `litF32`, the nearest `float32`;
  `float16`: `litF16`, the nearest `float16` to that `float32`; `bfloat16`: `litBF16`, the
  `bfloat16` nearest to the six-decimal text of that `float32` (every constant below `5·10⁻⁷` in
  magnitude is `±0`);
* an fp8 dtype: `litF8`, its `float32` constant when the fp8 dtype holds that value exactly (MLIR
  folds the `truncf` only then), and `none` otherwise, where Triton rejects the kernel on the host;
* an integer dtype: `c` modulo `2^w` (`tl.full` does not check the range: `tl.full([1], 300,
  tl.int8)` is `44`); `int1`: `c ≠ 0`; `none` for a `float` in an integer dtype and for a tensor.
Each float constant is `+0` for a zero, `-0.0` included, and a NaN keeps its sign and is quiet. -/
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
two integers of the same signedness; with mixed signedness, the unsigned one when it is at least as
wide, the signed one otherwise (`int1` counts as unsigned, of width 1). -/
def intPromote (a b : DType) : DType :=
  if a.isSigned = b.isSigned then (if a.intWidth > b.intWidth then a else b)
  else if !a.isSigned then (if a.intWidth ≥ b.intWidth then a else b)
  else (if b.intWidth ≥ a.intWidth then b else a)

/-- `computation_type_impl` (`semantic.py:68-116`): the dtype a binary operator computes in, from
its operands' dtypes (a Python number's own dtype) and which of them are Python numbers; `divMod`
for `/`, `//` and `%`. The first that applies:
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

/-- How the front end types an operator. -/
structure Typing where
  /-- The dtype both operands are converted to. -/
  opnd : DType
  /-- The result's dtype. -/
  res : DType
  /-- Whether the result is a block. -/
  block : Bool
  deriving DecidableEq, Repr

/-- `binary_op_type_checking_impl` (`semantic.py:175-210`) on two operands, at least one a
tensor: the computation dtype, after rejecting a negative Python number with an unsigned result and
a Python integer outside an integer result's range. -/
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
`min` and `max` of two operands when one is a tensor). -/
inductive BinOp where
  | add | sub | mul | div | floordiv | mod | band | bor | bxor | shl | shr
  | lt | le | gt | ge | eq | ne | minimum | maximum
  deriving DecidableEq, Repr

/-- `float32` for `bfloat16` (`_promote_bfloat16_to_float32`, `core.py:3083-3089`), applied to
each operand of `tl.minimum` and `tl.maximum`. -/
def bf16ToF32 : Operand → Operand
  | .tensor .bf16 b => .tensor .f32 b
  | o => o

/-- How the front end types `x op y` when at least one operand is a tensor (`none`: rejected on
the host).
* `+ - *`: `common?`; the result is that dtype.
* `/` (`semantic.py:292-315`): `common?` with `divMod`; two integers are both converted to
  `float32` (`int64` included).
* `//`: `common?` with `divMod`, integers only. `%`: `common?` with `divMod` (a float `%` is
  `frem`, which TVA does not model).
* `& | ^ << >>`: `common?`, integers only.
* comparisons: each Python number first becomes a tensor of its own dtype, then `common?`; the
  result is `int1`. So `x < 2**31` with an `int32` `x` compares in `uint32`.
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
keeps the dtype (`tl.neg` on a float; `0 - x`, wrapping, on an integer); `~x` needs an integer;
`not x` needs `int1`; `+x` is rejected. -/
def UnOp.typing? (op : UnOp) : Operand → Option Typing
  | .tensor d b =>
    match op with
    | .neg => some ⟨d, d, b⟩
    | .invert => if d.isTritonInt then some ⟨d, d, b⟩ else none
    | .not => if d == .i1 then some ⟨.i1, .i1, b⟩ else none
    | .pos => none
  | _ => none

/-- `tl.where(c, x, y)` (`semantic.py:1638-1652`): the condition becomes a tensor and is converted
to `int1` (`c != 0`, so a NaN condition is true; an fp8 condition is rejected); `x` and `y` are
typed by `common?`, Python numbers included; the result is a block if any of the three is. -/
def whereTyping? (c x y : Operand) : Option Typing := do
  let cd ← c.ownType?
  if cd.isFp8 then none else
  let r ← common? x y false
  some ⟨r, r, c.block || x.block || y.block⟩

/-- What `a and b` / `a or b` gives. -/
inductive BoolOpRes where
  /-- A Python value, at compile time. -/
  | python
  /-- A tensor of dtype `d`. -/
  | tensor (d : DType) (block : Bool)
  deriving DecidableEq, Repr

/-- `a and b` (`isAnd`) and `a or b` (`code_generator.py:1481-1531`): a Python value that decides
the result (false for `and`, true for `or`) is the result; another Python value is dropped, so
`x and True` is `x`; two tensors must both be `int1`, and the result is their elementwise `&`
(`|`). -/
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

end TVA
