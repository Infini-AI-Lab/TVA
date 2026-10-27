import TVA.Lang.Ops

/-!
# `tl.dot_scaled`: matrix product of microscaled operands

`tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math,
lhs_k_pack, rhs_k_pack)` (`core.py:2427-2476`, `semantic.py:1553-1626`) multiplies `lhs`
(`[M, K]`) by `rhs` (`[K, N]`) whose elements are in a microscaling format. Each group of
`group` elements along `K` shares a scale (`lhs_scale`: `[M, K / group]`; `rhs_scale`:
`[N, K / group]`, not transposed). The group is `16` or `32`, deduced from the scale shapes
(`Ops.cpp:368-425`). A scale is `e8m0` (an `i8` tensor: OCP MX) or a float (NVFP4's
`float8e4nv` scales with group `16`). The result and accumulator are `float32`.

Storage (`_bitcast_to_fp_type`): fp8 operands are fp8 tensors or `uint8` tensors holding the
bits. `e2m1` (fp4) operands are `uint8` tensors with two elements per byte, the first in the low
nibble, packed along `K` (or along `M`/`N` when `lhs_k_pack`/`rhs_k_pack` is false). `bf16` and
`fp16` operands are tensors of those dtypes.

The library unpacks the operands exactly (`lhsBits`, `rhsBits`, `scaleBits`). The product
itself is the platform's `P.dotScaled`: native instructions and the software emulation (which
upcasts to `bf16`) round differently. `MXFmt.value` and `e8m0Value` give the exact values, so
that accuracy can be stated as a rule (`Rules.Num.DotScaledAcc`). 3-D (batched) operands are
not covered (`docs/exclusions.md`).
-/
namespace TVA

open TVA.Math.Float

namespace MXFmt

/-- The binary format of an element. -/
def format : MXFmt → Format
  | .e2m1 => Format.e2m1 | .e4m3 => Format.e4m3 | .e5m2 => Format.e5m2
  | .bf16 => Format.bfloat16 | .fp16 => Format.binary16

/-- The exact value of an element's bits; `none` for NaN and infinities. -/
def value (fmt : MXFmt) (bits : Nat) : Option Rat :=
  if fmt.format.isFinite bits then some (fmt.format.toRat bits) else none

/-- Operand tensors of dtype `d` hold this format, as `tl.dot_scaled` accepts them
(`_bitcast_to_fp_type`, `semantic.py:1535-1551`): `e2m1` packed in `uint8`; `e5m2`/`e4m3` as
the fp8 dtype or `uint8`; `bf16`/`fp16` as the float dtype or `uint16`. -/
def StoredIn : MXFmt → DType → Prop
  | .e2m1, d => d = .u8
  | .e5m2, d => d = .f8e5 ∨ d = .u8
  | .e4m3, d => d = .f8e4nv ∨ d = .u8
  | .bf16, d => d = .bf16 ∨ d = .u16
  | .fp16, d => d = .f16 ∨ d = .u16

end MXFmt

/-- The value of an `e8m0` scale: `2^(b − 127)`; `0xFF` is NaN (`none`). This is the OCP
value. The emulation path decodes by a bit shift, so it reads `b = 0` as `0`, and with `fp16`
compute it rounds the scale to `fp16` (`DecomposeScaledBlocked.cpp:58-96`). Accuracy rules
judge the path the target takes. -/
def e8m0Value (b : Nat) : Option Rat :=
  if b = 255 then none else some ((2 : Rat) ^ ((b : Int) - 127))

namespace ScaleFmt

/-- The exact value of a scale's bits; `none` for NaN and infinities. -/
def value : ScaleFmt → Nat → Option Rat
  | .e8m0, b => e8m0Value b
  | .float d, b => match d.format? with
    | some f => if f.isFinite b then some (f.toRat b) else none
    | none => none

/-- Scale tensors of dtype `d` hold this format: `uint8`/`int8` for `e8m0`, the float dtype
itself otherwise. -/
def Stores : ScaleFmt → DType → Prop
  | .e8m0, d => d = .u8 ∨ d = .i8
  | .float f, d => d = f ∧ f.format?.isSome = true

end ScaleFmt

/-- The exact value of an element with its scale (`1` without one). -/
def scaledValue (fmt : MXFmt) (sf : ScaleFmt) (bits : Nat) (scale : Option Nat) : Option Rat :=
  match fmt.value bits, scale with
  | some v, none => some v
  | some v, some b => (sf.value b).map (v * ·)
  | none, _ => none

/-- Element `(r, c)` of a rank-2 tensor, or the default outside it. -/
def Tensor.getD2 {R C : Nat} {α : Type} [Inhabited α] (x : Tensor [R, C] α) (r c : Nat) : α :=
  if h : r < R ∧ c < C then x.get2 ⟨r, h.1⟩ ⟨c, h.2⟩ else default

namespace ScaledCfg

/-- The bits of logical element `(i, k)` of `lhs`, from its storage `raw`. -/
def lhsBits (cfg : ScaledCfg) {R C : Nat} {d : DType} (raw : Tensor [R, C] d.Val) (i k : Nat) : Nat :=
  if cfg.lhs = .e2m1 then
    if cfg.lhsKPack then (d.toBits (raw.getD2 i (k / 2)) >>> (4 * (k % 2))) % 16
    else (d.toBits (raw.getD2 (i / 2) k) >>> (4 * (i % 2))) % 16
  else d.toBits (raw.getD2 i k)

/-- The bits of logical element `(k, n)` of `rhs`, from its storage `raw`. -/
def rhsBits (cfg : ScaledCfg) {R C : Nat} {d : DType} (raw : Tensor [R, C] d.Val) (k n : Nat) : Nat :=
  if cfg.rhs = .e2m1 then
    if cfg.rhsKPack then (d.toBits (raw.getD2 (k / 2) n) >>> (4 * (k % 2))) % 16
    else (d.toBits (raw.getD2 k (n / 2)) >>> (4 * (n % 2))) % 16
  else d.toBits (raw.getD2 k n)

/-- The storage of an operand with logical shape `[rows, cols]` is `[R, C]`. `K` runs along
`cols` (`kAlongCols`) or along `rows`. An `e2m1` operand holds two elements per byte along `K`
(`kpack`) or along the other dimension. -/
def StorageOk (fmt : MXFmt) (kpack kAlongCols : Bool) (rows cols R C : Nat) : Prop :=
  if fmt = .e2m1 then
    (if kpack = kAlongCols then rows = R ∧ cols = 2 * C else rows = 2 * R ∧ cols = C)
  else rows = R ∧ cols = C

/-- Triton's shape checks: the storage shapes match the logical `[M, K]` and `[K, N]`, only
`e2m1` may be packed along a dimension other than `K` (`semantic.py:1594-1595`), and the scales
have `K / group` columns with a group of `16` or `32`. -/
def ShapeOk (cfg : ScaledCfg) (M N K Ra Ca Rb Cb G : Nat) : Prop :=
  (cfg.group = 16 ∨ cfg.group = 32) ∧ K = G * cfg.group ∧
    (cfg.lhsKPack = true ∨ cfg.lhs = .e2m1) ∧ (cfg.rhsKPack = true ∨ cfg.rhs = .e2m1) ∧
    StorageOk cfg.lhs cfg.lhsKPack true M K Ra Ca ∧ StorageOk cfg.rhs cfg.rhsKPack false K N Rb Cb

/-- Triton's dtype checks: each operand's storage dtype holds its format, and each side's scale
dtype holds its scale format. -/
def TypesOk (cfg : ScaledCfg) (dl dr dsl dsr : DType) : Prop :=
  cfg.lhs.StoredIn dl ∧ cfg.rhs.StoredIn dr ∧ cfg.lhsScale.Stores dsl ∧ cfg.rhsScale.Stores dsr

/-- The `lhs` element bits as a logical `[M, K]` tensor. -/
def unpackL (cfg : ScaledCfg) (M K : Nat) {R C : Nat} {d : DType} (raw : Tensor [R, C] d.Val) :
    Tensor [M, K] Nat :=
  Tensor.of2 fun i k => cfg.lhsBits raw i k

/-- The `rhs` element bits as a logical `[K, N]` tensor. -/
def unpackR (cfg : ScaledCfg) (K N : Nat) {R C : Nat} {d : DType} (raw : Tensor [R, C] d.Val) :
    Tensor [K, N] Nat :=
  Tensor.of2 fun k n => cfg.rhsBits raw k n

/-- The scale bits of each `lhs` element (`lhs_scale` is `[M, K / group]`). -/
def scalesL (cfg : ScaledCfg) (M K : Nat) {G : Nat} {ds : DType} (sc : Tensor [M, G] ds.Val) :
    Tensor [M, K] Nat :=
  Tensor.of2 fun i k => ds.toBits (sc.getD2 i (k / cfg.group))

/-- The scale bits of each `rhs` element (`rhs_scale` is `[N, K / group]`, not transposed). -/
def scalesR (cfg : ScaledCfg) (K N : Nat) {G : Nat} {ds : DType} (sc : Tensor [N, G] ds.Val) :
    Tensor [K, N] Nat :=
  Tensor.of2 fun k n => ds.toBits (sc.getD2 n (k / cfg.group))

end ScaledCfg

namespace tl

/-- `tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc)` with
configuration `cfg`, logical sizes `M`, `N`, `K`, and the source's storage tensors. The scale
tensors have dtypes `dsl`, `dsr` (`cfg.TypesOk`). A missing scale is `none`; a missing `acc` is
`tl.zeros`. -/
def dot_scaled (cfg : ScaledCfg) (M N K : Nat) {dl dr dsl dsr : DType} {Ra Ca Rb Cb G : Nat}
    (lhs : Tensor [Ra, Ca] dl.Val) (lhs_scale : Option (Tensor [M, G] dsl.Val))
    (rhs : Tensor [Rb, Cb] dr.Val) (rhs_scale : Option (Tensor [N, G] dsr.Val))
    (acc : Tensor [M, N] F32) : Prog (Tensor [M, N] F32) :=
  .pick (fun _ : Unit => cfg.ShapeOk M N K Ra Ca Rb Cb G ∧ cfg.TypesOk dl dr dsl dsr) fun _ =>
  .env fun e => .pick (fun r => e.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
      (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) acc r) .ret

/-- A source `tl.dot_scaled(…, acc=z) + c` (or `c + tl.dot_scaled(…)`) with a zero accumulator
`z` (the default `tl.zeros`, or any constant `±0.0`) and no other use of the product. Triton's
combine pass may fold `c` into the accumulator (`CombineDotScaledAddFPattern`,
`Combine.cpp:255-298`). Otherwise it is the product followed by the (fusable) `float32` addition. -/
def dot_scaled_add (cfg : ScaledCfg) (M N K : Nat) {dl dr dsl dsr : DType} {Ra Ca Rb Cb G : Nat}
    (lhs : Tensor [Ra, Ca] dl.Val) (lhs_scale : Option (Tensor [M, G] dsl.Val))
    (rhs : Tensor [Rb, Cb] dr.Val) (rhs_scale : Option (Tensor [N, G] dsr.Val))
    (c : Tensor [M, N] F32) (z : Tensor [M, N] F32 := Tensor.full (FloatV.ofPat 0)) :
    Prog (Tensor [M, N] F32) :=
  .pick (fun _ : Unit => cfg.ShapeOk M N K Ra Ca Rb Cb G ∧ cfg.TypesOk dl dr dsl dsr) fun _ =>
  .env fun e => .pick (fun r =>
    e.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
      (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) c r ∨
    ∃ d, e.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
      (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) z d ∧
      ∀ i, AddRel e.fusion (d.get i) (c.get i) (r.get i)) .ret

end tl

end TVA
