import TVA.Triton.Arith

/-!
# Matrix products: `tl.dot`, `tl.dot_scaled`

**`tl.dot(a, b, acc)`** is the platform's `Platform.dot cfg`: the tensor cores' accumulation
order, internal precision and rounding are the hardware's, and the device facts bound them. Its
configuration `DotCfg` holds the operand dtypes, the accumulator dtype (`out_dtype`), the source's
`input_precision` (`tf32` when omitted) and the resolved `max_num_imprecise_acc`. The only integer
`tl.dot` is `int8 × int8` into an `int32` result (`K ≥ 32`; `semantic.py:1429-1520`); with a
`uint32` accumulator the model bit-casts it to `int32` (`tl.bitcast`).

**The combine pass's rewrites.** Triton rewrites some shapes whatever `enable_fp_fusion` says, and
the operation for each allows both the rewritten and the unrewritten result:
* `tl.dot_add`: a source `tl.dot(a, b) + c` (or `c + tl.dot(a, b)`) whose dot accumulator folds to
  a zero and whose product has no other use (`Combine.cpp:255-298`);
* `tl.sum_bcast_mul`: a float reduction over `k` of `x[i, k] * y[k, j]`, however spelled
  (`x[:, :, None] * y[None, :, :]`, `tl.expand_dims`, `tl.broadcast_to`, `tl.reshape`, summed by
  `tl.sum(…, axis=1)` or a `tl.reduce` whose combine adds its two arguments), which the pass may
  turn into an IEEE-precision `tl.dot` (`Combine.cpp:112-200`).

**`tl.dot_scaled`** (`core.py:2427-2476`, `semantic.py:1553-1626`) multiplies `lhs` (`[M, K]`) by
`rhs` (`[K, N]`) whose elements are in a microscaling format; each group of `16` or `32` elements
along `K` shares a scale (`lhs_scale`: `[M, K / group]`; `rhs_scale`: `[N, K / group]`, not
transposed). A scale is `e8m0` (a `uint8`/`int8` tensor) or a float (NVFP4's `float8e4nv`). The
result and accumulator are `float32`. Storage: fp8 operands are fp8 tensors or `uint8` bits;
`e2m1` (fp4) operands are `uint8` tensors holding two elements per byte, the first in the low
nibble, packed along `K` (or along `M`/`N` when `lhs_k_pack`/`rhs_k_pack` is false); `bf16`/`fp16`
operands are tensors of those dtypes. TVA unpacks the operands exactly (`ScaledCfg.lhsBits`,
`rhsBits`) and leaves the product to `Platform.dotScaled`; `scaledValue` gives the exact values
the device facts speak about. 3-D (batched) operands are not covered.
-/
namespace TVA

/-- The addition of an accumulator dtype: `AddRel` for the standard float formats, wrapping `+`
for integers, nothing for others. -/
def AccAdd (fusion : Bool) : (d : DType) → d.Val → d.Val → d.Val → Prop
  | .f16, x, y, r => AddRel fusion x y r | .bf16, x, y, r => AddRel fusion x y r
  | .f32, x, y, r => AddRel fusion x y r | .f64, x, y, r => AddRel fusion x y r
  | .i8, x, y, r => r = x + y | .i16, x, y, r => r = x + y | .i32, x, y, r => r = x + y
  | .i64, x, y, r => r = x + y
  | .u8, x, y, r => r = x + y | .u16, x, y, r => r = x + y | .u32, x, y, r => r = x + y
  | .u64, x, y, r => r = x + y
  | _, _, _, _ => False

namespace tl

/-- The product of a standard float dtype: the IEEE product up to NaN payload (nothing for other
dtypes). -/
def MulRel : (d : DType) → d.Val → d.Val → d.Val → Prop
  | .f16, a, b, c => FloatV.NaNEq (FloatV.mul a b) c
  | .bf16, a, b, c => FloatV.NaNEq (FloatV.mul a b) c
  | .f32, a, b, c => FloatV.NaNEq (FloatV.mul a b) c
  | .f64, a, b, c => FloatV.NaNEq (FloatV.mul a b) c
  | _, _, _, _ => False

/-- `tl.dot(a, b, acc)` with configuration `cfg`: as the platform computes it. -/
def dot (cfg : DotCfg) {M N K : Nat} (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc : Tensor [M, N] cfg.acc.Val) : Prog (Tensor [M, N] cfg.acc.Val) :=
  .env fun e => .pick (e.P.dot cfg a b acc) .ret

/-- A source `tl.dot(a, b, acc=z) + c` (or `c + tl.dot(…)`) with a zero accumulator `z` (the
default, or any constant `±0`) and no other use of the product: either `tl.dot(a, b, c)` (the
combine pass's rewrite, for a float dot only when `max_num_imprecise_acc` is `0`) or the product
followed by the (fusable) addition. -/
def dot_add (cfg : DotCfg) {M N K : Nat} (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (c : Tensor [M, N] cfg.acc.Val) (z : Tensor [M, N] cfg.acc.Val := Tensor.full (cfg.acc.ofBits 0)) :
    Prog (Tensor [M, N] cfg.acc.Val) :=
  .env fun e => .pick (fun r => e.P.dot cfg a b c r ∨
    ∃ d, e.P.dot cfg a b z d ∧ ∀ i, AccAdd e.fusion cfg.acc (d.get i) (c.get i) (r.get i)) .ret

/-- A float reduction on axis 1 of the products `x[i, k] * y[k, j]` (`x` expanded on its last axis,
`y` on its first), in any spelling: each element is the products summed along a tree with fusable
additions (as `tl.sumc`), or, when the additions are at most 32 bits wide and `M, N ≥ 16`, Triton's
rewrite to `tl.dot(x, y)` with `input_precision="ieee"`. A sum with `dtype=` that converts the
products first is not this computation, nor is a `tl.reduce` whose combine adds an argument to
itself. -/
def sum_bcast_mul (d : DType) {M K N : Nat} (x : Tensor [M, K] d.Val) (y : Tensor [K, N] d.Val) :
    Prog (Tensor [M, N] d.Val) :=
  .env fun e => .pick (fun r : Tensor [M, N] d.Val =>
    (∀ i j, ∃ p : Nat → d.Val,
      (∀ k (hk : k < K), MulRel d (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) (p k)) ∧
      ∃ t : Tree, t.Covers K ∧ t.Eval (AccAdd e.fusion d) p (r.get2 i j)) ∨
    (d.bytes ≤ 4 ∧ 16 ≤ M ∧ 16 ≤ N ∧
      e.P.dot { a := d, b := d, acc := d, precision := .ieee } x y (Tensor.full (d.ofBits 0)) r)) .ret

end tl

/-! ## Microscaled operands -/

namespace MXFmt

/-- The binary format of an element. -/
def format : MXFmt → Format
  | .e2m1 => Format.e2m1 | .e4m3 => Format.e4m3 | .e5m2 => Format.e5m2
  | .bf16 => Format.bfloat16 | .fp16 => Format.binary16

/-- The exact value of an element's bits; `none` for NaN and infinities. -/
def value (fmt : MXFmt) (bits : Nat) : Option Rat :=
  if fmt.format.isFinite bits then some (fmt.format.toRat bits) else none

/-- Operand tensors of dtype `d` hold this format, as `tl.dot_scaled` accepts them
(`_bitcast_to_fp_type`, `semantic.py:1535-1551`): `e2m1` packed in `uint8`; `e5m2`/`e4m3` as the
fp8 dtype or `uint8`; `bf16`/`fp16` as the float dtype or `uint16`. -/
def StoredIn : MXFmt → DType → Prop
  | .e2m1, d => d = .u8
  | .e5m2, d => d = .f8e5 ∨ d = .u8
  | .e4m3, d => d = .f8e4nv ∨ d = .u8
  | .bf16, d => d = .bf16 ∨ d = .u16
  | .fp16, d => d = .f16 ∨ d = .u16

end MXFmt

/-- The value of an `e8m0` scale byte `b`: `2^(b − 127)`, and NaN (`none`) for `0xFF` (the OCP
value). Triton's emulation decodes by a bit shift, reading `b = 0` as `0`; the device facts judge
the path the target takes. -/
def e8m0Value (b : Nat) : Option Rat :=
  if b = 255 then none else some ((2 : Rat) ^ ((b : Int) - 127))

namespace ScaleFmt

/-- The exact value of a scale's bits; `none` for NaN and infinities. -/
def value : ScaleFmt → Nat → Option Rat
  | .e8m0, b => e8m0Value b
  | .float d, b => match d.format? with
    | some f => if f.isFinite b then some (f.toRat b) else none
    | none => none

/-- Scale tensors of dtype `d` hold this format: `uint8`/`int8` for `e8m0`, the float dtype itself
otherwise. -/
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

/-- `tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc)` with configuration
`cfg`, logical sizes `M`, `N`, `K`, and the source's storage tensors: as the platform computes it
on the unpacked element bits and each element's scale bits. A missing scale is `none`; a missing
`acc` is `tl.zeros`. Undefined unless Triton's checks hold (`ShapeOk`, `TypesOk`). -/
def dot_scaled (cfg : ScaledCfg) (M N K : Nat) {dl dr dsl dsr : DType} {Ra Ca Rb Cb G : Nat}
    (lhs : Tensor [Ra, Ca] dl.Val) (lhs_scale : Option (Tensor [M, G] dsl.Val))
    (rhs : Tensor [Rb, Cb] dr.Val) (rhs_scale : Option (Tensor [N, G] dsr.Val))
    (acc : Tensor [M, N] F32) : Prog (Tensor [M, N] F32) :=
  .pick (fun _ : Unit => cfg.ShapeOk M N K Ra Ca Rb Cb G ∧ cfg.TypesOk dl dr dsl dsr) fun _ =>
  .env fun e => .pick (fun r => e.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
      (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) acc r) .ret

/-- A source `tl.dot_scaled(…, acc=z) + c` (or `c + tl.dot_scaled(…)`) with a zero accumulator `z`
and no other use of the product: either `c` folded into the accumulator (the combine pass) or the
product followed by the (fusable) `float32` addition. -/
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
