import TVA.Lang.Promote
import TVA.Lemmas.Sm90

/-! lr-31a, round 31: can a model use the new constants as a kernel would? -/
namespace TVA

-- L140: literal shapes, the default proof found by itself
example (x : Tensor [4, 1] F32) : Tensor [4, 8] F32 := x.broadcastTo [4, 8]
example (x : Tensor [1, 1, 8] F32) : Tensor [2, 4, 8] F32 := x.broadcastTo [2, 4, 8]
-- a rank change or a differing dimension has no proof
example : ¬ Shape.bcastOk [4, 1] [8, 1] = true := by decide
example : ¬ Shape.bcastOk [4] [4, 4] = true := by decide

-- a `tl.constexpr` block size (a Lean parameter): the proof is the prover's
theorem bcastOk_col (m n : Nat) : Shape.bcastOk [m, 1] [m, n] = true := by
  simp [Shape.bcastOk]

example (m n : Nat) (x : Tensor [m, 1] F32) :
    x.broadcastTo [m, n] (bcastOk_col m n) = x.bcastCols n :=
  Tensor.broadcastTo_cols x _

-- L143: bit casts between literal dtypes
example (x : Tensor [16] F32) : Tensor [16] I32 := tl.bitcast .f32 .i32 x
example (x : Tensor [16] BF16) : Tensor [16] F16 := tl.bitcast .bf16 .f16 x
example : ¬ DType.bitwidth .f32 = DType.bitwidth .f16 := by decide
example : DType.bitwidth .i1 = 1 := by decide
example (x : Tensor [16] I32) : tl.bitcast .i32 .i32 x = x := tl.bitcast_self .i32 x rfl

-- the bit cast round trip: f32 -> i32 -> f32 is the identity
example (x : Tensor [16] F32) :
    tl.bitcast .i32 .f32 (tl.bitcast .f32 .i32 x) = x := by
  ext i
  simp only [tl.bitcast, Tensor.get_map, DType.ofBits, DType.toBits, IntV.ofNat, FloatV.ofPat, FloatV.pat]
  rcases x.get i with ⟨b⟩
  simp only [FloatV.mk.injEq]
  apply BitVec.eq_of_toNat_eq
  have hb : b.toNat < 2 ^ 32 := b.isLt
  simp only [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hb]
  exact Nat.mod_eq_of_lt b.isLt

-- L141: the fact, at a kernel's configuration (any precision, `maxImpreciseAcc = 0`)
example (P : Platform) (h : Rules.Num.DotI8 P) :
    Rules.Num.DotAcc P ⟨.i8, .i8, .i32, .tf32, 0⟩ (fun _ => True) (fun _ => 0) (fun _ => 0) := h .tf32
example : Rules.Num.DotI8 Platform.sm90ref := Platform.sm90ref.dotI8

#print axioms Platform.sm90ref.dotI8
#print axioms tl.bitcast_self
#print axioms Tensor.get_broadcastTo

end TVA
