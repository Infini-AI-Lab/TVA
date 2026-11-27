import Proof.Leaf

/-!
# `layer_norm_gated_fwd_bf16`: the lanes of one row, before the output

For a row `t` of program `p` (lane `i0 = t − p·BT`): the row's inputs `a j` (`x[t, j]` for
`j < D`, `+0` on the padded lanes), each lane's square (`lane_sq`), and the row's `rstd`
(`row_rstd`), from the facts the walk gives.
-/

set_option maxHeartbeats 4000000

namespace Claims.LayerNormGatedFwd

open TVA
open TVA TVA.Format

/-- The `[1, BD]` mask `o_d[None, :] < D`, lane by lane. -/
theorem get_md_row {d : Inst} (hD : d.D < 2 ^ 31) (hBD : d.BD < 2 ^ 31) (i : Idx [d.BT, d.BD]) :
    (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D : I32)).row d.BT).get i = decide (i.2.1.val < d.D) := by
  have h2 : (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D : I32)).row d.BT).get i
      = IntV.lt ((tl.arange 0 d.BD).get1 i.2.1) (IntV.ofNat d.D : I32) := rfl
  have hari : IntV.toInt ((tl.arange 0 d.BD).get1 i.2.1) = (i.2.1.val : Int) :=
    arange_toInt d.BD hBD i.2.1
  have hDval : IntV.toInt (IntV.ofNat d.D : I32) = (d.D : Int) := IntV.I32.toInt_ofNat hD
  rw [h2, IntV.lt_eq, decide_eq_decide, hari, hDval]
  norm_cast

/-- The value read from a present element is `rval`. -/
theorem rval_of_read {m : Mem} {b : Buf .bf16} {k : Nat} {v : BF16}
    (h : m.read .bf16 (b.ptr (k : Int)).addr = some v) : rval m b k = v := by
  unfold rval; rw [h]

/-- `+0` of `float32` is finite with value `0`. -/
theorem zero32 : (FloatV.ofPat 0 : F32).isFinite = true ∧ (FloatV.ofPat 0 : F32).toReal = 0 :=
  ⟨by decide, by rw [FloatV.toReal_eq]; have : (FloatV.ofPat 0 : F32).toRat = 0 := by decide +kernel
                 rw [this]; norm_num⟩

/-- `+0` of `bfloat16` is finite with value `0`. -/
theorem zero16 : (FloatV.ofPat 0 : BF16).isFinite = true ∧ (FloatV.ofPat 0 : BF16).toReal = 0 :=
  ⟨by decide, by rw [FloatV.toReal_eq]; have : (FloatV.ofPat 0 : BF16).toRat = 0 := by decide +kernel
                 rw [this]; norm_num⟩

/-- One lane's square: `b_xx = fl(b_xbar²)` is finite and within `u·a² + η` of `a²`, with
`a = x[t, j]` (`j < D`) or `+0` (`j ≥ D`), `b_xbar` the widened input or `+0`. -/
theorem lane_sq {a : BF16} {xb : F32} {xx : F32} (ha : a.isFinite = true)
    (hxb : xb.isFinite = true ∧ xb.toReal = a.toReal) (hroom : a.toReal ^ 2 ≤ (2 : ℝ) ^ (127 : ℤ))
    (hm : FloatV.NaNEq (FloatV.mul xb xb) xx) :
    xx.isFinite = true ∧
      |xx.toReal - a.toReal ^ 2| ≤ (2 : ℝ) ^ (-24 : ℤ) * a.toReal ^ 2 + (2 : ℝ) ^ (-150 : ℤ) := by
  have hq : |xb.toReal * xb.toReal| ≤ (2 : ℝ) ^ (127 : ℤ) := by
    rw [hxb.2, ← pow_two, abs_of_nonneg (sq_nonneg _)]; exact hroom
  obtain ⟨h1, h2⟩ := mul32 hm hxb.1 hxb.1 hq
  rw [hxb.2, ← pow_two, abs_of_nonneg (sq_nonneg a.toReal)] at h2
  exact ⟨h1, h2⟩

end Claims.LayerNormGatedFwd
