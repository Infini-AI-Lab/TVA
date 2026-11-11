import Proof.Defs

/-!
# The `int64` arithmetic of the kernel's id

The two range masks are the integer comparisons of the claim, and the product
`local_id * weight_stride0` is the signed product the claim's offsets use (it does not wrap).
-/

namespace Claims.VocabEmbedding

open TVA

set_option maxHeartbeats 800000

private theorem sfits64_of_bounds {x : Int} (h1 : 0 ≤ x) (h2 : x < 2 ^ 63) :
    IntV.SFits 64 x := by
  have hp : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
  rw [IntV.SFits, hp]
  exact ⟨by linarith, h2⟩

private theorem ofInt_toInt64' {x : Int} (h1 : -(2 ^ 63) ≤ x) (h2 : x < 2 ^ 63) :
    (IntV.ofInt x : I64).toInt = x := by
  have hp : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
  exact IntV.toInt_ofInt (by norm_num) (by rw [IntV.SFits, hp]; exact ⟨h1, h2⟩)

private theorem ofInt_toInt64 {x : Int} (h1 : 0 ≤ x) (h2 : x < 2 ^ 63) :
    (IntV.ofInt x : I64).toInt = x :=
  ofInt_toInt64' (by linarith) h2

private theorem sfits64_i32_toInt (x : I32) : IntV.SFits 64 x.toInt := by
  have hs := IntV.sfits_toInt x
  rw [IntV.SFits] at hs ⊢
  norm_num at hs ⊢
  constructor <;> linarith [hs.1, hs.2]

/-- The `valid` mask of a token is the integer disjunction of the two ranges. -/
theorem validT_item {d : Inst} (hd : d.ok) {c : I64} {v : Int} (hc : c.toInt = v) :
    (validT d c).item =
      decide (d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index ∨
        d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index) := by
  have hos := ofInt_toInt64 hd.org_start.1 hd.org_start.2
  have hoe := ofInt_toInt64 hd.org_end.1 hd.org_end.2
  have has := ofInt_toInt64 hd.add_start.1 hd.add_start.2
  have hae := ofInt_toInt64 hd.add_end.1 hd.add_end.2
  have hor : ∀ (x y : Tensor [] Bool), (x.or y).item = (x.item || y.item) := fun _ _ => rfl
  have hand : ∀ (x y : Tensor [] Bool), (x.and y).item = (x.item && y.item) := fun _ _ => rfl
  have hle : ∀ (x y : Tensor [] I64), (x.le y).item = IntV.le x.item y.item := fun _ _ => rfl
  have hlt : ∀ (x y : Tensor [] I64), (x.lt y).item = IntV.lt x.item y.item := fun _ _ => rfl
  have hsc : ∀ a : I64, (Tensor.scalar a).item = a := fun _ => rfl
  simp only [validT, orgT, addT, hor, hand, hle, hlt, hsc, IntV.le_eq, IntV.lt_eq, hc,
    hos, hoe, has, hae]
  rw [Bool.eq_iff_iff]
  simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq]

/-- A conversion of an `int32` or `int64` value to `int64` keeps its signed value. -/
theorem convRel_i64_toInt {P : Platform} {dt : DType} (hdt : dt = .i32 ∨ dt = .i64)
    {x : dt.Val} {r : I64} (h : ConvRel P dt .i64 x r) : r.toInt = ivOf x := by
  rcases hdt with rfl | rfl
  · simp only [ConvRel, ofIntRel] at h
    subst h
    rw [IntV.toInt_ofInt (by norm_num) (sfits64_i32_toInt x)]
    rfl
  · simp only [ConvRel, ofIntRel] at h
    subst h
    rw [IntV.toInt_ofInt (by norm_num) (IntV.sfits_toInt x)]
    rfl

private theorem orgT_item {d : Inst} (hd : d.ok) {c : I64} {v : Int} (hc : c.toInt = v) :
    (orgT d c).item = decide (d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index) := by
  have hos := ofInt_toInt64 hd.org_start.1 hd.org_start.2
  have hoe := ofInt_toInt64 hd.org_end.1 hd.org_end.2
  have hand : ∀ (x y : Tensor [] Bool), (x.and y).item = (x.item && y.item) := fun _ _ => rfl
  have hle : ∀ (x y : Tensor [] I64), (x.le y).item = IntV.le x.item y.item := fun _ _ => rfl
  have hlt : ∀ (x y : Tensor [] I64), (x.lt y).item = IntV.lt x.item y.item := fun _ _ => rfl
  have hsc : ∀ a : I64, (Tensor.scalar a).item = a := fun _ => rfl
  simp only [orgT, hand, hle, hlt, hsc, IntV.le_eq, IntV.lt_eq, hc, hos, hoe]
  rw [Bool.eq_iff_iff]
  simp only [Bool.and_eq_true, decide_eq_true_eq]

private theorem localIdT_item {d : Inst} (hd : d.ok) {c : I64} {v : Int} (hc : c.toInt = v) :
    (localIdT d c).item =
      (if d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index then
        c - (IntV.ofInt d.org_vocab_start_index : I64)
      else if d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index then
        c - (IntV.ofInt (addedOffset d) : I64)
      else (IntV.ofInt 0 : I64)) := by
  have hwh : ∀ (b : Tensor [] Bool) (x y : Tensor [] I64),
      (tl.where b x y).item = if b.item then x.item else y.item := fun _ _ _ => rfl
  have hsc : ∀ a : I64, (Tensor.scalar a).item = a := fun _ => rfl
  simp only [localIdT, hwh, hsc, validT_item hd hc, orgT_item hd hc]
  by_cases hP : (d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index)
  · by_cases hQ : (d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index) <;>
      simp [hP, hQ]
  · by_cases hQ : (d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index) <;>
      simp [hP, hQ]

private theorem strideI {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.weight_stride0 : I64).toInt = (d.weight_stride0 : Int) := by
  exact IntV.I64.toInt_ofNat hd.stride_lt

/-- The kernel's `local_id * weight_stride0`, read as an integer: the weight row of the id. -/
theorem localId_mul_toInt {d : Inst} (hd : d.ok) {c : I64} {v : Int} (hc : c.toInt = v)
    (hTok : TokOk d v) :
    ((localIdT d c).item * (IntV.ofNat d.weight_stride0 : I64)).toInt
      = laneSel d v * (d.weight_stride0 : Int) := by
  have hos := ofInt_toInt64 hd.org_start.1 hd.org_start.2
  have hao : (IntV.ofInt (addedOffset d) : I64).toInt = addedOffset d :=
    ofInt_toInt64' hd.offset.1 hd.offset.2
  have hz : (IntV.ofInt 0 : I64).toInt = (0 : Int) :=
    ofInt_toInt64 (by norm_num) (by norm_num)
  have hstr := strideI hd
  have hhid : (0 : Int) < (d.hidden_dim : Int) := by exact_mod_cast hd.hidden_pos
  have hhid0 : (0 : Int) ≤ (d.hidden_dim : Int) := le_of_lt hhid
  by_cases hs : d.weight_stride0 = 0
  · have hs0 : (d.weight_stride0 : Int) = 0 := by rw [hs]; rfl
    have hsf : IntV.SFits 64 ((localIdT d c).item.toInt *
        (IntV.ofNat d.weight_stride0 : I64).toInt) := by
      rw [hstr, hs0, Int.mul_zero, IntV.SFits]
      norm_num
    rw [IntV.toInt_mul rfl rfl hsf, hstr, hs0]
    simp
  · have hspos : 0 < d.weight_stride0 := Nat.pos_of_ne_zero hs
    have hstr1 : (1 : Int) ≤ (d.weight_stride0 : Int) := by
      have := hspos
      exact_mod_cast this
    have hstr0 : (0 : Int) ≤ (d.weight_stride0 : Int) := by linarith
    by_cases hP : (d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index)
    · have hb := hTok.1 hP
      have hnn : (0 : Int) ≤ v - d.org_vocab_start_index := by linarith [hP.1]
      have hle1 : v - d.org_vocab_start_index ≤
          (v - d.org_vocab_start_index) * (d.weight_stride0 : Int) := by
        nlinarith [hnn, hstr1]
      have hub : v - d.org_vocab_start_index < 2 ^ 63 := by nlinarith [hle1, hb.2, hhid0]
      have hsub : IntV.SFits 64 (v - d.org_vocab_start_index) := sfits64_of_bounds hnn hub
      have hpn : (0 : Int) ≤ (v - d.org_vocab_start_index) * (d.weight_stride0 : Int) :=
        Int.mul_nonneg hnn hstr0
      have hpu : (v - d.org_vocab_start_index) * (d.weight_stride0 : Int) < 2 ^ 63 := by
        linarith [hb.2, hhid0]
      have hpm : IntV.SFits 64 ((v - d.org_vocab_start_index) * (d.weight_stride0 : Int)) :=
        sfits64_of_bounds hpn hpu
      have hloc := localIdT_item hd hc
      rw [ite_eq_left hP] at hloc
      rw [hloc]
      have hsubi : (c - (IntV.ofInt d.org_vocab_start_index : I64)).toInt =
          v - d.org_vocab_start_index := by
        rw [IntV.toInt_sub hc hos hsub]
      have hmul : IntV.SFits 64
          ((c - (IntV.ofInt d.org_vocab_start_index : I64)).toInt *
            (IntV.ofNat d.weight_stride0 : I64).toInt) := by
        rw [hsubi, hstr]
        exact hpm
      rw [IntV.toInt_mul rfl rfl hmul, hsubi, hstr, laneSel_of_org hP]
    · by_cases hQ : (d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index)
      · have hb := hTok.2 hQ
        have hnn : (0 : Int) ≤ v - addedOffset d := hb.1
        have hle1 : v - addedOffset d ≤
            (v - addedOffset d) * (d.weight_stride0 : Int) := by
          nlinarith [hnn, hstr1]
        have hub : v - addedOffset d < 2 ^ 63 := by nlinarith [hle1, hb.2.2, hhid0]
        have hsub : IntV.SFits 64 (v - addedOffset d) := sfits64_of_bounds hnn hub
        have hpn : (0 : Int) ≤ (v - addedOffset d) * (d.weight_stride0 : Int) :=
          Int.mul_nonneg hnn hstr0
        have hpu : (v - addedOffset d) * (d.weight_stride0 : Int) < 2 ^ 63 := by
          linarith [hb.2.2, hhid0]
        have hpm : IntV.SFits 64 ((v - addedOffset d) * (d.weight_stride0 : Int)) :=
          sfits64_of_bounds hpn hpu
        have hloc := localIdT_item hd hc
        rw [ite_eq_right hP, ite_eq_left hQ] at hloc
        rw [hloc]
        have hsubi : (c - (IntV.ofInt (addedOffset d) : I64)).toInt =
            v - addedOffset d := by
          rw [IntV.toInt_sub hc hao hsub]
        have hmul : IntV.SFits 64
            ((c - (IntV.ofInt (addedOffset d) : I64)).toInt *
              (IntV.ofNat d.weight_stride0 : I64).toInt) := by
          rw [hsubi, hstr]
          exact hpm
        rw [IntV.toInt_mul rfl rfl hmul, hsubi, hstr, laneSel_of_add hQ hP]
      · have hloc := localIdT_item hd hc
        rw [ite_eq_right hP, ite_eq_right hQ] at hloc
        rw [hloc]
        have hsf : IntV.SFits 64 ((IntV.ofInt 0 : I64).toInt *
            (IntV.ofNat d.weight_stride0 : I64).toInt) := by
          rw [hz, IntV.SFits]
          norm_num
        rw [IntV.toInt_mul rfl rfl hsf, hz, hstr, laneSel_of_none hP hQ,
          Int.zero_mul]

end Claims.VocabEmbedding
