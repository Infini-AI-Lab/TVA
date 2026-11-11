import Claim
import Proof.Setup
import Proof.Addrs
import Proof.Cast
import TVA.Proof

/-!
# The second top value: `tl.sumc (capped * onehot i1)` is `C i1`

`onehot32` is the f32 conversion of `IntV.eq (arange 0 8) i1`, so its entries are `1` at `i1`
and `0` elsewhere (each up to its own rounding: `ofRat 0` is exactly `0` by
`roundRat_zero_spec`, `ofRat 1` is within `u + η` of `1` by `ofRat_err`).

So the products are `prod2_j = capped_j · onehot_j`: at `i1` the value `C i1` up to
`u·|C| + η`, everywhere else a value of magnitude at most `η`
(`|(mul x y).toRat - x.toRat * y.toRat| ≤ |x·y|·u + η` with `y = 0`, `mul_err`). The fused
sum of `8` entries, of depth at most `⌈log₂ 8⌉ = 3` (`SumDepth`), moves the total by at most
`3·(u·30 + η)`. With `|C| ≤ 30(1 + 2^-20)` the total is at most `4u·30 + 4η ≤ 2^-16`.
-/

namespace Claims.MoeRouter

open TVA

private def u32q : ℚ := Format.binary32.u

private def eta32q : ℚ := Format.binary32.eta

private def mfac : ℚ := 30 * (1 + 2 ^ (-20 : ℤ))

private theorem u32q_eq : u32q = (2 : ℚ) ^ (-24 : ℤ) :=
  Format.binary32_u

private theorem eta32q_eq : eta32q = (2 : ℚ) ^ (-150 : ℤ) :=
  Format.binary32_eta

private theorem u32q_pos : (0 : ℚ) < u32q := by
  rw [u32q_eq]; norm_num

private theorem eta32q_pos : (0 : ℚ) < eta32q := by
  rw [eta32q_eq]; norm_num

private theorem mfac_le : mfac ≤ 31 := by
  unfold mfac; norm_num

private theorem maxVal32_nonneg : (0 : ℚ) ≤ Format.binary32.maxVal := by
  rw [Format.binary32_maxVal]; norm_num

private theorem maxVal32_ge : (62 : ℚ) ≤ Format.binary32.maxVal := by
  rw [Format.binary32_maxVal]; norm_num

private theorem ofRat_zero {f : Format} (hf : f.Regular) :
    (FloatV.ofRat (0 : ℚ) : FloatV f).isFinite = true ∧
      (FloatV.ofRat (0 : ℚ) : FloatV f).toRat = 0 := by
  unfold FloatV.ofRat
  obtain ⟨hlt, hfin, hrat⟩ :=
    Format.roundRat_zero_spec hf (rnd := .rn) (sat := false) (zs := false)
  exact ⟨by rw [FloatV.isFinite_ofPat hlt]; exact hfin,
    by rw [FloatV.toRat_ofPat hlt]; exact hrat⟩

private theorem ofRat_one_spec :
    (FloatV.ofRat (1 : ℚ) : FloatV Format.binary32).isFinite = true ∧
      |(FloatV.ofRat (1 : ℚ) : FloatV Format.binary32).toRat - 1|
        ≤ u32q + eta32q := by
  have hq : |((1 : ℤ) : ℚ)| ≤ Format.binary32.maxVal := by
    simpa using Format.one_le_maxVal Format.binary32_regular
  obtain ⟨hfin, herr⟩ :=
    FloatV.ofRat_err (f := Format.binary32)
      Format.binary32_regular hq
  refine ⟨hfin, ?_⟩
  simpa only [Int.cast_one, abs_one, one_mul, u32q, eta32q] using herr

private theorem itof_eq {z : ℤ} (hz : z = 0 ∨ z = 1) {r : F32}
    (h : tl.ItofRel Format.binary32 z r) : r = FloatV.ofRat (z : ℚ) := by
  rcases hz with rfl | rfl
  · exact (tl.itofRel_binary32 (z := (0 : ℤ)) (by norm_num)).mp h
  · exact (tl.itofRel_binary32 (z := (1 : ℤ)) (by norm_num)).mp h

private theorem intv_eq_iff (n m : Nat) (hn : n < 8) (hm : m < 8) :
    ((IntV.ofInt (0 + (n : ℤ)) : I32).eq (IntV.ofNat m) = true) ↔ n = m := by
  unfold IntV.eq
  rw [beq_iff_eq]
  constructor
  · intro h
    have h3 := congrArg BitVec.toNat h
    simp only [IntV.ofInt, IntV.ofNat] at h3
    rw [BitVec.toNat_ofInt, BitVec.toNat_ofNat] at h3
    have h4 : (0 + (n : ℤ)) % (2 ^ 32 : Nat) = (n : ℤ) := by
      rw [Int.emod_eq_of_lt] <;> omega
    rw [h4] at h3
    omega
  · intro h
    subst h
    have hb : (BitVec.ofInt 32 (0 + (n : ℤ))) = (BitVec.ofNat 32 n) := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofInt, BitVec.toNat_ofNat]
      rw [Int.emod_eq_of_lt] <;> omega
    exact hb

private theorem intv_eq_decide (n m : Nat) (hn : n < 8) (hm : m < 8) :
    ((IntV.ofInt (0 + (n : ℤ)) : I32).eq (IntV.ofNat m) : Bool) = decide (n = m) := by
  rw [Bool.eq_iff_iff, decide_eq_true_iff]
  exact intv_eq_iff n m hn hm

private theorem mask_get {pair2 : FloatV Format.binary32 × I32} {i1 : Fin 8}
    (hk : IntV.ofNat i1.val = pair2.2) (i : ℕ) (hi : i < 8) :
    (((Tensor.map (fun k => k.eq pair2.2) (tl.arange 0 8)).get (⟨i, hi⟩, ())) : Bool)
      = decide (i = i1.val) := by
  simp only [tl.arange, Tensor.arange, Tensor.of1]
  rw [← hk]
  exact intv_eq_decide i i1.val hi i1.isLt

private theorem list_sum_range_congr (g : Nat → ℚ) (n : Nat) :
    ((List.range n).map g).sum = ∑ k ∈ Finset.range n, g k := by
  induction n with
  | zero => simp
  | succ m ih =>
    rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ, ih]
    simp

private theorem entry_spec {Pl : Platform}
    {capped : Tensor [8] (FloatV Format.binary32)}
    (hfin : ∀ j : Fin 8, (capped.get1 j).isFinite = true)
    (hsmall : ∀ j : Fin 8, |(capped.get1 j).toReal| ≤ 30 * (1 + 2 ^ (-20 : ℤ)))
    {pair2 : FloatV Format.binary32 × I32} {i1 : Fin 8}
    (hk : IntV.ofNat i1.val = pair2.2)
    {onehot32 : Tensor [8] DType.f32.Val}
    (honehot : ∀ j : Idx [8], ConvRel Pl DType.i1 DType.f32
      ((Tensor.map (fun k => k.eq pair2.2) (tl.arange 0 8)).get j) (onehot32.get j))
    {prod2 : Tensor [8] (FloatV Format.binary32)}
    (hprod2 : ∀ j : Idx [8], ((capped.get j).mul (onehot32.get j)).NaNEq (prod2.get j))
    (BIG DEV : ℚ)
    (hBIG : BIG = mfac * (1 + u32q + eta32q) * (1 + u32q) + eta32q)
    (hDEV : DEV = mfac * ((1 + u32q + eta32q) * u32q + u32q + eta32q) + eta32q)
    (i : ℕ) (hi : i < 8) :
    (prod2.get1 ⟨i, hi⟩).isFinite = true ∧
      |(prod2.get1 ⟨i, hi⟩).toRat| ≤ (if i = i1.val then BIG else eta32q) ∧
      (i = i1.val →
        |(prod2.get1 ⟨i, hi⟩).toRat - (capped.get1 ⟨i, hi⟩).toRat| ≤ DEV) := by
  have hu_pos : (0 : ℚ) < u32q := u32q_pos
  have he_pos : (0 : ℚ) < eta32q := eta32q_pos
  have hu_le1 : u32q ≤ 1 := by rw [u32q_eq]; norm_num
  have he_le1 : eta32q ≤ 1 := by rw [eta32q_eq]; norm_num
  have hcfin : (capped.get1 ⟨i, hi⟩).isFinite = true := hfin ⟨i, hi⟩
  have hcM : |(capped.get1 ⟨i, hi⟩).toRat| ≤ mfac := by
    have h := hsmall ⟨i, hi⟩
    simp only [FloatV.toReal] at h
    rw [← Rat.cast_le (K := ℝ)]
    push_cast
    simpa [mfac] using h
  have hmask : (((Tensor.map (fun k => k.eq pair2.2) (tl.arange 0 8)).get
      (⟨i, hi⟩, ())) : Bool) = decide (i = i1.val) := mask_get hk i hi
  have hconv := honehot (⟨i, hi⟩, ())
  set z : ℤ := (if (((Tensor.map (fun k => k.eq pair2.2) (tl.arange 0 8)).get
    (⟨i, hi⟩, ())) : Bool) then (1 : ℤ) else 0) with hz_def
  have hitof : tl.ItofRel Format.binary32 z (onehot32.get1 ⟨i, hi⟩) := by
    simpa only [ConvRel, ofIntRel, hz_def, Tensor.get1] using hconv
  have hz01 : z = 0 ∨ z = 1 := by
    rw [hz_def, hmask]
    by_cases h : i = i1.val <;> simp [h]
  have hoeq : onehot32.get1 ⟨i, hi⟩ = FloatV.ofRat (z : ℚ) := itof_eq hz01 hitof
  by_cases hi1 : i = i1.val
  · have hz1 : z = 1 := by rw [hz_def, hmask]; simp [hi1]
    have ho1 : onehot32.get1 ⟨i, hi⟩ =
        (FloatV.ofRat (1 : ℚ) : FloatV Format.binary32) := by
      rw [hoeq, hz1]; norm_num
    have hofin : (onehot32.get1 ⟨i, hi⟩).isFinite = true := by
      rw [ho1]; exact ofRat_one_spec.1
    have hoerr : |(onehot32.get1 ⟨i, hi⟩).toRat - 1| ≤ u32q + eta32q := by
      rw [ho1]; exact ofRat_one_spec.2
    have hoabs : |(onehot32.get1 ⟨i, hi⟩).toRat| ≤ 1 + u32q + eta32q := by
      calc |(onehot32.get1 ⟨i, hi⟩).toRat|
          = |((onehot32.get1 ⟨i, hi⟩).toRat - 1) + 1| := by ring_nf
        _ ≤ |(onehot32.get1 ⟨i, hi⟩).toRat - 1| + |(1 : ℚ)| := abs_add_le _ _
        _ ≤ (u32q + eta32q) + 1 := by rw [abs_one]; linarith
        _ = 1 + u32q + eta32q := by ring
    have hco_nn : (0 : ℚ) ≤ |(capped.get1 ⟨i, hi⟩).toRat| := abs_nonneg _
    have hmax : |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat|
        ≤ Format.binary32.maxVal := by
      rw [abs_mul]
      have hb : |(capped.get1 ⟨i, hi⟩).toRat| * |(onehot32.get1 ⟨i, hi⟩).toRat| ≤ 62 := by
        have hm := mfac_le
        have huhe : u32q + eta32q ≤ 1 := by rw [u32q_eq, eta32q_eq]; norm_num
        have h2 : (1 : ℚ) + u32q + eta32q ≤ 2 := by linarith
        nlinarith [hcM, hoabs, abs_nonneg (capped.get1 ⟨i, hi⟩).toRat,
          abs_nonneg (onehot32.get1 ⟨i, hi⟩).toRat]
      exact hb.trans maxVal32_ge
    have hmul := FloatV.mul_err (f := Format.binary32)
      Format.binary32_regular hcfin hofin hmax
    have hp := hprod2 (⟨i, hi⟩, ())
    have hnan : ((capped.get1 ⟨i, hi⟩).mul (onehot32.get1 ⟨i, hi⟩)).isNaN = false :=
      (FloatV.isFinite_iff.mp hmul.1).1
    have hpeq : prod2.get1 ⟨i, hi⟩ =
        (capped.get1 ⟨i, hi⟩).mul (onehot32.get1 ⟨i, hi⟩) := by
      simpa only [Tensor.get1] using FloatV.NaNEq.eq hp hnan
    have hperr : |(prod2.get1 ⟨i, hi⟩).toRat
        - (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat|
        ≤ |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| * u32q + eta32q := by
      rw [hpeq]
      exact hmul.2
    have hco : |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat|
        ≤ mfac * (1 + u32q + eta32q) := by
      rw [abs_mul]
      refine mul_le_mul hcM hoabs (abs_nonneg _) ?_
      exact le_trans (abs_nonneg _) hcM
    refine ⟨?_, ?_, ?_⟩
    · rw [hpeq]; exact hmul.1
    · rw [ite_eq_left hi1]
      have htri : |(prod2.get1 ⟨i, hi⟩).toRat|
          ≤ |(prod2.get1 ⟨i, hi⟩).toRat
              - (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat|
            + |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| := by
        calc |(prod2.get1 ⟨i, hi⟩).toRat|
            = |((prod2.get1 ⟨i, hi⟩).toRat
                - (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat)
                + (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| := by ring_nf
          _ ≤ _ := abs_add_le _ _
      have h1u : (0 : ℚ) ≤ 1 + u32q := by linarith
      have hstep : |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| * (1 + u32q)
          ≤ mfac * (1 + u32q + eta32q) * (1 + u32q) :=
        mul_le_mul_of_nonneg_right hco h1u
      have h4 : |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| * u32q
          ≤ mfac * (1 + u32q + eta32q) * u32q :=
        mul_le_mul_of_nonneg_right hco (le_of_lt hu_pos)
      calc |(prod2.get1 ⟨i, hi⟩).toRat|
          ≤ (|(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| * u32q + eta32q)
            + |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| := by
            linarith [htri, hperr]
        _ = |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| * (1 + u32q)
            + eta32q := by ring
        _ ≤ mfac * (1 + u32q + eta32q) * (1 + u32q) + eta32q := by linarith [hstep]
        _ = BIG := by rw [hBIG]
    · intro _
      have htri : |(prod2.get1 ⟨i, hi⟩).toRat - (capped.get1 ⟨i, hi⟩).toRat|
          ≤ |(prod2.get1 ⟨i, hi⟩).toRat
              - (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat|
            + |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat
              - (capped.get1 ⟨i, hi⟩).toRat| := by
        calc |(prod2.get1 ⟨i, hi⟩).toRat - (capped.get1 ⟨i, hi⟩).toRat|
            = |((prod2.get1 ⟨i, hi⟩).toRat
                - (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat)
                + ((capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat
                  - (capped.get1 ⟨i, hi⟩).toRat)| := by ring_nf
          _ ≤ _ := abs_add_le _ _
      have hstep2 : |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat
          - (capped.get1 ⟨i, hi⟩).toRat| ≤ mfac * (u32q + eta32q) := by
        calc |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat
              - (capped.get1 ⟨i, hi⟩).toRat|
            = |(capped.get1 ⟨i, hi⟩).toRat| * |(onehot32.get1 ⟨i, hi⟩).toRat - 1| := by
              rw [show (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat
                  - (capped.get1 ⟨i, hi⟩).toRat
                  = (capped.get1 ⟨i, hi⟩).toRat * ((onehot32.get1 ⟨i, hi⟩).toRat - 1) from by
                ring, abs_mul]
          _ ≤ mfac * (u32q + eta32q) :=
            mul_le_mul hcM hoerr (abs_nonneg _) (le_trans (abs_nonneg _) hcM)
      have hstep3 : |(prod2.get1 ⟨i, hi⟩).toRat
          - (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat|
          ≤ mfac * (1 + u32q + eta32q) * u32q + eta32q := by
        have h4 : |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| * u32q
            ≤ mfac * (1 + u32q + eta32q) * u32q :=
          mul_le_mul_of_nonneg_right hco (le_of_lt hu_pos)
        linarith [hperr]
      calc |(prod2.get1 ⟨i, hi⟩).toRat - (capped.get1 ⟨i, hi⟩).toRat|
          ≤ (mfac * (1 + u32q + eta32q) * u32q + eta32q) + mfac * (u32q + eta32q) := by
            linarith [htri, hstep2, hstep3]
        _ = DEV := by rw [hDEV]; ring
  · have hz0 : z = 0 := by rw [hz_def, hmask]; simp [hi1]
    have ho0 : onehot32.get1 ⟨i, hi⟩ =
        (FloatV.ofRat (0 : ℚ) : FloatV Format.binary32) := by
      rw [hoeq, hz0]; norm_num
    have hofin : (onehot32.get1 ⟨i, hi⟩).isFinite = true := by
      rw [ho0]; exact (ofRat_zero Format.binary32_regular).1
    have hoz : (onehot32.get1 ⟨i, hi⟩).toRat = 0 := by
      rw [ho0]; exact (ofRat_zero Format.binary32_regular).2
    have hmax : |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat|
        ≤ Format.binary32.maxVal := by
      rw [hoz, mul_zero, abs_zero]
      exact maxVal32_nonneg
    have hmul := FloatV.mul_err (f := Format.binary32)
      Format.binary32_regular hcfin hofin hmax
    have hp := hprod2 (⟨i, hi⟩, ())
    have hnan : ((capped.get1 ⟨i, hi⟩).mul (onehot32.get1 ⟨i, hi⟩)).isNaN = false :=
      (FloatV.isFinite_iff.mp hmul.1).1
    have hpeq : prod2.get1 ⟨i, hi⟩ =
        (capped.get1 ⟨i, hi⟩).mul (onehot32.get1 ⟨i, hi⟩) := by
      simpa only [Tensor.get1] using FloatV.NaNEq.eq hp hnan
    have hperr : |(prod2.get1 ⟨i, hi⟩).toRat
        - (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat|
        ≤ |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| * u32q + eta32q := by
      rw [hpeq]
      exact hmul.2
    refine ⟨?_, ?_, ?_⟩
    · rw [hpeq]; exact hmul.1
    · rw [ite_eq_right hi1]
      have hco0 : (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat = 0 := by
        rw [hoz, mul_zero]
      have htri : |(prod2.get1 ⟨i, hi⟩).toRat|
          ≤ |(prod2.get1 ⟨i, hi⟩).toRat
              - (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat|
            + |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| := by
        calc |(prod2.get1 ⟨i, hi⟩).toRat|
            = |((prod2.get1 ⟨i, hi⟩).toRat
                - (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat)
                + (capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| := by ring_nf
          _ ≤ _ := abs_add_le _ _
      calc |(prod2.get1 ⟨i, hi⟩).toRat|
          ≤ (|(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| * u32q + eta32q)
            + |(capped.get1 ⟨i, hi⟩).toRat * (onehot32.get1 ⟨i, hi⟩).toRat| := by
            linarith [htri, hperr]
        _ = eta32q := by rw [hco0]; simp
    · intro h; exact absurd h hi1

variable {d : Inst}

/-- The second top value, `tl.sumc (capped * onehot i1)`, is finite and within `2^-15` of `C i1`. -/
theorem top_sum_bound {d : Inst} {P : Platform}
    (hsum : Device.SumDepth ((launch d).env P).P)
    {capped : Tensor [8 - 0] (FloatV Format.binary32)}
    (hfin : ∀ j : Fin (8 - 0), (capped.get1 j).isFinite = true)
    (hsmall : ∀ j : Fin (8 - 0), |(capped.get1 j).toReal| ≤ 30 * (1 + 2 ^ (-20 : ℤ)))
    {pair2 : FloatV Format.binary32 × I32} {i1 : Fin (8 - 0)}
    (hk : IntV.ofNat i1.val = pair2.2)
    {onehot32 : Tensor [8] DType.f32.Val}
    (honehot : ∀ j : Idx [8], ConvRel ((launch d).env P).P DType.i1 DType.f32
      ((Tensor.map (fun k => k.eq pair2.2) (tl.arange 0 8)).get j) (onehot32.get j))
    {prod2 : Tensor [8 - 0] (FloatV Format.binary32)}
    (hprod2 : ∀ j : Idx [8 - 0], ((capped.get j).mul (onehot32.get j)).NaNEq (prod2.get j))
    {top2v : FloatV Format.binary32}
    (htop2v : ∃ tr, Tree.Covers (8 - 0) tr ∧
      tr.depth ≤ tl.sumBound ((launch d).env P).P (8 - 0) ∧
      Tree.Eval (AddRel ((launch d).env P).fusion)
        (fun j => if h : j < 8 - 0 then prod2.get1 ⟨j, h⟩ else default) tr top2v) :
    top2v.isFinite = true ∧
      |top2v.toReal - (capped.get1 i1).toReal| ≤ (2 : ℝ) ^ (-15 : ℤ) := by
  obtain ⟨tr, hcov, hdep, hev⟩ := htop2v
  have hclog : Nat.clog 2 (8 - 0) = 3 := by
    have h8 : (8 - 0 : Nat) = 2 ^ 3 := by norm_num
    rw [h8]
    exact Nat.clog_pow 2 3 (by norm_num)
  have hd3 : tr.depth ≤ 3 := by
    have h := depth_le_clog hsum hdep
    rwa [hclog] at h
  set x : Nat → FloatV Format.binary32 :=
    (fun j => if h : j < 8 - 0 then prod2.get1 ⟨j, h⟩ else default) with hx_def
  have hev' : Tree.Eval (AddRel ((launch d).env P).fusion) x tr top2v := by
    rw [hx_def]
    exact hev
  set BIG : ℚ := mfac * (1 + u32q + eta32q) * (1 + u32q) + eta32q with hBIG
  set DEV : ℚ := mfac * ((1 + u32q + eta32q) * u32q + u32q + eta32q) + eta32q with hDEV
  have hnumB : BIG + 7 * eta32q ≤ 31 := by
    rw [hBIG]
    unfold mfac
    rw [u32q_eq, eta32q_eq]
    norm_num
  have hnumD : DEV + 7 * eta32q + (300 * u32q + 100 * eta32q) ≤ 512 * u32q := by
    rw [hDEV]
    unfold mfac
    rw [u32q_eq, eta32q_eq]
    norm_num
  have hnumE : Tree.errBound (3 * u32q) (4 * eta32q) 3 7 32 ≤ 300 * u32q + 100 * eta32q := by
    show ((1 + 3 * u32q) ^ 3 - 1) * 32 + 4 * eta32q * ((7 : ℕ) : ℚ) * (1 + 3 * u32q) ^ 3
      ≤ 300 * u32q + 100 * eta32q
    rw [u32q_eq, eta32q_eq]
    norm_num
  have h512 : (512 : ℚ) * u32q = (2 : ℚ) ^ (-15 : ℤ) := by
    rw [u32q_eq]
    norm_num
  have hmax2 : (32 : ℚ) ≤ Format.binary32.maxVal / 2 - 2 * eta32q := by
    rw [Format.binary32_maxVal, eta32q_eq]
    norm_num
  have hu1 : 300 * u32q + 100 * eta32q ≤ 1 := by
    rw [u32q_eq, eta32q_eq]
    norm_num
  have hspec : ∀ i (hi : i < 8 - 0), (x i).isFinite = true ∧
      |(x i).toRat| ≤ (if i = i1.val then BIG else eta32q) ∧
      (i = i1.val → |(x i).toRat - (capped.get1 ⟨i, hi⟩).toRat| ≤ DEV) := by
    intro i hi
    have hxeq : x i = prod2.get1 ⟨i, hi⟩ := by
      rw [hx_def]
      simp only [dite_eq_left hi]
    have h := entry_spec (Pl := ((launch d).env P).P) (capped := capped)
      hfin hsmall hk honehot hprod2 BIG DEV hBIG hDEV i hi
    rw [← hxeq] at h
    exact h
  have hxfin : ∀ i, i < 8 - 0 → (x i).isFinite = true := fun i hi => (hspec i hi).1
  have hsumBIG : ∑ i ∈ Finset.range (8 - 0), (if i = i1.val then BIG else eta32q)
      = BIG + 7 * eta32q := by
    have hmem : i1.val ∈ Finset.range (8 - 0) := Finset.mem_range.mpr i1.isLt
    have hsplit := Finset.sum_erase_add (Finset.range (8 - 0))
      (fun i => if i = i1.val then BIG else eta32q) hmem
    have herase : ∑ i ∈ (Finset.range (8 - 0)).erase i1.val,
        (if i = i1.val then BIG else eta32q)
        = ∑ i ∈ (Finset.range (8 - 0)).erase i1.val, eta32q := by
      refine Finset.sum_congr rfl (fun i hi => ?_)
      exact ite_eq_right (Finset.ne_of_mem_erase hi)
    have hfa : (if i1.val = i1.val then BIG else eta32q) = BIG := by simp
    calc ∑ i ∈ Finset.range (8 - 0), (if i = i1.val then BIG else eta32q)
        = (∑ i ∈ (Finset.range (8 - 0)).erase i1.val,
            (if i = i1.val then BIG else eta32q)) + (if i1.val = i1.val then BIG else eta32q) :=
          hsplit.symm
      _ = (∑ i ∈ (Finset.range (8 - 0)).erase i1.val, eta32q) + BIG := by rw [herase, hfa]
      _ = ((Finset.range (8 - 0)).erase i1.val).card • eta32q + BIG := by rw [Finset.sum_const]
      _ = BIG + 7 * eta32q := by
            rw [Finset.card_erase_of_mem hmem, Finset.card_range, nsmul_eq_mul]
            push_cast
            ring
  have hAbound : ((List.range (8 - 0)).map fun i => |(x i).toRat|).sum ≤ BIG + 7 * eta32q := by
    rw [list_sum_range_congr]
    have h1 : ∑ i ∈ Finset.range (8 - 0), |(x i).toRat|
        ≤ ∑ i ∈ Finset.range (8 - 0), (if i = i1.val then BIG else eta32q) := by
      refine Finset.sum_le_sum (fun i hi => ?_)
      exact (hspec i (Finset.mem_range.mp hi)).2.1
    linarith
  have hAnonneg : (0 : ℚ) ≤ ((List.range (8 - 0)).map fun i => |(x i).toRat|).sum := by
    rw [list_sum_range_congr]
    exact Finset.sum_nonneg (fun i _ => abs_nonneg _)
  have hδ : (0 : ℚ) ≤ 3 * u32q := by linarith [u32q_pos]
  have hη : (0 : ℚ) ≤ 4 * eta32q := by linarith [eta32q_pos]
  have hmonoA : Tree.errBound (3 * u32q) (4 * eta32q) 3 7
      ((List.range (8 - 0)).map fun i => |(x i).toRat|).sum
      ≤ Tree.errBound (3 * u32q) (4 * eta32q) 3 7 (BIG + 7 * eta32q) :=
    Tree.errBound_mono hδ hη le_rfl le_rfl hAnonneg hAbound
  have hmonoB : Tree.errBound (3 * u32q) (4 * eta32q) 3 7 (BIG + 7 * eta32q)
      ≤ Tree.errBound (3 * u32q) (4 * eta32q) 3 7 32 :=
    Tree.errBound_mono hδ hη le_rfl le_rfl (by linarith [eta32q_pos]) (by linarith [hnumB])
  have hM : ((List.range (8 - 0)).map fun i => |(x i).toRat|).sum +
      Tree.errBound (3 * u32q) (4 * eta32q) 3 7
        ((List.range (8 - 0)).map fun i => |(x i).toRat|).sum
      ≤ Format.binary32.maxVal / 2 - 2 * eta32q := by
    linarith [hAbound, hmonoA, hmonoB, hnumE, hnumB, hu1, hmax2]
  obtain ⟨hresfin, hreserr⟩ := TVA.sumc_err_depth (f := Format.binary32)
    bin32_regular bin32_specials bin32_u_le ((launch d).env P).fusion x hcov hd3 hev' hxfin hM
  have hresE : |top2v.toRat - ((List.range (8 - 0)).map fun i => (x i).toRat).sum|
      ≤ 300 * u32q + 100 * eta32q :=
    hreserr.trans (hmonoA.trans (hmonoB.trans hnumE))
  have hSdev : |(∑ i ∈ Finset.range (8 - 0), (x i).toRat) - (capped.get1 i1).toRat|
      ≤ DEV + 7 * eta32q := by
    have hmem : i1.val ∈ Finset.range (8 - 0) := Finset.mem_range.mpr i1.isLt
    have hsplit := Finset.sum_erase_add (Finset.range (8 - 0)) (fun i => (x i).toRat) hmem
    have hxI1 : |(x i1.val).toRat - (capped.get1 i1).toRat| ≤ DEV :=
      (hspec i1.val i1.isLt).2.2 rfl
    have habs : |∑ i ∈ (Finset.range (8 - 0)).erase i1.val, (x i).toRat| ≤ 7 * eta32q := by
      have h1 := Finset.abs_sum_le_sum_abs (fun i => (x i).toRat)
        ((Finset.range (8 - 0)).erase i1.val)
      have h3 : ∀ i ∈ (Finset.range (8 - 0)).erase i1.val, |(x i).toRat| ≤ eta32q := by
        intro i hi
        have hne : i ≠ i1.val := Finset.ne_of_mem_erase hi
        have hlt : i < 8 - 0 := Finset.mem_range.mp (Finset.mem_of_mem_erase hi)
        have h := (hspec i hlt).2.1
        rwa [ite_eq_right hne] at h
      have h4 : ∑ i ∈ (Finset.range (8 - 0)).erase i1.val, eta32q = 7 * eta32q := by
        rw [Finset.sum_const, Finset.card_erase_of_mem hmem, Finset.card_range, nsmul_eq_mul]
        push_cast
        ring
      linarith [Finset.sum_le_sum h3]
    have hdecomp : (∑ i ∈ Finset.range (8 - 0), (x i).toRat) - (capped.get1 i1).toRat
        = (∑ i ∈ (Finset.range (8 - 0)).erase i1.val, (x i).toRat)
          + ((x i1.val).toRat - (capped.get1 i1).toRat) := by linarith [hsplit]
    rw [hdecomp]
    calc |(∑ i ∈ (Finset.range (8 - 0)).erase i1.val, (x i).toRat)
          + ((x i1.val).toRat - (capped.get1 i1).toRat)|
        ≤ |∑ i ∈ (Finset.range (8 - 0)).erase i1.val, (x i).toRat|
          + |(x i1.val).toRat - (capped.get1 i1).toRat| := abs_add_le _ _
      _ ≤ 7 * eta32q + DEV := by linarith
      _ = DEV + 7 * eta32q := by ring
  have hlist : ((List.range (8 - 0)).map fun i => (x i).toRat).sum
      = ∑ i ∈ Finset.range (8 - 0), (x i).toRat := list_sum_range_congr _ _
  have hresE' : |top2v.toRat - ∑ i ∈ Finset.range (8 - 0), (x i).toRat|
      ≤ 300 * u32q + 100 * eta32q := by
    rw [← hlist]
    exact hresE
  have hbound_q : |top2v.toRat - (capped.get1 i1).toRat| ≤ 512 * u32q := by
    have htri : |top2v.toRat - (capped.get1 i1).toRat|
        ≤ |top2v.toRat - ∑ i ∈ Finset.range (8 - 0), (x i).toRat|
          + |(∑ i ∈ Finset.range (8 - 0), (x i).toRat) - (capped.get1 i1).toRat| := by
      calc |top2v.toRat - (capped.get1 i1).toRat|
          = |(top2v.toRat - ∑ i ∈ Finset.range (8 - 0), (x i).toRat)
              + ((∑ i ∈ Finset.range (8 - 0), (x i).toRat) - (capped.get1 i1).toRat)| := by
            ring_nf
        _ ≤ _ := abs_add_le _ _
    linarith [htri, hresE', hSdev, hnumD]
  refine ⟨hresfin, ?_⟩
  have h1 : top2v.toReal = (top2v.toRat : ℝ) := rfl
  have h2 : (capped.get1 i1).toReal = ((capped.get1 i1).toRat : ℝ) := rfl
  rw [h1, h2]
  have hcast : |(top2v.toRat : ℝ) - ((capped.get1 i1).toRat : ℝ)|
      ≤ ((512 * u32q : ℚ) : ℝ) := by
    have h := Rat.cast_le (K := ℝ).mpr hbound_q
    rwa [Rat.cast_abs, Rat.cast_sub] at h
  refine hcast.trans_eq ?_
  rw [h512]
  norm_num

end Claims.MoeRouter
