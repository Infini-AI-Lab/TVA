import Proof.Setup

/-!
# `scaled_mm_int8`: the load masks and the integer offsets

Each `ltS` mask the kernel builds is the `decide` of the in-range condition for the lane, and the
`int32` scale offsets are the row/column indices `saOff`/`sbOff`. The `int64` bias offsets are
`pn * BN + j`.
-/

namespace Claims.ScaledMmInt8
open TVA
set_option maxHeartbeats 800000

/-! ### Bounds on the tiles and on the program indices -/

/-- In a launched program the row offset is below `2^31`. -/
theorem m_pmBM_add_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) {i : Nat}
    (hi : i < d.BM) : pm d p * d.BM + i < 2 ^ 31 := by
  have h1 : pm d p + 1 ≤ (d.M + d.BM - 1) / d.BM := pm_lt_gridM hd hp
  have h2 : ((d.M + d.BM - 1) / d.BM) * d.BM ≤ d.M + d.BM - 1 := Nat.div_mul_le_self _ _
  have h3 : (pm d p + 1) * d.BM ≤ d.M + d.BM - 1 :=
    le_trans (Nat.mul_le_mul_right _ h1) h2
  have h4 : (pm d p + 1) * d.BM = pm d p * d.BM + d.BM := by
    rw [Nat.add_mul, Nat.one_mul]
  have h5 : d.M + d.BM ≤ 2 ^ 31 := hd.M_le
  have h6 : 0 < d.BM := BM_pos hd
  omega

/-- The column offset `pn * BN + j` is below `2^31` for every program index. -/
theorem m_pnBN_add_lt {d : Inst} (hd : d.ok) (p : Nat) {j : Nat} (hj : j < d.BN) :
    pn d p * d.BN + j < 2 ^ 31 := by
  have h1 : pn d p + 1 ≤ gridN d := by
    have h := Nat.mod_lt p (gridN_pos hd); simp only [pn]; omega
  have h2 : gridN d * d.BN ≤ d.N + d.BN - 1 := by
    simp only [gridN]; exact Nat.div_mul_le_self _ _
  have h3 : (pn d p + 1) * d.BN ≤ d.N + d.BN - 1 :=
    le_trans (Nat.mul_le_mul_right _ h1) h2
  have h4 : (pn d p + 1) * d.BN = pn d p * d.BN + d.BN := by
    rw [Nat.add_mul, Nat.one_mul]
  have h5 : d.N + d.BN ≤ 2 ^ 31 := hd.N_le
  have h6 : 0 < d.BN := BN_pos hd
  omega

/-! ### The range of integer sums and products -/

/-- A value with an `int32` `toInt` fits `int32`. -/
theorem m_sfits_of_toInt {a : I32} {x : ℤ} (h : a.toInt = x) : IntV.SFits 32 x := by
  rw [← h]; exact IntV.sfits_toInt a

/-- `a + b` of naturals below `2^31` fits `int32`. -/
theorem m_sfits_add32 {a b : Nat} (h : a + b < 2 ^ 31) :
    IntV.SFits 32 ((a : ℤ) + (b : ℤ)) := by
  simp only [IntV.SFits]; push_cast; constructor <;> omega

/-- `a·b` of naturals below `2^31` fits `int32`. -/
theorem m_sfits_mul32 {a b : Nat} (h : a * b < 2 ^ 31) :
    IntV.SFits 32 ((a : ℤ) * (b : ℤ)) := by
  have e : ((a : ℤ) * (b : ℤ)) = ((a * b : Nat) : ℤ) := by push_cast; ring
  rw [e]; simp only [IntV.SFits]; push_cast; constructor <;> omega

/-- The value of `tl.arange 0 n` in lane `a`. -/
theorem m_arange_toInt {n : Nat} (hn : n < 2 ^ 31) (a : Fin n) :
    ((tl.arange 0 n).get (a, ())).toInt = ((a.1 : Nat) : ℤ) := by
  show (IntV.ofInt (0 + (a.1 : ℤ))).toInt = ((a.1 : Nat) : ℤ)
  rw [IntV.I32.toInt_ofInt (by constructor <;> omega)]
  ring

/-- The row mask of the prologue (`masks_am`). -/
theorem mask_am (d : Inst) (hd : d.ok) (p : Nat) (hp : p < grid d)
    (r4 r6 : Tensor [] I64) (r5 : Tensor [d.BM] I64)
    (h4 : r4.item.toInt = ((pm d p * d.BM : Nat) : ℤ))
    (h5 : ∀ i : Fin d.BM, (r5.get1 i).toInt = (i.1 : ℤ))
    (h6 : r6.item.toInt = ((d.M : Nat) : ℤ)) :
    ∀ i : Idx [d.BM], ((r4.item + r5).ltS r6.item).get i
      = decide (pm d p * d.BM + i.1 < d.M) := by
  intro i
  obtain ⟨a, u⟩ := i
  cases u
  have h5a : (r5.get (a, ())).toInt = ((a.1 : Nat) : ℤ) := h5 a
  have hb : pm d p * d.BM + a.1 < 2 ^ 31 := m_pmBM_add_lt hd hp a.2
  have hsum : ((r4.item + r5).get (a, ())).toInt = ((pm d p * d.BM + a.1 : Nat) : ℤ) := by
    show (r4.item + r5.get (a, ())).toInt = _
    rw [IntV.toInt_add rfl rfl (by
      rw [h4, h5a]; exact sfits_add_nat (by omega)), h4, h5a]
    push_cast; ring
  show IntV.lt ((r4.item + r5).get (a, ())) r6.item = _
  rw [IntV.lt_eq, hsum, h6]
  simp only [decide_eq_decide]; norm_cast

/-- The column mask of the prologue (`masks_bn`), also used for the bias load. -/
theorem mask_bn (d : Inst) (hd : d.ok) (p : Nat)
    (r7 r9 : Tensor [] I64) (r8 : Tensor [d.BN] I64)
    (h7 : r7.item.toInt = ((pn d p * d.BN : Nat) : ℤ))
    (h8 : ∀ j : Fin d.BN, (r8.get1 j).toInt = (j.1 : ℤ))
    (h9 : r9.item.toInt = ((d.N : Nat) : ℤ)) :
    ∀ i : Idx [d.BN], ((r7.item + r8).ltS r9.item).get i
      = decide (pn d p * d.BN + i.1 < d.N) := by
  intro i
  obtain ⟨a, u⟩ := i
  cases u
  have h8a : (r8.get (a, ())).toInt = ((a.1 : Nat) : ℤ) := h8 a
  have hb : pn d p * d.BN + a.1 < 2 ^ 31 := m_pnBN_add_lt hd p a.2
  have hsum : ((r7.item + r8).get (a, ())).toInt = ((pn d p * d.BN + a.1 : Nat) : ℤ) := by
    show (r7.item + r8.get (a, ())).toInt = _
    rw [IntV.toInt_add rfl rfl (by
      rw [h7, h8a]; exact sfits_add_nat (by omega)), h7, h8a]
    push_cast; ring
  show IntV.lt ((r7.item + r8).get (a, ())) r9.item = _
  rw [IntV.lt_eq, hsum, h9]
  simp only [decide_eq_decide]; norm_cast

/-- The `int64` column offsets `offsets_bn`, as integers. -/
theorem offs_bn_toInt (d : Inst) (hd : d.ok) (p : Nat)
    (r7 : Tensor [] I64) (r8 : Tensor [d.BN] I64)
    (h7 : r7.item.toInt = ((pn d p * d.BN : Nat) : ℤ))
    (h8 : ∀ j : Fin d.BN, (r8.get1 j).toInt = (j.1 : ℤ)) :
    ∀ j : Fin d.BN, ((r7.item + r8).get1 j).toInt = ((pn d p * d.BN + j.1 : Nat) : ℤ) := by
  intro j
  have h8j : (r8.get (j, ())).toInt = ((j.1 : Nat) : ℤ) := h8 j
  have hb : pn d p * d.BN + j.1 < 2 ^ 31 := m_pnBN_add_lt hd p j.2
  show (r7.item + r8.get (j, ())).toInt = ((pn d p * d.BN + j.1 : Nat) : ℤ)
  rw [IntV.toInt_add rfl rfl (by
    rw [h7, h8j]; exact sfits_add_nat (by omega)), h7, h8j]
  push_cast; ring

/-- The `int32` offsets of the `A` scale, as `saOff`. -/
theorem offs_sa_toInt (d : Inst) (hd : d.ok) (p : Nat) (hp : p < grid d)
    (pmT : Tensor [] I32) (hpm : pmT.item.toInt = (pm d p : ℤ)) :
    ∀ i : Idx [BSA d],
      ((tl.arange 0 (BSA d) +
        (IntV.ofNat (if 1 < BSA d then 1 else 0) : I32) * pmT.item *
          (IntV.ofNat d.BM : I32)).get i).toInt = ((saOff d p i.1 : Nat) : ℤ) := by
  intro i
  obtain ⟨a, u⟩ := i
  cases u
  have hBM31 : d.BM < 2 ^ 31 := by
    have := BM_le256 hd; have := BM_pos hd; omega
  have hBMv : (IntV.ofNat d.BM : I32).toInt = ((d.BM : Nat) : ℤ) :=
    IntV.I32.toInt_ofNat hBM31
  have hpmBM : pm d p * d.BM < 2 ^ 31 := by
    have := m_pmBM_add_lt hd hp (i := 0) (by exact BM_pos hd); omega
  cases hs : d.saVec
  · have hBSA : BSA d = 1 := by simp [BSA, hs]
    have hcv : (if 1 < BSA d then 1 else 0) = 0 := by rw [hBSA]; norm_num
    have ha0 : (a.1 : Nat) = 0 := by have h := a.2; omega
    have har : ((tl.arange 0 (BSA d)).get (a, ())).toInt = ((a.1 : Nat) : ℤ) :=
      m_arange_toInt (n := BSA d) (by rw [hBSA]; norm_num) a
    have h1 : ((IntV.ofNat 0 : I32) * pmT.item).toInt = (0 : ℤ) := by
      rw [IntV.toInt_mul rfl rfl (by
            rw [IntV.I32.toInt_ofNat (by norm_num), hpm]
            simp only [IntV.SFits, Nat.cast_zero, zero_mul]; norm_num),
          IntV.I32.toInt_ofNat (by norm_num), hpm]
      simp
    have hprod : (((IntV.ofNat 0 : I32) * pmT.item) * (IntV.ofNat d.BM : I32)).toInt
        = (0 : ℤ) := by
      rw [IntV.toInt_mul rfl rfl (by
            rw [h1, hBMv]
            simp only [IntV.SFits, zero_mul]; norm_num), h1, hBMv]
      simp
    rw [hcv]
    show (((tl.arange 0 (BSA d)).get (a, ())) +
        ((IntV.ofNat 0 : I32) * pmT.item) * (IntV.ofNat d.BM : I32)).toInt
      = ((saOff d p a.1 : Nat) : ℤ)
    rw [IntV.toInt_add rfl rfl (by
          rw [har, hprod]
          have hz : ((a.1 : Nat) : ℤ) = 0 := by rw [ha0]; norm_num
          rw [hz]; norm_num [IntV.SFits]), har, hprod]
    rw [saOff_eq hd p a.1, hs]
    norm_num
  · have hBSA : BSA d = d.BM := by simp [BSA, hs]
    have hcv : (if 1 < BSA d then 1 else 0) = 1 := by
      rw [hBSA]; simp [show 1 < d.BM by have := BM_ge2 hd; omega]
    have ha : (a.1 : Nat) < d.BM := by have h := a.2; omega
    have har : ((tl.arange 0 (BSA d)).get (a, ())).toInt = ((a.1 : Nat) : ℤ) :=
      m_arange_toInt (n := BSA d) (by rw [hBSA]; omega) a
    have h1 : ((IntV.ofNat 1 : I32) * pmT.item).toInt = ((pm d p : Nat) : ℤ) := by
      rw [IntV.toInt_mul rfl rfl (by
            rw [IntV.I32.toInt_ofNat (by norm_num), hpm]
            simpa using m_sfits_of_toInt (a := pmT.item) hpm),
          IntV.I32.toInt_ofNat (by norm_num), hpm]
      simp
    have hprod : (((IntV.ofNat 1 : I32) * pmT.item) * (IntV.ofNat d.BM : I32)).toInt
        = ((pm d p * d.BM : Nat) : ℤ) := by
      rw [IntV.toInt_mul rfl rfl (by
            rw [h1, hBMv]
            rw [show ((pm d p : ℤ) * ((d.BM : Nat) : ℤ)) = ((pm d p * d.BM : Nat) : ℤ) by
              push_cast; ring]
            exact m_sfits_mul32 hpmBM), h1, hBMv]
      push_cast; ring
    rw [hcv]
    show (((tl.arange 0 (BSA d)).get (a, ())) +
        ((IntV.ofNat 1 : I32) * pmT.item) * (IntV.ofNat d.BM : I32)).toInt
      = ((saOff d p a.1 : Nat) : ℤ)
    rw [IntV.toInt_add rfl rfl (by
          rw [har, hprod]
          exact m_sfits_add32 (by have := m_pmBM_add_lt hd hp ha; omega)), har, hprod]
    rw [saOff_eq hd p a.1, hs]
    simp only [ite_true]
    push_cast; ring

/-- The mask of the `A` scale (`masks_scale_am`). -/
theorem mask_sa (d : Inst) (hd : d.ok) (p : Nat) (hp : p < grid d) (pmT : Tensor [] I32)
    (hpm : pmT.item.toInt = (pm d p : ℤ)) :
    ∀ i : Idx [BSA d],
      ((tl.arange 0 (BSA d) +
        (IntV.ofNat (if 1 < BSA d then 1 else 0) : I32) * pmT.item *
          (IntV.ofNat d.BM : I32)).ltS (IntV.ofNat d.M : I32)).get i
      = decide (saOff d p i.1 < d.M) := by
  intro i
  obtain ⟨a, u⟩ := i
  cases u
  have hM31 : d.M < 2 ^ 31 := by have := hd.M_le; have := BM_pos hd; omega
  show IntV.lt ((tl.arange 0 (BSA d) +
        (IntV.ofNat (if 1 < BSA d then 1 else 0) : I32) * pmT.item *
          (IntV.ofNat d.BM : I32)).get (a, ())) (IntV.ofNat d.M) = _
  rw [IntV.lt_eq, offs_sa_toInt d hd p hp pmT hpm (a, ()), IntV.I32.toInt_ofNat hM31]
  simp only [decide_eq_decide]; norm_cast

/-- The `int32` offsets of the `B` scale, as `sbOff`. -/
theorem offs_sb_toInt (d : Inst) (hd : d.ok) (p : Nat) (pnT : Tensor [] I32)
    (hpn : pnT.item.toInt = (pn d p : ℤ)) :
    ∀ i : Idx [BSB d],
      ((tl.arange 0 (BSB d) +
        (IntV.ofNat (if 1 < BSB d then 1 else 0) : I32) * pnT.item *
          (IntV.ofNat d.BN : I32)).get i).toInt = ((sbOff d p i.1 : Nat) : ℤ) := by
  intro i
  obtain ⟨a, u⟩ := i
  cases u
  have hBN31 : d.BN < 2 ^ 31 := by
    have := BN_le256 hd; have := BN_pos hd; omega
  have hBNv : (IntV.ofNat d.BN : I32).toInt = ((d.BN : Nat) : ℤ) :=
    IntV.I32.toInt_ofNat hBN31
  have hpnBN : pn d p * d.BN < 2 ^ 31 := pnBN_lt31 hd p
  cases hs : d.sbVec
  · have hBSB : BSB d = 1 := by simp [BSB, hs]
    have hcv : (if 1 < BSB d then 1 else 0) = 0 := by rw [hBSB]; norm_num
    have ha0 : (a.1 : Nat) = 0 := by have h := a.2; omega
    have har : ((tl.arange 0 (BSB d)).get (a, ())).toInt = ((a.1 : Nat) : ℤ) :=
      m_arange_toInt (n := BSB d) (by rw [hBSB]; norm_num) a
    have h1 : ((IntV.ofNat 0 : I32) * pnT.item).toInt = (0 : ℤ) := by
      rw [IntV.toInt_mul rfl rfl (by
            rw [IntV.I32.toInt_ofNat (by norm_num), hpn]
            simp only [IntV.SFits, Nat.cast_zero, zero_mul]; norm_num),
          IntV.I32.toInt_ofNat (by norm_num), hpn]
      simp
    have hprod : (((IntV.ofNat 0 : I32) * pnT.item) * (IntV.ofNat d.BN : I32)).toInt
        = (0 : ℤ) := by
      rw [IntV.toInt_mul rfl rfl (by
            rw [h1, hBNv]
            simp only [IntV.SFits, zero_mul]; norm_num), h1, hBNv]
      simp
    rw [hcv]
    show (((tl.arange 0 (BSB d)).get (a, ())) +
        ((IntV.ofNat 0 : I32) * pnT.item) * (IntV.ofNat d.BN : I32)).toInt
      = ((sbOff d p a.1 : Nat) : ℤ)
    rw [IntV.toInt_add rfl rfl (by
          rw [har, hprod]
          have hz : ((a.1 : Nat) : ℤ) = 0 := by rw [ha0]; norm_num
          rw [hz]; norm_num [IntV.SFits]), har, hprod]
    rw [sbOff_eq hd p a.1, hs]
    norm_num
  · have hBSB : BSB d = d.BN := by simp [BSB, hs]
    have hcv : (if 1 < BSB d then 1 else 0) = 1 := by
      rw [hBSB]; simp [show 1 < d.BN by have := BN_ge2 hd; omega]
    have ha : (a.1 : Nat) < d.BN := by have h := a.2; omega
    have har : ((tl.arange 0 (BSB d)).get (a, ())).toInt = ((a.1 : Nat) : ℤ) :=
      m_arange_toInt (n := BSB d) (by rw [hBSB]; omega) a
    have h1 : ((IntV.ofNat 1 : I32) * pnT.item).toInt = ((pn d p : Nat) : ℤ) := by
      rw [IntV.toInt_mul rfl rfl (by
            rw [IntV.I32.toInt_ofNat (by norm_num), hpn]
            simpa using m_sfits_of_toInt (a := pnT.item) hpn),
          IntV.I32.toInt_ofNat (by norm_num), hpn]
      simp
    have hprod : (((IntV.ofNat 1 : I32) * pnT.item) * (IntV.ofNat d.BN : I32)).toInt
        = ((pn d p * d.BN : Nat) : ℤ) := by
      rw [IntV.toInt_mul rfl rfl (by
            rw [h1, hBNv]
            rw [show ((pn d p : ℤ) * ((d.BN : Nat) : ℤ)) = ((pn d p * d.BN : Nat) : ℤ) by
              push_cast; ring]
            exact m_sfits_mul32 hpnBN), h1, hBNv]
      push_cast; ring
    rw [hcv]
    show (((tl.arange 0 (BSB d)).get (a, ())) +
        ((IntV.ofNat 1 : I32) * pnT.item) * (IntV.ofNat d.BN : I32)).toInt
      = ((sbOff d p a.1 : Nat) : ℤ)
    rw [IntV.toInt_add rfl rfl (by
          rw [har, hprod]
          exact m_sfits_add32 (by have := m_pnBN_add_lt hd p ha; omega)), har, hprod]
    rw [sbOff_eq hd p a.1, hs]
    simp only [ite_true]
    push_cast; ring

/-- The mask of the `B` scale (`masks_scale_bn`). -/
theorem mask_sb (d : Inst) (hd : d.ok) (p : Nat) (pnT : Tensor [] I32)
    (hpn : pnT.item.toInt = (pn d p : ℤ)) :
    ∀ i : Idx [BSB d],
      ((tl.arange 0 (BSB d) +
        (IntV.ofNat (if 1 < BSB d then 1 else 0) : I32) * pnT.item *
          (IntV.ofNat d.BN : I32)).ltS (IntV.ofNat d.N : I32)).get i
      = decide (sbOff d p i.1 < d.N) := by
  intro i
  obtain ⟨a, u⟩ := i
  cases u
  have hN31 : d.N < 2 ^ 31 := by have := hd.N_le; have := BN_pos hd; omega
  show IntV.lt ((tl.arange 0 (BSB d) +
        (IntV.ofNat (if 1 < BSB d then 1 else 0) : I32) * pnT.item *
          (IntV.ofNat d.BN : I32)).get (a, ())) (IntV.ofNat d.N) = _
  rw [IntV.lt_eq, offs_sb_toInt d hd p pnT hpn (a, ()), IntV.I32.toInt_ofNat hN31]
  simp only [decide_eq_decide]; norm_cast

/-- The second factor of the two scale masks is `true` in every lane. -/
theorem mask_one (i : Idx [1, 1]) :
    (((tl.arange 0 1).ltS (IntV.ofInt 1 : I32)).col 1).get i = true := by
  obtain ⟨a, b, u⟩ := i
  cases u
  have ha : (a.1 : Nat) = 0 := by have := a.2; omega
  have hb : (b.1 : Nat) = 0 := by have := b.2; omega
  have har : ((tl.arange 0 1).get (a, ())).toInt = ((a.1 : Nat) : ℤ) :=
    m_arange_toInt (n := 1) (by norm_num) a
  show IntV.lt ((tl.arange 0 1).get (a, ())) (IntV.ofInt 1) = true
  rw [IntV.lt_eq, har, IntV.I32.toInt_ofInt (by constructor <;> norm_num)]
  simp

end Claims.ScaledMmInt8
