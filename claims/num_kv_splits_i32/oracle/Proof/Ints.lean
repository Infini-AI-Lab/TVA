import Proof.Defs

/-!
# `tl.cdiv` on positive operands, and the kernel's integer values

`safe_cdiv`: `tl.cdiv(x, y)` (Triton's `(x + (y - 1)) // y`) of a nonnegative `x` by a positive `y`,
without wrapping, is `cdivZ x y` lane by lane. Then the bounds of the kernel's integers: every
chunk size and split count is at least `1` and at most the largest length.
-/

namespace Claims.NumKvSplits

open TVA

section Cdiv
variable {α : Type} {E : Env} {R W : Set Int} {Q : α → Mem → Prop} {m : Mem}

/-- **`tl.cdiv(x, y)`** with every lane `0 ≤ x`, `0 < y`, `x + y - 1 < 2^(w-1)`: defined, and lane
`i` is `cdivZ x y`. -/
theorem safe_cdiv {s : Shape} {w : Nat} (hw : 2 ≤ w) {x y : Tensor s (IntV true w)}
    {k : Tensor s (IntV true w) → Prog α}
    (h : ∀ i, 0 ≤ (x.get i).toInt ∧ 0 < (y.get i).toInt ∧
      (x.get i).toInt + (y.get i).toInt - 1 < 2 ^ (w - 1))
    (hk : ∀ r : Tensor s (IntV true w),
      (∀ i, (r.get i).toInt = cdivZ (x.get i).toInt (y.get i).toInt) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.cdiv x y >>= k) := by
  unfold tl.cdiv
  have hnum : ∀ i, ((x + (y - Tensor.full 1)).get i).toInt =
      (x.get i).toInt + (y.get i).toInt - 1 := by
    intro i
    obtain ⟨h1, h2, h3⟩ := h i
    have hpos : (0 : ℤ) ≤ 2 ^ (w - 1) := by positivity
    have hsub : ((y - Tensor.full 1).get i).toInt = (y.get i).toInt - 1 := by
      rw [Tensor.get_sub, Tensor.get_full]
      exact IntV.toInt_sub rfl (IntV.toInt_one hw) (by rw [IntV.sfits_iff]; constructor <;> omega)
    rw [Tensor.get_add]
    rw [IntV.toInt_add rfl hsub (by rw [IntV.sfits_iff]; constructor <;> omega)]
    ring
  refine safe_idiv (fun i => ?_) fun r hr => hk r fun i => ?_
  · obtain ⟨h1, h2, -⟩ := h i
    obtain ⟨q, hq, -⟩ := IntV.div?_of_nonneg (a := (x + (y - Tensor.full 1)).get i) (b := y.get i)
      (by rw [hnum i]; omega) h2
    exact ⟨q, hq⟩
  · obtain ⟨h1, h2, -⟩ := h i
    rw [IntV.toInt_of_div? (hr i) (by rw [hnum i]; omega) h2, hnum i]
    rfl

end Cdiv

/-! ## Ceiling division -/

/-- `cdivZ a b ≥ 1` for `a, b ≥ 1`. -/
theorem cdivZ_pos {a b : ℤ} (ha : 1 ≤ a) (hb : 1 ≤ b) : 1 ≤ cdivZ a b := by
  unfold cdivZ
  rw [Int.le_ediv_iff_mul_le (by omega)]; omega

/-- `cdivZ a b ≤ a` for `a, b ≥ 1`. -/
theorem cdivZ_le {a b : ℤ} (ha : 1 ≤ a) (hb : 1 ≤ b) : cdivZ a b ≤ a := by
  unfold cdivZ
  rw [Int.ediv_le_iff_le_mul (by omega)]; nlinarith

/-- `cdivZ a b ≥ 0` for `a ≥ 0`, `b ≥ 1`. -/
theorem cdivZ_nonneg {a b : ℤ} (ha : 0 ≤ a) (hb : 1 ≤ b) : 0 ≤ cdivZ a b := by
  unfold cdivZ; exact Int.ediv_nonneg (by omega) (by omega)

/-! ## The kernel's integers -/

/-- `kv_chunk_size_1` is in `[1, maxLen]`. -/
theorem chunk1_bounds {d : Inst} (hd : d.ok) {m : Mem} (hpre : Pre d m) :
    1 ≤ chunk1 d m ∧ chunk1 d m ≤ maxLen d m := by
  obtain ⟨hM1, -, -⟩ := maxLen_bounds hd hpre
  obtain ⟨hm1, -⟩ := minLen_bounds hd hpre
  have hK := hd.K_pos
  have h1 : 1 ≤ min (cdivZ (maxLen d m) (minLen d m)) (d.K : ℤ) :=
    le_min (cdivZ_pos hM1 hm1) (by exact_mod_cast hK)
  exact ⟨cdivZ_pos hM1 h1, cdivZ_le hM1 h1⟩

/-- `token_grid` is in `[1, num_seq · num_group · num_head]`. -/
theorem tokenGrid_bounds {d : Inst} (hd : d.ok) :
    1 ≤ tokenGrid d ∧ tokenGrid d ≤ (d.S : ℤ) * d.G * d.H := by
  have hS := hd.S_pos
  have hG := hd.G_pos
  have hH : 1 ≤ d.H := le_trans hd.Hkv_pos hd.Hkv_le
  have hSG : (1 : ℤ) ≤ (d.S : ℤ) * d.G := by
    have : 1 ≤ d.S * d.G := Nat.one_le_iff_ne_zero.mpr (by positivity)
    exact_mod_cast this
  unfold tokenGrid
  split_ifs with hg
  · refine ⟨?_, le_rfl⟩
    have : (1 : ℤ) ≤ (d.S : ℤ) * d.G * d.H := by
      have : 1 ≤ d.S * d.G * d.H := Nat.one_le_iff_ne_zero.mpr (by positivity)
      exact_mod_cast this
    exact this
  · have hg1 : (1 : ℤ) ≤ (d.H : ℤ) / d.Hkv := by
      rw [Int.le_ediv_iff_mul_le (by exact_mod_cast hd.Hkv_pos)]
      have := hd.Hkv_le; omega
    have hb : 1 ≤ min 16 ((d.H : ℤ) / d.Hkv) := le_min (by norm_num) hg1
    have hc1 := cdivZ_pos (a := (d.H : ℤ)) (by exact_mod_cast hH) hb
    have hc2 := cdivZ_le (a := (d.H : ℤ)) (by exact_mod_cast hH) hb
    constructor
    · nlinarith
    · exact mul_le_mul_of_nonneg_left hc2 (by positivity)

/-- `kv_chunk_size_2` is in `[1, maxLen]` for `E ≥ 1`. -/
theorem chunk2_bounds {d : Inst} (hd : d.ok) {m : Mem} (hpre : Pre d m) {E : ℤ} (hE : 1 ≤ E) :
    1 ≤ chunk2 d m E ∧ chunk2 d m E ≤ maxLen d m := by
  obtain ⟨hM1, -, -⟩ := maxLen_bounds hd hpre
  obtain ⟨hT, -⟩ := tokenGrid_bounds hd
  have h1 : 1 ≤ min (cdivZ E (tokenGrid d)) (d.K : ℤ) :=
    le_min (cdivZ_pos hE hT) (by exact_mod_cast hd.K_pos)
  exact ⟨cdivZ_pos hM1 h1, cdivZ_le hM1 h1⟩

/-- A request's split count is in `[1, len]`. -/
theorem numSplits_bounds {d : Inst} (hd : d.ok) {m : Mem} (hpre : Pre d m) {E : ℤ} (hE : 1 ≤ E)
    {s : Nat} (hs : s < d.S) : 1 ≤ numSplits d m E s ∧ numSplits d m E s ≤ 2 ^ 22 := by
  obtain ⟨hl1, hl2⟩ := hpre.2.2.2 s hs
  obtain ⟨hc1, -⟩ := chunk1_bounds hd hpre
  obtain ⟨hc2, -⟩ := chunk2_bounds hd hpre hE
  unfold numSplits
  refine ⟨le_trans (cdivZ_pos hl1 hc1) (le_max_left _ _), max_le ?_ ?_⟩
  · exact le_trans (cdivZ_le hl1 hc1) hl2
  · exact le_trans (cdivZ_le hl1 hc2) hl2

end Claims.NumKvSplits
