import Proof.Spec

/-!
# Facts about a valid row

The bounds `Pre` gives about the row of new token `i` of request `b`: its score bound, its keys,
its values, and its trip counts; and the positions the loops have seen at their start and end.
-/

namespace Claims.ExtendAttn

open TVA

private theorem rf_kvh_lt {d : Inst} (hd : d.ok) {h : Nat} (hh : h < d.H) : kvh d h < d.HKV := by
  have hpos := hd.kv_pos
  have hkv : 0 < kvg d := Nat.div_pos (Nat.le_of_dvd (by omega) hd.kv_dvd) hpos
  unfold kvh
  rw [Nat.div_lt_iff_lt_mul hkv]
  have : d.HKV * kvg d = d.H := Nat.mul_div_cancel' hd.kv_dvd
  omega

private theorem rf_bm_pos {d : Inst} (hd : d.ok) : 0 < d.BM := by
  rw [hd.bm]; split_ifs <;> omega

private theorem rf_sig_nonneg {d : Inst} (hd : d.ok) : 0 ≤ sig d := by
  have h := hd.scale.2.1
  unfold sig
  have : (0 : ℚ) ≤ d.sm_scale.toRat := le_trans (by norm_num) h
  exact_mod_cast this

private theorem rf_scoreMag_nonneg {d : Inst} (hd : d.ok) (m : Mem) (b h i j : Nat) :
    0 ≤ scoreMag d m b h i j := by
  unfold scoreMag
  refine mul_nonneg (rf_sig_nonneg hd) (Finset.sum_nonneg fun e _ => ?_)
  have := mul_nonneg (Device.liftMag_nonneg .bf16 (qv d m b h i e))
    (Device.liftMag_nonneg .bf16 (keyK d m b h j e))
  exact_mod_cast this

private theorem rf_mem_keys {d : Inst} {m : Mem} {b i j : Nat} :
    j ∈ keys d m b i ↔ j < preOf d m b + extOf d m b ∧ attends d m b i j = true := by
  simp [keys, Finset.mem_filter, Finset.mem_range]

private theorem rf_own_mem {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat}
    (hb : b < d.batch) {i : Nat} (hi : i < extOf d init b) :
    preOf d init b + i ∈ keys d init b i := by
  rw [rf_mem_keys]
  refine ⟨by omega, ?_⟩
  unfold attends
  by_cases hcm : d.cm = true
  · have hm := (hpre.2.2.2.2.2.2.2.2.2.2.2.1 hcm b hb).2.2.2 i hi
    simp [hcm, hm.1]
  · simp [hcm]

private theorem rf_mem_keys_le {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat}
    (hb : b < d.batch) {i : Nat} (hi : i < extOf d init b) {j : Nat}
    (hj : j ∈ keys d init b i) : j < preOf d init b + i + 1 := by
  rw [rf_mem_keys] at hj
  obtain ⟨hj1, hj2⟩ := hj
  unfold attends at hj2
  by_cases hcm : d.cm = true
  · have hm := (hpre.2.2.2.2.2.2.2.2.2.2.2.1 hcm b hb).2.2.2 i hi
    simp only [hcm, ite_true, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hj2
    rcases hj2.1 with h | h
    · omega
    · by_contra hc
      have := hm.2 (j - preOf d init b) (by omega) (by omega)
      rw [this] at h; exact absurd h (by simp)
  · simp only [hcm, Bool.false_eq_true, ite_false, Bool.and_eq_true, Bool.or_eq_true,
      decide_eq_true_eq] at hj2
    rcases hj2.1 with h | h <;> omega

/-- **The bounds of a valid row.** The score bound `X` is in `[0, 2^12]`; the row has keys, at most
`2^18`; each key's `scoreMag` is at most `X`; each attended value component (`e < D`) is at most
`2^100`; the trips are at most `2^12 + 2` and the new-token trips at least one. -/
theorem row_facts {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {r : Nat} (hr : validRow d init i r) :
    0 ≤ rowX d init i r ∧ rowX d init i r ≤ 2 ^ 12 ∧
    (keys d init (bOf d i) (mOf d i * d.BM + r)).Nonempty ∧
    (keys d init (bOf d i) (mOf d i * d.BM + r)).card ≤ nmax ∧
    (∀ j ∈ keys d init (bOf d i) (mOf d i * d.BM + r),
      scoreMag d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) j ≤ rowX d init i r) ∧
    (∀ j ∈ keys d init (bOf d i) (mOf d i * d.BM + r), ∀ e, e < d.D →
      |rowVc d init (bOf d i) (hOf d i) e j| ≤ 2 ^ 100) ∧
    tripsPre d init (bOf d i) + tripsExt d init (bOf d i) (mOf d i * d.BM + r) ≤ 2 ^ 12 + 2 ∧
    1 ≤ tripsExt d init (bOf d i) (mOf d i * d.BM + r) := by
  obtain ⟨hb, hh, -⟩ := prog_range hi
  have hbm := rf_bm_pos hd
  have hexti : mOf d i * d.BM + r < extOf d init (bOf d i) := hr
  have hown := rf_own_mem hpre hb hexti
  have hsz := req_sizes hpre hb
  have hv := hpre.2.2.2.2.2.2.2.2.2.2.2.2 (bOf d i) hb
  obtain ⟨-, -, hvp, hsc⟩ := hv
  obtain ⟨-, hvk, hvpre, -⟩ := hpre.2.2.2.2.2.2.2.2.2.2.2.2 (bOf d i) hb
  have hsc' : ∀ j ∈ keys d init (bOf d i) (mOf d i * d.BM + r),
      scoreMag d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) j ≤ 2 ^ 12 :=
    fun j hj => hsc _ _ j hh hexti hj
  have hX : Xrow d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) ≤ 2 ^ 12 := by
    unfold Xrow
    refine Finset.sup'_le _ _ fun j hj => ?_
    rcases Finset.mem_insert.1 hj with rfl | hj
    · exact hsc' _ hown
    · exact hsc' _ hj
  have hX0 : 0 ≤ Xrow d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) := by
    unfold Xrow
    exact le_trans (rf_scoreMag_nonneg hd init _ _ _ (preOf d init (bOf d i) + (mOf d i * d.BM + r)))
      (Finset.le_sup' (fun j => scoreMag d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) j)
        (Finset.mem_insert_self _ _))
  refine ⟨hX0, hX, ⟨_, hown⟩, ?_, ?_, ?_, ?_, ?_⟩
  · calc (keys d init (bOf d i) (mOf d i * d.BM + r)).card
        ≤ (Finset.range (preOf d init (bOf d i) + extOf d init (bOf d i))).card :=
          Finset.card_le_card (Finset.filter_subset _ _)
      _ ≤ nmax := by simp only [Finset.card_range, nmax]; exact hsz.2.2.2
  · intro j hj
    exact Finset.le_sup' (fun j => scoreMag d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) j)
      (Finset.mem_insert_of_mem hj)
  · intro j hj e he
    have hj1 := (rf_mem_keys.1 hj).1
    have hg := rf_kvh_lt hd hh
    unfold rowVc keyV
    split_ifs with hjp
    · have := (hvpre j (kvh d (hOf d i)) e hjp hg he).2.2
      have h2 : |valB init d.vb (vbIdx d (slotOf d init (bOf d i) j) (kvh d (hOf d i)) e)| ≤ 2 ^ 100 := this
      exact_mod_cast h2
    · have := (hvk (j - preOf d init (bOf d i)) (kvh d (hOf d i)) e (by omega) hg he).2.2
      exact_mod_cast this
  · have hT : tripsPre d init (bOf d i) + tripsExt d init (bOf d i) (mOf d i * d.BM + r) ≤ 2 ^ 12 + 2 := by
      unfold tripsPre tripsExt BN
      have h1 : min (extOf d init (bOf d i)) (((mOf d i * d.BM + r) / d.BM + 1) * d.BM) ≤
          extOf d init (bOf d i) := min_le_left _ _
      have := hsz.2.2.2
      omega
    exact hT
  · unfold tripsExt BN
    have h2 : mOf d i * d.BM + r < ((mOf d i * d.BM + r) / d.BM + 1) * d.BM := by
      have h3 := Nat.mod_lt (mOf d i * d.BM + r) hbm
      have h4 := Nat.div_add_mod (mOf d i * d.BM + r) d.BM
      have h5 := Nat.mul_comm d.BM ((mOf d i * d.BM + r) / d.BM)
      rw [Nat.add_mul, Nat.one_mul]
      omega
    have : 1 ≤ min (extOf d init (bOf d i)) (((mOf d i * d.BM + r) / d.BM + 1) * d.BM) := by
      rw [Nat.le_min]; omega
    omega

/-- Every key of a valid row is at most its own position: `j < pre + (m * BM + r) + 1`. -/
theorem mem_keys_lt {d : Inst} (_hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {r : Nat} (hr : validRow d init i r) {j : Nat}
    (hj : j ∈ keys d init (bOf d i) (mOf d i * d.BM + r)) :
    j < preOf d init (bOf d i) + (mOf d i * d.BM + r) + 1 := by
  exact rf_mem_keys_le hpre (prog_range hi).1 hr hj

/-- **The new-token loop starts where the cached-part loop ended**: no key of a valid row lies in the
new tokens skipped by `extend_start` (a window), so the keys below `pre + extStart` are the
cached-part's keys. -/
theorem seen_start {d : Inst} (_hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {r : Nat} (hr : validRow d init i r) :
    seenE d init (bOf d i) (mOf d i * d.BM + r) (extStart d i) =
      seenP d init (bOf d i) (mOf d i * d.BM + r) (tripsPre d init (bOf d i)) := by
  obtain ⟨hb, -, -⟩ := prog_range hi
  have hi' : mOf d i * d.BM + r < extOf d init (bOf d i) := hr
  unfold seenE seenP
  refine Finset.filter_congr fun j hj => ?_
  have hjl := rf_mem_keys_le hpre hb hi' hj
  have hatt := (rf_mem_keys.1 hj).2
  unfold attends at hatt
  unfold tripsPre extStart BN
  by_cases hsw : 0 < d.sw
  · simp only [hsw, ite_true]
    simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hatt
    rcases hatt.2 with h | h
    · omega
    · omega
  · simp only [hsw, ite_false]
    omega

/-- **The new-token loop sees every key**: once it has passed the block's end, the keys below
`pre + es + 64 K` are all of the row's keys. -/
theorem seen_end {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {r : Nat} (hr : validRow d init i r) (hrb : r < d.BM) {es K : Nat}
    (hcov : blockEnd d init i ≤ es + BN * K) :
    seenE d init (bOf d i) (mOf d i * d.BM + r) (es + BN * K) =
      keys d init (bOf d i) (mOf d i * d.BM + r) := by
  obtain ⟨hb, -, -⟩ := prog_range hi
  have hbm := rf_bm_pos hd
  have hi' : mOf d i * d.BM + r < extOf d init (bOf d i) := hr
  unfold seenE
  refine Finset.filter_true_of_mem fun j hj => ?_
  have hjl := rf_mem_keys_le hpre hb hi' hj
  unfold blockEnd at hcov
  have h1 : (mOf d i + 1) * d.BM = mOf d i * d.BM + d.BM := by ring
  unfold BN at hcov ⊢
  omega

end Claims.ExtendAttn
