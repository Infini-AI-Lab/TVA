import Claim
import TVA.Proof
import Proof.Foot

/-!
# `Proof.Addrs`

The addresses a tile touches: they name elements of the matrices, inside the buffers, and are injective; `mag16` and chunk bounds in `float16` range.
-/

/-!
# The accesses of the `a`, `b` and `c` lanes

The `tl.load`s of `a` and `b` and the `tl.store` / `tl.atomic_add` of `c` address, at each lane
`(i, j)`, element `(pid_m * 128 + i) * stride_m + (off_2 * 128 + j) * stride_2` of the buffer, when
the lane's mask enables it. This file proves the `Buf.Addrs` witness of each access, and the
consequences the load, store and atomic rules of `TVA.Lemmas.Access` need: the enabled lanes are
inside the matrix and inside the buffer, and the `c` lanes are injective on the enabled lanes.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

set_option maxHeartbeats 400000

/-! ## The `Buf.Addrs` witnesses -/

/-- The lane addresses of the masked load of an `a` tile are the elements of `a` they name. -/
theorem aAddrs (hd : d.ok) {pm kk : I32} {u v : Nat}
    (hpm : pm.toInt = (pidMOf d u : Int)) (hkk : kk.toInt = (v : Int))
    (hu : u < tilesM d * tilesN d) (hv : v < steps d.K) :
    d.a.Addrs
      (tl.addrs (ptrsA (d.a.ptr 0) (offs1 pm 128) (offs1 kk 64) (IntV.ofNat d.stride_am) (IntV.ofNat d.stride_ak))
        (some (maskA (offs1 pm 128) (offs1 kk 64) (IntV.ofNat d.M) (IntV.ofNat d.K))))
      (fun i => (pidMOf d u * 128 + i.1.val) * d.stride_am + (v * 64 + i.2.1.val) * d.stride_ak)
      (fun i => decide (pidMOf d u * 128 + i.1.val < d.M) && decide (v * 64 + i.2.1.val < d.K)) :=
  Buf.addrs_of_get d.a
    (fun i => maskA_get hd hpm hkk i.1 i.2.1 (pidM_bound hd hu i.1) (kk_bound hd hv i.2.1))
    (fun i hi => by
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hi
      obtain ⟨h1, h2⟩ := hi
      exact ptrsA_lane hd hpm hkk i.1 i.2.1 h1 h2)

/-- A column lane of a tile column below `tilesN d` is below `2 ^ 31`. -/
theorem nLane_bound {w : Nat} (hd : d.ok) (hw : w < tilesN d) (j : Fin 128) :
    w * 128 + j.val < 2 ^ 31 := by
  have h1 : w + 1 ≤ tilesN d := hw
  have h2 : (w + 1) * 128 ≤ tilesN d * 128 := Nat.mul_le_mul_right _ h1
  have h3 : tilesN d * 128 ≤ d.N + 127 := by
    unfold tilesN
    exact Nat.div_mul_le_self _ _
  have h4 := hd.N_le
  have h5 : j.val < 128 := j.isLt
  have h6 : (w + 1) * 128 = w * 128 + 128 := by ring
  omega

/-- The lane addresses of the masked load of a `b` tile are the elements of `b` they name. -/
theorem bAddrs (hd : d.ok) {kk pn : I32} {v w : Nat}
    (hkk : kk.toInt = (v : Int)) (hpn : pn.toInt = (w : Int))
    (hv : v < steps d.K) (hw : w < tilesN d) :
    d.b.Addrs
      (tl.addrs (ptrsB (d.b.ptr 0) (offs1 kk 64) (offs1 pn 128) (IntV.ofNat d.stride_bk) (IntV.ofNat d.stride_bn))
        (some (maskB (offs1 kk 64) (offs1 pn 128) (IntV.ofNat d.K) (IntV.ofNat d.N))))
      (fun i => (v * 64 + i.1.val) * d.stride_bk + (w * 128 + i.2.1.val) * d.stride_bn)
      (fun i => decide (v * 64 + i.1.val < d.K) && decide (w * 128 + i.2.1.val < d.N)) :=
  Buf.addrs_of_get d.b
    (fun i => maskB_get hd hkk hpn i.1 i.2.1 (kk_bound hd hv i.1) (nLane_bound hd hw i.2.1))
    (fun i hi => by
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hi
      obtain ⟨h1, h2⟩ := hi
      exact ptrsB_lane hd hkk hpn i.1 i.2.1 h1 h2)

/-- The lane addresses of the masked store or atomic of a `c` tile are the elements of `c` they name. -/
theorem cAddrs (hd : d.ok) {pm pn : I32} {u : Nat}
    (hpm : pm.toInt = (pidMOf d u : Int)) (hpn : pn.toInt = (pidNOf d u : Int))
    (hu : u < tilesM d * tilesN d) :
    d.c.Addrs
      (tl.addrs (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm) (IntV.ofNat d.stride_cn))
        (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N))))
      (fun i => (pidMOf d u * 128 + i.1.val) * d.stride_cm + (pidNOf d u * 128 + i.2.1.val) * d.stride_cn)
      (fun i => decide (pidMOf d u * 128 + i.1.val < d.M) && decide (pidNOf d u * 128 + i.2.1.val < d.N)) :=
  Buf.addrs_of_get d.c
    (fun i => maskC_get hd hpm hpn i.1 i.2.1 (pidM_bound hd hu i.1) (pidN_bound hd hu i.2.1))
    (fun i hi => by
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hi
      obtain ⟨h1, h2⟩ := hi
      exact ptrsC_lane hd hpm hpn i.1 i.2.1 h1 h2)

/-! ## The image of the enabled lanes -/

/-- Every element an `a` tile loads lies in the matrix `a`. -/
theorem aImg_sub (_hd : d.ok) {u v : Nat} (_hu : u < tilesM d * tilesN d) (_hv : v < steps d.K) :
    Buf.img (fun i : Idx [128, 64] =>
        (pidMOf d u * 128 + i.1.val) * d.stride_am + (v * 64 + i.2.1.val) * d.stride_ak)
      (fun i : Idx [128, 64] => decide (pidMOf d u * 128 + i.1.val < d.M) && decide (v * 64 + i.2.1.val < d.K))
      ⊆ aIdx d := by
  rintro e ⟨i, he, rfl⟩
  simp only [Bool.and_eq_true, decide_eq_true_eq] at he
  obtain ⟨h1, h2⟩ := he
  exact ⟨_, _, h1, h2, rfl⟩

/-- Every element a `b` tile loads lies in the matrix `b`. -/
theorem bImg_sub (_hd : d.ok) {v w : Nat} (_hv : v < steps d.K) (_hw : w < tilesN d) :
    Buf.img (fun i : Idx [64, 128] =>
        (v * 64 + i.1.val) * d.stride_bk + (w * 128 + i.2.1.val) * d.stride_bn)
      (fun i : Idx [64, 128] => decide (v * 64 + i.1.val < d.K) && decide (w * 128 + i.2.1.val < d.N))
      ⊆ bIdx d := by
  rintro e ⟨i, he, rfl⟩
  simp only [Bool.and_eq_true, decide_eq_true_eq] at he
  obtain ⟨h1, h2⟩ := he
  exact ⟨_, _, h1, h2, rfl⟩

/-- Every element a `c` tile stores lies in the matrix `c`. -/
theorem cImg_sub (_hd : d.ok) {u : Nat} (_hu : u < tilesM d * tilesN d) :
    Buf.img (fun i : Idx [128, 128] =>
        (pidMOf d u * 128 + i.1.val) * d.stride_cm + (pidNOf d u * 128 + i.2.1.val) * d.stride_cn)
      (fun i : Idx [128, 128] => decide (pidMOf d u * 128 + i.1.val < d.M) && decide (pidNOf d u * 128 + i.2.1.val < d.N))
      ⊆ cIdx d := by
  rintro e ⟨i, he, rfl⟩
  simp only [Bool.and_eq_true, decide_eq_true_eq] at he
  obtain ⟨h1, h2⟩ := he
  exact ⟨_, _, h1, h2, rfl⟩

/-! ## The enabled lanes are inside the buffers -/

/-- The elements of an `a` tile are inside the buffer. -/
theorem aLt_len (hd : d.ok) {u v : Nat} (_hu : u < tilesM d * tilesN d) (_hv : v < steps d.K)
    (i : Idx [128, 64])
    (h : (decide (pidMOf d u * 128 + i.1.val < d.M) && decide (v * 64 + i.2.1.val < d.K)) = true) :
    (pidMOf d u * 128 + i.1.val) * d.stride_am + (v * 64 + i.2.1.val) * d.stride_ak < d.a.len := by
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨h1, h2⟩ := h
  exact hd.a_len _ _ h1 h2

/-- The elements of a `b` tile are inside the buffer. -/
theorem bLt_len (hd : d.ok) {v w : Nat} (_hv : v < steps d.K) (_hw : w < tilesN d)
    (i : Idx [64, 128])
    (h : (decide (v * 64 + i.1.val < d.K) && decide (w * 128 + i.2.1.val < d.N)) = true) :
    (v * 64 + i.1.val) * d.stride_bk + (w * 128 + i.2.1.val) * d.stride_bn < d.b.len := by
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨h1, h2⟩ := h
  exact hd.b_len _ _ h1 h2

/-- The elements of a `c` tile are inside the buffer. -/
theorem cLt_len (hd : d.ok) {u : Nat} (_hu : u < tilesM d * tilesN d)
    (i : Idx [128, 128])
    (h : (decide (pidMOf d u * 128 + i.1.val < d.M) && decide (pidNOf d u * 128 + i.2.1.val < d.N)) = true) :
    (pidMOf d u * 128 + i.1.val) * d.stride_cm + (pidNOf d u * 128 + i.2.1.val) * d.stride_cn < d.c.len := by
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨h1, h2⟩ := h
  exact hd.c_len _ _ h1 h2

/-! ## The `c` lanes are injective on the enabled lanes -/

/-- With `hd.c_rows`, a column offset below `N` contributes less than one row: `J * stride_cn` is
below `stride_cm`, so the row is read off the address `I * stride_cm + J * stride_cn`. -/
theorem cCol_lt (hd : d.ok) {J : Nat} (hJ : J < d.N) : J * d.stride_cn < d.stride_cm := by
  have hNpos : 0 < d.N := hd.N_pos
  have hscn : 0 < d.stride_cn := hd.cn.1
  have hrow : d.N * d.stride_cn ≤ d.stride_cm := hd.c_rows
  have h1 : J ≤ d.N - 1 := by omega
  have h2 : J * d.stride_cn ≤ (d.N - 1) * d.stride_cn := Nat.mul_le_mul_right _ h1
  have h3 : (d.N - 1) * d.stride_cn < d.N * d.stride_cn :=
    Nat.mul_lt_mul_of_pos_right (by omega) hscn
  omega

/-- Enabled lanes with the same address of a `c` tile are the same lane. -/
theorem cInj (hd : d.ok) {u : Nat} (_hu : u < tilesM d * tilesN d) :
    ∀ i j : Idx [128, 128],
      (decide (pidMOf d u * 128 + i.1.val < d.M) && decide (pidNOf d u * 128 + i.2.1.val < d.N)) = true →
      (decide (pidMOf d u * 128 + j.1.val < d.M) && decide (pidNOf d u * 128 + j.2.1.val < d.N)) = true →
      ((pidMOf d u * 128 + i.1.val) * d.stride_cm + (pidNOf d u * 128 + i.2.1.val) * d.stride_cn) =
      ((pidMOf d u * 128 + j.1.val) * d.stride_cm + (pidNOf d u * 128 + j.2.1.val) * d.stride_cn) →
      i = j := by
  intro i j hi hj heq
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hi hj
  obtain ⟨hiM, hiN⟩ := hi
  obtain ⟨hjM, hjN⟩ := hj
  have hscn : 0 < d.stride_cn := hd.cn.1
  have hCi := cCol_lt hd hiN
  have hCj := cCol_lt hd hjN
  have hle1 : (pidMOf d u * 128 + i.1.val) * d.stride_cm <
      (pidMOf d u * 128 + j.1.val + 1) * d.stride_cm := by
    have e : (pidMOf d u * 128 + j.1.val + 1) * d.stride_cm
        = (pidMOf d u * 128 + j.1.val) * d.stride_cm + d.stride_cm := by ring
    omega
  have hle2 : (pidMOf d u * 128 + j.1.val) * d.stride_cm <
      (pidMOf d u * 128 + i.1.val + 1) * d.stride_cm := by
    have e : (pidMOf d u * 128 + i.1.val + 1) * d.stride_cm
        = (pidMOf d u * 128 + i.1.val) * d.stride_cm + d.stride_cm := by ring
    omega
  have hI : pidMOf d u * 128 + i.1.val = pidMOf d u * 128 + j.1.val := by
    have h1 := Nat.lt_of_mul_lt_mul_right hle1
    have h2 := Nat.lt_of_mul_lt_mul_right hle2
    omega
  have hA : (pidMOf d u * 128 + i.1.val) * d.stride_cm =
      (pidMOf d u * 128 + j.1.val) * d.stride_cm := by rw [hI]
  have hC : (pidNOf d u * 128 + i.2.1.val) * d.stride_cn =
      (pidNOf d u * 128 + j.2.1.val) * d.stride_cn := by omega
  have hJ : pidNOf d u * 128 + i.2.1.val = pidNOf d u * 128 + j.2.1.val :=
    Nat.mul_right_cancel hscn hC
  have hi1 : i.1.val = j.1.val := by omega
  have hi2 : i.2.1.val = j.2.1.val := by omega
  exact Prod.ext (Fin.ext hi1) (Prod.ext (Fin.ext hi2) rfl)

end Claims.StreamKMatmul


/-!
# Magnitude bounds of the operands, the partial sums and the accumulator

The numeric side of the proof needs a uniform cap on

* the `f16` operands (`|x| ≤ 65504`),
* one chunk sum (`chunkMag ≤ 2 ^ 38`),
* a partial sum deviation (`|dev| ≤ 2 ^ 45`),
* one program's `fineOne` error (`≤ 2 ^ 38`),
* a `float32` accumulator (`≤ 2 ^ 46`),

and the two "room" inequalities the platform rules are applied under: the
`DotLift.dotAdd_step` hypothesis `hroom` (`dot_room`) and, in `Val.lean`, the
`replay_err_shared` room (there the bound `Format.binary32.maxVal` is so much larger than
everything that `dot_room`'s constant suffices).

Also here: the reindexing of one chunk (`Fin 64` ↔ `chunk K v`), which needs `K % 64 = 0`.
-/

namespace Claims.StreamKMatmul

open TVA

/-! ## The `f16` operands -/

/-- A finite `f16` has magnitude at most `65504`. -/
theorem f16_toRat_abs_le {x : F16} (hx : x.isFinite = true) : |x.toRat| ≤ 65504 := by
  have h := FloatV.abs_toRat_le_maxVal (f := Format.binary16)
    (TVA.Format.binary16_regular) hx
  rwa [TVA.Format.binary16_maxVal] at h

/-- `mag16` of a bounded rational is bounded by `65504`. -/
theorem mag16_le {q : ℚ} (hq : |q| ≤ 65504) : mag16 q ≤ 65504 := by
  rw [mag16_eq_liftMag]
  unfold Device.liftMag Device.liftFloor
  split
  · norm_num
  · refine max_le hq ?_
    norm_num

/-- `liftMag .f16` of a bounded rational is bounded by `65504`. -/
theorem liftMag_f16_le {q : ℚ} (hq : |q| ≤ 65504) : Device.liftMag .f16 q ≤ 65504 := by
  unfold Device.liftMag Device.liftFloor
  split
  · norm_num
  · refine max_le hq ?_
    norm_num

/-! ## Bounds on the sums -/

/-- One chunk's magnitude sum is at most `2 ^ 38`. -/
theorem chunkMag_le {va vb : Nat → F16} {K t : Nat}
    (hva : ∀ k, k < K → |(va k).toRat| ≤ 65504)
    (hvb : ∀ k, k < K → |(vb k).toRat| ≤ 65504) :
    chunkMag va vb K t ≤ 2 ^ 38 := by
  have hterm : ∀ k ∈ chunk K t, mag16 (va k).toRat * mag16 (vb k).toRat ≤ 2 ^ 32 := by
    intro k hk
    have hkK : k < K := Finset.mem_range.mp (chunk_subset_range K t hk)
    have h1 : mag16 (va k).toRat ≤ 65504 := mag16_le (hva k hkK)
    have h2 : mag16 (vb k).toRat ≤ 65504 := mag16_le (hvb k hkK)
    have h1n : (0 : ℚ) ≤ mag16 (va k).toRat := mag16_nonneg _
    have h2n : (0 : ℚ) ≤ mag16 (vb k).toRat := mag16_nonneg _
    calc mag16 (va k).toRat * mag16 (vb k).toRat ≤ 65504 * 65504 :=
          mul_le_mul h1 h2 h2n (by norm_num)
      _ ≤ 2 ^ 32 := by norm_num
  have hcard : (chunk K t).card ≤ 64 := by
    have h1 : min (64 * t + 64) K ≤ 64 * t + 64 := min_le_left _ _
    unfold chunk
    rw [Nat.card_Ico]
    omega
  unfold chunkMag
  calc ∑ k ∈ chunk K t, mag16 (va k).toRat * mag16 (vb k).toRat
      ≤ (chunk K t).card • (2 ^ 32 : ℚ) := Finset.sum_le_card_nsmul _ _ _ hterm
    _ = ((chunk K t).card : ℚ) * 2 ^ 32 := by rw [nsmul_eq_mul]
    _ ≤ (64 : ℚ) * 2 ^ 32 := by
        have hc' : ((chunk K t).card : ℚ) ≤ 64 := by exact_mod_cast hcard
        exact mul_le_mul_of_nonneg_right hc' (by norm_num)
    _ = 2 ^ 38 := by norm_num

/-- A partial-sum deviation is at most `2 ^ 45`. -/
theorem dev_abs_le {va vb : Nat → F16} {K t : Nat} (hK : K ≤ 2 ^ 13)
    (hva : ∀ k, k < K → |(va k).toRat| ≤ 65504)
    (hvb : ∀ k, k < K → |(vb k).toRat| ≤ 65504) :
    |dev va vb K t| ≤ 2 ^ 45 := by
  have hcm : ∀ u, chunkMag va vb K u ≤ 2 ^ 38 := fun u => chunkMag_le hva hvb
  have hcs : ∀ u, |chunkSum va vb K u| ≤ 2 ^ 38 := fun u =>
    (abs_chunkSum_le_chunkMag va vb K u).trans (hcm u)
  have hzero : ∀ u, steps K ≤ u → chunkSum va vb K u = 0 := by
    intro u hu
    have hempty : chunk K u = ∅ := by
      rw [chunk, Finset.Ico_eq_empty_iff]
      unfold steps at hu
      omega
    simp [chunkSum, hempty]
  have hpre_min : ∀ m, pre va vb K m = pre va vb K (min m (steps K)) := by
    intro m
    induction m with
    | zero => simp
    | succ m ih =>
      rw [pre_succ, ih]
      by_cases h : m + 1 ≤ steps K
      · have h1 : min (m + 1) (steps K) = m + 1 := min_eq_left h
        have h2 : min m (steps K) = m := min_eq_left (by omega)
        rw [h1, h2, pre_succ]
      · have h1 : min (m + 1) (steps K) = steps K := min_eq_right (by omega)
        have h2 : min m (steps K) = steps K := min_eq_right (by omega)
        rw [h1, h2, hzero m (by omega), add_zero]
  have hkey : ∀ s u, s ≤ u → |pre va vb K u - pre va vb K s| ≤ 2 ^ 45 := by
    intro s u hsu
    have hst' : min s (steps K) ≤ min u (steps K) := min_le_min hsu le_rfl
    have hsub : pre va vb K u - pre va vb K s
        = ∑ k ∈ Finset.Ico (min s (steps K)) (min u (steps K)), chunkSum va vb K k := by
      rw [hpre_min u, hpre_min s]
      unfold pre
      rw [← Finset.sum_Ico_eq_sub (fun k => chunkSum va vb K k) hst']
    rw [hsub]
    have h1 : |∑ k ∈ Finset.Ico (min s (steps K)) (min u (steps K)), chunkSum va vb K k|
        ≤ ∑ k ∈ Finset.Ico (min s (steps K)) (min u (steps K)), |chunkSum va vb K k| :=
      Finset.abs_sum_le_sum_abs _ _
    have h2 : ∑ k ∈ Finset.Ico (min s (steps K)) (min u (steps K)), |chunkSum va vb K k|
        ≤ (Finset.Ico (min s (steps K)) (min u (steps K))).card • (2 ^ 38 : ℚ) :=
      Finset.sum_le_card_nsmul _ _ _ (fun k _ => hcs k)
    have hcard : (Finset.Ico (min s (steps K)) (min u (steps K))).card ≤ 128 := by
      rw [Nat.card_Ico]
      have h1 : min u (steps K) ≤ steps K := min_le_right _ _
      have h2 : steps K ≤ 128 := steps_le_128 hK
      omega
    have h3 : ((Finset.Ico (min s (steps K)) (min u (steps K))).card : ℚ) * 2 ^ 38 ≤
        128 * 2 ^ 38 := by
      have hc' : ((Finset.Ico (min s (steps K)) (min u (steps K))).card : ℚ) ≤ 128 := by
        exact_mod_cast hcard
      exact mul_le_mul_of_nonneg_right hc' (by norm_num)
    calc |∑ k ∈ Finset.Ico (min s (steps K)) (min u (steps K)), chunkSum va vb K k|
        ≤ ∑ k ∈ Finset.Ico (min s (steps K)) (min u (steps K)), |chunkSum va vb K k| := h1
      _ ≤ (Finset.Ico (min s (steps K)) (min u (steps K))).card • (2 ^ 38 : ℚ) := h2
      _ = ((Finset.Ico (min s (steps K)) (min u (steps K))).card : ℚ) * 2 ^ 38 := by
          rw [nsmul_eq_mul]
      _ ≤ 128 * 2 ^ 38 := h3
      _ = 2 ^ 45 := by norm_num
  have hdev0 : 0 ≤ dev va vb K t := dev_nonneg va vb K t
  rw [abs_of_nonneg hdev0]
  unfold dev
  refine Finset.sup'_le _ _ ?_
  intro s hs
  have hst : s ≤ t := by
    have := Finset.mem_range.mp hs
    omega
  exact hkey s t hst

/-- One program's `fineOne` error over at most `128` chunks is at most `2 ^ 38`. -/
theorem fineOne_le {va vb : Nat → F16} {K c0 n : Nat} (hK : K ≤ 2 ^ 13) (hn : n ≤ 128)
    (hva : ∀ k, k < K → |(va k).toRat| ≤ 65504)
    (hvb : ∀ k, k < K → |(vb k).toRat| ≤ 65504) :
    fineOne va vb K c0 n ≤ 2 ^ 38 := by
  have hdev : ∀ t, |dev va vb K t| ≤ 2 ^ 45 := fun t => dev_abs_le hK hva hvb
  have hcm : ∀ t, chunkMag va vb K t ≤ 2 ^ 38 := fun t => chunkMag_le hva hvb
  have hg : ∀ t, rho * (dev va vb K t + chunkMag va vb K t) + alpha ≤ 2 ^ 30 := by
    intro t
    have hd := hdev t
    have hc := hcm t
    have hd0 : (0 : ℚ) ≤ dev va vb K t := dev_nonneg va vb K t
    have hc0 : (0 : ℚ) ≤ chunkMag va vb K t := chunkMag_nonneg va vb K t
    have hdle : dev va vb K t ≤ 2 ^ 45 := (abs_le.mp hd).2
    have hs0 : (0 : ℚ) ≤ dev va vb K t + chunkMag va vb K t := by linarith
    have hmul : rho * (dev va vb K t + chunkMag va vb K t) ≤
        (1 / 60000) * (2 ^ 45 + 2 ^ 38) :=
      mul_le_mul rho_le (by linarith) hs0 (by norm_num)
    have hconst : (1 / 60000) * (2 ^ 45 + 2 ^ 38) + (2 : ℚ) ^ (-119 : ℤ) ≤ 2 ^ 30 := by
      norm_num
    have h4 : rho * (dev va vb K t + chunkMag va vb K t) + alpha ≤
        (1 / 60000) * (2 ^ 45 + 2 ^ 38) + (2 : ℚ) ^ (-119 : ℤ) :=
      add_le_add hmul alpha_le
    exact h4.trans hconst
  have hmain : ∀ m, m ≤ 128 → fineOne va vb K c0 m ≤ 2 ^ 31 * (m : ℚ) := by
    intro m
    induction m with
    | zero => intro _; simp [fineOne]
    | succ m ih =>
      intro hm
      have hm' : m ≤ 128 := by omega
      have ihm := ih hm'
      rw [fineOne_step]
      have hstep := hg (c0 + m)
      have h1 : (1 + rho) * fineOne va vb K c0 m ≤ (1 + rho) * (2 ^ 31 * (m : ℚ)) :=
        mul_le_mul_of_nonneg_left ihm (by linarith [rho_nonneg])
      have hrn : rho * (m : ℚ) ≤ 1 / 2 := by
        have hm128 : (m : ℚ) ≤ 128 := by exact_mod_cast hm'
        have hmul := mul_le_mul rho_le hm128 (by positivity : (0 : ℚ) ≤ (m : ℚ))
          (by norm_num : (0 : ℚ) ≤ 1 / 60000)
        linarith [hmul]
      have h3 : rho * (2 ^ 31 * (m : ℚ)) ≤ (1 / 2) * 2 ^ 31 := by
        have h4 := mul_le_mul_of_nonneg_left hrn (by norm_num : (0 : ℚ) ≤ 2 ^ 31)
        nlinarith [h4]
      have h2 : (1 + rho) * (2 ^ 31 * (m : ℚ)) + 2 ^ 30 ≤ 2 ^ 31 * ((m : ℚ) + 1) := by
        nlinarith [h3]
      push_cast
      linarith
  have hfin := hmain n hn
  have hn128 : (n : ℚ) ≤ 128 := by exact_mod_cast hn
  have hlast : (2 : ℚ) ^ 31 * (n : ℚ) ≤ 2 ^ 38 := by
    have h : (2 : ℚ) ^ 31 * (n : ℚ) ≤ 2 ^ 31 * 128 :=
      mul_le_mul_of_nonneg_left hn128 (by norm_num)
    calc (2 : ℚ) ^ 31 * (n : ℚ) ≤ 2 ^ 31 * 128 := h
      _ = 2 ^ 38 := by norm_num
  linarith

/-- The magnitude sum of one tile's dot is at most `2 ^ 38`. -/
theorem sum_liftMag_le {va vb : Fin 64 → ℚ} (hva : ∀ k, |va k| ≤ 65504)
    (hvb : ∀ k, |vb k| ≤ 65504) :
    (∑ k : Fin 64, Device.liftMag .f16 (va k) * Device.liftMag .f16 (vb k)) ≤ 2 ^ 38 := by
  have hterm : ∀ k : Fin 64, Device.liftMag .f16 (va k) * Device.liftMag .f16 (vb k) ≤
      2 ^ 32 := by
    intro k
    have h1 : Device.liftMag .f16 (va k) ≤ 65504 := liftMag_f16_le (hva k)
    have h2 : Device.liftMag .f16 (vb k) ≤ 65504 := liftMag_f16_le (hvb k)
    have h1n : (0 : ℚ) ≤ Device.liftMag .f16 (va k) := Device.liftMag_nonneg _ _
    have h2n : (0 : ℚ) ≤ Device.liftMag .f16 (vb k) := Device.liftMag_nonneg _ _
    calc Device.liftMag .f16 (va k) * Device.liftMag .f16 (vb k) ≤ 65504 * 65504 :=
          mul_le_mul h1 h2 h2n (by norm_num)
      _ ≤ 2 ^ 32 := by norm_num
  calc (∑ k : Fin 64, Device.liftMag .f16 (va k) * Device.liftMag .f16 (vb k))
      ≤ ∑ _k : Fin 64, (2 ^ 32 : ℚ) := Finset.sum_le_sum (fun k _ => hterm k)
    _ = 64 * 2 ^ 32 := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        norm_num
    _ = 2 ^ 38 := by norm_num

/-- The room hypothesis of `DotLift.dotAdd_step`, at the cap `2 ^ 46` for the accumulator. -/
theorem dot_room {c s : ℚ} (hc : |c| ≤ 2 ^ 46) (hs : s ≤ 2 ^ 38) (hs0 : 0 ≤ s) :
    2 * ((1 + Device.relDot16 64) * (|c| + s) + Device.absDot16 64 +
      2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
  have hrel1 : Device.relDot16 64 ≤ 1 / 2 := by
    unfold Device.relDot16; norm_num
  have habs1 : Device.absDot16 64 ≤ 1 := by
    unfold Device.absDot16; norm_num
  have heta1 : Format.binary32.eta ≤ 1 := by
    rw [Format.binary32_eta]; norm_num
  have hbase0 : (0 : ℚ) ≤ |c| + s := by linarith [abs_nonneg c]
  have hle : |c| + s ≤ 2 ^ 46 + 2 ^ 38 := by linarith
  have hmul : (1 + Device.relDot16 64) * (|c| + s) ≤ (3 / 2) * (2 ^ 46 + 2 ^ 38) :=
    mul_le_mul (by linarith) hle hbase0 (by norm_num)
  have hnum : (3 / 2) * (2 ^ 46 + 2 ^ 38) + 1 + 2 * 1 ≤ 2 ^ 49 := by norm_num
  have hinner : (1 + Device.relDot16 64) * (|c| + s) + Device.absDot16 64 +
      2 * Format.binary32.eta ≤ 2 ^ 49 := by linarith
  have h2 : 2 * ((1 + Device.relDot16 64) * (|c| + s) + Device.absDot16 64 +
      2 * Format.binary32.eta) ≤ 2 ^ 50 := by linarith
  have hmax : (2 : ℚ) ^ 50 ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]
    norm_num
  linarith

/-! ## Reindexing one chunk -/

/-- A chunk of a multiple of `64` is a block of `64` consecutive indices. -/
theorem chunk_eq_range {K v : Nat} (hK : K % 64 = 0) (hv : v < steps K) :
    chunk K v = (Finset.range 64).image (fun k => v * 64 + k) := by
  have hK64 : K = 64 * (K / 64) := by omega
  have hsteps : steps K = K / 64 := by unfold steps; omega
  have hv' : v < K / 64 := by rw [← hsteps]; exact hv
  have hle : 64 * v + 64 ≤ K := by
    rw [hK64]
    have hv1 : v + 1 ≤ K / 64 := hv'
    calc 64 * v + 64 = 64 * (v + 1) := by ring
      _ ≤ 64 * (K / 64) := Nat.mul_le_mul_left 64 hv1
  have hmin : min (64 * v + 64) K = 64 * v + 64 := min_eq_left hle
  have himg : (Finset.range 64).image (fun k => v * 64 + k) =
      Finset.Ico (v * 64) (v * 64 + 64) := by
    rw [Finset.range_eq_Ico]
    simpa using Finset.image_add_left_Ico (a := 0) (b := 64) (c := v * 64)
  unfold chunk
  rw [hmin, himg]
  rw [show (64 : Nat) * v = v * 64 from Nat.mul_comm 64 v]

/-- A `Fin 64` sum over the lanes of chunk `v` is the sum over the chunk. -/
theorem sum_fin_chunk {f : Nat → ℚ} {K v : Nat} (hK : K % 64 = 0) (hv : v < steps K) :
    (∑ k : Fin 64, f (v * 64 + k.val)) = ∑ k ∈ chunk K v, f k := by
  rw [chunk_eq_range hK hv, Fin.sum_univ_eq_sum_range (fun k => f (v * 64 + k)) 64]
  exact (Finset.sum_image (f := f) (g := fun k => v * 64 + k)
    (fun a _ b _ h => Nat.add_left_cancel h)).symm

/-- The kernel's `Fin 64` magnitude sum of chunk `v` is `chunkMag`. -/
theorem sum_fin_chunkMag {va vb : Nat → F16} {K v : Nat} (hK : K % 64 = 0) (hv : v < steps K) :
    (∑ k : Fin 64, Device.liftMag .f16 ((va (v * 64 + k.val)).toRat) *
      Device.liftMag .f16 ((vb (v * 64 + k.val)).toRat)) = chunkMag va vb K v := by
  unfold chunkMag
  rw [chunk_eq_range hK hv]
  simp only [mag16_eq_liftMag]
  rw [Fin.sum_univ_eq_sum_range (fun k => Device.liftMag .f16 ((va (v * 64 + k)).toRat) *
    Device.liftMag .f16 ((vb (v * 64 + k)).toRat)) 64]
  exact (Finset.sum_image (f := fun k => Device.liftMag .f16 ((va k).toRat) *
    Device.liftMag .f16 ((vb k).toRat)) (g := fun k => v * 64 + k)
    (fun a _ b _ h => Nat.add_left_cancel h)).symm

end Claims.StreamKMatmul

