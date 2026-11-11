import Claim
import TVA.Proof
import Proof.Bounds

/-!
# The numeric bound of one `float32` scan position

`Tree.sum_err_list` bounds a tree's error by `Tree.errBound`; the relaxation below turns it into the
claim's tolerance. `scan_scale_bound` adds the one rounding of the scale multiplication
(`FloatV.mul_err`), which is a `NaNEq` result (`tl.mul` is `upToNaN`).
-/

namespace Claims.ChunkCumsum

open TVA

/-- Every tree has a leaf. -/
theorem Tree.leaves_length_pos (t : Tree) : 0 < t.leaves.length := by
  induction t with
  | leaf i => simp [Tree.leaves]
  | node l r ihl ihr => simp only [Tree.leaves, List.length_append]; omega

/-- The number of leaves is at most `2^21` under the claim's depth bound. -/
private lemma scan_len_le (L : List Nat) (hm : L.length - 1 ≤ 2 ^ 20) :
    (L.length : ℚ) ≤ 2 ^ 21 := by
  have h1 : L.length ≤ 2 ^ 20 + 1 := by omega
  have h2 : (L.length : ℚ) ≤ ((2 ^ 20 + 1 : Nat) : ℚ) := by exact_mod_cast h1
  have h3 : ((2 ^ 20 + 1 : Nat) : ℚ) = (2 : ℚ) ^ 20 + 1 := by push_cast; ring
  have h4 : (2 : ℚ) ^ 20 + 1 ≤ 2 ^ 21 := by norm_num
  linarith

/-- The `float32` error bound at most `2^20` steps is within the claim's tolerance. -/
private lemma scan_errBound_relax (A : ℚ) (hA : 0 ≤ A) (m : Nat) (hm : m ≤ 2 ^ 20) :
    Tree.errBound Format.binary32.u Format.binary32.eta m m A ≤
      ((m : ℚ) + 1) * 2 ^ (-23 : ℤ) * A + 2 ^ (-128 : ℤ) := by
  rw [Format.binary32_u, Format.binary32_eta]
  have h24 : (2 : ℚ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have h23 : (2 : ℚ) ^ (-23 : ℤ) = 1 / 2 ^ 23 := by norm_num
  have h128 : (2 : ℚ) ^ (-128 : ℤ) = 1 / 2 ^ 128 := by norm_num
  have h150 : (2 : ℚ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  rw [h24, h23, h128, h150] at *
  unfold Tree.errBound
  have hmQ : (m : ℚ) ≤ 2 ^ 20 := by exact_mod_cast hm
  have hu0 : (0 : ℚ) ≤ 1 / 2 ^ 24 := by norm_num
  have hm1 : 2 * (m : ℚ) * (1 / 2 ^ 24) ≤ 1 := by nlinarith
  have hpow := one_add_pow_le (1 / 2 ^ 24) hu0 m hm1
  have hGtop : ((1 : ℚ) + 1 / 2 ^ 24) ^ m ≤ 9 / 8 := by
    have h : 2 * (m : ℚ) * (1 / 2 ^ 24) ≤ 1 / 8 := by nlinarith
    linarith
  have hterm : ((1 + 1 / 2 ^ 24) ^ m - 1) * A ≤ (m : ℚ) * (1 / 2 ^ 23) * A := by
    have hle : (1 + 1 / 2 ^ 24) ^ m - 1 ≤ (m : ℚ) * (1 / 2 ^ 23) := by nlinarith
    exact mul_le_mul_of_nonneg_right hle hA
  have hmG : (m : ℚ) * (1 + 1 / 2 ^ 24) ^ m ≤ 2 ^ 20 * (9 / 8) :=
    mul_le_mul hmQ hGtop (by positivity) (by positivity)
  have heta : 1 / 2 ^ 150 * ((m : ℚ) * (1 + 1 / 2 ^ 24) ^ m) ≤ 1 / 2 ^ 128 := by
    calc 1 / 2 ^ 150 * ((m : ℚ) * (1 + 1 / 2 ^ 24) ^ m)
        ≤ 1 / 2 ^ 150 * (2 ^ 20 * (9 / 8)) := by
          apply mul_le_mul_of_nonneg_left hmG (by norm_num)
      _ ≤ 1 / 2 ^ 128 := by norm_num
  nlinarith

/-- The hypothesis of `TVA.sum_err_list` holds under the claim's assumptions. -/
private lemma scan_maxVal (A : ℚ) (hA0 : 0 ≤ A) (hA1 : A ≤ 2 ^ 126) (m : Nat)
    (hm : m ≤ 2 ^ 20) :
    A + Tree.errBound Format.binary32.u Format.binary32.eta m m A ≤ Format.binary32.maxVal := by
  have hrelax := scan_errBound_relax A hA0 m hm
  have h23 : (2 : ℚ) ^ (-23 : ℤ) = 1 / 2 ^ 23 := by norm_num
  have h128 : (2 : ℚ) ^ (-128 : ℤ) = 1 / 2 ^ 128 := by norm_num
  rw [h23, h128] at hrelax
  have hmQ : (m : ℚ) ≤ 2 ^ 20 := by exact_mod_cast hm
  have hmq : (m : ℚ) * (1 / 2 ^ 23) ≤ 1 / 4 := by nlinarith
  have hstep : (m : ℚ) * (1 / 2 ^ 23) * A ≤ (1 / 4) * A :=
    mul_le_mul_of_nonneg_right hmq hA0
  rw [Format.binary32_maxVal]
  have hnum : (2 : ℚ) ^ 126 + (1 / 4) * 2 ^ 126 + 1 / 2 ^ 128 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by
    norm_num
  nlinarith

/-- A scan position of a `float32` tree: finite, and within the tree bound of the exact sum of
its leaves, relaxed as the claim's tolerance. -/
theorem scan_bound {t : Tree} {L : List Nat} {x : Nat → F32} {r : F32} {S A : ℚ}
    (hL : t.leaves.Perm L) (h : t.Eval (AddRel false) x r)
    (hx : ∀ i ∈ L, (x i).isFinite = true)
    (hS : (L.map fun i => (x i).toRat).sum = S)
    (hA : (L.map fun i => |(x i).toRat|).sum = A)
    (hA0 : 0 ≤ A) (hA1 : A ≤ 2 ^ 126) (hm : L.length - 1 ≤ 2 ^ 20) :
    r.isFinite = true ∧
      |r.toRat - S| ≤ (L.length : ℚ) * 2 ^ (-23 : ℤ) * A + 2 ^ (-128 : ℤ) := by
  have hM : (L.map fun i => |(x i).toRat|).sum +
      Tree.errBound Format.binary32.u Format.binary32.eta (L.length - 1) (L.length - 1)
        (L.map fun i => |(x i).toRat|).sum ≤ Format.binary32.maxVal := by
    rw [hA]
    exact scan_maxVal A hA0 hA1 (L.length - 1) hm
  obtain ⟨hfin, herr⟩ :=
    sum_err_list (f := Format.binary32) Format.binary32_regular x hL h hx hM
  rw [hS, hA] at herr
  have hlen : 0 < L.length := by
    rw [← hL.length_eq]
    exact Tree.leaves_length_pos t
  have hrelax := scan_errBound_relax A hA0 (L.length - 1) hm
  have hcast : ((L.length - 1 : Nat) : ℚ) + 1 = (L.length : ℚ) := by
    have h1 : L.length - 1 + 1 = L.length := Nat.sub_add_cancel hlen
    have h2 := congrArg (fun n : Nat => (n : ℚ)) h1
    push_cast at h2 ⊢
    exact h2
  rw [hcast] at hrelax
  exact ⟨hfin, herr.trans hrelax⟩

/-- With one more rounding (the scale multiplication) the tolerance grows by `|σ|`. -/
private lemma scan_scale_relax (A : ℚ) (hA : 0 ≤ A) (m : Nat) (hm : m ≤ 2 ^ 20) :
    2 ^ (-24 : ℤ) * A +
        (1 + 2 ^ (-24 : ℤ)) * Tree.errBound (2 ^ (-24 : ℤ)) (2 ^ (-150 : ℤ)) m m A +
        2 ^ (-150 : ℤ) ≤ ((m : ℚ) + 1) * 2 ^ (-23 : ℤ) * A + 2 ^ (-128 : ℤ) := by
  have h24 : (2 : ℚ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have h23 : (2 : ℚ) ^ (-23 : ℤ) = 1 / 2 ^ 23 := by norm_num
  have h128 : (2 : ℚ) ^ (-128 : ℤ) = 1 / 2 ^ 128 := by norm_num
  have h150 : (2 : ℚ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  rw [h24, h23, h128, h150] at *
  unfold Tree.errBound
  have hmQ : (m : ℚ) ≤ 2 ^ 20 := by exact_mod_cast hm
  have hu0 : (0 : ℚ) ≤ 1 / 2 ^ 24 := by norm_num
  have hm1 : 2 * ((m : ℚ) + 1) * (1 / 2 ^ 24) ≤ 1 := by nlinarith
  have hm1' : 2 * ((m + 1 : Nat) : ℚ) * (1 / 2 ^ 24) ≤ 1 := by push_cast; linarith
  have hpow := one_add_pow_le (1 / 2 ^ 24) hu0 (m + 1) hm1'
  have hGtop : ((1 : ℚ) + 1 / 2 ^ 24) ^ (m + 1) ≤ 5 / 4 := by
    have h : 2 * ((m + 1 : Nat) : ℚ) * (1 / 2 ^ 24) ≤ 1 / 4 := by
      push_cast
      nlinarith
    linarith
  have e : 1 / 2 ^ 24 * A +
      (1 + 1 / 2 ^ 24) * (((1 + 1 / 2 ^ 24) ^ m - 1) * A +
        1 / 2 ^ 150 * (m : ℚ) * (1 + 1 / 2 ^ 24) ^ m) + 1 / 2 ^ 150 =
      ((1 + 1 / 2 ^ 24) ^ (m + 1) - 1) * A +
        1 / 2 ^ 150 * ((m : ℚ) * (1 + 1 / 2 ^ 24) ^ (m + 1) + 1) := by
    rw [pow_succ]
    ring
  rw [e]
  have hterm : ((1 + 1 / 2 ^ 24) ^ (m + 1) - 1) * A ≤
      ((m : ℚ) + 1) * (1 / 2 ^ 23) * A := by
    have hle : (1 + 1 / 2 ^ 24) ^ (m + 1) - 1 ≤ ((m : ℚ) + 1) * (1 / 2 ^ 23) := by
      have hc : ((m + 1 : Nat) : ℚ) = (m : ℚ) + 1 := by push_cast; ring
      nlinarith
    exact mul_le_mul_of_nonneg_right hle hA
  have hmul : (m : ℚ) * (1 + 1 / 2 ^ 24) ^ (m + 1) ≤ 2 ^ 20 * (5 / 4) :=
    mul_le_mul hmQ hGtop (by positivity) (by positivity)
  have heta : 1 / 2 ^ 150 * ((m : ℚ) * (1 + 1 / 2 ^ 24) ^ (m + 1) + 1) ≤ 1 / 2 ^ 128 := by
    have hstep : 1 / 2 ^ 150 * ((m : ℚ) * (1 + 1 / 2 ^ 24) ^ (m + 1) + 1) ≤
        1 / 2 ^ 150 * (2 ^ 20 * (5 / 4) + 1) := by
      apply mul_le_mul_of_nonneg_left _ (by norm_num)
      linarith
    have hnum : (1 / 2 ^ 150 : ℚ) * (2 ^ 20 * (5 / 4) + 1) ≤ 1 / 2 ^ 128 := by norm_num
    linarith
  nlinarith

/-- The absolute sum of a list of rationals is at most the sum of absolute values. -/
private lemma list_abs_sum_le (lst : List ℚ) : |lst.sum| ≤ (lst.map fun q => |q|).sum := by
  induction lst with
  | nil => simp
  | cons a l ih =>
    simp only [List.sum_cons, List.map_cons]
    have h1 : |a + l.sum| ≤ |a| + |l.sum| := abs_add_le a l.sum
    linarith

/-- The scan followed by the scale multiplication (up to NaN payload): finite, and within the
claim's tolerance for `sig * S`. -/
theorem scan_scale_bound {t : Tree} {L : List Nat} {x : Nat → F32} {r v σ : F32} {S A sig : ℚ}
    (hL : t.leaves.Perm L) (h : t.Eval (AddRel false) x r)
    (hx : ∀ i ∈ L, (x i).isFinite = true)
    (hS : (L.map fun i => (x i).toRat).sum = S)
    (hA : (L.map fun i => |(x i).toRat|).sum = A)
    (hA0 : 0 ≤ A) (hA1 : A ≤ 2 ^ 126) (hm : L.length - 1 ≤ 2 ^ 20)
    (hσf : σ.isFinite = true) (hσA : |σ.toRat| * A ≤ 2 ^ 126)
    (hv : FloatV.NaNEq (FloatV.mul r σ) v) :
    v.isFinite = true ∧
      |v.toRat - sig * S| ≤
        (L.length : ℚ) * 2 ^ (-23 : ℤ) * |σ.toRat| * A + |sig - σ.toRat| * A +
          2 ^ (-128 : ℤ) * (1 + |σ.toRat|) := by
  have hM : (L.map fun i => |(x i).toRat|).sum +
      Tree.errBound Format.binary32.u Format.binary32.eta (L.length - 1) (L.length - 1)
        (L.map fun i => |(x i).toRat|).sum ≤ Format.binary32.maxVal := by
    rw [hA]
    exact scan_maxVal A hA0 hA1 (L.length - 1) hm
  obtain ⟨hfinr, herr⟩ :=
    sum_err_list (f := Format.binary32) Format.binary32_regular x hL h hx hM
  rw [hS, hA] at herr
  rw [Format.binary32_u, Format.binary32_eta] at herr
  set E : ℚ := Tree.errBound (2 ^ (-24 : ℤ)) (2 ^ (-150 : ℤ)) (L.length - 1) (L.length - 1) A
    with hEdef
  have herrE : |r.toRat - S| ≤ E := by rw [hEdef]; exact herr
  have hlen : 0 < L.length := by
    rw [← hL.length_eq]
    exact Tree.leaves_length_pos t
  have hcast : ((L.length - 1 : Nat) : ℚ) + 1 = (L.length : ℚ) := by
    have h1 : L.length - 1 + 1 = L.length := Nat.sub_add_cancel hlen
    have h2 := congrArg (fun n : Nat => (n : ℚ)) h1
    push_cast at h2 ⊢
    exact h2
  have hrelax := scan_errBound_relax A hA0 (L.length - 1) hm
  rw [hcast] at hrelax
  rw [Format.binary32_u, Format.binary32_eta] at hrelax
  have hErelax : E ≤ (L.length : ℚ) * 2 ^ (-23 : ℤ) * A + 2 ^ (-128 : ℤ) := by
    rw [hEdef]; exact hrelax
  have hlenQ : (L.length : ℚ) ≤ 2 ^ 21 := scan_len_le L hm
  have hE4 : E ≤ (1 / 4) * A + 1 / 2 ^ 128 := by
    have h23 : (2 : ℚ) ^ (-23 : ℤ) = 1 / 2 ^ 23 := by norm_num
    have h128 : (2 : ℚ) ^ (-128 : ℤ) = 1 / 2 ^ 128 := by norm_num
    have h := hErelax
    rw [h23, h128] at h
    have hq : (L.length : ℚ) * (1 / 2 ^ 23) ≤ 1 / 4 := by nlinarith
    have hstep : (L.length : ℚ) * (1 / 2 ^ 23) * A ≤ (1 / 4) * A :=
      mul_le_mul_of_nonneg_right hq hA0
    linarith
  have hSle : |S| ≤ A := by
    have h := list_abs_sum_le (L.map fun i => (x i).toRat)
    rw [hS] at h
    have h2 : (List.map (fun q : ℚ => |q|) (List.map (fun i => (x i).toRat) L)).sum =
        (List.map (fun i => |(x i).toRat|) L).sum := by
      rw [List.map_map]
      rfl
    rw [h2, hA] at h
    exact h
  have hrle : |r.toRat| ≤ A + E := by
    have h := abs_add_le (r.toRat - S) S
    have h' : |r.toRat| ≤ |r.toRat - S| + |S| := by
      rwa [sub_add_cancel] at h
    linarith
  have hσabs : |σ.toRat| ≤ Format.binary32.maxVal :=
    FloatV.abs_toRat_le_maxVal Format.binary32_regular hσf
  have hmax : |r.toRat * σ.toRat| ≤ Format.binary32.maxVal := by
    rw [abs_mul, Format.binary32_maxVal]
    have hσle : |σ.toRat| ≤ (2 ^ 24 - 1) * 2 ^ 104 := by
      rw [← Format.binary32_maxVal]; exact hσabs
    have hAσ : A * |σ.toRat| ≤ 2 ^ 126 := by
      have h := hσA
      rw [mul_comm] at h
      exact h
    have h1 : |r.toRat| * |σ.toRat| ≤ (A + E) * |σ.toRat| :=
      mul_le_mul_of_nonneg_right hrle (abs_nonneg _)
    have hEσ : E * |σ.toRat| ≤ ((1 / 4) * A + 1 / 2 ^ 128) * |σ.toRat| :=
      mul_le_mul_of_nonneg_right hE4 (abs_nonneg _)
    have h4 : (1 / 2 ^ 128 : ℚ) * |σ.toRat| ≤ 1 := by
      have hle := mul_le_mul_of_nonneg_left hσle (by norm_num : (0 : ℚ) ≤ 1 / 2 ^ 128)
      have hnum : (1 / 2 ^ 128 : ℚ) * ((2 ^ 24 - 1) * 2 ^ 104) ≤ 1 := by norm_num
      linarith
    have h3 : (1 / 4) * (A * |σ.toRat|) ≤ (1 / 4) * 2 ^ 126 := by linarith
    have hnum : (2 : ℚ) ^ 126 + (1 / 4) * 2 ^ 126 + 1 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by
      norm_num
    nlinarith
  obtain ⟨hwfin, hwerr⟩ :=
    FloatV.mul_err (f := Format.binary32) Format.binary32_regular hfinr hσf hmax
  have hweq : v = FloatV.mul r σ := hv.eq (FloatV.isFinite_iff.mp hwfin).1
  refine ⟨?_, ?_⟩
  · rw [hweq]; exact hwfin
  · rw [hweq]
    have hu : Format.binary32.u = 1 / 2 ^ 24 := by rw [Format.binary32_u]; norm_num
    have he : Format.binary32.eta = 1 / 2 ^ 150 := by rw [Format.binary32_eta]; norm_num
    have hwerr' : |(FloatV.mul r σ).toRat - r.toRat * σ.toRat| ≤
        |r.toRat * σ.toRat| * (1 / 2 ^ 24) + 1 / 2 ^ 150 := by
      rw [hu, he] at hwerr
      exact hwerr
    have htri : |(FloatV.mul r σ).toRat - sig * S| ≤
        |(FloatV.mul r σ).toRat - r.toRat * σ.toRat| + |r.toRat * σ.toRat - σ.toRat * S| +
          |σ.toRat * S - sig * S| := by
      have h1 := abs_add_le ((FloatV.mul r σ).toRat - r.toRat * σ.toRat)
        ((r.toRat * σ.toRat - σ.toRat * S) + (σ.toRat * S - sig * S))
      have h2 := abs_add_le (r.toRat * σ.toRat - σ.toRat * S) (σ.toRat * S - sig * S)
      have hc : ((FloatV.mul r σ).toRat - r.toRat * σ.toRat) +
          ((r.toRat * σ.toRat - σ.toRat * S) + (σ.toRat * S - sig * S)) =
          (FloatV.mul r σ).toRat - sig * S := by ring
      rw [hc] at h1
      linarith
    have ht2 : |r.toRat * σ.toRat - σ.toRat * S| ≤ |σ.toRat| * E := by
      have hc : r.toRat * σ.toRat - σ.toRat * S = σ.toRat * (r.toRat - S) := by ring
      rw [hc, abs_mul]
      exact mul_le_mul_of_nonneg_left herrE (abs_nonneg _)
    have ht3 : |σ.toRat * S - sig * S| ≤ |sig - σ.toRat| * A := by
      have hc : σ.toRat * S - sig * S = (σ.toRat - sig) * S := by ring
      rw [hc, abs_mul, abs_sub_comm σ.toRat sig]
      exact mul_le_mul_of_nonneg_left hSle (abs_nonneg _)
    have hrσ : |r.toRat * σ.toRat| ≤ (A + E) * |σ.toRat| := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_right hrle (abs_nonneg _)
    have hb1 : |(FloatV.mul r σ).toRat - r.toRat * σ.toRat| ≤
        (A + E) * |σ.toRat| * (1 / 2 ^ 24) + 1 / 2 ^ 150 := by
      have h := hwerr'
      have := mul_le_mul_of_nonneg_right hrσ (by norm_num : (0 : ℚ) ≤ 1 / 2 ^ 24)
      linarith
    have hscale := scan_scale_relax A hA0 (L.length - 1) hm
    rw [hcast] at hscale
    have hscale' : 2 ^ (-24 : ℤ) * A + (1 + 2 ^ (-24 : ℤ)) * E + 2 ^ (-150 : ℤ) ≤
        (L.length : ℚ) * 2 ^ (-23 : ℤ) * A + 2 ^ (-128 : ℤ) := by
      rw [hEdef]; exact hscale
    have h24 : (2 : ℚ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
    have h23 : (2 : ℚ) ^ (-23 : ℤ) = 1 / 2 ^ 23 := by norm_num
    have h128 : (2 : ℚ) ^ (-128 : ℤ) = 1 / 2 ^ 128 := by norm_num
    have h150 : (2 : ℚ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
    rw [h24, h23, h128, h150] at hscale'
    rw [h23, h128]
    have hσ0 : (0 : ℚ) ≤ |σ.toRat| := abs_nonneg _
    have hmul := mul_le_mul_of_nonneg_left hscale' hσ0
    have hfac : (A + E) * |σ.toRat| * (1 / 2 ^ 24) + 1 / 2 ^ 150 + |σ.toRat| * E =
        |σ.toRat| * (1 / 2 ^ 24 * A + (1 + 1 / 2 ^ 24) * E) + 1 / 2 ^ 150 := by ring
    have hcomb : |(FloatV.mul r σ).toRat - r.toRat * σ.toRat| +
        |r.toRat * σ.toRat - σ.toRat * S| + |σ.toRat * S - sig * S| ≤
        |σ.toRat| * (1 / 2 ^ 24 * A + (1 + 1 / 2 ^ 24) * E) + 1 / 2 ^ 150 +
          |sig - σ.toRat| * A := by
      linarith [hb1, ht2, ht3, hfac]
    have heta1 : (1 / 2 ^ 150 : ℚ) ≤ 1 / 2 ^ 128 + |σ.toRat| * (1 / 2 ^ 150) := by
      nlinarith [hσ0]
    have hmul2 : |σ.toRat| * (1 / 2 ^ 24 * A + (1 + 1 / 2 ^ 24) * E) + 1 / 2 ^ 150 ≤
        |σ.toRat| * ((L.length : ℚ) * (1 / 2 ^ 23) * A + 1 / 2 ^ 128) + 1 / 2 ^ 128 := by
      nlinarith [hmul, heta1, mul_nonneg hσ0 (by norm_num : (0 : ℚ) ≤ 1 / 2 ^ 150)]
    calc |(FloatV.mul r σ).toRat - sig * S|
        ≤ |(FloatV.mul r σ).toRat - r.toRat * σ.toRat| + |r.toRat * σ.toRat - σ.toRat * S| +
            |σ.toRat * S - sig * S| := htri
      _ ≤ |σ.toRat| * (1 / 2 ^ 24 * A + (1 + 1 / 2 ^ 24) * E) + 1 / 2 ^ 150 +
            |sig - σ.toRat| * A := hcomb
      _ ≤ |σ.toRat| * ((L.length : ℚ) * (1 / 2 ^ 23) * A + 1 / 2 ^ 128) + 1 / 2 ^ 128 +
            |sig - σ.toRat| * A := by linarith [hmul2]
      _ = (L.length : ℚ) * (1 / 2 ^ 23) * |σ.toRat| * A + |sig - σ.toRat| * A +
            1 / 2 ^ 128 * (1 + |σ.toRat|) := by ring

end Claims.ChunkCumsum
