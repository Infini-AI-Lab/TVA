import Claim
import TVA.Proof
import Proof.Basic

/-!
# The numeric bound of one output element

An output element is the evaluation of one tree of IEEE `float32` additions (`AddRel false`):
the block's scan at its lane, joined with the running sum's tree. `tree_err` bounds any such
tree of depth and node count at most `2^23` by `D·2^-23·A + 2^-126` (`Tree.errBound` relaxed
with `one_add_pow_le`); `good_of_tree` adds the scale's product (`FloatV.mul_err`) and gives
`Good`.
-/

namespace Claims.ChunkGlobalCumsum

open TVA

/-- `(1 + u)^m ≤ 1 + 2·m·u` whenever `2·m·u ≤ 1`. -/
theorem one_add_pow_le (u : ℚ) (hu0 : 0 ≤ u) (m : Nat) (hm : 2 * (m : ℚ) * u ≤ 1) :
    (1 + u) ^ m ≤ 1 + 2 * (m : ℚ) * u := by
  induction m with
  | zero => simp
  | succ k ih =>
    have hm' : 2 * ((k : ℚ) + 1) * u ≤ 1 := by push_cast at hm; linarith
    have hk : 2 * (k : ℚ) * u ≤ 1 := by
      have h1 : 2 * (k : ℚ) * u ≤ 2 * ((k : ℚ) + 1) * u := by nlinarith
      linarith
    have ih' := ih hk
    have hku : 2 * (k : ℚ) * u * u ≤ u := by
      have h1 := mul_le_mul_of_nonneg_right hk hu0
      simpa using h1
    have hmul := mul_le_mul_of_nonneg_right ih' (show (0 : ℚ) ≤ 1 + u by linarith)
    have hstep : (1 + 2 * (k : ℚ) * u) * (1 + u) ≤ 1 + 2 * ((k : ℚ) + 1) * u := by
      nlinarith
    rw [pow_succ]
    push_cast
    linarith

/-- The tree bound of `float32` at depth and node count at most `2^23`. -/
theorem errBound_relax (A : ℚ) (hA : 0 ≤ A) (D N : Nat) (hD : D ≤ 2 ^ 23) (hN : N ≤ 2 ^ 23) :
    Tree.errBound Format.binary32.u Format.binary32.eta D N A ≤
      (D : ℚ) * 2 ^ (-23 : ℤ) * A + 2 ^ (-126 : ℤ) := by
  rw [Format.binary32_u, Format.binary32_eta]
  have h24 : (2 : ℚ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have h23 : (2 : ℚ) ^ (-23 : ℤ) = 1 / 2 ^ 23 := by norm_num
  have h126 : (2 : ℚ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  have h150 : (2 : ℚ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  rw [h24, h23, h126, h150]
  unfold Tree.errBound
  have hDQ : (D : ℚ) ≤ 2 ^ 23 := by exact_mod_cast hD
  have hNQ : (N : ℚ) ≤ 2 ^ 23 := by exact_mod_cast hN
  have hu0 : (0 : ℚ) ≤ 1 / 2 ^ 24 := by norm_num
  have hm1 : 2 * (D : ℚ) * (1 / 2 ^ 24) ≤ 1 := by nlinarith
  have hpow := one_add_pow_le (1 / 2 ^ 24) hu0 D hm1
  have hG : ((1 : ℚ) + 1 / 2 ^ 24) ^ D ≤ 2 := by linarith
  have hterm : ((1 + 1 / 2 ^ 24) ^ D - 1) * A ≤ (D : ℚ) * (1 / 2 ^ 23) * A := by
    have hle : (1 + 1 / 2 ^ 24) ^ D - 1 ≤ (D : ℚ) * (1 / 2 ^ 23) := by nlinarith
    exact mul_le_mul_of_nonneg_right hle hA
  have hNG : (N : ℚ) * (1 + 1 / 2 ^ 24) ^ D ≤ 2 ^ 23 * 2 :=
    mul_le_mul hNQ hG (by positivity) (by positivity)
  have heta : 1 / 2 ^ 150 * (N : ℚ) * (1 + 1 / 2 ^ 24) ^ D ≤ 1 / 2 ^ 126 := by
    have h1 : 1 / 2 ^ 150 * (N : ℚ) * (1 + 1 / 2 ^ 24) ^ D
        = 1 / 2 ^ 150 * ((N : ℚ) * (1 + 1 / 2 ^ 24) ^ D) := by ring
    rw [h1]
    calc 1 / 2 ^ 150 * ((N : ℚ) * (1 + 1 / 2 ^ 24) ^ D)
        ≤ 1 / 2 ^ 150 * (2 ^ 23 * 2) := mul_le_mul_of_nonneg_left hNG (by norm_num)
      _ = 1 / 2 ^ 126 := by norm_num
  linarith

/-- **A tree of `float32` additions** of depth and node count at most `2^23` over finite leaves
whose magnitudes sum to at most `2^126`: finite, within `D·2^-23·A + 2^-126` of the exact sum. -/
theorem tree_err {x : Nat → F32} {t : Tree} {r : F32} (h : t.Eval (AddRel false) x r)
    (hx : ∀ q ∈ t.leaves, (x q).isFinite = true)
    (hA : Tree.absSumOf FloatV.toRat x t ≤ 2 ^ 126) (hD : t.depth ≤ 2 ^ 23)
    (hN : t.nodes ≤ 2 ^ 23) :
    r.isFinite = true ∧
      |r.toRat - Tree.sumOf FloatV.toRat x t| ≤
        (t.depth : ℚ) * 2 ^ (-23 : ℤ) * Tree.absSumOf FloatV.toRat x t + 2 ^ (-126 : ℤ) := by
  have hA0 := Tree.absSumOf_nonneg FloatV.toRat x t
  have hrel := errBound_relax _ hA0 t.depth t.nodes hD hN
  have hM : Tree.absSumOf FloatV.toRat x t +
      Tree.errBound Format.binary32.u Format.binary32.eta t.depth t.nodes
        (Tree.absSumOf FloatV.toRat x t) ≤ Format.binary32.maxVal := by
    have hDQ : (t.depth : ℚ) ≤ 2 ^ 23 := by exact_mod_cast hD
    have h23 : (2 : ℚ) ^ (-23 : ℤ) = 1 / 2 ^ 23 := by norm_num
    have h126 : (2 : ℚ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
    rw [h23, h126] at hrel
    have hq : (t.depth : ℚ) * (1 / 2 ^ 23) ≤ 1 := by nlinarith
    have hstep : (t.depth : ℚ) * (1 / 2 ^ 23) * Tree.absSumOf FloatV.toRat x t ≤
        1 * Tree.absSumOf FloatV.toRat x t := mul_le_mul_of_nonneg_right hq hA0
    rw [Format.binary32_maxVal]
    have hnum : (2 : ℚ) ^ 126 + 2 ^ 126 + 1 / 2 ^ 126 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  obtain ⟨hfin, herr⟩ := sum_err Format.binary32_regular x h hx hM
  exact ⟨hfin, herr.trans hrel⟩

/-- **An output element** computed along a tree of depth at most `depth` (and at most `2^23`
nodes), whose leaves give `refSum` and `absSum` and are finite when the element is in range, and
then scaled (or not) as the kernel does, is `Good`. -/
theorem good_of_tree {d : Inst} {n p h : Nat} {x : Nat → F32} {t : Tree} {r v : F32}
    (hE : t.Eval (AddRel false) x r)
    (hx : InRange d n p h → ∀ q ∈ t.leaves, (x q).isFinite = true)
    (hS : Tree.sumOf FloatV.toRat x t = refSum d n p h)
    (hAs : Tree.absSumOf FloatV.toRat x t = absSum d n p h)
    (hD : t.depth ≤ depth d n p) (hD23 : depth d n p ≤ 2 ^ 23) (hN : t.nodes ≤ 2 ^ 23)
    (hv : if d.hasScale then FloatV.NaNEq (FloatV.mul r (sigma32 d)) v else v = r) :
    Good d n p h v := by
  intro hIR
  have hA126 : absSum d n p h ≤ 2 ^ 126 := hIR.2.1
  have hA0 : 0 ≤ absSum d n p h := by
    unfold absSum; exact Finset.sum_nonneg (fun q _ => abs_nonneg _)
  obtain ⟨hrf, hre⟩ := tree_err hE (hx hIR) (by rw [hAs]; exact hA126) (le_trans hD hD23) hN
  rw [hS, hAs] at hre
  set A := absSum d n p h with hAdef
  set S := refSum d n p h with hSdef
  have hSA : |S| ≤ A := by
    rw [← hS, ← hAs]; exact Tree.abs_sumOf_le FloatV.toRat x t
  have hDQ : (t.depth : ℚ) ≤ (depth d n p : ℚ) := by exact_mod_cast hD
  have hD23Q : (depth d n p : ℚ) ≤ 2 ^ 23 := by exact_mod_cast hD23
  have h23 : (2 : ℚ) ^ (-23 : ℤ) = 1 / 2 ^ 23 := by norm_num
  have h126 : (2 : ℚ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  have h125 : (2 : ℚ) ^ (-125 : ℤ) = 1 / 2 ^ 125 := by norm_num
  rw [h23, h126] at hre
  -- the tree's error, relaxed to `depth`
  have hE1 : |r.toRat - S| ≤ (depth d n p : ℚ) * (1 / 2 ^ 23) * A + 1 / 2 ^ 126 := by
    have : (t.depth : ℚ) * (1 / 2 ^ 23) * A ≤ (depth d n p : ℚ) * (1 / 2 ^ 23) * A := by
      apply mul_le_mul_of_nonneg_right _ hA0
      exact mul_le_mul_of_nonneg_right hDQ (by norm_num)
    linarith
  by_cases hs : d.hasScale = true
  · rw [ite_eq_left hs] at hv
    have hσf : (sigma32 d).isFinite = true := (hIR.2.2 hs).2.1
    have hσA : |(sigma32 d).toRat| * A ≤ 2 ^ 126 := (hIR.2.2 hs).2.2
    set σ := (sigma32 d).toRat with hσdef
    have hσ0 : (0 : ℚ) ≤ |σ| := abs_nonneg _
    have hσmax : |σ| ≤ (2 ^ 24 - 1) * 2 ^ 104 := by
      rw [← Format.binary32_maxVal]
      exact FloatV.abs_toRat_le_maxVal Format.binary32_regular hσf
    have hdA : (depth d n p : ℚ) * (1 / 2 ^ 23) * A ≤ A := by
      have hq : (depth d n p : ℚ) * (1 / 2 ^ 23) ≤ 1 := by nlinarith
      nlinarith
    have hrle : |r.toRat| ≤ 2 * A + 1 / 2 ^ 126 := by
      have h := abs_add_le (r.toRat - S) S
      rw [sub_add_cancel] at h
      linarith
    have hmax : |r.toRat * σ| ≤ Format.binary32.maxVal := by
      rw [abs_mul, Format.binary32_maxVal]
      have h1 : |r.toRat| * |σ| ≤ (2 * A + 1 / 2 ^ 126) * |σ| :=
        mul_le_mul_of_nonneg_right hrle hσ0
      have h2 : (1 / 2 ^ 126 : ℚ) * |σ| ≤ 4 := by
        have := mul_le_mul_of_nonneg_left hσmax (by norm_num : (0 : ℚ) ≤ 1 / 2 ^ 126)
        have hnum : (1 / 2 ^ 126 : ℚ) * ((2 ^ 24 - 1) * 2 ^ 104) ≤ 4 := by norm_num
        linarith
      have h3 : A * |σ| ≤ 2 ^ 126 := by rw [mul_comm]; exact hσA
      have hnum : (2 : ℚ) * 2 ^ 126 + 4 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
      nlinarith
    obtain ⟨hwfin, hwerr⟩ :=
      FloatV.mul_err (f := Format.binary32) Format.binary32_regular hrf hσf hmax
    have hweq : v = FloatV.mul r (sigma32 d) := hv.eq (FloatV.isFinite_iff.mp hwfin).1
    rw [Format.binary32_u, Format.binary32_eta] at hwerr
    have h24 : (2 : ℚ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
    have h150 : (2 : ℚ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
    rw [h24, h150] at hwerr
    refine ⟨by rw [hweq]; exact hwfin, ?_⟩
    rw [hweq]
    have hsig32 : sig32 d = σ := by rw [sig32, ite_eq_left hs]
    unfold tol
    rw [hsig32, h23, h125, ← hAdef]
    -- the triangle inequality over the product's rounding, the tree, and the scale
    have htri : |(FloatV.mul r (sigma32 d)).toRat - sig d * S| ≤
        |(FloatV.mul r (sigma32 d)).toRat - r.toRat * σ| + |σ| * |r.toRat - S| +
          |sig d - σ| * A := by
      have h1 := abs_add_le ((FloatV.mul r (sigma32 d)).toRat - r.toRat * σ)
        ((r.toRat * σ - σ * S) + (σ * S - sig d * S))
      have h2 := abs_add_le (r.toRat * σ - σ * S) (σ * S - sig d * S)
      have hc : ((FloatV.mul r (sigma32 d)).toRat - r.toRat * σ) +
          ((r.toRat * σ - σ * S) + (σ * S - sig d * S)) =
          (FloatV.mul r (sigma32 d)).toRat - sig d * S := by ring
      rw [hc] at h1
      have h3 : |r.toRat * σ - σ * S| = |σ| * |r.toRat - S| := by
        rw [show r.toRat * σ - σ * S = σ * (r.toRat - S) by ring, abs_mul]
      have h4 : |σ * S - sig d * S| ≤ |sig d - σ| * A := by
        rw [show σ * S - sig d * S = (σ - sig d) * S by ring, abs_mul, abs_sub_comm σ (sig d)]
        exact mul_le_mul_of_nonneg_left hSA (abs_nonneg _)
      linarith
    have hrσ : |r.toRat * σ| ≤ (A + ((depth d n p : ℚ) * (1 / 2 ^ 23) * A + 1 / 2 ^ 126)) * |σ| := by
      rw [abs_mul]
      apply mul_le_mul_of_nonneg_right _ hσ0
      have h := abs_add_le (r.toRat - S) S
      rw [sub_add_cancel] at h
      linarith
    have hb1 : |(FloatV.mul r (sigma32 d)).toRat - r.toRat * σ| ≤
        (A + ((depth d n p : ℚ) * (1 / 2 ^ 23) * A + 1 / 2 ^ 126)) * |σ| * (1 / 2 ^ 24) +
          1 / 2 ^ 150 := by
      have := mul_le_mul_of_nonneg_right hrσ (by norm_num : (0 : ℚ) ≤ 1 / 2 ^ 24)
      linarith
    have hb2 : |σ| * |r.toRat - S| ≤
        |σ| * ((depth d n p : ℚ) * (1 / 2 ^ 23) * A + 1 / 2 ^ 126) :=
      mul_le_mul_of_nonneg_left hE1 hσ0
    -- the relative part: `u·(1 + D·2^-23) + D·2^-23 ≤ (D + 1)·2^-23` for `D ≤ 2^23`
    have hrelA : (A + (depth d n p : ℚ) * (1 / 2 ^ 23) * A) * |σ| * (1 / 2 ^ 24) +
        |σ| * ((depth d n p : ℚ) * (1 / 2 ^ 23) * A) ≤
        ((depth d n p : ℚ) + 1) * (1 / 2 ^ 23) * (|σ| * A) := by
      have hσA0 : 0 ≤ |σ| * A := mul_nonneg hσ0 hA0
      have hq : (1 + (depth d n p : ℚ) * (1 / 2 ^ 23)) * (1 / 2 ^ 24) ≤ 1 / 2 ^ 23 := by
        nlinarith
      have e1 : (A + (depth d n p : ℚ) * (1 / 2 ^ 23) * A) * |σ| * (1 / 2 ^ 24) +
          |σ| * ((depth d n p : ℚ) * (1 / 2 ^ 23) * A) =
          ((1 + (depth d n p : ℚ) * (1 / 2 ^ 23)) * (1 / 2 ^ 24)) * (|σ| * A) +
            (depth d n p : ℚ) * (1 / 2 ^ 23) * (|σ| * A) := by ring
      have e2 : ((depth d n p : ℚ) + 1) * (1 / 2 ^ 23) * (|σ| * A) =
          (1 / 2 ^ 23) * (|σ| * A) + (depth d n p : ℚ) * (1 / 2 ^ 23) * (|σ| * A) := by ring
      rw [e1, e2]
      have := mul_le_mul_of_nonneg_right hq hσA0
      linarith
    -- the absolute part
    have hrelB : (1 / 2 ^ 126 : ℚ) * |σ| * (1 / 2 ^ 24) + 1 / 2 ^ 150 + |σ| * (1 / 2 ^ 126) ≤
        1 / 2 ^ 125 * (1 + |σ|) := by nlinarith
    have hcast : ((depth d n p : ℕ) : ℚ) + 1 = (depth d n p : ℚ) + 1 := rfl
    nlinarith [htri, hb1, hb2, hrelA, hrelB]
  · rw [ite_eq_right hs] at hv
    subst hv
    have hsig : sig d = 1 := by rw [sig, ite_eq_right hs]
    have hsig32 : sig32 d = 1 := by rw [sig32, ite_eq_right hs]
    refine ⟨hrf, ?_⟩
    unfold tol
    rw [hsig, hsig32, one_mul, ← hAdef, h23, h125]
    have : (depth d n p : ℚ) * (1 / 2 ^ 23) * A ≤
        ((depth d n p : ℚ) + 1) * (1 / 2 ^ 23) * (|(1 : ℚ)| * A) := by
      rw [abs_one, one_mul]
      have := mul_nonneg (by norm_num : (0 : ℚ) ≤ 1 / 2 ^ 23) hA0
      nlinarith
    have h0 : |(1 : ℚ) - 1| * A = 0 := by simp
    have hpos : (1 / 2 ^ 126 : ℚ) ≤ 1 / 2 ^ 125 * (1 + |(1 : ℚ)|) := by norm_num
    linarith

end Claims.ChunkGlobalCumsum
