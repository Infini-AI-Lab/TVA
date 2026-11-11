import TVA.Proof

/-!
# A sharp error bound for a contractible addition

`FloatV.AddRel.err` bounds a contractible `x + y` by `3u(|x| + |y|) + 4η`, the worst case over
all of `AddRel`, because `AddRel.wide` forgets which operand the compiler left exact. For a
`tl.subc a b` with `b` a broadcast scalar (our `capped - max`, and `top2 - top1` in the
weight) the operand `b` is itself an exact value, so only `a` may be unrounded.

Writing `M = max |x| |y|` and `S = |x| + |y|`, a fused addition then costs

  `u (S + M) + 2u² M + 4η`

rather than `3u S + 4η`: only one operand's rounding (`u M`) and the fusion of the other
(`u S + u² M`) appear. For `float32` with `S ≤ 90.0001` this is `5.365e-6 < 3u S = 1.61e-5`.

The proof splits `AddRel` itself (not `AddRel.wide`). The statement lives at the root
namespace: inside `namespace TVA.FloatV` the names `max` and `abs` would refer to the tensor
ops there rather than to the order-theoretic ones on `ℚ`.
-/

open TVA

open TVA

/-- A fusable addition (`AddRel true`) of finite operands has the error of one fused rounding: `3u·|x+y| + 4η`. -/
theorem addrel_err_fusion {f : Format} (hf : f.Regular) (hs : f.specials ≠ .finite)
    (hu : f.u ≤ 1 / 16) {fusion : Bool} {x y r : FloatV f} (h : AddRel fusion x y r)
    (_hfus : fusion = true) (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hM : |x.toRat| + |y.toRat| ≤ f.maxVal / 2 - 2 * f.eta) :
    r.isFinite = true ∧
      |r.toRat - (x.toRat + y.toRat)| ≤
        f.u * (|x.toRat| + |y.toRat| + max (|x.toRat|) (|y.toRat|))
          + 2 * f.u * f.u * max (|x.toRat|) (|y.toRat|) + 4 * f.eta := by
  have hu0 : (0 : ℚ) ≤ f.u := Format.u_nonneg (f := f)
  have he0 : (0 : ℚ) ≤ f.eta := Format.eta_nonneg (f := f)
  have hmax0 : (0 : ℚ) < f.maxVal := Format.maxVal_pos hf
  have hxy : |x.toRat + y.toRat| ≤ |x.toRat| + |y.toRat| := abs_add_le _ _
  have hxu : 0 ≤ |x.toRat| := abs_nonneg _
  have hyu : 0 ≤ |y.toRat| := abs_nonneg _
  have hS0 : 0 ≤ |x.toRat| + |y.toRat| := by linarith
  have hMu : 0 ≤ max (|x.toRat|) (|y.toRat|) := le_trans hxu (le_max_left _ _)
  have hxM : |x.toRat| ≤ max (|x.toRat|) (|y.toRat|) := le_max_left _ _
  have hyM : |y.toRat| ≤ max (|x.toRat|) (|y.toRat|) := le_max_right _ _
  have hu1 : f.u ≤ 1 := by linarith
  have hmaxS : max (|x.toRat|) (|y.toRat|) ≤ |x.toRat| + |y.toRat| :=
    max_le (by linarith) (by linarith)
  have hum : f.u * max (|x.toRat|) (|y.toRat|) ≤ f.maxVal / 2 := by
    have h1 : f.u * max (|x.toRat|) (|y.toRat|) ≤ 1 * max (|x.toRat|) (|y.toRat|) :=
      mul_le_mul_of_nonneg_right hu1 hMu
    linarith
  rcases h with hne | ⟨-, hfu⟩
  · -- the IEEE sum
    obtain ⟨hfin, herr⟩ := FloatV.add_err hf hx hy (by linarith)
    rw [hne.eq (FloatV.isFinite_iff.mp hfin).1]
    refine ⟨hfin, herr.trans ?_⟩
    have h1 : |x.toRat + y.toRat| * f.u ≤ (|x.toRat| + |y.toRat|) * f.u :=
      mul_le_mul_of_nonneg_right hxy hu0
    have h2 : 0 ≤ f.u * f.u * max (|x.toRat|) (|y.toRat|) :=
      mul_nonneg (mul_nonneg hu0 hu0) hMu
    have h3 : (0 : ℚ) ≤ f.eta := he0
    have h4 : 0 ≤ |x.toRat| + |y.toRat| + max (|x.toRat|) (|y.toRat|) := by linarith
    have h5 : 0 ≤ f.u * (|x.toRat| + |y.toRat| + max (|x.toRat|) (|y.toRat|)) :=
      mul_nonneg hu0 h4
    nlinarith
  · rcases hfu with ⟨p, q, hp, hq, hpq, z, hne⟩ | ⟨p, hp, hyi, -⟩ | ⟨q, hq, hxi, -⟩
    · -- one operand exact, the other an exact value that rounds to it
      rcases hpq with hpx | hqy
      · subst hpx
        have hqle := RoundsTo.abs_le hf hs hq hy
        have hxqM : |x.toRat + q| ≤ f.maxVal := by
          have := abs_add_le x.toRat q
          nlinarith
        obtain ⟨hfin, herr⟩ := FloatV.ofPat_roundRat_err hf false z hxqM
        rw [hne.eq (FloatV.isFinite_iff.mp hfin).1]
        refine ⟨hfin, ?_⟩
        have hqe := RoundsTo.err hf hs hq hy
        have hxq : |x.toRat + q| ≤ |x.toRat| + |y.toRat| + f.u * |y.toRat| + f.eta := by
          have := abs_add_le x.toRat q
          nlinarith
        have e : (FloatV.ofPat (f.roundRat .rn false z (x.toRat + q)) : FloatV f).toRat
            - (x.toRat + y.toRat) =
            ((FloatV.ofPat (f.roundRat .rn false z (x.toRat + q)) : FloatV f).toRat
                - (x.toRat + q))
              + (q - y.toRat) := by ring
        rw [e]
        have t := abs_add_le
          ((FloatV.ofPat (f.roundRat .rn false z (x.toRat + q)) : FloatV f).toRat
            - (x.toRat + q))
          (q - y.toRat)
        have hb : |x.toRat + q| * f.u ≤
            (|x.toRat| + |y.toRat| + f.u * |y.toRat| + f.eta) * f.u :=
          mul_le_mul_of_nonneg_right hxq hu0
        have hy1 : f.u * |y.toRat| ≤ f.u * max (|x.toRat|) (|y.toRat|) :=
          mul_le_mul_of_nonneg_left hyM hu0
        have hy2 : f.u * f.u * |y.toRat| ≤ f.u * f.u * max (|x.toRat|) (|y.toRat|) :=
          mul_le_mul_of_nonneg_left hyM (mul_nonneg hu0 hu0)
        nlinarith [mul_nonneg hu0 hS0, mul_nonneg (mul_nonneg hu0 hu0) hMu]
      · subst hqy
        have hple := RoundsTo.abs_le hf hs hp hx
        have hpyM : |p + y.toRat| ≤ f.maxVal := by
          have := abs_add_le p y.toRat
          nlinarith
        obtain ⟨hfin, herr⟩ := FloatV.ofPat_roundRat_err hf false z hpyM
        rw [hne.eq (FloatV.isFinite_iff.mp hfin).1]
        refine ⟨hfin, ?_⟩
        have hpxe := RoundsTo.err hf hs hp hx
        have hpy : |p + y.toRat| ≤ |x.toRat| + |y.toRat| + f.u * |x.toRat| + f.eta := by
          have := abs_add_le p y.toRat
          nlinarith
        have e : (FloatV.ofPat (f.roundRat .rn false z (p + y.toRat)) : FloatV f).toRat
            - (x.toRat + y.toRat) =
            ((FloatV.ofPat (f.roundRat .rn false z (p + y.toRat)) : FloatV f).toRat
                - (p + y.toRat))
              + (p - x.toRat) := by ring
        rw [e]
        have t := abs_add_le
          ((FloatV.ofPat (f.roundRat .rn false z (p + y.toRat)) : FloatV f).toRat
            - (p + y.toRat))
          (p - x.toRat)
        have hb : |p + y.toRat| * f.u ≤
            (|x.toRat| + |y.toRat| + f.u * |x.toRat| + f.eta) * f.u :=
          mul_le_mul_of_nonneg_right hpy hu0
        have hx1 : f.u * |x.toRat| ≤ f.u * max (|x.toRat|) (|y.toRat|) :=
          mul_le_mul_of_nonneg_left hxM hu0
        have hx2 : f.u * f.u * |x.toRat| ≤ f.u * f.u * max (|x.toRat|) (|y.toRat|) :=
          mul_le_mul_of_nonneg_left hxM (mul_nonneg hu0 hu0)
        nlinarith [mul_nonneg hu0 hS0, mul_nonneg (mul_nonneg hu0 hu0) hMu]
    · exfalso; rw [(FloatV.isFinite_iff.mp hy).2] at hyi; cases hyi
    · exfalso; rw [(FloatV.isFinite_iff.mp hx).2] at hxi; cases hxi
