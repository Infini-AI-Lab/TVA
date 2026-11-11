import TVA.Proof.Float.Tree

/-!
# Errors of fusable additions

With `enable_fp_fusion` (the default) the compiler may contract a product and an addition into
an `fma`, so the product enters the addition unrounded. The model does not track which operand
is a product: `AddRel fusion x y r` (`TVA.Triton.Arith`) allows the IEEE sum, and also the
rounding of `p + q` for exact `p`, `q` that round to the operands (`RoundsTo`). Such a result is
not within `u·|x + y| + η` of `x + y` (with `x = 1`, `y = −1` it may be about `2^-24` in
`float32` while `x + y = 0`), but it is within a bound relative to the operands' magnitudes:

* `RoundsTo.err`: an exact value that rounds to a finite `x` is within `|x|·u + η` of it, also
  above `maxVal`; `RoundsTo.abs_le`: so `|p| ≤ (1 + u)·|x| + η`.
* `FloatV.AddRel.err`: a fusable `x + y` of finite operands with `|x| + |y| ≤ maxVal/2 − 2η` is
  finite and within `3u·(|x| + |y|) + 4η` of `x + y`. A tree of such nodes is bounded by
  `Tree.eval_err_abs` with `δ = 3u` and `4η` (`TVA.Proof.Float.Sum`).
* `FloatV.mulAdd_err`: `tl.mul_add a b c` (two roundings, or one `fma`) is within
  `3u·(|a·b| + |c|) + 4η` of `a·b + c`.

The format must have infinities or a NaN (`specials ≠ .finite`), so that a finite rounding is not
an overflow, and `u ≤ 1/16` (at least 3 fraction bits): `float16`, `bfloat16`, `float32`,
`float64` and `float8e4nv`, not `float8e5` or `e2m1`.
-/
namespace TVA

variable {f : Format}

/-- **An exact value that rounds to a finite `x` is within one rounding's bound of `x`**,
`|x|·u + η`. -/
theorem RoundsTo.err (hf : f.Regular) (hs : f.specials ≠ .finite) {p : ℚ} {x : FloatV f}
    (h : RoundsTo p x) (hx : x.isFinite = true) : |p - x.toRat| ≤ |x.toRat| * f.u + f.eta := by
  rcases h with ⟨hfin, -, heq⟩ | ⟨hinf, heq⟩
  · have := Format.roundRat_rn_err_of_finite hf hs false hfin
    rw [heq] at this
    rw [abs_sub_comm]; exact this
  · exfalso
    have hxi : x.isInf = true := by unfold FloatV.isInf; rw [← heq]; exact hinf
    rw [(FloatV.isFinite_iff.mp hx).2] at hxi
    cases hxi

/-- An exact value that rounds to a finite `x` is at most `(1 + u)·|x| + η` in magnitude. -/
theorem RoundsTo.abs_le (hf : f.Regular) (hs : f.specials ≠ .finite) {p : ℚ} {x : FloatV f}
    (h : RoundsTo p x) (hx : x.isFinite = true) : |p| ≤ (1 + f.u) * |x.toRat| + f.eta := by
  have e := RoundsTo.err hf hs h hx
  have := abs_sub_abs_le_abs_sub p x.toRat
  nlinarith [abs_nonneg x.toRat]

namespace FloatV

/-- **A fusable addition** (`AddRel`, either `fusion`) of finite operands with room is finite
and within `3u·(|x| + |y|) + 4η` of `x + y`. Use it for `tl.addc` and for each node of a
reduction of products (`tl.sumc`). -/
theorem AddRel.err (hf : f.Regular) (hs : f.specials ≠ .finite) (hu : f.u ≤ 1 / 16) {fusion : Bool}
    {x y r : FloatV f} (h : AddRel fusion x y r) (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hM : |x.toRat| + |y.toRat| ≤ f.maxVal / 2 - 2 * f.eta) :
    r.isFinite = true ∧
      |r.toRat - (x.toRat + y.toRat)| ≤ 3 * f.u * (|x.toRat| + |y.toRat|) + 4 * f.eta := by
  have hu0 := Format.u_nonneg (f := f)
  have he0 := Format.eta_nonneg (f := f)
  have hmax0 := Format.maxVal_pos hf
  have hxy : |x.toRat + y.toRat| ≤ |x.toRat| + |y.toRat| := abs_add_le _ _
  have hS0 : 0 ≤ |x.toRat| + |y.toRat| := by positivity
  rcases h with hne | ⟨-, hfu⟩
  · -- the IEEE sum
    obtain ⟨hfin, herr⟩ := add_err hf hx hy (by linarith)
    rw [hne.eq (isFinite_iff.mp hfin).1]
    refine ⟨hfin, herr.trans ?_⟩
    nlinarith [mul_le_mul_of_nonneg_right hxy hu0]
  · rcases hfu with ⟨p, q, hp, hq, -, z, hne⟩ | ⟨p, -, hyi, -⟩ | ⟨q, -, hxi, -⟩
    · have ep := RoundsTo.err hf hs hp hx
      have eq := RoundsTo.err hf hs hq hy
      have hpq : |p + q| ≤ (1 + f.u) * (|x.toRat| + |y.toRat|) + 2 * f.eta := by
        have h1 := abs_sub_abs_le_abs_sub p x.toRat
        have h2 := abs_sub_abs_le_abs_sub q y.toRat
        have := abs_add_le p q
        nlinarith
      have hpqM : |p + q| ≤ f.maxVal := by nlinarith
      obtain ⟨hfin, herr⟩ := ofPat_roundRat_err hf false z hpqM
      rw [hne.eq (isFinite_iff.mp hfin).1]
      refine ⟨hfin, ?_⟩
      have e : (ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (x.toRat + y.toRat) =
          ((ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (p + q)) +
            (p - x.toRat) + (q - y.toRat) := by ring
      rw [e]
      have t1 := abs_add_le (((ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (p + q)) +
        (p - x.toRat)) (q - y.toRat)
      have t2 := abs_add_le ((ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (p + q))
        (p - x.toRat)
      have hb : |p + q| * f.u ≤ ((1 + f.u) * (|x.toRat| + |y.toRat|) + 2 * f.eta) * f.u :=
        mul_le_mul_of_nonneg_right hpq hu0
      nlinarith [mul_nonneg hu0 hS0, mul_nonneg (mul_nonneg hu0 hu0) hS0, mul_nonneg hu0 he0]
    · exfalso; rw [(isFinite_iff.mp hy).2] at hyi; cases hyi
    · exfalso; rw [(isFinite_iff.mp hx).2] at hxi; cases hxi

/-- **`tl.mul_add a b c`**: the product rounded then the sum rounded, or one `fma`. Finite
operands with room give a finite result within `3u·(|a·b| + |c|) + 4η` of `a·b + c`. -/
theorem mulAdd_err (hf : f.Regular) (hu : f.u ≤ 1 / 16) {fusion : Bool} {a b c r : FloatV f}
    (h : NaNEq (add (mul a b) c) r ∨ (fusion = true ∧ NaNEq (fma a b c) r))
    (ha : a.isFinite = true) (hb : b.isFinite = true) (hc : c.isFinite = true)
    (hM : |a.toRat * b.toRat| + |c.toRat| ≤ f.maxVal / 2 - 2 * f.eta) :
    r.isFinite = true ∧
      |r.toRat - (a.toRat * b.toRat + c.toRat)| ≤ 3 * f.u * (|a.toRat * b.toRat| + |c.toRat|) + 4 * f.eta := by
  have hu0 := Format.u_nonneg (f := f)
  have he0 := Format.eta_nonneg (f := f)
  have hmax0 := Format.maxVal_pos hf
  have hS0 : 0 ≤ |a.toRat * b.toRat| + |c.toRat| := by positivity
  have hsum : |a.toRat * b.toRat + c.toRat| ≤ |a.toRat * b.toRat| + |c.toRat| := abs_add_le _ _
  have hab0 := abs_nonneg (a.toRat * b.toRat)
  have hc0 := abs_nonneg c.toRat
  have hu1 : f.u * |a.toRat * b.toRat| ≤ 1 / 16 * |a.toRat * b.toRat| := mul_le_mul_of_nonneg_right hu hab0
  rcases h with hne | ⟨-, hne⟩
  · obtain ⟨hpf, hpe⟩ := mul_err hf ha hb (by linarith)
    have hp : |(mul a b).toRat| ≤ (1 + f.u) * |a.toRat * b.toRat| + f.eta := by
      have := abs_sub_abs_le_abs_sub (mul a b).toRat (a.toRat * b.toRat)
      nlinarith [abs_nonneg (a.toRat * b.toRat)]
    have hpc : |(mul a b).toRat + c.toRat| ≤ (1 + f.u) * |a.toRat * b.toRat| + f.eta + |c.toRat| :=
      (abs_add_le _ _).trans (by linarith)
    obtain ⟨hrf, hre⟩ := add_err hf hpf hc (by linarith)
    rw [hne.eq (isFinite_iff.mp hrf).1]
    refine ⟨hrf, ?_⟩
    have e : (add (mul a b) c).toRat - (a.toRat * b.toRat + c.toRat) =
        ((add (mul a b) c).toRat - ((mul a b).toRat + c.toRat)) + ((mul a b).toRat - a.toRat * b.toRat) := by
      ring
    rw [e]
    have t := abs_add_le ((add (mul a b) c).toRat - ((mul a b).toRat + c.toRat))
      ((mul a b).toRat - a.toRat * b.toRat)
    have hb2 : |(mul a b).toRat + c.toRat| * f.u ≤
        ((1 + f.u) * |a.toRat * b.toRat| + f.eta + |c.toRat|) * f.u := mul_le_mul_of_nonneg_right hpc hu0
    nlinarith [mul_nonneg hu0 (abs_nonneg (a.toRat * b.toRat)), mul_nonneg hu0 (abs_nonneg c.toRat),
      mul_nonneg (mul_nonneg hu0 hu0) (abs_nonneg (a.toRat * b.toRat)), mul_nonneg hu0 he0]
  · obtain ⟨hrf, hre⟩ := fma_err hf ha hb hc (by linarith)
    rw [hne.eq (isFinite_iff.mp hrf).1]
    refine ⟨hrf, hre.trans ?_⟩
    nlinarith [mul_le_mul_of_nonneg_right hsum hu0]

end FloatV

end TVA
