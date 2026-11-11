import Claim
import TVA.Proof

/-!
# The rounding error of one rotated channel

A rotated channel of `bfloat16` input is computed in `float32`: two products are rounded, added
(each product may be fused into the addition, so `AddRel` covers the results) and the sum is
rounded to `bfloat16`. `tol_sub` and `tol_add` bound the result against the exact rotation,
`tol` (`Claim.lean`): one `bfloat16` rounding of the rotation, the `float32` arithmetic, and the
subnormal floors.
-/

namespace Claims.RotaryVarlen

open TVA

set_option maxHeartbeats 1000000

private theorem ieee_arith {u e r x y X Y : ℚ}
    (hu : 0 < u) (_he : 0 < e)
    (hr : |r - (x + y)| ≤ |x + y| * u + e)
    (hx : |x - X| ≤ u * |X| + e)
    (hy : |y - Y| ≤ u * |Y| + e)
    (hcoef : 2 * u + u * u ≤ (7/2) * u)
    (heta : 3 * e + 2 * (u * e) ≤ 6 * e) :
    |r - (X + Y)| ≤ (7/2) * u * (|X| + |Y|) + 6 * e := by
  have hu0 : (0:ℚ) ≤ u := le_of_lt hu
  have hT0 : (0:ℚ) ≤ |X| + |Y| := by positivity
  have hxy : |x + y| ≤ (|X| + |Y|) + (u * (|X| + |Y|) + 2 * e) := by
    have h1 : |X + Y| ≤ |X| + |Y| := abs_add_le _ _
    have h2 : |(x + y) - (X + Y)| ≤ |x - X| + |y - Y| := by
      have e' : (x + y) - (X + Y) = (x - X) + (y - Y) := by ring
      rw [e']; exact abs_add_le _ _
    have h3 : |x + y| ≤ |X + Y| + |(x + y) - (X + Y)| := by
      have e' : x + y = (X + Y) + ((x + y) - (X + Y)) := by ring
      exact le_of_eq_of_le (congrArg abs e') (abs_add_le _ _)
    linarith
  have ht1 : |x + y| * u ≤ (u + u * u) * (|X| + |Y|) + 2 * (u * e) := by
    have h1a : |x + y| * u ≤ ((|X| + |Y|) + (u * (|X| + |Y|) + 2 * e)) * u :=
      mul_le_mul_of_nonneg_right hxy hu0
    have h1b : ((|X| + |Y|) + (u * (|X| + |Y|) + 2 * e)) * u =
        (u + u * u) * (|X| + |Y|) + 2 * (u * e) := by ring
    linarith
  have ht2 : |(x + y) - (X + Y)| ≤ u * (|X| + |Y|) + 2 * e := by
    have e' : (x + y) - (X + Y) = (x - X) + (y - Y) := by ring
    calc |(x + y) - (X + Y)| = |(x - X) + (y - Y)| := by rw [e']
      _ ≤ |x - X| + |y - Y| := abs_add_le _ _
      _ ≤ (u * |X| + e) + (u * |Y| + e) := by linarith
      _ = u * (|X| + |Y|) + 2 * e := by ring
  have ht3 : |r - (X + Y)| ≤ |r - (x + y)| + |(x + y) - (X + Y)| := by
    have e' : r - (X + Y) = (r - (x + y)) + ((x + y) - (X + Y)) := by ring
    calc |r - (X + Y)| = |(r - (x + y)) + ((x + y) - (X + Y))| := by rw [e']
      _ ≤ |r - (x + y)| + |(x + y) - (X + Y)| := abs_add_le _ _
  have hcoefT : (2 * u + u * u) * (|X| + |Y|) ≤ ((7/2) * u) * (|X| + |Y|) :=
    mul_le_mul_of_nonneg_right hcoef hT0
  have hmid : |r - (X + Y)| ≤ (2 * u + u * u) * (|X| + |Y|) + (3 * e + 2 * (u * e)) := by
    nlinarith [ht1, ht2, ht3, hr]
  have hsum : (2 * u + u * u) * (|X| + |Y|) + (3 * e + 2 * (u * e)) ≤
      (7/2) * u * (|X| + |Y|) + 6 * e := by
    nlinarith [hcoefT, heta]
  exact hmid.trans hsum

private theorem fusion_arith {u e r x y X Y p' q' : ℚ}
    (hu : 0 < u) (_he : 0 < e)
    (hr : |r - (p' + q')| ≤ |p' + q'| * u + e)
    (hp : |p' - x| ≤ u * |x| + e)
    (hq : |q' - y| ≤ u * |y| + e)
    (hX : |x - X| ≤ u * |X| + e)
    (hY : |y - Y| ≤ u * |Y| + e)
    (hex : p' = x ∨ q' = y)
    (h1 : u * u ≤ (3/2) * u) (h2 : 3 * (u * u) + u * u * u ≤ (1/2) * u)
    (h3 : 5 * (u * e) + (u * u) * e ≤ e) :
    |r - (X + Y)| ≤ (7/2) * u * (|X| + |Y|) + 6 * e := by
  have hu0 : (0:ℚ) ≤ u := le_of_lt hu
  have htri : ∀ a b c : ℚ, |a - c| ≤ |a - b| + |b - c| := by
    intro a b c
    have e' : a - c = (a - b) + (b - c) := by ring
    calc |a - c| = |(a - b) + (b - c)| := by rw [e']
      _ ≤ |a - b| + |b - c| := abs_add_le _ _
  have ht3 : |r - (X + Y)| ≤ |r - (p' + q')| + |(p' + q') - (X + Y)| := htri _ _ _
  have ht4 : |(p' + q') - (X + Y)| ≤ |p' - X| + |q' - Y| := by
    have e' : (p' + q') - (X + Y) = (p' - X) + (q' - Y) := by ring
    calc |(p' + q') - (X + Y)| = |(p' - X) + (q' - Y)| := by rw [e']
      _ ≤ |p' - X| + |q' - Y| := abs_add_le _ _
  have ht2 : |p' + q'| ≤ (|X| + |Y|) + (|p' - X| + |q' - Y|) := by
    have h1' : |X + Y| ≤ |X| + |Y| := abs_add_le _ _
    have h2' : |p' + q'| ≤ |X + Y| + |(p' + q') - (X + Y)| := by
      have e' : p' + q' = (X + Y) + ((p' + q') - (X + Y)) := by ring
      exact le_of_eq_of_le (congrArg abs e') (abs_add_le _ _)
    linarith
  have hX0 : (0:ℚ) ≤ |X| := abs_nonneg X
  have hY0 : (0:ℚ) ≤ |Y| := abs_nonneg Y
  have h1X : (u * u) * |X| ≤ ((3/2) * u) * |X| := mul_le_mul_of_nonneg_right h1 hX0
  have h1Y : (u * u) * |Y| ≤ ((3/2) * u) * |Y| := mul_le_mul_of_nonneg_right h1 hY0
  have h2X : (3 * (u * u) + u * u * u) * |X| ≤ ((1/2) * u) * |X| :=
    mul_le_mul_of_nonneg_right h2 hX0
  have h2Y : (3 * (u * u) + u * u * u) * |Y| ≤ ((1/2) * u) * |Y| :=
    mul_le_mul_of_nonneg_right h2 hY0
  rcases hex with hex | hex
  · have hX' : |p' - X| ≤ u * |X| + e := by rw [hex]; exact hX
    have hqY : |q' - Y| ≤ (2 * u + u * u) * |Y| + 3 * e + u * e := by
      have ht : |q' - Y| ≤ |q' - y| + |y - Y| := htri _ _ _
      have hy : |y| ≤ |Y| + (u * |Y| + e) := by
        have := abs_sub_abs_le_abs_sub y Y
        linarith
      have hm : u * |y| ≤ u * (|Y| + (u * |Y| + e)) := mul_le_mul_of_nonneg_left hy hu0
      nlinarith [ht, hq, hm, hY]
    have hD : |p' - X| + |q' - Y| ≤ u * |X| + (2 * u + u * u) * |Y| + 4 * e + u * e := by
      linarith [hX', hqY]
    have hEr : |r - (X + Y)| ≤
        u * ((|X| + |Y|) + (u * |X| + (2 * u + u * u) * |Y| + 4 * e + u * e)) + e +
          (u * |X| + (2 * u + u * u) * |Y| + 4 * e + u * e) := by
      nlinarith [ht2, ht3, ht4, hr, hD, hu]
    have hA : |r - (X + Y)| ≤
        (2 * u + u * u) * |X| + (3 * u + 3 * (u * u) + u * u * u) * |Y| +
          (5 * e + 5 * (u * e) + (u * u) * e) := by
      have hring : u * ((|X| + |Y|) + (u * |X| + (2 * u + u * u) * |Y| + 4 * e + u * e)) + e +
          (u * |X| + (2 * u + u * u) * |Y| + 4 * e + u * e) =
          (2 * u + u * u) * |X| + (3 * u + 3 * (u * u) + u * u * u) * |Y| +
            (5 * e + 5 * (u * e) + (u * u) * e) := by
        ring
      rw [hring] at hEr
      exact hEr
    nlinarith [hA, h1X, h2Y, h3, hX0, hY0]
  · have hY' : |q' - Y| ≤ u * |Y| + e := by rw [hex]; exact hY
    have hpX : |p' - X| ≤ (2 * u + u * u) * |X| + 3 * e + u * e := by
      have ht : |p' - X| ≤ |p' - x| + |x - X| := htri _ _ _
      have hx : |x| ≤ |X| + (u * |X| + e) := by
        have := abs_sub_abs_le_abs_sub x X
        linarith
      have hm : u * |x| ≤ u * (|X| + (u * |X| + e)) := mul_le_mul_of_nonneg_left hx hu0
      nlinarith [ht, hp, hm, hX]
    have hD : |p' - X| + |q' - Y| ≤ (2 * u + u * u) * |X| + u * |Y| + 4 * e + u * e := by
      linarith [hpX, hY']
    have hEr : |r - (X + Y)| ≤
        u * ((|X| + |Y|) + ((2 * u + u * u) * |X| + u * |Y| + 4 * e + u * e)) + e +
          ((2 * u + u * u) * |X| + u * |Y| + 4 * e + u * e) := by
      nlinarith [ht2, ht3, ht4, hr, hD, hu]
    have hA : |r - (X + Y)| ≤
        (3 * u + 3 * (u * u) + u * u * u) * |X| + (2 * u + u * u) * |Y| +
          (5 * e + 5 * (u * e) + (u * u) * e) := by
      have hring : u * ((|X| + |Y|) + ((2 * u + u * u) * |X| + u * |Y| + 4 * e + u * e)) + e +
          ((2 * u + u * u) * |X| + u * |Y| + 4 * e + u * e) =
          (3 * u + 3 * (u * u) + u * u * u) * |X| + (2 * u + u * u) * |Y| +
            (5 * e + 5 * (u * e) + (u * u) * e) := by
        ring
      rw [hring] at hEr
      exact hEr
    nlinarith [hA, h2X, h1Y, h3, hX0, hY0]

private theorem mul_bnd {f : Format} (hf : f.Regular) {x y p : FloatV f}
    (hxf : x.isFinite = true) (hyf : y.isFinite = true)
    (hp : FloatV.NaNEq (FloatV.mul x y) p)
    (hmax : |x.toRat * y.toRat| ≤ f.maxVal) :
    p.isFinite = true ∧ |p.toRat - x.toRat * y.toRat| ≤ f.u * |x.toRat * y.toRat| + f.eta := by
  obtain ⟨hfin, herr⟩ := FloatV.mul_err hf hxf hyf hmax
  rw [hp.eq (FloatV.isFinite_iff.mp hfin).1]
  have hcomm : |x.toRat * y.toRat| * f.u = f.u * |x.toRat * y.toRat| := mul_comm _ _
  rw [hcomm] at herr
  exact ⟨hfin, herr⟩

private theorem add_rel_bnd {f : Format} (hf : f.Regular) (hs : f.specials ≠ .finite)
    {x y r : FloatV f} {X Y : ℚ}
    (hxf : x.isFinite = true) (hyf : y.isFinite = true)
    (h : AddRel true x y r)
    (hX : |x.toRat - X| ≤ f.u * |X| + f.eta)
    (hY : |y.toRat - Y| ≤ f.u * |Y| + f.eta)
    (hM : |x.toRat| + |y.toRat| ≤ f.maxVal / 2 - 2 * f.eta)
    (hu : 0 < f.u) (he : 0 < f.eta)
    (hroom : (1 + f.u) * (f.maxVal / 2 - 2 * f.eta) + 2 * f.eta ≤ f.maxVal)
    (hu1 : f.u * f.u ≤ (3/2) * f.u)
    (hu2 : 3 * (f.u * f.u) + f.u * f.u * f.u ≤ (1/2) * f.u)
    (he1 : 5 * (f.u * f.eta) + (f.u * f.u) * f.eta ≤ f.eta)
    (hcoef : 2 * f.u + f.u * f.u ≤ (7/2) * f.u)
    (heta : 3 * f.eta + 2 * (f.u * f.eta) ≤ 6 * f.eta) :
    r.isFinite = true ∧ |r.toRat - (X + Y)| ≤ (7/2) * f.u * (|X| + |Y|) + 6 * f.eta := by
  have hu0 : (0:ℚ) ≤ f.u := le_of_lt hu
  have hmax0 : 0 < f.maxVal := Format.maxVal_pos hf
  have hxy : |x.toRat + y.toRat| ≤ |x.toRat| + |y.toRat| := abs_add_le _ _
  rcases h with hne | ⟨-, hfu⟩
  · obtain ⟨hfin, herr⟩ := FloatV.add_err hf hxf hyf (by linarith)
    rw [hne.eq (FloatV.isFinite_iff.mp hfin).1]
    exact ⟨hfin, ieee_arith hu he herr hX hY hcoef heta⟩
  · rcases hfu with ⟨p, q, hp, hq, hconst, z, hne⟩ | ⟨p, -, hyi, -⟩ | ⟨q, -, hxi, -⟩
    · have ep := RoundsTo.err hf hs hp hxf
      have eq := RoundsTo.err hf hs hq hyf
      rw [mul_comm (|x.toRat|) f.u] at ep
      rw [mul_comm (|y.toRat|) f.u] at eq
      have hpq : |p + q| ≤ (1 + f.u) * (|x.toRat| + |y.toRat|) + 2 * f.eta := by
        have h1 := abs_sub_abs_le_abs_sub p x.toRat
        have h2 := abs_sub_abs_le_abs_sub q y.toRat
        have := abs_add_le p q
        nlinarith
      have hpqM : |p + q| ≤ f.maxVal := by
        have hD : (1 + f.u) * (|x.toRat| + |y.toRat|) ≤
            (1 + f.u) * (f.maxVal / 2 - 2 * f.eta) :=
          mul_le_mul_of_nonneg_left hM (by linarith)
        linarith
      obtain ⟨hfin, herr⟩ := FloatV.ofPat_roundRat_err hf false z hpqM
      rw [hne.eq (FloatV.isFinite_iff.mp hfin).1]
      exact ⟨hfin, fusion_arith hu he herr ep eq hX hY hconst hu1 hu2 he1⟩
    · exfalso; rw [(FloatV.isFinite_iff.mp hyf).2] at hyi; cases hyi
    · exfalso; rw [(FloatV.isFinite_iff.mp hxf).2] at hxi; cases hxi

private theorem cvt_room {u e T S r c : ℚ}
    (hu : 0 ≤ u) (_hT0 : 0 ≤ T)
    (hT : T ≤ (2:ℚ)^126)
    (hnum : (1 + (7/2) * u) * (2:ℚ)^126 + 6 * e ≤ c)
    (hE : |r - S| ≤ (7/2) * u * T + 6 * e)
    (hS : |S| ≤ T) :
    |r| ≤ c := by
  have h1 : |r| ≤ T + ((7/2) * u * T + 6 * e) := by
    have := abs_sub_abs_le_abs_sub r S
    linarith
  have hm : (7/2) * u * T ≤ (7/2) * u * (2:ℚ)^126 :=
    mul_le_mul_of_nonneg_left hT (mul_nonneg (by norm_num) hu)
  have e1 : T + ((7/2) * u * T + 6 * e) ≤ (2:ℚ)^126 + (7/2) * u * (2:ℚ)^126 + 6 * e := by
    linarith
  have e2 : (2:ℚ)^126 + (7/2) * u * (2:ℚ)^126 + 6 * e =
      (1 + (7/2) * u) * (2:ℚ)^126 + 6 * e := by ring
  linarith

private theorem finish_arith {u e u8 e8 c22 ef r v S T : ℚ}
    (hu8 : 0 ≤ u8) (hT0 : 0 ≤ T)
    (hE8 : (1 + u8) * ((7/2) * u) ≤ c22)
    (he8 : (1 + u8) * (6 * e) + e8 ≤ ef)
    (hE : |r - S| ≤ (7/2) * u * T + 6 * e)
    (hvr : |v - r| ≤ |r| * u8 + e8) :
    |v - S| ≤ u8 * |S| + c22 * T + ef := by
  have htri : |v - S| ≤ |v - r| + |r - S| := by
    have e' : v - S = (v - r) + (r - S) := by ring
    calc |v - S| = |(v - r) + (r - S)| := by rw [e']
      _ ≤ |v - r| + |r - S| := abs_add_le _ _
  have hrS2 : |r| ≤ |S| + ((7/2) * u * T + 6 * e) := by
    have := abs_sub_abs_le_abs_sub r S
    linarith
  have hmul : |r| * u8 ≤ (|S| + ((7/2) * u * T + 6 * e)) * u8 :=
    mul_le_mul_of_nonneg_right hrS2 hu8
  have hcoefT : ((1 + u8) * ((7/2) * u)) * T ≤ c22 * T :=
    mul_le_mul_of_nonneg_right hE8 hT0
  have h1 : |v - S| ≤ |r| * u8 + e8 + ((7/2) * u * T + 6 * e) := by linarith
  have h2 : |r| * u8 + e8 + ((7/2) * u * T + 6 * e) ≤
      u8 * |S| + (1 + u8) * ((7/2) * u) * T + ((1 + u8) * (6 * e) + e8) := by
    nlinarith [hmul]
  have h3 : u8 * |S| + (1 + u8) * ((7/2) * u) * T + ((1 + u8) * (6 * e) + e8) ≤
      u8 * |S| + c22 * T + ef := by linarith
  linarith

/-- The rotated first channel: `x0 · cos − x1 · sin`, where `x0`, `x1` are a pair of `bfloat16`
inputs, `cos`, `sin` its table entries (`af`…`sf` are their exact `float32` widenings). -/
theorem tol_sub {a b cs sn : BF16} {af bf cf sf : F32}
    (haf : af.toRat = a.toRat) (hbf : bf.toRat = b.toRat)
    (hcf : cf.toRat = cs.toRat) (hsf : sf.toRat = sn.toRat)
    (hafin : af.isFinite = true) (hbfin : bf.isFinite = true)
    (hcfin : cf.isFinite = true) (hsfn : sf.isFinite = true)
    {p q r : F32} {v : BF16}
    (hp : FloatV.NaNEq (FloatV.mul af cf) p)
    (hq : FloatV.NaNEq (FloatV.mul bf sf) q)
    (hr : AddRel true p (FloatV.neg q) r)
    (hv : FloatV.NaNEq (FloatV.cvtWith .rn r) v)
    (ha : XOk a) (hb : XOk b) (hcs : TabOk cs) (hsn : TabOk sn) :
    v.isFinite = true ∧
      |v.toRat - rotRef true a.toRat b.toRat cs.toRat sn.toRat| ≤
        tol true a.toRat b.toRat cs.toRat sn.toRat := by
  have hu : (0:ℚ) < Format.binary32.u := Format.u_pos
  have he : (0:ℚ) < Format.binary32.eta := Format.eta_pos
  have hcbound : |a.toRat * cs.toRat| ≤ (2:ℚ)^125 := by
    rw [abs_mul]
    nlinarith [ha.2, hcs.2, abs_nonneg a.toRat, abs_nonneg cs.toRat]
  have hmaxp : |af.toRat * cf.toRat| ≤ Format.binary32.maxVal := by
    rw [haf, hcf]
    calc |a.toRat * cs.toRat| ≤ (2:ℚ)^125 := hcbound
      _ ≤ Format.binary32.maxVal := by rw [Format.binary32_maxVal]; norm_num
  obtain ⟨hpmul, hperr⟩ := mul_bnd Format.binary32_regular hafin hcfin hp hmaxp
  rw [haf, hcf] at hperr
  have hdbound : |b.toRat * sn.toRat| ≤ (2:ℚ)^125 := by
    rw [abs_mul]
    nlinarith [hb.2, hsn.2, abs_nonneg b.toRat, abs_nonneg sn.toRat]
  have hmaxq : |bf.toRat * sf.toRat| ≤ Format.binary32.maxVal := by
    rw [hbf, hsf]
    calc |b.toRat * sn.toRat| ≤ (2:ℚ)^125 := hdbound
      _ ≤ Format.binary32.maxVal := by rw [Format.binary32_maxVal]; norm_num
  obtain ⟨hqmul, hqerr⟩ := mul_bnd Format.binary32_regular hbfin hsfn hq hmaxq
  rw [hbf, hsf] at hqerr
  have hnegfin : (FloatV.neg q).isFinite = true := by
    rw [FloatV.isFinite_neg]; exact hqmul
  have hnegerr : |(FloatV.neg q).toRat - -(b.toRat * sn.toRat)| ≤
      Format.binary32.u * |-(b.toRat * sn.toRat)| + Format.binary32.eta := by
    rw [FloatV.toRat_neg]
    rw [show -q.toRat - -(b.toRat * sn.toRat) = -(q.toRat - b.toRat * sn.toRat) by ring,
      abs_neg, abs_neg]
    exact hqerr
  have hT0 : (0:ℚ) ≤ |a.toRat * cs.toRat| + |-(b.toRat * sn.toRat)| := by
    have h1 := abs_nonneg (a.toRat * cs.toRat)
    have h2 := abs_nonneg (-(b.toRat * sn.toRat))
    linarith
  have hTre : |a.toRat * cs.toRat| + |-(b.toRat * sn.toRat)| ≤ (2:ℚ)^126 := by
    have h1 := hcbound
    have h2 : |-(b.toRat * sn.toRat)| ≤ (2:ℚ)^125 := by rw [abs_neg]; exact hdbound
    linarith
  have hpabs : |p.toRat| ≤
      (1 + Format.binary32.u) * |a.toRat * cs.toRat| + Format.binary32.eta := by
    have := abs_sub_abs_le_abs_sub p.toRat (a.toRat * cs.toRat)
    linarith [hperr]
  have hqabs : |(FloatV.neg q).toRat| ≤
      (1 + Format.binary32.u) * |-(b.toRat * sn.toRat)| + Format.binary32.eta := by
    have := abs_sub_abs_le_abs_sub (FloatV.neg q).toRat (-(b.toRat * sn.toRat))
    linarith [hnegerr]
  have hM : |p.toRat| + |(FloatV.neg q).toRat| ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have hsum : |p.toRat| + |(FloatV.neg q).toRat| ≤
        (1 + Format.binary32.u) * (|a.toRat * cs.toRat| + |-(b.toRat * sn.toRat)|) +
          2 * Format.binary32.eta := by
      nlinarith [hpabs, hqabs]
    have hmul2 : (1 + Format.binary32.u) *
        (|a.toRat * cs.toRat| + |-(b.toRat * sn.toRat)|) ≤
        (1 + Format.binary32.u) * (2:ℚ)^126 :=
      mul_le_mul_of_nonneg_left hTre (by linarith)
    have hnum : (1 + Format.binary32.u) * (2:ℚ)^126 + 2 * Format.binary32.eta ≤
        Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
      rw [Format.binary32_u, Format.binary32_eta, Format.binary32_maxVal]
      norm_num
    linarith
  obtain ⟨hrfin, hrerr⟩ := add_rel_bnd Format.binary32_regular
    (by decide : Format.binary32.specials ≠ .finite) hpmul hnegfin hr hperr hnegerr hM hu he
    (by rw [Format.binary32_u, Format.binary32_eta, Format.binary32_maxVal]; norm_num)
    (by rw [Format.binary32_u]; norm_num)
    (by rw [Format.binary32_u]; norm_num)
    (by rw [Format.binary32_u, Format.binary32_eta]; norm_num)
    (by rw [Format.binary32_u]; norm_num)
    (by rw [Format.binary32_u, Format.binary32_eta]; norm_num)
  have hSabs : |a.toRat * cs.toRat + -(b.toRat * sn.toRat)| ≤
      |a.toRat * cs.toRat| + |-(b.toRat * sn.toRat)| := abs_add_le _ _
  have hrmax : |r.toRat| ≤ Format.bfloat16.maxVal :=
    cvt_room (u := Format.binary32.u) (e := Format.binary32.eta)
      (T := |a.toRat * cs.toRat| + |-(b.toRat * sn.toRat)|)
      (S := a.toRat * cs.toRat + -(b.toRat * sn.toRat))
      (r := r.toRat) (c := Format.bfloat16.maxVal)
      (le_of_lt hu) hT0 hTre
      (by rw [Format.binary32_u, Format.binary32_eta, Format.bfloat16_maxVal]; norm_num)
      hrerr hSabs
  have hv' : FloatV.NaNEq (FloatV.cvt r) v := by
    simpa only [FloatV.cvtWith_rn] using hv
  obtain ⟨hvfin, hverr⟩ := FloatV.cvt_err (f := Format.binary32) (g := Format.bfloat16)
    Format.bfloat16_regular hrfin hrmax
  have hveq : v = FloatV.cvt r := hv'.eq (FloatV.isFinite_iff.mp hvfin).1
  have habs : |-(b.toRat * sn.toRat)| = |b.toRat * sn.toRat| := abs_neg _
  have harg : a.toRat * cs.toRat + -(b.toRat * sn.toRat) =
      a.toRat * cs.toRat - b.toRat * sn.toRat := by ring
  have hE : |r.toRat - (a.toRat * cs.toRat - b.toRat * sn.toRat)| ≤
      (7/2) * Format.binary32.u * (|a.toRat * cs.toRat| + |b.toRat * sn.toRat|) +
        6 * Format.binary32.eta := by
    rw [habs, harg] at hrerr
    exact hrerr
  have hT0' : (0:ℚ) ≤ |a.toRat * cs.toRat| + |b.toRat * sn.toRat| := by
    have h1 := abs_nonneg (a.toRat * cs.toRat)
    have h2 := abs_nonneg (b.toRat * sn.toRat)
    linarith
  have hE8 : (1 + Format.bfloat16.u) * ((7/2) * Format.binary32.u) ≤
      (2:ℚ)^(-22:ℤ) := by
    rw [Format.bfloat16_u, Format.binary32_u]; norm_num
  have he8 : (1 + Format.bfloat16.u) * (6 * Format.binary32.eta) + Format.bfloat16.eta ≤
      (2:ℚ)^(-132:ℤ) := by
    rw [Format.bfloat16_u, Format.binary32_eta, Format.bfloat16_eta]; norm_num
  have hfin := finish_arith (u := Format.binary32.u) (e := Format.binary32.eta)
    (u8 := Format.bfloat16.u) (e8 := Format.bfloat16.eta)
    (c22 := (2:ℚ)^(-22:ℤ)) (ef := (2:ℚ)^(-132:ℤ))
    (r := r.toRat) (v := (FloatV.cvt r).toRat)
    (S := a.toRat * cs.toRat - b.toRat * sn.toRat)
    (T := |a.toRat * cs.toRat| + |b.toRat * sn.toRat|)
    (le_of_lt (Format.u_pos (f := Format.bfloat16))) hT0' hE8 he8 hE hverr
  refine ⟨?_, ?_⟩
  · rw [hveq]; exact hvfin
  · subst hveq
    rw [Format.bfloat16_u] at hfin
    rw [show rotRef true a.toRat b.toRat cs.toRat sn.toRat =
        a.toRat * cs.toRat - b.toRat * sn.toRat from rfl]
    rw [show tol true a.toRat b.toRat cs.toRat sn.toRat =
        (2:ℚ)^(-8:ℤ) * |a.toRat * cs.toRat - b.toRat * sn.toRat| +
          (2:ℚ)^(-22:ℤ) * (|a.toRat * cs.toRat| + |b.toRat * sn.toRat|) +
          (2:ℚ)^(-132:ℤ) from by rw [tol]; rfl]
    exact hfin

/-- The rotated second channel: `x0 · cos + x1 · sin`, as `tol_sub` (the addition takes
`x0 · sin` and `x1 · cos`, so the operands enter in the other order). -/
theorem tol_add {a b cs sn : BF16} {af bf cf sf : F32}
    (haf : af.toRat = a.toRat) (hbf : bf.toRat = b.toRat)
    (hcf : cf.toRat = cs.toRat) (hsf : sf.toRat = sn.toRat)
    (hafin : af.isFinite = true) (hbfin : bf.isFinite = true)
    (hcfin : cf.isFinite = true) (hsfn : sf.isFinite = true)
    {p q r : F32} {v : BF16}
    (hp : FloatV.NaNEq (FloatV.mul bf sf) p)
    (hq : FloatV.NaNEq (FloatV.mul af cf) q)
    (hr : AddRel true p q r)
    (hv : FloatV.NaNEq (FloatV.cvtWith .rn r) v)
    (ha : XOk a) (hb : XOk b) (hcs : TabOk cs) (hsn : TabOk sn) :
    v.isFinite = true ∧
      |v.toRat - rotRef false a.toRat b.toRat cs.toRat sn.toRat| ≤
        tol false a.toRat b.toRat cs.toRat sn.toRat := by
  have hu : (0:ℚ) < Format.binary32.u := Format.u_pos
  have he : (0:ℚ) < Format.binary32.eta := Format.eta_pos
  have hdbound : |b.toRat * sn.toRat| ≤ (2:ℚ)^125 := by
    rw [abs_mul]
    nlinarith [hb.2, hsn.2, abs_nonneg b.toRat, abs_nonneg sn.toRat]
  have hmaxp : |bf.toRat * sf.toRat| ≤ Format.binary32.maxVal := by
    rw [hbf, hsf]
    calc |b.toRat * sn.toRat| ≤ (2:ℚ)^125 := hdbound
      _ ≤ Format.binary32.maxVal := by rw [Format.binary32_maxVal]; norm_num
  obtain ⟨hpmul, hperr⟩ := mul_bnd Format.binary32_regular hbfin hsfn hp hmaxp
  rw [hbf, hsf] at hperr
  have hcbound : |a.toRat * cs.toRat| ≤ (2:ℚ)^125 := by
    rw [abs_mul]
    nlinarith [ha.2, hcs.2, abs_nonneg a.toRat, abs_nonneg cs.toRat]
  have hmaxq : |af.toRat * cf.toRat| ≤ Format.binary32.maxVal := by
    rw [haf, hcf]
    calc |a.toRat * cs.toRat| ≤ (2:ℚ)^125 := hcbound
      _ ≤ Format.binary32.maxVal := by rw [Format.binary32_maxVal]; norm_num
  obtain ⟨hqmul, hqerr⟩ := mul_bnd Format.binary32_regular hafin hcfin hq hmaxq
  rw [haf, hcf] at hqerr
  have hT0 : (0:ℚ) ≤ |b.toRat * sn.toRat| + |a.toRat * cs.toRat| := by
    have h1 := abs_nonneg (b.toRat * sn.toRat)
    have h2 := abs_nonneg (a.toRat * cs.toRat)
    linarith
  have hTre : |b.toRat * sn.toRat| + |a.toRat * cs.toRat| ≤ (2:ℚ)^126 := by
    have h1 := hdbound
    have h2 := hcbound
    linarith
  have hpabs : |p.toRat| ≤
      (1 + Format.binary32.u) * |b.toRat * sn.toRat| + Format.binary32.eta := by
    have := abs_sub_abs_le_abs_sub p.toRat (b.toRat * sn.toRat)
    linarith [hperr]
  have hqabs : |q.toRat| ≤
      (1 + Format.binary32.u) * |a.toRat * cs.toRat| + Format.binary32.eta := by
    have := abs_sub_abs_le_abs_sub q.toRat (a.toRat * cs.toRat)
    linarith [hqerr]
  have hM : |p.toRat| + |q.toRat| ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have hsum : |p.toRat| + |q.toRat| ≤
        (1 + Format.binary32.u) * (|b.toRat * sn.toRat| + |a.toRat * cs.toRat|) +
          2 * Format.binary32.eta := by
      nlinarith [hpabs, hqabs]
    have hmul2 : (1 + Format.binary32.u) *
        (|b.toRat * sn.toRat| + |a.toRat * cs.toRat|) ≤
        (1 + Format.binary32.u) * (2:ℚ)^126 :=
      mul_le_mul_of_nonneg_left hTre (by linarith)
    have hnum : (1 + Format.binary32.u) * (2:ℚ)^126 + 2 * Format.binary32.eta ≤
        Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
      rw [Format.binary32_u, Format.binary32_eta, Format.binary32_maxVal]
      norm_num
    linarith
  obtain ⟨hrfin, hrerr⟩ := add_rel_bnd Format.binary32_regular
    (by decide : Format.binary32.specials ≠ .finite) hpmul hqmul hr hperr hqerr hM hu he
    (by rw [Format.binary32_u, Format.binary32_eta, Format.binary32_maxVal]; norm_num)
    (by rw [Format.binary32_u]; norm_num)
    (by rw [Format.binary32_u]; norm_num)
    (by rw [Format.binary32_u, Format.binary32_eta]; norm_num)
    (by rw [Format.binary32_u]; norm_num)
    (by rw [Format.binary32_u, Format.binary32_eta]; norm_num)
  have hSabs : |b.toRat * sn.toRat + a.toRat * cs.toRat| ≤
      |b.toRat * sn.toRat| + |a.toRat * cs.toRat| := abs_add_le _ _
  have hrmax : |r.toRat| ≤ Format.bfloat16.maxVal :=
    cvt_room (u := Format.binary32.u) (e := Format.binary32.eta)
      (T := |b.toRat * sn.toRat| + |a.toRat * cs.toRat|)
      (S := b.toRat * sn.toRat + a.toRat * cs.toRat)
      (r := r.toRat) (c := Format.bfloat16.maxVal)
      (le_of_lt hu) hT0 hTre
      (by rw [Format.binary32_u, Format.binary32_eta, Format.bfloat16_maxVal]; norm_num)
      hrerr hSabs
  have hv' : FloatV.NaNEq (FloatV.cvt r) v := by
    simpa only [FloatV.cvtWith_rn] using hv
  obtain ⟨hvfin, hverr⟩ := FloatV.cvt_err (f := Format.binary32) (g := Format.bfloat16)
    Format.bfloat16_regular hrfin hrmax
  have hveq : v = FloatV.cvt r := hv'.eq (FloatV.isFinite_iff.mp hvfin).1
  have harg : b.toRat * sn.toRat + a.toRat * cs.toRat =
      a.toRat * cs.toRat + b.toRat * sn.toRat := by ring
  have hE : |r.toRat - (a.toRat * cs.toRat + b.toRat * sn.toRat)| ≤
      (7/2) * Format.binary32.u * (|a.toRat * cs.toRat| + |b.toRat * sn.toRat|) +
        6 * Format.binary32.eta := by
    have hTcomm : |b.toRat * sn.toRat| + |a.toRat * cs.toRat| =
        |a.toRat * cs.toRat| + |b.toRat * sn.toRat| := by ring
    rw [harg, hTcomm] at hrerr
    exact hrerr
  have hT0' : (0:ℚ) ≤ |a.toRat * cs.toRat| + |b.toRat * sn.toRat| := by
    have h1 := abs_nonneg (a.toRat * cs.toRat)
    have h2 := abs_nonneg (b.toRat * sn.toRat)
    linarith
  have hE8 : (1 + Format.bfloat16.u) * ((7/2) * Format.binary32.u) ≤
      (2:ℚ)^(-22:ℤ) := by
    rw [Format.bfloat16_u, Format.binary32_u]; norm_num
  have he8 : (1 + Format.bfloat16.u) * (6 * Format.binary32.eta) + Format.bfloat16.eta ≤
      (2:ℚ)^(-132:ℤ) := by
    rw [Format.bfloat16_u, Format.binary32_eta, Format.bfloat16_eta]; norm_num
  have hfin := finish_arith (u := Format.binary32.u) (e := Format.binary32.eta)
    (u8 := Format.bfloat16.u) (e8 := Format.bfloat16.eta)
    (c22 := (2:ℚ)^(-22:ℤ)) (ef := (2:ℚ)^(-132:ℤ))
    (r := r.toRat) (v := (FloatV.cvt r).toRat)
    (S := a.toRat * cs.toRat + b.toRat * sn.toRat)
    (T := |a.toRat * cs.toRat| + |b.toRat * sn.toRat|)
    (le_of_lt (Format.u_pos (f := Format.bfloat16))) hT0' hE8 he8 hE hverr
  refine ⟨?_, ?_⟩
  · rw [hveq]; exact hvfin
  · subst hveq
    rw [Format.bfloat16_u] at hfin
    rw [show rotRef false a.toRat b.toRat cs.toRat sn.toRat =
        a.toRat * cs.toRat + b.toRat * sn.toRat from rfl]
    rw [show tol false a.toRat b.toRat cs.toRat sn.toRat =
        (2:ℚ)^(-8:ℤ) * |a.toRat * cs.toRat + b.toRat * sn.toRat| +
          (2:ℚ)^(-22:ℤ) * (|a.toRat * cs.toRat| + |b.toRat * sn.toRat|) +
          (2:ℚ)^(-132:ℤ) from by rw [tol]; rfl]
    exact hfin

end Claims.RotaryVarlen
