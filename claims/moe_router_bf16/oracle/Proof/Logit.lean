import Claim
import TVA.Proof
import Proof.Setup
import Proof.Addrs
import Proof.Cast

/-!
# The logit bound: `tl.sum1c (w * x)` is within `tolL` of the reference `mu`

`prod[e][j] = w[e][h] * x[h]` (a `binary32` rounding of the exact product: the `bf16` load is
exact in `binary32`), and `logits[e] = Σ_j prod[e][j]` along a summation tree of depth at most
`tl.sumBound P d.BLOCK`. Padded lanes (`hidden ≤ h < BLOCK`) contribute exactly zero: at least one
factor is `+0`. So

  `|logits[e] − μ| ≤ (u·S + hidden·η) + errBound (3u) (4η) D (BLOCK−1) S'`,

with `S = absS init d t e`, `S' = Σ_j |prod[j]| ≤ (1+u)·S + hidden·η`, and `D ≤ ⌈log₂ BLOCK⌉`.
The first bracketed term is the product rounding, the second the summation; together they are
exactly `tolL d S`.
-/

namespace Claims.MoeRouter
open TVA
open Block

variable {d : Inst}

private theorem wVal_eq {m : Mem} {d : Inst} {e h : Nat} {v : F32}
    (hr : m.read .f32 (d.w.ptr ((e * d.hidden + h : Nat) : Int)).addr = some v) :
    wVal m d e h = v.toReal := by
  unfold wVal; rw [hr]

private theorem c32_zero_rat : (c32 0).toRat = 0 := by decide +kernel
private theorem c32_zero_fin : (c32 0).isFinite = true := by decide +kernel
private theorem litBF16_zero : FloatV.litBF16 (FloatV.ofPat 0) = (FloatV.ofPat 0 : BF16) := rfl

private theorem maxVal_pos_q : (0 : ℚ) < Format.binary32.maxVal := by
  rw [Format.binary32_maxVal]; norm_num

private theorem maxVal_ge_120 : (2 : ℚ) ^ 120 ≤ Format.binary32.maxVal := by
  rw [Format.binary32_maxVal]; norm_num

private theorem mul_toRat_eq_zero {f : Format} (hf : f.Regular) {a b : FloatV f}
    (ha : a.toRat = 0) (hb : b.isFinite = true) : (FloatV.mul a b).toRat = 0 := by
  have hza : f.isZero a.pat = true := Format.toRat_eq_zero_iff.mp ha
  have hmaga : f.mag a.pat = 0 := by
    unfold Format.isZero at hza
    simpa using hza
  have hfa : a.isFinite = true :=
    Format.isFinite_of_mag_le hf (by rw [hmaga]; exact Nat.zero_le _)
  have hmul : f.mul .rn a.pat b.pat =
      f.roundRat .rn false (f.sign a.pat != f.sign b.pat) (f.toRat a.pat * f.toRat b.pat) :=
    Format.mul_of_finite .rn hfa hb
  have hz : f.toRat a.pat * f.toRat b.pat = 0 := by
    have h : f.toRat a.pat = 0 := ha
    rw [h, zero_mul]
  have hspec := Format.roundRat_zero_spec (f := f) hf
    (rnd := .rn) (sat := false) (zs := f.sign a.pat != f.sign b.pat)
  show (FloatV.ofPat (f.mul .rn a.pat b.pat)).toRat = 0
  rw [hmul, hz, FloatV.toRat_ofPat hspec.1, hspec.2.2]

private theorem list_sum_range_congr (g : Nat → ℚ) (n : Nat) :
    ((List.range n).map g).sum = ∑ k ∈ Finset.range n, g k := by
  induction n with
  | zero => simp
  | succ m ih =>
    rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ, ih]
    simp

private theorem finset_sum_range_eq_of_zero (g : Nat → ℚ) {L N : Nat} (hLN : L ≤ N)
    (hz : ∀ k, L ≤ k → k < N → g k = 0) :
    ∑ k ∈ Finset.range N, g k = ∑ k ∈ Finset.range L, g k := by
  refine (Finset.sum_subset (f := g) (s₁ := Finset.range L) (s₂ := Finset.range N) ?_ ?_).symm
  · intro x hx
    simp only [Finset.mem_range] at hx ⊢
    omega
  · intro x hxN hxL
    have hxL' : L ≤ x := by
      simp only [Finset.mem_range, not_lt] at hxL; exact hxL
    exact hz x hxL' (by simpa using hxN)

private theorem clog_block_le (d : Inst) (hd : d.ok) : Nat.clog 2 d.BLOCK ≤ 13 := by
  obtain ⟨e, he⟩ := hd.block_pow2
  have he13 : e ≤ 13 := by
    have hble := hd.block_le
    rw [he] at hble
    exact (Nat.pow_le_pow_iff_right (by norm_num)).mp hble
  rw [he, Nat.clog_pow 2 e (by norm_num)]
  exact he13

private theorem errBound_cast (δ η : ℚ) (d N : Nat) (A : ℚ) :
    ((Tree.errBound δ η d N A : ℚ) : ℝ) =
      ((1 + (δ : ℝ)) ^ d - 1) * (A : ℝ) + (η : ℝ) * (N : ℝ) * (1 + (δ : ℝ)) ^ d := by
  show ((((1 + δ) ^ d - 1) * A + η * (N : ℚ) * (1 + δ) ^ d : ℚ) : ℝ) = _
  push_cast
  ring

set_option maxHeartbeats 2000000 in
/-- **The logit bound.** Each of the 8 logits the row sum returns is finite and within `tolL` of the exact `μ_e`. -/
theorem logit_bound {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform}
    (hsum : Device.SumDepth P) (i : Fin (launch d).size)
    {wv : Tensor [8 - 0, d.BLOCK] DType.f32.Val}
    (hwrd : ∀ (j : Idx [8 - 0, d.BLOCK]), decide (↑j.2.1 < d.hidden) = true →
      init.read DType.f32 (d.w.ptr ↑(gW d j)).addr = some (wv.get j))
    (hwpad : ∀ (j : Idx [8 - 0, d.BLOCK]), decide (↑j.2.1 < d.hidden) = false →
      ∀ (o : Tensor [8 - 0, d.BLOCK] DType.f32.Val), some (bc (c32 0)) = some o → wv.get j = o.get j)
    {xv : Tensor [d.BLOCK - 0] DType.bf16.Val}
    (hxrd : ∀ (j : Idx [d.BLOCK - 0]), decide (↑j.1 < d.hidden) = true →
      init.read DType.bf16 (d.x.ptr ↑(gX d (↑i) j)).addr = some (xv.get j))
    (hxpad : ∀ (j : Idx [d.BLOCK - 0]), decide (↑j.1 < d.hidden) = false →
      ∀ (o : Tensor [d.BLOCK - 0] DType.bf16.Val),
        some (bc (FloatV.litBF16 (FloatV.ofPat 0))) = some o → xv.get j = o.get j)
    {x32 : Tensor [d.BLOCK] (FloatV Format.binary32)}
    (hx32 : ∀ (j : Idx [d.BLOCK]), (FloatV.cvtWith RoundMode.rn (xv.get j)).NaNEq (x32.get j))
    {prod : Tensor [8 - 0, d.BLOCK] (FloatV Format.binary32)}
    (hprod : ∀ (j : Idx [8 - 0, d.BLOCK]),
      ((wv.get j).mul ((x32.row 8).get j)).NaNEq (prod.get j))
    {logits : Tensor [8 - 0] (FloatV Format.binary32)}
    (hlogits : ∀ (e : Fin (8 - 0)),
      ∃ t, Tree.Covers d.BLOCK t ∧
        t.depth ≤ tl.sumBound ((launch d).env P).P d.BLOCK ∧
          Tree.Eval (AddRel ((launch d).env P).fusion)
            (fun j => if h : j < d.BLOCK then prod.get2 e ⟨j, h⟩ else default) t (logits.get1 e))
    (e : Fin (8 - 0)) :
    (logits.get1 e).isFinite = true ∧
      |(logits.get1 e).toReal - mu init d (i : Nat) e| ≤ tolL d (absS init d (i : Nat) e) := by
  classical
  obtain ⟨t, hcov, hdepth, hev⟩ := hlogits e
  have hbs : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
  have he8 : (e : Nat) < 8 := by have := e.isLt; omega
  have hBLOCK : 0 < d.BLOCK := lt_of_lt_of_le hd.hidden_pos hd.hidden_le
  have hdt : t.depth ≤ Nat.clog 2 d.BLOCK :=
    depth_le_clog hsum hdepth
  have hD13 : Nat.clog 2 d.BLOCK ≤ 13 := clog_block_le d hd
  let u : ℚ := Format.binary32.u
  let η : ℚ := Format.binary32.eta
  let af : Nat → F32 := fun k =>
    if hk : k < d.BLOCK then wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) else default
  let bf : Nat → F32 := fun k =>
    if hk : k < d.BLOCK then (x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) else default
  let xf : Nat → F32 := fun k =>
    if hk : k < d.BLOCK then prod.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) else default
  have hev' : Tree.Eval (AddRel ((launch d).env P).fusion) xf t (logits.get1 e) := hev
  -- padded `w`
  have hapad : ∀ (k : Nat), d.hidden ≤ k → k < d.BLOCK → af k = c32 0 := by
    intro k hkL hk
    have hdec : decide (↑((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]).2.1 < d.hidden) = false := by
      show decide (k < d.hidden) = false
      simp [Nat.not_lt.mpr hkL]
    have h := hwpad ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) hdec (bc (c32 0)) rfl
    have h' : wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) = c32 0 := h
    have hafk : af k = wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
      simp only [af, dite_eq_left hk]
    rw [hafk, h']
  -- padded `x`
  have hxvpad : ∀ (k : Nat) (hk : k < d.BLOCK), d.hidden ≤ k →
      xv.get ((⟨k, hk⟩, ()) : Idx [d.BLOCK]) = (FloatV.ofPat 0 : BF16) := by
    intro k hk hkL
    have hdec : decide (↑((⟨k, hk⟩, ()) : Idx [d.BLOCK - 0]).1 < d.hidden) = false := by
      show decide (k < d.hidden) = false
      simp [Nat.not_lt.mpr hkL]
    have h := hxpad ((⟨k, hk⟩, ()) : Idx [d.BLOCK - 0]) hdec
      (bc (FloatV.litBF16 (FloatV.ofPat 0))) rfl
    have h' : xv.get ((⟨k, hk⟩, ()) : Idx [d.BLOCK]) = FloatV.litBF16 (FloatV.ofPat 0) := h
    rw [h', litBF16_zero]
  -- finiteness of the `x` row entries
  have hxvfin : ∀ (k : Nat) (hk : k < d.BLOCK),
      (xv.get ((⟨k, hk⟩, ()) : Idx [d.BLOCK])).isFinite = true := by
    intro k hk
    by_cases hkl : k < d.hidden
    · have hdec : decide (↑((⟨k, hk⟩, ()) : Idx [d.BLOCK - 0]).1 < d.hidden) = true := by
        show decide (k < d.hidden) = true
        simp [hkl]
      have hr : init.read DType.bf16 (d.x.ptr ((↑i * d.hidden + k : Nat) : Int)).addr
          = some (xv.get ((⟨k, hk⟩, ()) : Idx [d.BLOCK])) :=
        hxrd ((⟨k, hk⟩, ()) : Idx [d.BLOCK - 0]) hdec
      obtain ⟨v, hv, hvf⟩ := hpre.2.2.2.2.1 (i : Nat) k hbs hkl
      rw [hv] at hr
      exact (Option.some.inj hr) ▸ hvf
    · have hkL : d.hidden ≤ k := by omega
      rw [hxvpad k hk hkL]
      exact zero_bf16_fin
  -- finiteness of the `w` entries
  have hwvfin : ∀ (k : Nat) (hk : k < d.BLOCK),
      (wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])).isFinite = true := by
    intro k hk
    by_cases hkl : k < d.hidden
    · have hdec : decide (↑((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]).2.1 < d.hidden) = true := by
        show decide (k < d.hidden) = true
        simp [hkl]
      have hr : init.read DType.f32 (d.w.ptr ((↑e * d.hidden + k : Nat) : Int)).addr
          = some (wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])) :=
        hwrd ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) hdec
      obtain ⟨v, hv, hvf⟩ := hpre.2.2.2.2.2.1 (e : Nat) k he8 hkl
      rw [hv] at hr
      exact (Option.some.inj hr) ▸ hvf
    · have hkL : d.hidden ≤ k := by omega
      have hafk : af k = wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
        simp only [af, dite_eq_left hk]
      rw [← hafk, hapad k hkL hk]
      exact c32_zero_fin
  -- the conversion is the `x32` value
  have hx32eq : ∀ (k : Nat) (hk : k < d.BLOCK),
      (x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])
        = FloatV.cvtWith RoundMode.rn (xv.get ((⟨k, hk⟩, ()) : Idx [d.BLOCK])) := by
    intro k hk
    have hcvt := FloatV.toRat_cvt_of_wider (f := Format.bfloat16)
      (g := Format.binary32) Format.bfloat16_regular
      Format.binary32_regular Format.wider_bfloat16_binary32
      RoundMode.rn (hxvfin k hk)
    exact (hx32 ((⟨k, hk⟩, ()) : Idx [d.BLOCK])).eq
      (Format.not_isNaN_of_isFinite hcvt.1)
  -- `bf` finiteness
  have hbfin : ∀ (k : Nat) (hk : k < d.BLOCK), (bf k).isFinite = true := by
    intro k hk
    have hbfk : bf k = (x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
      simp only [bf, dite_eq_left hk]
    rw [hbfk, hx32eq k hk]
    exact (FloatV.toRat_cvt_of_wider (f := Format.bfloat16)
      (g := Format.binary32) Format.bfloat16_regular
      Format.binary32_regular Format.wider_bfloat16_binary32
      RoundMode.rn (hxvfin k hk)).1
  -- `af` finiteness
  have hafin : ∀ (k : Nat) (hk : k < d.BLOCK), (af k).isFinite = true := by
    intro k hk
    have hafk : af k = wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
      simp only [af, dite_eq_left hk]
    rw [hafk]
    exact hwvfin k hk
  -- `af` toRat for padded lanes
  have hafpad : ∀ (k : Nat) (hk : k < d.BLOCK), d.hidden ≤ k → (af k).toRat = 0 := by
    intro k hk hkL
    rw [hapad k hkL hk]
    exact c32_zero_rat
  -- `bf` toRat for padded lanes
  have hbfpad : ∀ (k : Nat) (hk : k < d.BLOCK), d.hidden ≤ k → (bf k).toRat = 0 := by
    intro k hk hkL
    have hbfk : bf k = (x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
      simp only [bf, dite_eq_left hk]
    have hcvt := FloatV.toRat_cvt_of_wider (f := Format.bfloat16)
      (g := Format.binary32) Format.bfloat16_regular
      Format.binary32_regular Format.wider_bfloat16_binary32
      RoundMode.rn (hxvfin k hk)
    rw [hbfk, hx32eq k hk, hcvt.2, hxvpad k hk hkL]
    exact zero_bf16_rat
  -- the `x` value matches `bf`
  have hxval : ∀ (k : Nat) (hk : k < d.BLOCK), k < d.hidden →
      xVal init d (i : Nat) k = ((bf k).toRat : ℝ) := by
    intro k hk hkl
    have hdec : decide (↑((⟨k, hk⟩, ()) : Idx [d.BLOCK - 0]).1 < d.hidden) = true := by
      show decide (k < d.hidden) = true
      simp [hkl]
    have hr : init.read DType.bf16 (d.x.ptr ((↑i * d.hidden + k : Nat) : Int)).addr
        = some (xv.get ((⟨k, hk⟩, ()) : Idx [d.BLOCK])) :=
      hxrd ((⟨k, hk⟩, ()) : Idx [d.BLOCK - 0]) hdec
    have hbfk : bf k = (x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
      simp only [bf, dite_eq_left hk]
    have hcvt := FloatV.toRat_cvt_of_wider (f := Format.bfloat16)
      (g := Format.binary32) Format.bfloat16_regular
      Format.binary32_regular Format.wider_bfloat16_binary32
      RoundMode.rn (hxvfin k hk)
    rw [hbfk, hx32eq k hk, hcvt.2]
    exact xVal_eq hr
  -- the `w` value matches `af`
  have hwval : ∀ (k : Nat) (hk : k < d.BLOCK), k < d.hidden →
      wVal init d (e : Nat) k = ((af k).toRat : ℝ) := by
    intro k hk hkl
    have hdec : decide (↑((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]).2.1 < d.hidden) = true := by
      show decide (k < d.hidden) = true
      simp [hkl]
    have hr : init.read DType.f32 (d.w.ptr ((↑e * d.hidden + k : Nat) : Int)).addr
        = some (wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])) :=
      hwrd ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) hdec
    have hafk : af k = wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
      simp only [af, dite_eq_left hk]
    rw [hafk]
    exact wVal_eq hr
  -- ## Stage C: the reduction's error
  have hu_pos : (0 : ℚ) < u := by
    show (0 : ℚ) < Format.binary32.u
    exact bin32_u_pos
  have hu_le : u ≤ 1 / 16 := by
    show Format.binary32.u ≤ 1 / 16
    exact bin32_u_le
  have hu_nn : (0 : ℚ) ≤ u := le_of_lt hu_pos
  have heta_pos : (0 : ℚ) < η := by
    show (0 : ℚ) < Format.binary32.eta
    rw [Format.binary32_eta]
    norm_num
  have heta_nn : (0 : ℚ) ≤ η := le_of_lt heta_pos
  have heta150 : η ≤ (2 : ℚ) ^ (-150 : ℤ) := by
    show Format.binary32.eta ≤ (2 : ℚ) ^ (-150 : ℤ)
    rw [Format.binary32_eta]
  have hSle : absS init d (i : Nat) (e : Nat) ≤ (2 : ℝ) ^ 120 :=
    (hpre.2.2.2.2.2.2 (i : Nat) (e : Nat) hbs he8).1
  -- every lane's product is below `binary32`'s largest value
  have hprod_mag : ∀ (k : Nat) (hk : k < d.BLOCK),
      |(af k).toRat * (bf k).toRat| ≤ Format.binary32.maxVal := by
    intro k hk
    by_cases hkl : k < d.hidden
    · have hcast : ((|(af k).toRat * (bf k).toRat| : ℚ) : ℝ)
          = |xVal init d (i : Nat) k * wVal init d (e : Nat) k| := by
        rw [Rat.cast_abs, Rat.cast_mul, ← hwval k hk hkl, ← hxval k hk hkl, mul_comm]
      have hsingle : |xVal init d (i : Nat) k * wVal init d (e : Nat) k|
          ≤ absS init d (i : Nat) (e : Nat) := by
        unfold absS
        exact Finset.single_le_sum
          (f := fun h => |xVal init d (i : Nat) h * wVal init d (e : Nat) h|)
          (fun h _ => abs_nonneg _) (Finset.mem_range.mpr hkl)
      refine Rat.cast_le (K := ℝ).mp ?_
      rw [hcast]
      calc |xVal init d (i : Nat) k * wVal init d (e : Nat) k|
          ≤ absS init d (i : Nat) (e : Nat) := hsingle
        _ ≤ (2 : ℝ) ^ 120 := hSle
        _ ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
            have h := Rat.cast_le (K := ℝ).mpr maxVal_ge_120
            have h2 : (((2 : ℚ) ^ 120 : ℚ) : ℝ) = (2 : ℝ) ^ 120 := by push_cast; ring
            rwa [h2] at h
    · have hkL : d.hidden ≤ k := by omega
      rw [hafpad k hk hkL, zero_mul, abs_zero]
      exact le_of_lt maxVal_pos_q
  -- each product's rounding error
  have hperr : ∀ (k : Nat) (hk : k < d.BLOCK),
      (prod.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])).isFinite = true ∧
      |(prod.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])).toRat
          - (af k).toRat * (bf k).toRat| ≤ |(af k).toRat * (bf k).toRat| * u + η := by
    intro k hk
    have hafk : af k = wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
      simp only [af, dite_eq_left hk]
    have hbfk : bf k = (x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
      simp only [bf, dite_eq_left hk]
    have hmul := FloatV.mul_err (f := Format.binary32) bin32_regular
      (hafin k hk) (hbfin k hk) (hprod_mag k hk)
    have hmul2 : ((wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])).mul
          ((x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]))).isFinite = true ∧
        |((wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])).mul
            ((x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]))).toRat
          - (af k).toRat * (bf k).toRat| ≤ |(af k).toRat * (bf k).toRat| * u + η := by
      have h' : ((af k).mul (bf k)) =
          ((wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])).mul
            ((x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]))) := by
        rw [hafk, hbfk]
      rw [h'] at hmul
      exact hmul
    have hne := (hprod ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])).eq
      (Format.not_isNaN_of_isFinite hmul2.1)
    exact ⟨by rw [hne]; exact hmul2.1, by rw [hne]; exact hmul2.2⟩
  -- padded product lanes are zero
  have hxfpad : ∀ (k : Nat) (hk : k < d.BLOCK), d.hidden ≤ k → (xf k).toRat = 0 := by
    intro k hk hkL
    have hxfk : xf k = prod.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
      simp only [xf, dite_eq_left hk]
    have hafk : af k = wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
      simp only [af, dite_eq_left hk]
    have hbfk : bf k = (x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
      simp only [bf, dite_eq_left hk]
    have hmul := mul_toRat_eq_zero (f := Format.binary32) bin32_regular
      (hafpad k hk hkL) (hbfin k hk)
    have h' : ((af k).mul (bf k)) =
        ((wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])).mul
          ((x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]))) := by
      rw [hafk, hbfk]
    have hmul2 : ((wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])).mul
        ((x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]))).toRat = 0 := by
      rw [h'] at hmul
      exact hmul
    have hfin : ((wv.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])).mul
        ((x32.row 8).get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]))).isFinite = true := by
      rw [← h']
      exact (FloatV.mul_err (f := Format.binary32) bin32_regular
        (hafin k hk) (hbfin k hk) (hprod_mag k hk)).1
    have hne := (hprod ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK])).eq
      (Format.not_isNaN_of_isFinite hfin)
    rw [hxfk, hne, hmul2]
  have hxffin : ∀ (k : Nat), k < d.BLOCK → (xf k).isFinite = true := by
    intro k hk
    have hxfk : xf k = prod.get ((e, (⟨k, hk⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
      simp only [xf, dite_eq_left hk]
    rw [hxfk]
    exact (hperr k hk).1
  -- the aggregates
  let AP : ℚ := ∑ k ∈ Finset.range d.BLOCK, |(xf k).toRat|
  let Ps : ℚ := ∑ k ∈ Finset.range d.BLOCK, (xf k).toRat
  let BL : ℚ := ∑ k ∈ Finset.range d.hidden, |(af k).toRat * (bf k).toRat|
  let Sl : ℚ := ∑ k ∈ Finset.range d.hidden, (af k).toRat * (bf k).toRat
  let A' : ℚ := (1 + u) * BL + ((d.hidden : ℕ) : ℚ) * η
  have hAPlist : (List.map (fun k => |(xf k).toRat|) (List.range d.BLOCK)).sum = AP := by
    rw [list_sum_range_congr]
  have hPslist : (List.map (fun k => (xf k).toRat) (List.range d.BLOCK)).sum = Ps := by
    rw [list_sum_range_congr]
  have hAPsplit : AP = ∑ k ∈ Finset.range d.hidden, |(xf k).toRat| := by
    show (∑ k ∈ Finset.range d.BLOCK, |(xf k).toRat|) = _
    rw [finset_sum_range_eq_of_zero (g := fun k => |(xf k).toRat|) hd.hidden_le
      (fun k hkL hk => by rw [hxfpad k hk hkL, abs_zero])]
  have hPssplit : Ps = ∑ k ∈ Finset.range d.hidden, (xf k).toRat := by
    show (∑ k ∈ Finset.range d.BLOCK, (xf k).toRat) = _
    rw [finset_sum_range_eq_of_zero (g := fun k => (xf k).toRat) hd.hidden_le
      (fun k hkL hk => hxfpad k hk hkL)]
  have hAPnn : (0 : ℚ) ≤ AP := by
    show (0 : ℚ) ≤ ∑ k ∈ Finset.range d.BLOCK, |(xf k).toRat|
    exact Finset.sum_nonneg (fun k _ => abs_nonneg _)
  have hBLnn : (0 : ℚ) ≤ BL := by
    show (0 : ℚ) ≤ ∑ k ∈ Finset.range d.hidden, |(af k).toRat * (bf k).toRat|
    exact Finset.sum_nonneg (fun k _ => abs_nonneg _)
  -- the products' total rounding error
  have hPle : |Ps - Sl| ≤ u * BL + ((d.hidden : ℕ) : ℚ) * η := by
    have hsub : Ps - Sl = ∑ k ∈ Finset.range d.hidden,
        ((xf k).toRat - (af k).toRat * (bf k).toRat) := by
      rw [hPssplit]
      show (∑ k ∈ Finset.range d.hidden, (xf k).toRat) - Sl = _
      rw [show Sl = ∑ k ∈ Finset.range d.hidden, (af k).toRat * (bf k).toRat from rfl]
      rw [Finset.sum_sub_distrib]
    rw [hsub]
    refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
    have hterm : ∀ k ∈ Finset.range d.hidden,
        |(xf k).toRat - (af k).toRat * (bf k).toRat|
          ≤ |(af k).toRat * (bf k).toRat| * u + η := by
      intro k hk
      have hkl : k < d.hidden := Finset.mem_range.mp hk
      have hk' : k < d.BLOCK := lt_of_lt_of_le hkl hd.hidden_le
      have hxfk : xf k = prod.get ((e, (⟨k, hk'⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
        simp only [xf, dite_eq_left hk']
      rw [hxfk]
      exact (hperr k hk').2
    refine le_trans (Finset.sum_le_sum hterm) ?_
    have hsplit : (∑ k ∈ Finset.range d.hidden, (|(af k).toRat * (bf k).toRat| * u + η))
        = (∑ k ∈ Finset.range d.hidden, |(af k).toRat * (bf k).toRat|) * u
          + ((d.hidden : ℕ) : ℚ) * η := by
      rw [Finset.sum_add_distrib, Finset.sum_mul, Finset.sum_const, Finset.card_range,
        nsmul_eq_mul]
    rw [hsplit]
    show (∑ k ∈ Finset.range d.hidden, |(af k).toRat * (bf k).toRat|) * u
        + ((d.hidden : ℕ) : ℚ) * η ≤ u * BL + ((d.hidden : ℕ) : ℚ) * η
    rw [show BL = ∑ k ∈ Finset.range d.hidden, |(af k).toRat * (bf k).toRat| from rfl,
      mul_comm u]
  -- the sum of the leaves' magnitudes
  have hAPle : AP ≤ A' := by
    have hterm : ∀ k ∈ Finset.range d.hidden,
        |(xf k).toRat| ≤ (1 + u) * |(af k).toRat * (bf k).toRat| + η := by
      intro k hk
      have hkl : k < d.hidden := Finset.mem_range.mp hk
      have hk' : k < d.BLOCK := lt_of_lt_of_le hkl hd.hidden_le
      have hxfk : xf k = prod.get ((e, (⟨k, hk'⟩, ())) : Idx [8 - 0, d.BLOCK]) := by
        simp only [xf, dite_eq_left hk']
      have hper := (hperr k hk').2
      rw [← hxfk] at hper
      calc |(xf k).toRat|
          = |((xf k).toRat - (af k).toRat * (bf k).toRat) + (af k).toRat * (bf k).toRat| := by
            rw [sub_add_cancel]
        _ ≤ |(xf k).toRat - (af k).toRat * (bf k).toRat| + |(af k).toRat * (bf k).toRat| :=
            abs_add_le _ _
        _ ≤ (|(af k).toRat * (bf k).toRat| * u + η) + |(af k).toRat * (bf k).toRat| := by
            linarith
        _ = (1 + u) * |(af k).toRat * (bf k).toRat| + η := by ring
    calc AP = ∑ k ∈ Finset.range d.hidden, |(xf k).toRat| := hAPsplit
      _ ≤ ∑ k ∈ Finset.range d.hidden, ((1 + u) * |(af k).toRat * (bf k).toRat| + η) :=
          Finset.sum_le_sum hterm
      _ = A' := by
          show _ = (1 + u) * BL + ((d.hidden : ℕ) : ℚ) * η
          rw [show BL = ∑ k ∈ Finset.range d.hidden, |(af k).toRat * (bf k).toRat| from rfl]
          rw [Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_const, Finset.card_range,
            nsmul_eq_mul]
  -- the aggregates, cast to `ℝ`
  have hBLcast : ((∑ k ∈ Finset.range d.hidden, |(af k).toRat * (bf k).toRat| : ℚ) : ℝ)
      = absS init d (i : Nat) (e : Nat) := by
    unfold absS
    rw [Rat.cast_sum]
    refine Finset.sum_congr rfl ?_
    intro k hk
    have hkl : k < d.hidden := Finset.mem_range.mp hk
    have hk' : k < d.BLOCK := lt_of_lt_of_le hkl hd.hidden_le
    rw [Rat.cast_abs, Rat.cast_mul, ← hwval k hk' hkl, ← hxval k hk' hkl, mul_comm]
  have hSlcast : ((∑ k ∈ Finset.range d.hidden, (af k).toRat * (bf k).toRat : ℚ) : ℝ)
      = mu init d (i : Nat) (e : Nat) := by
    unfold mu
    rw [Rat.cast_sum]
    refine Finset.sum_congr rfl ?_
    intro k hk
    have hkl : k < d.hidden := Finset.mem_range.mp hk
    have hk' : k < d.BLOCK := lt_of_lt_of_le hkl hd.hidden_le
    rw [Rat.cast_mul, ← hwval k hk' hkl, ← hxval k hk' hkl, mul_comm]
  -- growth of the reduction tree
  have hG : (1 + 3 * u) ^ Nat.clog 2 d.BLOCK ≤ 3 := by
    have hu : u = (2 : ℚ) ^ (-24 : ℤ) := by
      show Format.binary32.u = (2 : ℚ) ^ (-24 : ℤ)
      exact Format.binary32_u
    rw [hu]
    calc (1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ Nat.clog 2 d.BLOCK
        ≤ (1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ 13 := pow_le_pow_right₀ (by norm_num) hD13
      _ ≤ 3 := by norm_num
  -- the size of the leaves
  have hAP121 : AP ≤ (2 : ℚ) ^ 121 := by
    have hBLle : BL ≤ (2 : ℚ) ^ 120 := by
      have h2 : ((BL : ℚ) : ℝ) ≤ (((2 : ℚ) ^ 120 : ℚ) : ℝ) := by
        show ((∑ k ∈ Finset.range d.hidden, |(af k).toRat * (bf k).toRat| : ℚ) : ℝ) ≤ _
        rw [hBLcast, show (((2 : ℚ) ^ 120 : ℚ) : ℝ) = (2 : ℝ) ^ 120 by push_cast; ring]
        exact hSle
      exact Rat.cast_le (K := ℝ).mp h2
    have h1u : (1 + u) ≤ 1 + 1 / 16 := by linarith
    have hstep1 : (1 + u) * BL ≤ (1 + 1 / 16) * (2 : ℚ) ^ 120 :=
      mul_le_mul h1u hBLle hBLnn (by norm_num)
    have hL17 : ((d.hidden : ℕ) : ℚ) ≤ 2 ^ 17 := by
      have := hd.hidden_le17
      exact_mod_cast this
    have hstep2 : ((d.hidden : ℕ) : ℚ) * η ≤ (2 : ℚ) ^ 17 * (2 : ℚ) ^ (-150 : ℤ) :=
      mul_le_mul hL17 heta150 heta_nn (by norm_num)
    have hnum : (1 + 1 / 16) * (2 : ℚ) ^ 120 + (2 : ℚ) ^ 17 * (2 : ℚ) ^ (-150 : ℤ)
        ≤ (2 : ℚ) ^ 121 := by norm_num
    have hA'le : A' ≤ (2 : ℚ) ^ 121 := by
      show (1 + u) * BL + ((d.hidden : ℕ) : ℚ) * η ≤ (2 : ℚ) ^ 121
      have h := add_le_add hstep1 hstep2
      exact le_trans h hnum
    exact le_trans hAPle hA'le
  -- the reduction's error bound
  have hE : Tree.errBound (3 * u) (4 * η) (Nat.clog 2 d.BLOCK) (d.BLOCK - 1) AP
      ≤ (2 : ℚ) ^ 122 + (2 : ℚ) ^ 120 := by
    show ((1 + 3 * u) ^ Nat.clog 2 d.BLOCK - 1) * AP +
        4 * η * ((d.BLOCK - 1 : ℕ) : ℚ) * (1 + 3 * u) ^ Nat.clog 2 d.BLOCK
      ≤ (2 : ℚ) ^ 122 + (2 : ℚ) ^ 120
    have hG1 : (1 : ℚ) ≤ (1 + 3 * u) ^ Nat.clog 2 d.BLOCK := one_le_pow₀ (by linarith)
    have hG0 : (0 : ℚ) ≤ (1 + 3 * u) ^ Nat.clog 2 d.BLOCK := le_trans (by norm_num) hG1
    have hGm1 : (1 + 3 * u) ^ Nat.clog 2 d.BLOCK - 1 ≤ 2 := by linarith
    have hNle : ((d.BLOCK - 1 : ℕ) : ℚ) ≤ 2 ^ 13 := by
      have h1 : d.BLOCK - 1 ≤ d.BLOCK := Nat.sub_le _ _
      have h2 : ((d.BLOCK : ℕ) : ℚ) ≤ 2 ^ 13 := by
        have := hd.block_le
        exact_mod_cast this
      have h3 : ((d.BLOCK - 1 : ℕ) : ℚ) ≤ ((d.BLOCK : ℕ) : ℚ) := by exact_mod_cast h1
      linarith
    have hNnn : (0 : ℚ) ≤ ((d.BLOCK - 1 : ℕ) : ℚ) := by positivity
    have ht1 : ((1 + 3 * u) ^ Nat.clog 2 d.BLOCK - 1) * AP ≤ (2 : ℚ) ^ 122 := by
      calc ((1 + 3 * u) ^ Nat.clog 2 d.BLOCK - 1) * AP ≤ 2 * (2 : ℚ) ^ 121 :=
            mul_le_mul hGm1 hAP121 hAPnn (by norm_num)
        _ = (2 : ℚ) ^ 122 := by norm_num
    have ht2 : 4 * η * ((d.BLOCK - 1 : ℕ) : ℚ) * (1 + 3 * u) ^ Nat.clog 2 d.BLOCK
        ≤ (2 : ℚ) ^ 120 := by
      have hNG : ((d.BLOCK - 1 : ℕ) : ℚ) * (1 + 3 * u) ^ Nat.clog 2 d.BLOCK
          ≤ 2 ^ 13 * 3 := mul_le_mul hNle hG hG0 (by norm_num)
      have h4 : 4 * η ≤ 4 * (2 : ℚ) ^ (-150 : ℤ) := by linarith
      have hle : 4 * η * (((d.BLOCK - 1 : ℕ) : ℚ) * (1 + 3 * u) ^ Nat.clog 2 d.BLOCK)
          ≤ 4 * (2 : ℚ) ^ (-150 : ℤ) * (2 ^ 13 * 3) :=
        mul_le_mul h4 hNG (mul_nonneg hNnn hG0) (by norm_num)
      have hnum : (4 : ℚ) * (2 : ℚ) ^ (-150 : ℤ) * (2 ^ 13 * 3) ≤ (2 : ℚ) ^ 120 := by
        norm_num
      calc 4 * η * ((d.BLOCK - 1 : ℕ) : ℚ) * (1 + 3 * u) ^ Nat.clog 2 d.BLOCK
          = 4 * η * (((d.BLOCK - 1 : ℕ) : ℚ) * (1 + 3 * u) ^ Nat.clog 2 d.BLOCK) := by ring
        _ ≤ 4 * (2 : ℚ) ^ (-150 : ℤ) * (2 ^ 13 * 3) := hle
        _ ≤ (2 : ℚ) ^ 120 := hnum
    exact add_le_add ht1 ht2
  -- the lemma's side condition
  have hM : (List.map (fun k => |(xf k).toRat|) (List.range d.BLOCK)).sum +
      Tree.errBound (3 * u) (4 * η) (Nat.clog 2 d.BLOCK) (d.BLOCK - 1)
        (List.map (fun k => |(xf k).toRat|) (List.range d.BLOCK)).sum
      ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    simp only [hAPlist]
    have hmax : (2 : ℚ) ^ 121 + ((2 : ℚ) ^ 122 + (2 : ℚ) ^ 120)
        ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
      rw [Format.binary32_maxVal, Format.binary32_eta]
      norm_num
    linarith [hAP121, hE, hmax]
  -- apply the reduction's error lemma
  have hres := sumc_err_depth (f := Format.binary32) bin32_regular bin32_specials
    bin32_u_le ((launch d).env P).fusion xf hcov hdt hev' hxffin hM
  have hres1 := hres.1
  have hres2 := hres.2
  simp only [hAPlist, hPslist] at hres2
  refine ⟨hres1, ?_⟩
  have h1 : |(logits.get1 e).toReal - (Ps : ℝ)| ≤
      ((Tree.errBound (3 * u) (4 * η) (Nat.clog 2 d.BLOCK) (d.BLOCK - 1) AP : ℚ) : ℝ) := by
    have hc : ((|(logits.get1 e).toRat - Ps| : ℚ) : ℝ) = |(logits.get1 e).toReal - (Ps : ℝ)| := by
      simp only [Rat.cast_abs, Rat.cast_sub]
      rfl
    rw [← hc]
    exact Rat.cast_le (K := ℝ).mpr hres2
  have h2 : |(Ps : ℝ) - mu init d (i : Nat) (e : Nat)| ≤
      ((u * BL + ((d.hidden : ℕ) : ℚ) * η : ℚ) : ℝ) := by
    have h := Rat.cast_le (K := ℝ).mpr hPle
    have hc : ((|Ps - Sl| : ℚ) : ℝ) = |(Ps : ℝ) - (Sl : ℝ)| := by
      simp only [Rat.cast_abs, Rat.cast_sub]
    rw [hc] at h
    have hsl : ((Sl : ℚ) : ℝ) = mu init d (i : Nat) (e : Nat) := by
      show ((∑ k ∈ Finset.range d.hidden, (af k).toRat * (bf k).toRat : ℚ) : ℝ) = _
      exact hSlcast
    rw [hsl] at h
    exact h
  have h3 : ((Tree.errBound (3 * u) (4 * η) (Nat.clog 2 d.BLOCK) (d.BLOCK - 1) AP : ℚ) : ℝ) ≤
      ((Tree.errBound (3 * u) (4 * η) (Nat.clog 2 d.BLOCK) (d.BLOCK - 1) A' : ℚ) : ℝ) := by
    exact Rat.cast_le (K := ℝ).mpr (Tree.errBound_mono (δ := 3 * u) (η := 4 * η)
      (d := Nat.clog 2 d.BLOCK) (d' := Nat.clog 2 d.BLOCK)
      (N := d.BLOCK - 1) (N' := d.BLOCK - 1) (A := AP) (A' := A')
      (by linarith) (by linarith) (le_refl _) (le_refl _) hAPnn hAPle)
  have h4 : ((Tree.errBound (3 * u) (4 * η) (Nat.clog 2 d.BLOCK) (d.BLOCK - 1) A' : ℚ) : ℝ) +
      ((u * BL + ((d.hidden : ℕ) : ℚ) * η : ℚ) : ℝ) =
      tolL d (absS init d (i : Nat) (e : Nat)) := by
    rw [errBound_cast]
    simp only [A', BL, u, η]
    push_cast
    have hBLR : (∑ k ∈ Finset.range d.hidden,
        |((af k).toRat : ℝ) * ((bf k).toRat : ℝ)|)
        = absS init d (i : Nat) (e : Nat) := by
      rw [← hBLcast, Rat.cast_sum]
      refine Finset.sum_congr rfl ?_
      intro k hk
      rw [Rat.cast_abs, Rat.cast_mul]
    rw [hBLR, u32_eq, eta32_eq]
    unfold tolL
    push_cast
    ring
  have htri : |(logits.get1 e).toReal - mu init d (i : Nat) (e : Nat)| ≤
      |(logits.get1 e).toReal - (Ps : ℝ)| + |(Ps : ℝ) - mu init d (i : Nat) (e : Nat)| := by
    have hab := abs_add_le ((logits.get1 e).toReal - (Ps : ℝ))
      ((Ps : ℝ) - mu init d (i : Nat) (e : Nat))
    rwa [show (logits.get1 e).toReal - (Ps : ℝ) + ((Ps : ℝ) - mu init d (i : Nat) (e : Nat))
        = (logits.get1 e).toReal - mu init d (i : Nat) (e : Nat) from by ring] at hab
  linarith [h1, h2, h3, h4, htri]

end Claims.MoeRouter
