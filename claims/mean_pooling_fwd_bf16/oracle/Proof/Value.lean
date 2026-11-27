import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Num

/-!
# The value of one output element

`lane_value`: a column's leaves (the chunk's `L` inputs widened to `float32`, then `+0`), summed
along a tree of depth `log₂ BT`, divided by the exact `float32` `L` with the platform's division,
and rounded to `bfloat16`, give an element within `tol` of the chunk's mean (`Num.mean_err`).
-/

namespace Claims.MeanPoolingFwd

open TVA

/-- A list sum over `range n` is the `Finset` sum. -/
theorem list_sum_range (f : ℕ → ℚ) (n : ℕ) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ, ih]; simp

/-- A sum over `range n` of a function that vanishes from `L ≤ n` on is the sum over `range L`. -/
theorem sum_range_trunc (f : ℕ → ℚ) {L n : ℕ} (hL : L ≤ n) (h0 : ∀ r, L ≤ r → r < n → f r = 0) :
    ∑ r ∈ Finset.range n, f r = ∑ r ∈ Finset.range L, f r := by
  induction n with
  | zero => have : L = 0 := by omega
            subst this; rfl
  | succ n ih =>
    rcases Nat.eq_or_lt_of_le hL with h | h
    · rw [h]
    · rw [Finset.sum_range_succ, h0 n (by omega) (by omega), add_zero]
      exact ih (by omega) fun r h1 h2 => h0 r h1 (by omega)

/-- A natural number below `2^24` converts to `float32` exactly. -/
theorem ofRat_nat_f32 {z : ℤ} (hz : |z| < 2 ^ 24) :
    (FloatV.ofRat (z : ℚ) : F32).isFinite = true ∧ (FloatV.ofRat (z : ℚ) : F32).toRat = z := by
  have hf := Format.binary32_regular
  have hq : |(z : ℚ)| ≤ Format.binary32.maxVal := by
    have : |(z : ℚ)| < 2 ^ 24 := by exact_mod_cast hz
    rw [Format.binary32_maxVal]; norm_num; linarith
  unfold FloatV.ofRat
  by_cases h0 : (z : ℚ) = 0
  · obtain ⟨h1, h2, h3⟩ := Format.roundRat_zero_spec (f := Format.binary32) (rnd := .rn)
      (sat := false) (zs := false) hf
    rw [h0]
    refine ⟨?_, ?_⟩
    · unfold FloatV.isFinite; rw [FloatV.pat_ofPat h1]; exact h2
    · rw [FloatV.toRat_ofPat h1, h3]
  obtain ⟨h1, h2, h3⟩ := Format.roundRat_spec hf .rn false false h0 hq
  refine ⟨?_, ?_⟩
  · unfold FloatV.isFinite; rw [FloatV.pat_ofPat h1]; exact h2
  rw [FloatV.toRat_ofPat h1, h3]
  have ha : 0 < |(z : ℚ)| := abs_pos.mpr h0
  have hlog : Int.log 2 |(z : ℚ)| ≤ 23 := by
    have : |(z : ℚ)| < (2 : ℚ) ^ (24 : ℤ) := by exact_mod_cast hz
    have := Format.log_lt_of_lt ha this
    omega
  have hQ : Format.binary32.qexp |(z : ℚ)| ≤ 0 := by
    rw [Format.qexp_eq ha]
    have : Format.binary32.emin = -126 := by decide
    have hm : (Format.binary32.m : ℤ) = 23 := rfl
    rw [this, hm]; omega
  set Q := Format.binary32.qexp |(z : ℚ)| with hQdef
  have hK : |(z : ℚ)| * (2 : ℚ) ^ (-Q) = ((z.natAbs * 2 ^ (-Q).toNat : ℕ) : ℚ) := by
    push_cast
    rw [← zpow_natCast, Int.toNat_of_nonneg (by omega)]
    congr 1
    rw [Nat.cast_natAbs, Int.cast_abs]
  rw [Format.roundSig_exact .rn hK]
  by_cases hneg : (z : ℚ) < 0
  · rw [ite_eq_left hneg, abs_of_neg hneg]; ring
  · rw [ite_eq_right hneg, abs_of_nonneg (not_lt.mp hneg), one_mul]

/-- **The divisor as a `float32`** (`tl.itof` of the `int64` `L`): exact for `L < 2^24`. -/
theorem itof_exact {L : ℕ} (hL : L < 2 ^ 24) {r : F32} (h : tl.ItofRel .binary32 (L : ℤ) r) :
    r.isFinite = true ∧ r.toRat = L := by
  have hz : |(L : ℤ)| < 2 ^ 53 := by
    rw [abs_of_nonneg (by positivity)]; have : (L : ℤ) < 2 ^ 24 := by exact_mod_cast hL
    linarith
  rw [tl.itofRel_binary32 hz] at h
  subst h
  obtain ⟨hfin, hval⟩ := ofRat_nat_f32 (z := L) (by
    rw [abs_of_nonneg (by positivity)]; exact_mod_cast hL)
  exact ⟨hfin, by rw [hval]; simp⟩

/-- `+0` holds `0`. -/
theorem zero_toRat : (FloatV.ofPat 0 : F32).toRat = 0 := by
  apply FloatV.toRat_of_isZero
  rw [FloatV.pat_ofPat (by decide)]
  exact Format.isZero_zero

/-- `+0` is finite. -/
theorem zero_isFinite : (FloatV.ofPat 0 : F32).isFinite = true := by
  rw [FloatV.isFinite_ofPat (by decide)]
  decide

/-- **One output element.** The leaves `y r`, `r < BT`, are the chunk's inputs `x[start + r]`
widened (`r < L`) and `+0` (`r ≥ L`), up to NaN payload; `s` is their sum along a tree covering
`BT` leaves of depth at most `log₂ BT`; `L32` is the exact `float32` `L`; `q` a platform quotient
of `s` by `L32`; `c` the `bfloat16` rounding of `q`. When the element is in range, `c` is finite
and within `tol` of the mean. -/
theorem lane_value {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < gx d * gy d) {P : Platform} (hP : Device P) (j : Nat)
    {y : Nat → F32}
    (hy1 : ∀ r, r < Lof d i →
      FloatV.NaNEq (FloatV.cvt (xAt d (start d (outRow d i) + r) (hOf d i) j)) (y r))
    (hy0 : ∀ r, Lof d i ≤ r → r < d.BT → FloatV.NaNEq (FloatV.ofPat 0) (y r))
    {t : Tree} (hcov : t.Covers d.BT) (hdep : t.depth ≤ Nat.log 2 d.BT) {s : F32}
    (hs : t.Eval (AddRel false) y s)
    {L32 : F32} (hL32f : L32.isFinite = true) (hL32 : L32.toRat = (Lof d i : ℚ))
    {q : F32} (hq : P.div .binary32 s L32 q)
    {c : BF16} (hc : FloatV.NaNEq (FloatV.cvtWith .rn q) c)
    (hin : InRange d (outRow d i) (hOf d i) j) :
    c.isFinite = true ∧
      |c.toRat - mean d (outRow d i) (hOf d i) j| ≤
        tol d.BT (mean d (outRow d i) (hOf d i) j) (absMean d (outRow d i) (hOf d i) j) := by
  obtain ⟨hL1, hLB⟩ := Lof_bounds hd hT hi
  obtain ⟨hlen, hstop⟩ := len_eq hd hT hi
  set g := outRow d i with hg
  set st := start d g with hst
  set L := Lof d i with hLdef
  obtain ⟨hfinx, hAle⟩ := hin
  -- the leaves
  have hxfin : ∀ r, r < L → (xAt d (st + r) (hOf d i) j).isFinite = true := fun r hr =>
    hfinx _ (by rw [Finset.mem_Ico, hstop]; omega)
  have hyv : ∀ r, r < d.BT → (y r).isFinite = true ∧
      (y r).toRat = if r < L then (xAt d (st + r) (hOf d i) j).toRat else 0 := by
    intro r hr
    by_cases hrL : r < L
    · obtain ⟨h1, h2, -⟩ := FloatV.cvt_bf16_f32_exact (hxfin r hrL)
      obtain ⟨h3, h4, -⟩ := (hy1 r hrL).exact_of_isFinite h1
      exact ⟨h3, by rw [ite_eq_left hrL, h4, h2]⟩
    · obtain ⟨h3, h4, -⟩ := (hy0 r (by omega) hr).exact_of_isFinite zero_isFinite
      exact ⟨h3, by rw [ite_eq_right hrL, h4, zero_toRat]⟩
  have hSum : ((List.range d.BT).map fun r => (y r).toRat).sum = refSum d g (hOf d i) j := by
    rw [list_sum_range]
    rw [Finset.sum_congr rfl (fun r hr => (hyv r (Finset.mem_range.mp hr)).2)]
    rw [sum_range_trunc _ hLB (fun r h1 h2 => by rw [ite_eq_right (by omega)])]
    unfold refSum
    rw [hstop, Finset.sum_Ico_eq_sum_range]
    rw [show st + L - st = L by omega]
    exact Finset.sum_congr rfl fun r hr => by rw [ite_eq_left (Finset.mem_range.mp hr)]
  have hAbs : ((List.range d.BT).map fun r => |(y r).toRat|).sum = absSum d g (hOf d i) j := by
    rw [list_sum_range]
    rw [Finset.sum_congr rfl (fun r hr => by rw [(hyv r (Finset.mem_range.mp hr)).2])]
    rw [sum_range_trunc _ hLB (fun r h1 h2 => by rw [ite_eq_right (by omega), abs_zero])]
    unfold absSum
    rw [hstop, Finset.sum_Ico_eq_sum_range]
    rw [show st + L - st = L by omega]
    exact Finset.sum_congr rfl fun r hr => by rw [ite_eq_left (Finset.mem_range.mp hr)]
  have hA0 : 0 ≤ absSum d g (hOf d i) j := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hSA : |refSum d g (hOf d i) j| ≤ absSum d g (hOf d i) j := Finset.abs_sum_le_sum_abs _ _
  obtain ⟨he16a, he16⟩ := bt_log hd
  have hN : d.BT - 1 < 2 ^ 16 := by have := bt_le hd; omega
  -- the sum
  obtain ⟨hsf, hserr⟩ := sum_err_depth Format.binary32_regular y hcov hdep hs
    (fun r hr => (hyv r hr).1) (by rw [hAbs]; exact sum_room he16 hN hA0 hAle)
  rw [hSum, hAbs, Format.binary32_u, Format.binary32_eta] at hserr
  -- the division
  have hLq : (1 : ℚ) ≤ (L : ℚ) := by exact_mod_cast hL1
  obtain ⟨hroom, -⟩ := div_room he16 hN hLq hSA hAle hserr
  have hL0 : L32.toRat ≠ 0 := by rw [hL32]; positivity
  obtain ⟨hqf, hqerr⟩ := hP.divFull s L32 q hq hsf hL32f hL0 (by rw [hL32]; exact hroom)
  rw [hL32] at hqerr
  -- the rounding to `bfloat16`
  have hqroom := cvt_room hroom hqerr
  obtain ⟨hcf0, hcerr0⟩ := FloatV.cvt_err Format.bfloat16_regular hqf hqroom
  rw [FloatV.cvtWith_rn] at hc
  obtain ⟨hcf, hcv, -⟩ := hc.exact_of_isFinite hcf0
  rw [Format.bfloat16_u, Format.bfloat16_eta] at hcerr0
  refine ⟨hcf, ?_⟩
  rw [hcv]
  have key := mean_err he16 hN hLq hSA hserr hqerr hcerr0
  unfold tol mean absMean
  rw [hlen]
  exact key

end Claims.MeanPoolingFwd
