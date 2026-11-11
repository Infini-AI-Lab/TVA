import Proof.Spec
import Proof.RowFacts
import Proof.Ops
import Proof.ScoreLane
import Proof.DotLemmas

/-!
# One new-token trip, from the float facts to the rows' states

`ext_trip`: the relations of one new-token trip (`ExtData`) and the tiles it loaded carry every
valid row's state after `k` new-token trips to its state after `k + 1`.
-/

namespace Claims.ExtendAttn

open TVA

/-- The lanes' positions after one more trip. -/
theorem ext_seen_step (d : Inst) (m : Mem) (b i x : ℕ) :
    seenE d m b i (x + 64) = seenE d m b i x ∪
      laneSet (preOf d m b + x) (fun c : Fin 64 => decide (preOf d m b + (x + (c : ℕ)) ∈ keys d m b i)) ∧
    Disjoint (seenE d m b i x)
      (laneSet (preOf d m b + x) (fun c : Fin 64 => decide (preOf d m b + (x + (c : ℕ)) ∈ keys d m b i))) := by
  constructor
  · ext j
    simp only [seenE, Finset.mem_union, Finset.mem_filter, mem_laneSet, decide_eq_true_eq]
    constructor
    · rintro ⟨hj, hlt⟩
      by_cases h : j < preOf d m b + x
      · exact Or.inl ⟨hj, h⟩
      · right
        refine ⟨⟨j - (preOf d m b + x), by omega⟩, ?_, by simp; omega⟩
        have : preOf d m b + (x + (j - (preOf d m b + x))) = j := by omega
        simpa [this] using hj
    · rintro (⟨hj, h⟩ | ⟨n, hn, rfl⟩)
      · exact ⟨hj, by omega⟩
      · refine ⟨?_, by have := n.isLt; omega⟩
        have : preOf d m b + x + (n : ℕ) = preOf d m b + (x + (n : ℕ)) := by omega
        rw [this]; exact hn
  · rw [Finset.disjoint_left]
    intro j hj hj2
    obtain ⟨n, -, rfl⟩ := mem_laneSet.mp hj2
    simp only [seenE, Finset.mem_filter] at hj
    omega

/-- A sum over `Fin BD` masked to the first `D` entries equals the sum over `range D`. -/
theorem ext_sum_fin {BD D : ℕ} (hD : D ≤ BD) (g : ℕ → ℝ) :
    ∑ e : Fin BD, (if (e : ℕ) < D then g e else 0) = ∑ e ∈ Finset.range D, g e := by
  rw [Fin.sum_univ_eq_sum_range (fun e => if e < D then g e else 0) BD, ← Finset.sum_filter]
  refine Finset.sum_congr ?_ fun _ _ => rfl
  ext e; simp; omega

/-- The f32 pattern 0 reads back as the rational 0. -/
theorem ext_zero_f32 : DType.val? .f32 (FloatV.ofPat 0 : F32) = some 0 :=
  DType.val?_f32.mpr ⟨by decide, FloatV.toRat_ofPat_zero⟩

/-- The score of a lane. -/
theorem ext_score_glue {P : Platform} (hP : Device P) {M BD D : ℕ} (hDB : D ≤ BD) (hBD : BD ≤ 256)
    {sig X : ℝ} {ss : F32} (hsig1 : 1 / 2 ^ 64 ≤ sig) (hsig2 : sig ≤ 2 ^ 64)
    (hssf : ss.isFinite = true) (hss : |ss.toReal - sig| ≤ sig / 2 ^ 20)
    {q : Tensor [M, BD] BF16} {k : Tensor [BD, 64] BF16} {qk0 qk1 : Tensor [M, 64] F32}
    (hdot : P.dot cfgBF16 q k (tl.zeros [M, 64] .f32) qk0) (r : Fin M) (n : Fin 64)
    (hmul : FloatV.NaNEq (FloatV.mul (qk0.get2 r n) ss) (qk1.get2 r n))
    (qa ka : ℕ → ℚ)
    (hq : ∀ e : Fin BD, DType.val? .bf16 (q.get2 r e) = some (if (e : ℕ) < D then qa e else 0))
    (hk : ∀ e : Fin BD, DType.val? .bf16 (k.get2 e n) = some (if (e : ℕ) < D then ka e else 0))
    (hX : sig * ∑ e ∈ Finset.range D, ((Device.liftMag .bf16 (qa e) * Device.liftMag .bf16 (ka e) : ℚ) : ℝ) ≤ X)
    (hX2 : X ≤ 2 ^ 12) :
    (qk1.get2 r n).isFinite = true ∧
      |(qk1.get2 r n).toReal - sig * ∑ e ∈ Finset.range D, ((qa e : ℚ) : ℝ) * ((ka e : ℚ) : ℝ)| ≤ dlt BD X ∧
      |(qk1.get2 r n).toReal| ≤ X + dlt BD X := by
  have hqf : ∀ e, (q.get2 r e).isFinite = true := fun e => (DType.val?_bf16.mp (hq e)).1
  have hkf : ∀ e, (k.get2 e n).isFinite = true := fun e => (DType.val?_bf16.mp (hk e)).1
  have hqr : ∀ e : Fin BD, (q.get2 r e).toReal = if (e : ℕ) < D then ((qa e : ℚ) : ℝ) else 0 := by
    intro e
    have := (DType.val?_bf16.mp (hq e)).2
    unfold FloatV.toReal; rw [this]; split_ifs <;> simp
  have hkr : ∀ e : Fin BD, (k.get2 e n).toReal = if (e : ℕ) < D then ((ka e : ℚ) : ℝ) else 0 := by
    intro e
    have := (DType.val?_bf16.mp (hk e)).2
    unfold FloatV.toReal; rw [this]; split_ifs <;> simp
  have hL : liftSumK q k r n = ∑ e ∈ Finset.range D, ((Device.liftMag .bf16 (qa e) * Device.liftMag .bf16 (ka e) : ℚ) : ℝ) := by
    unfold liftSumK
    rw [← ext_sum_fin hDB (fun e => ((Device.liftMag .bf16 (qa e) * Device.liftMag .bf16 (ka e) : ℚ) : ℝ))]
    refine Finset.sum_congr rfl fun e _ => ?_
    have h1 := (DType.val?_bf16.mp (hq e)).2
    have h2 := (DType.val?_bf16.mp (hk e)).2
    rw [h1, h2]
    split_ifs <;> simp
  have hL0 : 0 ≤ ∑ e ∈ Finset.range D, ((Device.liftMag .bf16 (qa e) * Device.liftMag .bf16 (ka e) : ℚ) : ℝ) := by
    rw [← hL]
    exact Finset.sum_nonneg fun e _ => by
      have := mul_nonneg (Device.liftMag_nonneg .bf16 (q.get2 r e).toRat) (Device.liftMag_nonneg .bf16 (k.get2 e n).toRat)
      exact_mod_cast this
  have hsig0 : 0 < sig := lt_of_lt_of_le (by positivity) hsig1
  have hLle : ∑ e ∈ Finset.range D, ((Device.liftMag .bf16 (qa e) * Device.liftMag .bf16 (ka e) : ℚ) : ℝ) ≤ 2 ^ 76 := by
    have h1 : sig * (∑ e ∈ Finset.range D, ((Device.liftMag .bf16 (qa e) * Device.liftMag .bf16 (ka e) : ℚ) : ℝ)) ≤ 2 ^ 12 := hX.trans hX2
    have h2 : (1 / 2 ^ 64 : ℝ) * (∑ e ∈ Finset.range D, ((Device.liftMag .bf16 (qa e) * Device.liftMag .bf16 (ka e) : ℚ) : ℝ)) ≤ sig * (∑ e ∈ Finset.range D, ((Device.liftMag .bf16 (qa e) * Device.liftMag .bf16 (ka e) : ℚ) : ℝ)) :=
      mul_le_mul_of_nonneg_right hsig1 hL0
    have : (∑ e ∈ Finset.range D, ((Device.liftMag .bf16 (qa e) * Device.liftMag .bf16 (ka e) : ℚ) : ℝ)) ≤ 2 ^ 76 := by
      have e : (2 : ℝ) ^ 76 = 2 ^ 64 * 2 ^ 12 := by norm_num
      nlinarith
    exact this
  obtain ⟨hDf, hD⟩ := dot_zero_realK hP hBD hdot r n ext_zero_f32 (fun e => hqf e) (fun e => hkf e) (by rw [hL]; linarith [hLle, (by norm_num : (2 : ℝ) ^ 76 ≤ 2 ^ 120)])
  have hSum : ∑ e : Fin BD, (q.get2 r e).toReal * (k.get2 e n).toReal =
      ∑ e ∈ Finset.range D, ((qa e : ℚ) : ℝ) * ((ka e : ℚ) : ℝ) := by
    rw [← ext_sum_fin hDB (fun e => ((qa e : ℚ) : ℝ) * ((ka e : ℚ) : ℝ))]
    refine Finset.sum_congr rfl fun e _ => ?_
    rw [hqr, hkr]; split_ifs <;> simp
  have hS : |∑ e ∈ Finset.range D, ((qa e : ℚ) : ℝ) * ((ka e : ℚ) : ℝ)| ≤
      ∑ e ∈ Finset.range D, ((Device.liftMag .bf16 (qa e) * Device.liftMag .bf16 (ka e) : ℚ) : ℝ) := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun e _ => ?_)
    have h1 := Device.abs_le_liftMag .bf16 (qa e)
    have h2 := Device.abs_le_liftMag .bf16 (ka e)
    have := mul_le_mul h1 h2 (abs_nonneg _) (Device.liftMag_nonneg _ _)
    rw [← abs_mul] at this
    have h3 : ((|qa e * ka e| : ℚ) : ℝ) ≤ ((Device.liftMag .bf16 (qa e) * Device.liftMag .bf16 (ka e) : ℚ) : ℝ) := by
      exact_mod_cast this
    rw [← Rat.cast_mul, ← Rat.cast_abs]; exact h3
  rw [hSum, hL] at hD
  rw [hL] at *
  exact score_lane hBD hsig1 hsig2 hL0 hX hX2 hS hDf hD hssf hss hmul

/-- One accumulator element of a new-token trip. -/
theorem ext_acc_glue {P : Platform} (hP : Device P) {fusion : Bool} {M N : ℕ} {valid : Fin 64 → Bool}
    {A : Tensor [M, 64] BF16} {B : Tensor [64, N] BF16} {a1 acc' : Tensor [M, N] F32}
    (hdot : P.dot cfgBF16 A B a1 acc' ∨ ∃ d', P.dot cfgBF16 A B (Tensor.full (cfgBF16.acc.ofBits 0)) d' ∧
      ∀ x, AccAdd fusion .f32 (d'.get x) (a1.get x) (acc'.get x))
    (r : Fin M) (e : Fin N) {a re : F32}
    (hz : ∀ n, valid n = false → (A.get2 r n).toReal = 0)
    (hpb : ∀ n, valid n = true → |(A.get2 r n).toReal| ≤ 2)
    (hvb : ∀ n, valid n = true → |(B.get2 n e).toReal| ≤ 2 ^ 100)
    (hAf : ∀ n, (A.get2 r n).isFinite = true) (hBf : ∀ n, (B.get2 n e).isFinite = true)
    (haf : a.isFinite = true) (hab : |a.toReal| ≤ 2 ^ 119) (hre : re.isFinite = true)
    (hreb : |re.toReal| ≤ 1 + 1 / 2 ^ 20) (hcx : FloatV.NaNEq (FloatV.mul a re) (a1.get2 r e)) :
    (acc'.get2 r e).isFinite = true ∧
      |(acc'.get2 r e).toReal - (re.toReal * a.toReal + sgSum valid (fun n => A.get2 r n) (fun n => B.get2 n e))| ≤
        epsE1 * |re.toReal * a.toReal| + eps2 * lpSum valid (fun n => A.get2 r n) (fun n => B.get2 n e) +
          eps1 * (1 + v1Sum valid (fun n => B.get2 n e)) := by
  classical
  have hra : |a.toReal * re.toReal| ≤ 2 ^ 119 * (1 + 1 / 2 ^ 20) := by
    rw [abs_mul]
    exact mul_le_mul hab hreb (abs_nonneg _) (by positivity)
  obtain ⟨hcf, hce⟩ := mul_real haf hre (hra.trans (by norm_num)) hcx
  have hcb : |(a1.get2 r e).toReal| ≤ 2 ^ 119 + 2 ^ 100 := by
    have h0 := abs_sub_abs_le_abs_sub (a1.get2 r e).toReal (a.toReal * re.toReal)
    have h1 : 1 / 2 ^ 24 * |a.toReal * re.toReal| ≤ 1 / 2 ^ 24 * (2 ^ 119 * (1 + 1 / 2 ^ 20)) :=
      mul_le_mul_of_nonneg_left hra (by positivity)
    have : (2 : ℝ) ^ 119 * (1 + 1 / 2 ^ 20) + 1 / 2 ^ 24 * (2 ^ 119 * (1 + 1 / 2 ^ 20)) + 1 / 2 ^ 150 ≤
        2 ^ 119 + 2 ^ 100 := by norm_num
    linarith
  have hAle := liftA_le hz hpb (v := fun n => B.get2 n e)
  have hLp : lpSum valid (fun n => A.get2 r n) (fun n => B.get2 n e) ≤ 2 ^ 107 := by
    unfold lpSum
    calc _ ≤ ∑ _n : Fin 64, (2 : ℝ) * 2 ^ 100 := Finset.sum_le_sum fun n _ => by
          by_cases hv : valid n = true
          · simp only [hv, ite_true]
            exact mul_le_mul (hpb n hv) (hvb n hv) (abs_nonneg _) (by norm_num)
          · simp only [hv]; norm_num
      _ = 2 ^ 107 := by simp; norm_num
  have hV1 : v1Sum valid (fun n => B.get2 n e) ≤ 2 ^ 106 := by
    unfold v1Sum
    calc _ ≤ ∑ _n : Fin 64, (2 : ℝ) ^ 100 := Finset.sum_le_sum fun n _ => by
          by_cases hv : valid n = true
          · simp only [hv, ite_true]; exact hvb n hv
          · simp only [hv]; norm_num
      _ = 2 ^ 106 := by simp; norm_num
  have hroom : |(a1.get2 r e).toReal| + liftSum A B r e ≤ 2 ^ 120 := by
    have h1 : liftSum A B r e ≤ 2 ^ 107 + 1 / 2 ^ 118 * (1 + 2 ^ 106) := by
      have hLp' : (∑ n : Fin 64, if valid n = true then |(A.get2 r n).toReal| * |(B.get2 n e).toReal| else 0) ≤ 2 ^ 107 := hLp
      have hV1 : (∑ n : Fin 64, if valid n = true then |(B.get2 n e).toReal| else 0) ≤ 2 ^ 106 := hV1
      have h2 : 1 / 2 ^ 118 * (1 + ∑ n : Fin 64, if valid n = true then |(B.get2 n e).toReal| else 0) ≤
          1 / 2 ^ 118 * (1 + 2 ^ 106) := mul_le_mul_of_nonneg_left (by linarith) (by positivity)
      show ∑ k : Fin 64, ((Device.liftMag .bf16 (A.get2 r k).toRat * Device.liftMag .bf16 (B.get2 k e).toRat : ℚ) : ℝ) ≤ _
      linarith
    have : (2 : ℝ) ^ 119 + 2 ^ 100 + (2 ^ 107 + 1 / 2 ^ 118 * (1 + 2 ^ 106)) ≤ 2 ^ 120 := by norm_num
    linarith
  have hd := dot_add_real hP (c := a1) (z := Tensor.full (cfgBF16.acc.ofBits 0)) hdot r e ext_zero_f32
    hAf hBf hcf hroom
  exact col_ext hP hz hpb haf hab hre (by linarith [hreb, (by norm_num : (1 : ℝ) / 2 ^ 20 ≤ 1)]) hcx hd


/-- **One new-token trip**, at the new token `sn = es + 64 k` of a loop started at `es`
(`extStart`), for the block's end `cbe`. -/
theorem ext_trip {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {k : Nat}
    (hk1 : extStart d i + BN * k < blockEnd d init i) {sm32 : F32}
    (hsm32 : sm32 = FloatV.hostF32 d.sm_scale) (R : ExtData P true d.BM d.BD sm32)
    (hq : QFacts d init i R.q) (hK : KFactsE d init i (extStart d i + BN * k) (blockEnd d init i) R.k)
    (hV : VFactsE d init i (extStart d i + BN * k) (blockEnd d init i) R.v)
    (hFM : ∀ (r : Fin d.BM) (c : Fin BN),
      R.FM.get2 r c = (decide (validRow d init i r) && vE d init i r (extStart d i + BN * k) c))
    (hold : ∀ r : Fin d.BM, validRow d init i r →
      RowFl (rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + r))
        (rowVc d init (bOf d i) (hOf d i)) d.D d.BD (dlt d.BD (rowX d init i r))
        (rowX d init i r + dlt d.BD (rowX d init i r)) (tripsPre d init (bOf d i) + k)
        (g2 + tripsPre d init (bOf d i) * g1p + k * g1e)
        (seenE d init (bOf d i) (mOf d i * d.BM + r) (extStart d i + BN * k)) (R.e_max.get1 r)
        (R.deno.get1 r) (fun e => R.acc.get2 r e)) :
    ∀ r : Fin d.BM, validRow d init i r →
      RowFl (rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + r))
        (rowVc d init (bOf d i) (hOf d i)) d.D d.BD (dlt d.BD (rowX d init i r))
        (rowX d init i r + dlt d.BD (rowX d init i r)) (tripsPre d init (bOf d i) + (k + 1))
        (g2 + tripsPre d init (bOf d i) * g1p + (k + 1 : Nat) * g1e)
        (seenE d init (bOf d i) (mOf d i * d.BM + r) (extStart d i + BN * (k + 1)))
        (R.n_e_max.get1 r) (R.deno'.get1 r) (fun e => R.acc'.get2 r e) := by
  classical
  intro r hr
  obtain ⟨hX0, hX2, hne, hcard, hmag, hvb, htr, hTe⟩ := row_facts hd hpre hi (r := (r : ℕ)) hr
  obtain ⟨m, l, G, hrow, hdF, hdV, hemaxF, hemaxE, hcols⟩ := hold r hr
  have hBN : BN = 64 := rfl
  have hBM : 0 < d.BM := by have := hd.bm; split_ifs at this <;> omega
  -- numbers
  have hdiv : (mOf d i * d.BM + (r : ℕ)) / d.BM = mOf d i := by
    rw [Nat.mul_comm, Nat.mul_add_div hBM, Nat.div_eq_of_lt r.isLt]; simp
  have hTe' : tripsExt d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) =
      (blockEnd d init i + 63) / 64 := by
    unfold tripsExt blockEnd
    rw [hdiv]; rfl
  have hkTe : k + 1 ≤ tripsExt d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) := by
    rw [hTe']; rw [hBN] at hk1; omega
  have ht : tripsPre d init (bOf d i) + k + 1 ≤ tmax := by unfold tmax; omega
  -- positions
  obtain ⟨hseen, hdisj⟩ := ext_seen_step d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) (extStart d i + BN * k)
  have hvalid : ∀ n : Fin 64, R.FM.get2 r n = decide (preOf d init (bOf d i) + (extStart d i + BN * k + (n : ℕ)) ∈
      keys d init (bOf d i) (mOf d i * d.BM + (r : ℕ))) := by
    intro n
    rw [hFM r n]
    have : decide (validRow d init i (r : ℕ)) = true := decide_eq_true hr
    simp only [this, Bool.true_and, vE]
  have hfun : (fun n : Fin 64 => R.FM.get2 r n) = (fun c : Fin 64 => decide (preOf d init (bOf d i) +
      (extStart d i + BN * k + (c : ℕ)) ∈ keys d init (bOf d i) (mOf d i * d.BM + (r : ℕ)))) := funext hvalid
  have hmem : ∀ n : Fin 64, R.FM.get2 r n = true → preOf d init (bOf d i) +
      (extStart d i + BN * k + (n : ℕ)) ∈ keys d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) := by
    intro n hn; rw [hvalid n] at hn; exact of_decide_eq_true hn
  have hltE : ∀ n : Fin 64, R.FM.get2 r n = true → extStart d i + BN * k + (n : ℕ) < blockEnd d init i := by
    intro n hn
    have h1 := mem_keys_lt hd hpre hi hr (hmem n hn)
    have h2 := r.isLt
    unfold blockEnd
    have : (mOf d i + 1) * d.BM = mOf d i * d.BM + d.BM := by ring
    omega
  have hSV : seenE d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) (extStart d i + BN * k) ∪
      laneSet (preOf d init (bOf d i) + (extStart d i + BN * k)) (fun n : Fin 64 => R.FM.get2 r n) ⊆
      keys d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) := by
    intro j hj
    rcases Finset.mem_union.mp hj with h | h
    · exact (Finset.mem_filter.mp h).1
    · obtain ⟨n, hn, rfl⟩ := mem_laneSet.mp h
      have := hmem n hn
      rwa [← Nat.add_assoc] at this
  have hcardU : (seenE d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) (extStart d i + BN * k) ∪
      laneSet (preOf d init (bOf d i) + (extStart d i + BN * k)) (fun n : Fin 64 => R.FM.get2 r n)).card ≤ nmax :=
    (Finset.card_le_card hSV).trans hcard
  have hdisj' : Disjoint (seenE d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) (extStart d i + BN * k))
      (laneSet (preOf d init (bOf d i) + (extStart d i + BN * k)) (fun n : Fin 64 => R.FM.get2 r n)) := by
    rw [hfun]; exact hdisj
  have hnew : seenE d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) (extStart d i + BN * (k + 1)) =
      seenE d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) (extStart d i + BN * k) ∪
      laneSet (preOf d init (bOf d i) + (extStart d i + BN * k)) (fun n : Fin 64 => R.FM.get2 r n) := by
    have : extStart d i + BN * (k + 1) = extStart d i + BN * k + 64 := by rw [hBN]; ring
    rw [this, hseen, hfun]
  -- scale
  obtain ⟨hsf, hslo, hshi⟩ := hd.scale
  have hsmR : (d.sm_scale.toReal) = sig d := rfl
  obtain ⟨hssf, hss⟩ := scale_ext hsf (by
      show (1 : ℝ) / 2 ^ 64 ≤ ((d.sm_scale.toRat : ℚ) : ℝ)
      have := (Rat.cast_le (K := ℝ)).mpr hslo; push_cast at this; exact this) (by
      show ((d.sm_scale.toRat : ℚ) : ℝ) ≤ 2 ^ 64
      exact_mod_cast hshi)
  rw [← hsm32] at hssf hss
  have hsig1 : (1 : ℝ) / 2 ^ 64 ≤ sig d := by
    show (1 : ℝ) / 2 ^ 64 ≤ ((d.sm_scale.toRat : ℚ) : ℝ)
    have := (Rat.cast_le (K := ℝ)).mpr hslo; push_cast at this; exact this
  have hsig2 : sig d ≤ 2 ^ 64 := by
    show ((d.sm_scale.toRat : ℚ) : ℝ) ≤ 2 ^ 64; exact_mod_cast hshi
  have hss' : |sm32.toReal - sig d| ≤ sig d / 2 ^ 20 := hss
  -- the lanes' scores
  have hscore : ∀ n : Fin 64, R.FM.get2 r n = true →
      (R.qk1.get2 r n).isFinite = true ∧
      |(R.qk1.get2 r n).toReal - rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + (r : ℕ))
        (preOf d init (bOf d i) + (extStart d i + BN * k + (n : ℕ)))| ≤ dlt d.BD (rowX d init i r) ∧
      |(R.qk1.get2 r n).toReal| ≤ rowX d init i r + dlt d.BD (rowX d init i r) := by
    intro n hn
    have hlt := hltE n hn
    refine ext_score_glue hP hd.D_le hd.bd_le hsig1 hsig2 hssf hss' R.hdot1 r n (R.hmul1 (r, n, ()))
      (fun e => qv d init (bOf d i) (hOf d i) (mOf d i * d.BM + (r : ℕ)) e)
      (fun e => keyK d init (bOf d i) (hOf d i) (preOf d init (bOf d i) + (extStart d i + BN * k + (n : ℕ))) e)
      (fun e => ?_) (fun e => ?_) ?_ hX2
    · have := hq r e
      simpa only [hr, true_and] using this
    · have := hK e n
      simp only [hlt, true_and] at this
      convert this using 3
      rfl
    · have := hmag _ (hmem n hn)
      exact this
  have hXp0 : 0 ≤ rowX d init i r + dlt d.BD (rowX d init i r) := by
    have : 0 ≤ dlt d.BD (rowX d init i r) := by unfold dlt; positivity
    linarith
  have hdlt : dlt d.BD (rowX d init i r) ≤ 1 := by
    have hbd : (d.BD : ℝ) ≤ 256 := by exact_mod_cast hd.bd_le
    unfold dlt
    nlinarith [hX0, hX2]
  have hdl0 : 0 ≤ dlt d.BD (rowX d init i r) := by unfold dlt; positivity
  have hXp2 : rowX d init i r + dlt d.BD (rowX d init i r) ≤ 2 ^ 13 := by linarith
  -- masked scores
  have hqkT : ∀ n : Fin 64, R.FM.get2 r n = true →
      (tl.where R.FM R.qk1 (bc ninf32)).get2 r n = R.qk1.get2 r n := by
    intro n hn
    show (if R.FM.get2 r n = true then R.qk1.get2 r n else ninf32) = _
    simp [hn]
  have hqkF : ∀ n : Fin 64, R.FM.get2 r n = false →
      (tl.where R.FM R.qk1 (bc ninf32)).get2 r n = FloatV.negInf := by
    intro n hn
    show (if R.FM.get2 r n = true then R.qk1.get2 r n else ninf32) = _
    rw [hn, ninf32_eq]; rfl
  have hq' : ∀ n : Fin 64, (R.FM.get2 r n = true →
      ((tl.where R.FM R.qk1 (bc ninf32)).get2 r n).isFinite = true ∧
      |((tl.where R.FM R.qk1 (bc ninf32)).get2 r n).toReal - rowSc d init (bOf d i) (hOf d i)
        (mOf d i * d.BM + (r : ℕ)) (preOf d init (bOf d i) + (extStart d i + BN * k) + (n : ℕ))| ≤
          dlt d.BD (rowX d init i r) ∧
      |((tl.where R.FM R.qk1 (bc ninf32)).get2 r n).toReal| ≤ rowX d init i r + dlt d.BD (rowX d init i r)) ∧
      (R.FM.get2 r n = false → (tl.where R.FM R.qk1 (bc ninf32)).get2 r n = FloatV.negInf) := by
    intro n
    refine ⟨fun hn => ?_, hqkF n⟩
    rw [hqkT n hn, Nat.add_assoc]
    exact hscore n hn
  have hmax' := fun i' : Fin d.BM => R.hmax i'
  obtain ⟨m', g, f, δ', hT, hround, hl'F, hreF, hpF, hp16F, hp16z, hnmF, hnn, hne'⟩ :=
    row_stat hP (fusion := true) (sc := rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + (r : ℕ)))
      (dl := dlt d.BD (rowX d init i r)) (Xp := rowX d init i r + dlt d.BD (rowX d init i r))
      (t := tripsPre d init (bOf d i) + k)
      (S := seenE d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) (extStart d i + BN * k))
      (base := preOf d init (bOf d i) + (extStart d i + BN * k)) (valid := fun n : Fin 64 => R.FM.get2 r n)
      (m := m) (l := l) (G := G) (deno := R.deno.get1 r)
      (l' := R.deno'.get1 r) (rmax := R.row_max.get1 r) (nmx := R.n_e_max.get1 r) (re := R.re.get1 r)
      (Ssum := R.sm.get1 r) (qk := fun n => (tl.where R.FM R.qk1 (bc ninf32)).get2 r n)
      (p := fun n => R.p.get2 r n) (p16 := fun n => R.p16.get2 r n)
      hXp0 hXp2 hdl0 ht hcardU hdisj' hrow hdF hdV hemaxF hemaxE hq' (R.hmax r) (R.hnm (r, ()))
      ⟨_, R.hsub1 (r, ()), R.hexp1 (r, ())⟩ (fun n => ⟨_, R.hsub2 (r, n, ()), R.hexp2 (r, n, ())⟩)
      (R.hsum r) (R.hden (r, ())) (fun n => R.hcvt (r, n, ()))
  have hVne : ∀ n : Fin 64, R.FM.get2 r n = true →
      (seenE d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) (extStart d i + BN * k) ∪
      laneSet (preOf d init (bOf d i) + (extStart d i + BN * k)) (fun n : Fin 64 => R.FM.get2 r n)).Nonempty := by
    intro n hn
    exact ⟨_, Finset.mem_union_right _ (mem_laneSet.mpr ⟨n, hn, rfl⟩)⟩
  have hpb : ∀ n : Fin 64, R.FM.get2 r n = true → |(R.p16.get2 r n).toReal| ≤ 2 := by
    intro n hn
    have hj : preOf d init (bOf d i) + (extStart d i + BN * k) + (n : ℕ) ∈
        laneSet (preOf d init (bOf d i) + (extStart d i + BN * k)) (fun n : Fin 64 => R.FM.get2 r n) :=
      mem_laneSet.mpr ⟨n, hn, rfl⟩
    have hl := hT.lanes _ hj
    have hw := hT.weight _ hj
    have hsm := hT.ref_ge.1 _ hj
    simp only [ext64_lane] at hl hw hsm
    obtain ⟨-, hs, hδ⟩ := hl
    have hm' := hT.m'_bound hrow (hVne n hn)
    obtain ⟨hw0, hw1⟩ := new_weight hXp2 hm' hs hsm hw.1
    have hpn : |(R.p.get2 r n).toReal| ≤ 1 + 1 / 2 ^ 20 + eps0 := by
      have h1 := abs_le.mp hw.2
      rw [abs_le]; constructor <;> linarith [h1.1, h1.2, (by unfold eps0; norm_num : (0 : ℝ) ≤ eps0)]
    have hr1 := hround n hn
    have hd1 : |1 + δ' (preOf d init (bOf d i) + (extStart d i + BN * k) + (n : ℕ))| ≤ 1 + ub := by
      have := abs_le.mp hδ
      rw [abs_le]; constructor <;> linarith [this.1, this.2]
    have h2 : |(R.p.get2 r n).toReal * (1 + δ' (preOf d init (bOf d i) + (extStart d i + BN * k) + (n : ℕ)))| ≤
        (1 + 1 / 2 ^ 20 + eps0) * (1 + ub) := by
      rw [abs_mul]; exact mul_le_mul hpn hd1 (abs_nonneg _) (by unfold eps0; positivity)
    have h3 := abs_sub_abs_le_abs_sub (R.p16.get2 r n).toReal ((R.p.get2 r n).toReal * (1 + δ' (preOf d init (bOf d i) + (extStart d i + BN * k) + (n : ℕ))))
    unfold eps0 ub etab at *
    norm_num at h2 hr1 h3 ⊢
    linarith
  have hvq : ∀ (n : Fin 64) (e : Fin d.BD), (R.v.get2 n e).toReal =
      if extStart d i + BN * k + (n : ℕ) < blockEnd d init i ∧ (e : ℕ) < d.D then
        ((keyV d init (bOf d i) (hOf d i) (preOf d init (bOf d i) + (extStart d i + BN * k + (n : ℕ))) e : ℚ) : ℝ)
      else 0 := by
    intro n e
    have h2 : FloatV.toRat (R.v.get2 n e) =
        (if extStart d i + BN * k + (n : ℕ) < blockEnd d init i ∧ (e : ℕ) < d.D then
          keyV d init (bOf d i) (hOf d i) (preOf d init (bOf d i) + (extStart d i + BN * k + (n : ℕ))) e else 0) :=
      (DType.val?_bf16.mp (hV n e)).2
    unfold FloatV.toReal; rw [h2]; split_ifs <;> simp
  have hvfin : ∀ (n : Fin 64) (e : Fin d.BD), (R.v.get2 n e).isFinite = true :=
    fun n e => (DType.val?_bf16.mp (hV n e)).1
  have hvto : ∀ (e : Fin d.BD), (e : ℕ) < d.D → ∀ n : Fin 64, R.FM.get2 r n = true →
      (R.v.get2 n e).toReal = rowVc d init (bOf d i) (hOf d i) e
        (preOf d init (bOf d i) + (extStart d i + BN * k) + (n : ℕ)) := by
    intro e he n hn
    rw [hvq]
    simp only [hltE n hn, he, and_self, ite_true]
    unfold rowVc
    rw [Nat.add_assoc (preOf d init (bOf d i))]
  have hvb' : ∀ (e : Fin d.BD) (n : Fin 64), R.FM.get2 r n = true → |(R.v.get2 n e).toReal| ≤ 2 ^ 100 := by
    intro e n hn
    by_cases he : (e : ℕ) < d.D
    · rw [hvto e he n hn]
      have hm : preOf d init (bOf d i) + (extStart d i + BN * k) + (n : ℕ) ∈
          keys d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) := by
        rw [Nat.add_assoc (preOf d init (bOf d i))]; exact hmem n hn
      exact hvb _ hm e he
    · rw [hvq]; simp only [he, and_false, ite_false]; norm_num
  have hreb := hT.scale_bd
  have hacc : ∀ e : Fin d.BD, (e : ℕ) < d.D → (R.acc'.get2 r e).isFinite = true ∧
      |(R.acc'.get2 r e).toReal - ((R.re.get1 r).toReal * (R.acc.get2 r e).toReal +
        sgSum (fun n : Fin 64 => R.FM.get2 r n) (fun n => R.p16.get2 r n) (fun n => R.v.get2 n e))| ≤
        epsE1 * |(R.re.get1 r).toReal * (R.acc.get2 r e).toReal| +
        eps2 * lpSum (fun n : Fin 64 => R.FM.get2 r n) (fun n => R.p16.get2 r n) (fun n => R.v.get2 n e) +
        eps1 * (1 + v1Sum (fun n : Fin 64 => R.FM.get2 r n) (fun n => R.v.get2 n e)) := by
    intro e he
    obtain ⟨haf, hab, -⟩ := hcols e he
    exact ext_acc_glue hP (fusion := true) (valid := fun n : Fin 64 => R.FM.get2 r n) (A := R.p16) (B := R.v)
      (a1 := R.a1) (acc' := R.acc') R.hdot2 r e hp16z hpb (fun n hn => hvb' e n hn) hp16F (fun n => hvfin n e)
      haf hab hreF hreb (R.hmul2 (r, e, ()))
  obtain ⟨hε0, hε1, hg1a, hg1b, hg1c⟩ := epsE1_ok
  have hg2 : 0 ≤ g2 := by unfold g2; norm_num
  have hgp : 0 ≤ g1p := by unfold g1p; norm_num
  have hTp : (tripsPre d init (bOf d i) : ℝ) + k + 1 ≤ 4098 := by
    have : tripsPre d init (bOf d i) + k + 1 ≤ 4098 := by omega
    exact_mod_cast this
  have hTp0 : (0 : ℝ) ≤ tripsPre d init (bOf d i) := Nat.cast_nonneg _
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  have hγ0 : g2 + tripsPre d init (bOf d i) * g1p + k * g1e ≤ 1 / 2 ^ 3 := by
    unfold g2 g1p g1e at *
    nlinarith
  have hγ1 : g2 + tripsPre d init (bOf d i) * g1p + k * g1e + g1e ≤ 1 / 2 ^ 3 := by
    unfold g2 g1p g1e at *
    nlinarith
  have key := RowFl.trip (sc := rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + (r : ℕ)))
    (vcol := rowVc d init (bOf d i) (hOf d i)) (D := d.D) (BD := d.BD)
    (dl := dlt d.BD (rowX d init i r)) (Xp := rowX d init i r + dlt d.BD (rowX d init i r))
    (t := tripsPre d init (bOf d i) + k) (γ := g2 + tripsPre d init (bOf d i) * g1p + k * g1e)
    (g1 := g1e) (ε1 := epsE1) (S := seenE d init (bOf d i) (mOf d i * d.BM + (r : ℕ)) (extStart d i + BN * k))
    (base := preOf d init (bOf d i) + (extStart d i + BN * k)) (valid := fun n : Fin 64 => R.FM.get2 r n)
    (deno := R.deno.get1 r) (l' := R.deno'.get1 r) (re := R.re.get1 r)
    (nmx := R.n_e_max.get1 r) (acc := fun e => R.acc.get2 r e) (acc' := fun e => R.acc'.get2 r e)
    (p16 := fun n => R.p16.get2 r n) (vT := fun e n => R.v.get2 n e) (δ' := δ') (f := f)
    (m := m) (l := l) (m' := m') (g := g) (Ssum := (R.sm.get1 r).toReal) (G := G)
    (qkr := fun n => ((tl.where R.FM R.qk1 (bc ninf32)).get2 r n).toReal) (pr := fun n => (R.p.get2 r n).toReal)
    hXp0 hXp2 hdl0 ht hcardU hdisj' hrow hcols hT hround hl'F (fun h => ⟨hnmF, hnn h⟩) hne' (fun e he n hn => hvto e he n hn)
    (fun e he j hj => hvb j (hSV hj) e he) hacc ⟨hε0, hε1⟩ ⟨hg1a, hg1b, hg1c⟩
    (by linarith [mul_nonneg hTp0 hgp, mul_nonneg hk0 hg1c]) (by linarith) hγ1 hdF hdV
  rw [hnew]
  have hγeq : g2 + (tripsPre d init (bOf d i) : ℝ) * g1p + ((k + 1 : ℕ) : ℝ) * g1e =
      g2 + tripsPre d init (bOf d i) * g1p + k * g1e + g1e := by push_cast; ring
  rw [hγeq]
  exact key

end Claims.ExtendAttn
