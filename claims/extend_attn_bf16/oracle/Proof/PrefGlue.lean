import Proof.Spec
import Proof.RowFacts
import Proof.Ops
import Proof.ScoreLane
import Proof.DotLemmas
import Proof.RowFl

/-!
# One cached-part trip, from the float facts to the rows' states

`pref_trip`: the relations of one cached-part trip (`PrefData`) and the tiles it loaded carry every
valid row's state after `n` trips to its state after `n + 1`.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- A sum over `Fin BD` masked to the first `D` entries equals the sum over `range D`. -/
theorem pg_sum_fin_range (g : ℕ → ℝ) {D BD : ℕ} (h : D ≤ BD) :
    ∑ e : Fin BD, (if (e : ℕ) < D then g e else 0) = ∑ e ∈ Finset.range D, g e := by
  rw [Fin.sum_univ_eq_sum_range (fun e => if e < D then g e else 0) BD, ← Finset.sum_filter]
  refine Finset.sum_congr ?_ fun _ _ => rfl
  ext e; simp only [Finset.mem_filter, Finset.mem_range]; omega

/-- The set identity of a cached-part trip. -/
theorem pg_seenP_succ (d : Inst) (m : Mem) (b i : Nat) (n : Nat) (valid : Fin 64 → Bool)
    (hv : ∀ c : Fin 64, valid c = true ↔ (BN * n + c ∈ keys d m b i ∧ BN * n + c < preOf d m b)) :
    seenP d m b i (n + 1) = seenP d m b i n ∪ laneSet (BN * n) valid ∧
      Disjoint (seenP d m b i n) (laneSet (BN * n) valid) := by
  have key : ∀ j, j ∈ laneSet (BN * n) valid ↔ ∃ c : Fin 64, (BN * n + c ∈ keys d m b i ∧
      BN * n + c < preOf d m b) ∧ j = BN * n + c := by
    intro j
    rw [mem_laneSet]
    exact exists_congr fun c => and_congr_left' (hv c)
  have hBN : BN = 64 := rfl
  have e1 : BN * (n + 1) = BN * n + 64 := by rw [hBN]; ring
  constructor
  · ext j
    simp only [seenP, Finset.mem_union, Finset.mem_filter, key, lt_min_iff]
    constructor
    · rintro ⟨hj, h1, h2⟩
      by_cases hlt : j < BN * n
      · exact Or.inl ⟨hj, hlt, h2⟩
      · right
        refine ⟨⟨j - BN * n, by omega⟩, ?_, ?_⟩
        · have : BN * n + (j - BN * n) = j := by omega
          simp only [this]; exact ⟨hj, h2⟩
        · simp only; omega
    · rintro (⟨hj, h1, h2⟩ | ⟨c, ⟨h1, h2⟩, rfl⟩)
      · exact ⟨hj, by omega, h2⟩
      · exact ⟨h1, by have := c.isLt; omega, h2⟩
  · rw [Finset.disjoint_left]
    intro j hj hj'
    obtain ⟨c, -, rfl⟩ := (key j).mp hj'
    simp only [seenP, Finset.mem_filter, lt_min_iff] at hj
    omega


/-- A bf16 value read as `some y` is finite with rational value `y` and real value `y`. -/
theorem pg_val_bf16 {x : BF16} {y : ℚ} (h : DType.val? .bf16 x = some y) :
    x.isFinite = true ∧ x.toRat = y ∧ x.toReal = (y : ℝ) := by
  obtain ⟨h1, h2⟩ := DType.val?_bf16.mp h
  exact ⟨h1, h2, by unfold FloatV.toReal; rw [h2]⟩

/-- **The score of a valid lane.** -/
theorem pg_score {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {n : Nat}
    {q16 : Tensor [d.BM, d.BD] BF16} {k : Tensor [d.BD, 64] BF16}
    {qk0 qk1 : Tensor [d.BM, 64] F32} {ss : F32}
    (hq : QFacts d init i q16) (hk : KFactsP d init i n k)
    (hdot : P.dot cfgBF16 q16 k (tl.zeros [d.BM, 64] .f32) qk0)
    (hssf : ss.isFinite = true) (hss : |ss.toReal - sig d| ≤ sig d / 2 ^ 20)
    (hmul : ∀ x, FloatV.NaNEq (FloatV.mul (qk0.get x) ss) (qk1.get x))
    (r : Fin d.BM) (hr : validRow d init i r) (c : Fin 64)
    (hc : vP d init i r n c = true) :
    (qk1.get2 r c).isFinite = true ∧
      |(qk1.get2 r c).toReal - rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + r)
        (BN * n + c)| ≤ dlt d.BD (rowX d init i r) ∧
      |(qk1.get2 r c).toReal| ≤ rowX d init i r + dlt d.BD (rowX d init i r) := by
  have hc' : BN * n + c ∈ keys d init (bOf d i) (mOf d i * d.BM + r) ∧
      BN * n + c < preOf d init (bOf d i) := by
    simpa [vP] using hc
  obtain ⟨hX0, hX1, -, -, hsc, -, -, -⟩ := row_facts hd hpre hi hr
  have hscm := hsc _ hc'.1
  obtain ⟨hsf, hs1, hs2⟩ := hd.scale
  have hsig1 : 1 / 2 ^ 64 ≤ sig d := by
    unfold sig
    have := (Rat.cast_le (K := ℝ)).mpr hs1
    push_cast at this; exact this
  have hsig2 : sig d ≤ 2 ^ 64 := by
    unfold sig
    have := (Rat.cast_le (K := ℝ)).mpr hs2
    push_cast at this; exact this
  have hqe : ∀ e : Fin d.BD, (q16.get2 r e).isFinite = true ∧
      (q16.get2 r e).toRat = (if (e : ℕ) < d.D then
        qv d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e else 0) := by
    intro e
    obtain ⟨h1, h2, -⟩ := pg_val_bf16 (hq r e)
    refine ⟨h1, ?_⟩
    exact h2.trans (by simp only [hr, true_and])
  have hke : ∀ e : Fin d.BD, (k.get2 e c).isFinite = true ∧
      (k.get2 e c).toRat = (if (e : ℕ) < d.D then
        keyK d init (bOf d i) (hOf d i) (BN * n + c) e else 0) := by
    intro e
    obtain ⟨h1, h2, -⟩ := pg_val_bf16 (hk e c)
    refine ⟨h1, ?_⟩
    exact h2.trans (by simp only [hc'.2, true_and])
  set L : ℝ := liftSumK q16 k r c with hLdef
  set S : ℝ := ∑ e ∈ Finset.range d.D, ((qv d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e : ℚ) : ℝ) *
      ((keyK d init (bOf d i) (hOf d i) (BN * n + c) e : ℚ) : ℝ) with hSdef
  have hL : L = ∑ e ∈ Finset.range d.D, ((Device.liftMag .bf16
      (qv d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e) * Device.liftMag .bf16
      (keyK d init (bOf d i) (hOf d i) (BN * n + c) e) : ℚ) : ℝ) := by
    rw [← pg_sum_fin_range _ hd.D_le, hLdef]
    unfold liftSumK
    refine Finset.sum_congr rfl fun e _ => ?_
    rw [(hqe e).2, (hke e).2]
    by_cases he : (e : ℕ) < d.D <;> simp [he]
  have hSS : ∑ e : Fin d.BD, (q16.get2 r e).toReal * (k.get2 e c).toReal = S := by
    rw [hSdef, ← pg_sum_fin_range _ hd.D_le]
    refine Finset.sum_congr rfl fun e _ => ?_
    unfold FloatV.toReal
    rw [(hqe e).2, (hke e).2]
    by_cases he : (e : ℕ) < d.D <;> simp [he]
  have hL0 : 0 ≤ L := by rw [hLdef]; exact (by
    unfold liftSumK
    exact Finset.sum_nonneg fun e _ => by
      have := mul_nonneg (Device.liftMag_nonneg .bf16 (q16.get2 r e).toRat)
        (Device.liftMag_nonneg .bf16 (k.get2 e c).toRat)
      exact_mod_cast this)
  have hLX : sig d * L ≤ rowX d init i r := by
    rw [hL]; exact hscm
  have hX : rowX d init i r ≤ 2 ^ 12 := hX1
  have hSL : |S| ≤ L := by
    rw [hL, hSdef]
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun e _ => ?_)
    rw [← Rat.cast_mul, ← Rat.cast_abs]
    exact_mod_cast Device.abs_mul_le_liftMag .bf16 .bf16 _ _
  have hroom : liftSumK q16 k r c ≤ 2 ^ 120 := by
    have h1 : L * (1 / 2 ^ 64) ≤ sig d * L := by
      rw [mul_comm]; exact mul_le_mul_of_nonneg_right hsig1 hL0
    show L ≤ 2 ^ 120
    norm_num at h1 ⊢
    linarith
  have hzero : DType.val? .f32 ((tl.zeros [d.BM, 64] .f32).get2 r c) = some 0 := by
    show DType.val? .f32 (FloatV.ofPat 0) = some 0
    exact DType.val?_zero_f32
  obtain ⟨hDf, hD⟩ := dot_zero_realK hP hd.bd_le hdot r c hzero (fun e => (hqe e).1) (fun e => (hke e).1) hroom
  rw [hSS] at hD
  have := score_lane (D := qk0.get2 r c) (qk1 := qk1.get2 r c) (S := S) (X := rowX d init i r) hd.bd_le
    hsig1 hsig2 hL0 hLX hX hSL hDf hD hssf hss (hmul (r, c, ()))
  convert this using 3
  unfold rowSc score
  rfl


/-- For `0 <= X <= 2^12` and `BD <= 256`, `dlt BD X` is nonnegative and `X + dlt BD X` is at most 2^13. -/
theorem pg_dlt {BD : ℕ} {X : ℝ} (hBD : BD ≤ 256) (h0 : 0 ≤ X) (h1 : X ≤ 2 ^ 12) :
    0 ≤ dlt BD X ∧ X + dlt BD X ≤ 2 ^ 13 := by
  have hB : (BD : ℝ) ≤ 256 := by exact_mod_cast hBD
  have hB0 : (0 : ℝ) ≤ BD := Nat.cast_nonneg BD
  unfold dlt
  constructor
  · positivity
  · have : ((BD : ℝ) + 8) * (1 / 2 ^ 22) * X ≤ (256 + 8) * (1 / 2 ^ 22) * 2 ^ 12 := by
      apply mul_le_mul (mul_le_mul_of_nonneg_right (by linarith) (by norm_num)) h1 h0 (by norm_num)
    norm_num at this ⊢
    linarith

/-- The lifted magnitude sum of a row of `p16 · v` is at most `2^108`. -/
theorem pg_liftA {valid : Fin 64 → Bool} {p16 v : Fin 64 → BF16}
    (hz : ∀ n, valid n = false → (p16 n).toReal = 0) (hpb : ∀ n, valid n = true → |(p16 n).toReal| ≤ 2)
    (hvb : ∀ n, valid n = true → |(v n).toReal| ≤ 2 ^ 100) :
    ∑ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat * Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ) ≤ 2 ^ 108 := by
  have hLp : lpSum valid p16 v ≤ 2 ^ 107 := by
    unfold lpSum
    calc _ ≤ ∑ _n : Fin 64, (2 : ℝ) * 2 ^ 100 := Finset.sum_le_sum fun n _ => by
          by_cases hv : valid n = true
          · simp only [hv, ite_true]
            exact mul_le_mul (hpb n hv) (hvb n hv) (abs_nonneg _) (by norm_num)
          · simp only [hv]; norm_num
      _ = 2 ^ 107 := by simp; norm_num
  have hV1 : v1Sum valid v ≤ 2 ^ 106 := by
    unfold v1Sum
    calc _ ≤ ∑ _n : Fin 64, (2 : ℝ) ^ 100 := Finset.sum_le_sum fun n _ => by
          by_cases hv : valid n = true
          · simp only [hv, ite_true]; exact hvb n hv
          · simp only [hv]; norm_num
      _ = 2 ^ 106 := by simp; norm_num
  have h1 : (1 : ℝ) / 2 ^ 118 * (1 + v1Sum valid v) ≤ 1 / 2 ^ 118 * (1 + 2 ^ 106) :=
    mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  have h2 : (1 : ℝ) / 2 ^ 118 * (1 + 2 ^ 106) ≤ 1 := by norm_num
  have : (∑ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat * Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ)) ≤
      lpSum valid p16 v + 1 / 2 ^ 118 * (1 + v1Sum valid v) := liftA_le hz hpb
  have h3 : (2 : ℝ) ^ 107 + 1 ≤ 2 ^ 108 := by norm_num
  linarith


/-- **One cached-part trip.** -/
theorem pref_trip {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {n : Nat}
    (hn : n < tripsPre d init (bOf d i)) {sm32 : F32} (hsm32 : sm32 = FloatV.hostF32 d.sm_scale)
    (R : PrefData P true d.BM d.BD sm32) (hq : QFacts d init i R.q16)
    (hk : KFactsP d init i n R.k) (hv : VFactsP d init i n R.v)
    (hFM : ∀ (r : Fin d.BM) (c : Fin BN),
      R.FM.get2 r c = (decide (validRow d init i r) && vP d init i r n c))
    (hold : ∀ r : Fin d.BM, validRow d init i r →
      RowFl (rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + r))
        (rowVc d init (bOf d i) (hOf d i)) d.D d.BD (dlt d.BD (rowX d init i r))
        (rowX d init i r + dlt d.BD (rowX d init i r)) n (g2 + n * g1p)
        (seenP d init (bOf d i) (mOf d i * d.BM + r) n) (R.e_max.get1 r) (R.deno.get1 r)
        (fun e => R.acc.get2 r e)) :
    ∀ r : Fin d.BM, validRow d init i r →
      RowFl (rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + r))
        (rowVc d init (bOf d i) (hOf d i)) d.D d.BD (dlt d.BD (rowX d init i r))
        (rowX d init i r + dlt d.BD (rowX d init i r)) (n + 1) (g2 + (n + 1 : Nat) * g1p)
        (seenP d init (bOf d i) (mOf d i * d.BM + r) (n + 1)) (R.n_e_max.get1 r)
        (R.deno'.get1 r) (fun e => R.acc'.get2 r e) := by
  classical
  intro r hr
  subst hsm32
  obtain ⟨hX0, hX1, -, hcardK, -, hvK, htr, hext⟩ := row_facts hd hpre hi hr
  obtain ⟨hdl, hXp2⟩ := pg_dlt hd.bd_le hX0 hX1
  set valid : Fin 64 → Bool := fun c => R.FM.get2 r c with hvalid
  have hvalid' : ∀ c : Fin 64, valid c = true ↔ (BN * n + c ∈ keys d init (bOf d i)
      (mOf d i * d.BM + r) ∧ BN * n + c < preOf d init (bOf d i)) := by
    intro c
    have := hFM r c
    simp only [hvalid, this, hr, decide_true, Bool.true_and, vP, decide_eq_true_eq]
  obtain ⟨hset, hdisj⟩ := pg_seenP_succ d init (bOf d i) (mOf d i * d.BM + r) n valid hvalid'
  have hvP : ∀ c : Fin 64, valid c = true → vP d init i r n c = true := by
    intro c hc
    have := (hvalid' c).mp hc
    simpa [vP] using this
  -- the scale
  obtain ⟨hsf, hs1, hs2⟩ := hd.scale
  have hlo : (1 : ℝ) / 2 ^ 64 ≤ d.sm_scale.toReal := by
    have := (Rat.cast_le (K := ℝ)).mpr hs1
    unfold FloatV.toReal; push_cast at this ⊢; exact this
  have hhi : d.sm_scale.toReal ≤ 2 ^ 64 := by
    have := (Rat.cast_le (K := ℝ)).mpr hs2
    unfold FloatV.toReal; push_cast at this ⊢; exact this
  obtain ⟨hssf, hss⟩ := scale_pref hsf hlo hhi R.hss
  have hss' : |R.ss.toReal - sig d| ≤ sig d / 2 ^ 20 := hss
  -- the old state
  obtain ⟨m, l, G, hrow, hdF, hdV, hemaxF, hemaxE, hcols⟩ := hold r hr
  -- the scores
  have hsc := fun c (hc : valid c = true) => pg_score hP hd hpre hi hq hk R.hdot1 hssf hss' R.hmul1 r hr c (hvP c hc)
  have hqkv : ∀ c : Fin 64, valid c = true →
      (tl.where R.FM R.qk1 (bc ninf32)).get2 r c = R.qk1.get2 r c := by
    intro c hc
    show (if R.FM.get2 r c then _ else _) = _
    have h2 : R.FM.get2 r c = true := hc
    rw [h2]; rfl
  have hqkf : ∀ c : Fin 64, valid c = false →
      (tl.where R.FM R.qk1 (bc ninf32)).get2 r c = FloatV.negInf := by
    intro c hc
    show (if R.FM.get2 r c then _ else _) = _
    have h2 : R.FM.get2 r c = false := hc
    rw [h2, ninf32_eq]
    rfl
  have hn1 : n + 1 ≤ 2 ^ 12 + 1 := by omega
  have hts : n + 1 ≤ tmax := by unfold tmax; omega
  have hcardU : (seenP d init (bOf d i) (mOf d i * d.BM + r) n ∪ laneSet (BN * n) valid).card ≤ nmax := by
    rw [← hset]
    exact (Finset.card_le_card (Finset.filter_subset _ _)).trans hcardK
  have hrs := row_stat hP (fusion := true) (sc := rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + r))
    (dl := dlt d.BD (rowX d init i r)) (Xp := rowX d init i r + dlt d.BD (rowX d init i r))
    (t := n) (S := seenP d init (bOf d i) (mOf d i * d.BM + r) n) (base := BN * n) (valid := valid)
    (deno := R.deno.get1 r) (l' := R.deno'.get1 r)
    (rmax := R.row_max.get1 r) (nmx := R.n_e_max.get1 r) (re := R.re.get1 r) (Ssum := R.sm.get1 r)
    (qk := fun c => (tl.where R.FM R.qk1 (bc ninf32)).get2 r c) (p := fun c => R.p.get2 r c)
    (p16 := fun c => R.p16.get2 r c)
    (by unfold dlt; have := hX0; positivity) hXp2 hdl hts hcardU hdisj hrow hdF hdV hemaxF hemaxE
    (fun c => ⟨fun hc => by
        rw [hqkv c hc]; exact hsc c hc, hqkf c⟩)
    (R.hmax r) (R.hnm (r, ())) ⟨R.x1.get (r, ()), R.hsub1 (r, ()), R.hexp1 (r, ())⟩
    (fun c => ⟨R.x2.get (r, c, ()), R.hsub2 (r, c, ()), R.hexp2 (r, c, ())⟩) (R.hsum r)
    (R.hden (r, ())) (fun c => R.hcvt (r, c, ()))
  obtain ⟨m', g, f, δ', hT, hround, hl'F, hreF, hpF, hp16F, hp16z, hnmxF, hnn, hne⟩ := hrs
  have hp2 : ∀ c : Fin 64, valid c = true → |(R.p16.get2 r c).toReal| ≤ 2 := by
    intro c hc
    have hj : BN * n + (c : ℕ) ∈ laneSet (BN * n) valid := mem_laneSet.mpr ⟨c, hc, rfl⟩
    have hne' : (seenP d init (bOf d i) (mOf d i * d.BM + r) n ∪ laneSet (BN * n) valid).Nonempty :=
      ⟨_, Finset.mem_union_right _ hj⟩
    have hm' := hT.m'_bound hrow hne'
    obtain ⟨-, hl2, hl3⟩ := hT.lanes _ hj
    obtain ⟨hf, hpw⟩ := hT.weight _ hj
    have hle := hT.ref_ge.1 _ hj
    obtain ⟨hw0, hw1⟩ := new_weight hXp2 hm' hl2 hle hf
    have hpj : |ext64 (BN * n) (fun c => (R.p.get2 r c).toReal) (BN * n + (c : ℕ))| ≤
        1 + 1 / 2 ^ 20 + eps0 := by
      have h5 := abs_le.mp hpw
      unfold eps0 at h5 ⊢
      rw [abs_le]; constructor <;> linarith [h5.1, h5.2]
    rw [ext64_lane] at hpj
    have hr1 := hround c hc
    have hd1 : |1 + δ' (BN * n + (c : ℕ))| ≤ 1 + ub := by
      have := abs_le.mp hl3
      unfold ub at this ⊢
      rw [abs_le]; constructor <;> linarith [this.1, this.2]
    have h1 : |(R.p.get2 r c).toReal * (1 + δ' (BN * n + (c : ℕ)))| ≤ (1 + 1 / 2 ^ 20 + eps0) * (1 + ub) := by
      rw [abs_mul]
      exact mul_le_mul hpj hd1 (abs_nonneg _) (by unfold eps0; positivity)
    have h2 := abs_sub_abs_le_abs_sub (R.p16.get2 r c).toReal ((R.p.get2 r c).toReal * (1 + δ' (BN * n + (c : ℕ))))
    unfold eps0 ub etab at *
    norm_num at h1 hr1 h2 ⊢
    linarith
  have hvl : ∀ (e : Fin d.BD), (e : ℕ) < d.D → ∀ c : Fin 64, valid c = true →
      (R.v.get2 c e).toReal = rowVc d init (bOf d i) (hOf d i) e (BN * n + (c : ℕ)) := by
    intro e he c hc
    have hc' := (hvalid' c).mp hc
    obtain ⟨-, -, h3⟩ := pg_val_bf16 (hv c e)
    refine h3.trans ?_
    simp only [hc'.2, he, and_self, ite_true]
    rfl
  have hvbl : ∀ (e : Fin d.BD), (e : ℕ) < d.D → ∀ c : Fin 64, valid c = true →
      |(R.v.get2 c e).toReal| ≤ 2 ^ 100 := by
    intro e he c hc
    rw [hvl e he c hc]
    exact hvK _ ((hvalid' c).mp hc).1 e he
  have hacc : ∀ e : Fin d.BD, (e : ℕ) < d.D → (R.acc'.get2 r e).isFinite = true ∧
      |(R.acc'.get2 r e).toReal - ((R.re.get1 r).toReal * (R.acc.get2 r e).toReal +
        sgSum valid (fun c => R.p16.get2 r c) (fun c => R.v.get2 c e))| ≤
        epsP1 * |(R.re.get1 r).toReal * (R.acc.get2 r e).toReal| +
          eps2 * lpSum valid (fun c => R.p16.get2 r c) (fun c => R.v.get2 c e) +
          eps1 * (1 + v1Sum valid (fun c => R.v.get2 c e)) := by
    intro e he
    obtain ⟨haF, hab, -⟩ := hcols e he
    have hroom := pg_liftA (valid := valid) (p16 := fun c => R.p16.get2 r c)
      (v := fun c => R.v.get2 c e) hp16z hp2 (hvbl e he)
    have hz : DType.val? .f32 ((tl.zeros [d.BM, d.BD] .f32).get2 r e) = some 0 := by
      show DType.val? .f32 (FloatV.ofPat 0) = some 0
      exact DType.val?_zero_f32
    obtain ⟨hDf, hD⟩ := dot_zero_real hP R.hdot2 r e hz (fun k => hp16F k)
      (fun k => (pg_val_bf16 (hv k e)).1) (by
        unfold liftSum
        exact hroom.trans (by norm_num))
    exact col_pref hP (valid := valid) (p16 := fun c => R.p16.get2 r c) (v := fun c => R.v.get2 c e)
      hp16z hp2 (hvbl e he) haF hab hreF (hT.scale_bd.trans (by norm_num)) hDf hD (R.hmul2 (r, e, ()))
      (R.hmul3 (r, e, ())) (R.hadd (r, e, ()))
  have hn4 : (n : ℝ) + 1 ≤ 4097 := by exact_mod_cast (by omega : n + 1 ≤ 4097)
  have hg0 : (0 : ℝ) ≤ n * g1p := by unfold g1p; positivity
  have hγ1 : g2 + (n : ℝ) * g1p + g1p ≤ 1 / 2 ^ 3 := by
    unfold g1p g2; nlinarith
  have hγ0 : g2 + (n : ℝ) * g1p ≤ 1 / 2 ^ 3 := by
    unfold g1p at hg0 hγ1 ⊢; unfold g2 at *; nlinarith
  obtain ⟨he1, he2, he3, he4, he5⟩ := epsP1_ok
  have key := RowFl.trip (sc := rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + r))
    (vcol := rowVc d init (bOf d i) (hOf d i)) (D := d.D) (BD := d.BD)
    (dl := dlt d.BD (rowX d init i r)) (Xp := rowX d init i r + dlt d.BD (rowX d init i r))
    (t := n) (γ := g2 + n * g1p) (g1 := g1p) (ε1 := epsP1)
    (S := seenP d init (bOf d i) (mOf d i * d.BM + r) n) (base := BN * n) (valid := valid)
    (deno := R.deno.get1 r) (l' := R.deno'.get1 r) (re := R.re.get1 r)
    (nmx := R.n_e_max.get1 r) (acc := fun e => R.acc.get2 r e) (acc' := fun e => R.acc'.get2 r e)
    (p16 := fun c => R.p16.get2 r c) (vT := fun e c => R.v.get2 c e) (δ' := δ') (f := f)
    (qkr := fun c => (tl.where R.FM R.qk1 (bc ninf32)).get2 r c |>.toReal)
    (pr := fun c => (R.p.get2 r c).toReal)
    (by have := hX0; linarith) hXp2 hdl hts hcardU hdisj hrow hcols hT hround hl'F
    (fun h => ⟨hnmxF, hnn h⟩) hne hvl
    (fun e he j hj => by
      rw [← hset] at hj
      exact hvK j (Finset.mem_filter.mp hj).1 e he)
    hacc ⟨he1, he2⟩ ⟨he3, he4, he5⟩ (by unfold g2; have := hg0; linarith) hγ0 hγ1 hdF hdV
  have ecast : g2 + ((n + 1 : Nat) : ℝ) * g1p = g2 + n * g1p + g1p := by push_cast; ring
  rw [ecast, hset]
  exact key

end
end Claims.ExtendAttn
